"""
experiments/frontier_scaling/navitrit_scale_model.py: Scalable NaviTrit Architecture.

Gate 15 (Track D-4): Frontier Scaling supporting both 10M and 100M parameter scales.

Key Components:
1. Flexible Model Scaling (10M and 100M presets):
     - 10M  : d_model=192, L=4, n_heads=3, d_ff=512   (~11.7M params, < 1.0 GB VRAM)
     - 100M : d_model=768, L=12, n_heads=12, d_ff=2048 (~134.1M params, ~3.5 GB VRAM)
2. Stationary Language Graph G = (V, E):
     Nodes 2l     : Attn_l (Hop 1) for l in 0..L-1
     Nodes 2l + 1 : FFN_l  (Hop 2) for l in 0..L-1
     Node 2L      : REASONING_CORE (Latent Recurrent Denoiser with FPF)
     Node 2L + 1  : EXIT (Dynamic Termination & Logit Readout)
3. Hop-Conditioned Tile Modulation (FiLM):
     (gamma_t, beta_t) = MLP_hop(Embedding_hop(t)) in R^{2 x d}
     h_tilde^{(t)} = (1 + gamma_t) * LN(h^{(t)}) + beta_t
     Adapts stationary ternary weights to temporal processing depth.
4. Dedicated Recurrent Reasoning Core (v_reason):
     Weight-tied continuous attractor denoiser executing internal relaxation:
     s^{(k+1)} = s^{(k)} + dt * D_reason(s^{(k)} | h^{(t)})
     with certified contraction stopping ||s^{(k+1)} - s^{(k)}||_inf < eps_reason.
5. Hidden-State Contraction Regularization (L_state_fpf):
     Directly penalizes token representation divergence on self-loops.
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.bitlinear import BitLinear
from experiments.bitroute_model import (
    BitRouteConfig,
    BitRouteRMSNorm,
    BitRouteAttention,
    BitRouteFFN,
)

# Transition type identifiers
TRANSITION_FORWARD = "FORWARD_HOP"
TRANSITION_BACKWARD = "BACKWARD_HOP"
TRANSITION_SELF_LOOP = "SELF_LOOP"
TRANSITION_INTRA_LAYER = "INTRA_LAYER_HOP"
TRANSITION_REASONING = "REASONING_CORE"
TRANSITION_EXIT = "EXIT"


@dataclass
class NaviTritScaleConfig:
    """Configuration for Scalable NaviTrit model supporting 10M and 100M presets."""
    model_size: str = "10m"
    vocab_size: int = 50257
    hidden_size: int = 192
    intermediate_size: int = 512
    num_layers: int = 4                # 4 Attn + 4 FFN = 8 language tiles
    num_attention_heads: int = 3
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    block_size: int = 256
    max_hops: int = 6                  # Max trajectory length T_max
    d_nav_route: int = 128             # Routing state vector dimension
    flow_dt: float = 0.5               # Velocity integration step
    eps_contraction: float = 1e-3      # Halting threshold for router
    
    # Hop Modulation
    use_hop_modulation: bool = True    # Enable FiLM hop affine conditioning
    d_hop_embed: int = 64              # Embedding dimension for hop index
    
    # Recurrent Reasoning Core
    use_reasoning_core: bool = True    # Dedicated recurrent latent core (Node 2L)
    reasoning_sub_steps: int = 3       # Internal recurrent steps per core visitation
    reasoning_contraction_eps: float = 1e-2 # Internal early exit threshold
    
    # Regularization
    min_attn_ratio: float = 0.40       # Minimum total Attention visitation budget
    lambda_attn_div: float = 0.50      # Base penalty for starving Attention tiles (at 10M reference)
    lambda_layer_entropy: float = 0.20 # Base penalty for layer collapse (at 10M reference)
    lambda_cohere: float = 0.10        # Supervised coherence loss weight
    lambda_hop: float = 0.02           # Step budget penalty
    lambda_fpf: float = 0.01           # Internal reasoning core contraction loss
    lambda_state_fpf: float = 0.05     # Hidden-state contraction penalty on self-loops
    tau_cohere: float = 0.85           # Exit coherence threshold

    # Scale-Adaptive Generalization Framework
    use_scale_adaptive: bool = True    # Enable scale-adaptive regularization and anti-gravity shield
    d_ref: int = 192                   # Reference hidden dimension (NaviTrit-10M baseline)
    l_ref: int = 4                     # Reference layer count (NaviTrit-10M baseline)

    @property
    def scale_dim_ratio(self) -> float:
        """Ratio of hidden dimension to reference 10M baseline: R_dim = d / d_0."""
        return self.hidden_size / float(self.d_ref)

    @property
    def scale_depth_ratio(self) -> float:
        """Ratio of depth (layers) to reference 10M baseline: R_depth = L / L_0."""
        return self.num_layers / float(self.l_ref)

    @property
    def scale_adaptive_lambda_attn_div(self) -> float:
        """Dimension & depth scaled attention diversity weight:
        lambda_attn(d, L) = lambda_0 * sqrt(R_dim) * sqrt(R_depth)
        Balances the O(sqrt(d)) growth of ||nabla_h L_CE|| against the dimensionless diversity penalty.
        """
        return self.lambda_attn_div * math.sqrt(self.scale_dim_ratio) * math.sqrt(self.scale_depth_ratio)

    @property
    def scale_adaptive_lambda_entropy(self) -> float:
        """Depth scaled layer entropy weight."""
        return self.lambda_layer_entropy * math.sqrt(self.scale_depth_ratio)

    @property
    def scale_adaptive_ffn_damping(self) -> float:
        """Topological Anti-Gravity Shield:
        Logit bias damping consecutive FFN transitions to counteract the steep FFN representational basin:
        alpha_damp(d) = ln(1 + R_dim)
        """
        return math.log(1.0 + self.scale_dim_ratio)


def get_scale_config(model_size: str = "10m", max_hops: int = 6, use_scale_adaptive: bool = True) -> NaviTritScaleConfig:
    """Factory helper to return verified configurations for 10M or 100M scale."""
    if model_size == "10m":
        return NaviTritScaleConfig(
            model_size="10m",
            vocab_size=50257,
            hidden_size=192,
            intermediate_size=512,
            num_layers=4,
            num_attention_heads=3,
            max_hops=max_hops,
            d_nav_route=64,
            d_hop_embed=32,
            use_scale_adaptive=use_scale_adaptive,
        )
    elif model_size == "100m":
        return NaviTritScaleConfig(
            model_size="100m",
            vocab_size=50257,
            hidden_size=768,
            intermediate_size=2048,
            num_layers=12,
            num_attention_heads=12,
            max_hops=max_hops,
            d_nav_route=128,
            d_hop_embed=64,
            use_scale_adaptive=use_scale_adaptive,
        )
    else:
        raise ValueError(f"Unknown model_size: {model_size}. Choose '10m' or '100m'.")


class HopConditionedModulation(nn.Module):
    """Hop-Conditioned Tile Modulation (FiLM).
    
    Computes per-channel affine parameters (gamma_t, beta_t) based on discrete hop step t.
    h_tilde = (1 + gamma_t) * LN(h) + beta_t
    """
    def __init__(self, max_hops: int, d_hop_embed: int, hidden_size: int):
        super().__init__()
        self.max_hops = max_hops
        self.hidden_size = hidden_size
        self.hop_embed = nn.Embedding(max_hops + 1, d_hop_embed)
        self.mlp = nn.Sequential(
            nn.Linear(d_hop_embed, d_hop_embed * 2),
            nn.SiLU(),
            nn.Linear(d_hop_embed * 2, hidden_size * 2),
        )
        # Initialize output linear layer to near-zero so modulation starts as near-identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t_idx = min(t, self.max_hops)
        t_tensor = torch.tensor([t_idx], device=device, dtype=torch.long)
        embed = self.hop_embed(t_tensor)           # (1, d_hop_embed)
        mod = self.mlp(embed)                       # (1, 2 * hidden_size)
        gamma, beta = mod.chunk(2, dim=-1)         # (1, hidden_size) each
        return gamma, beta


class RecurrentReasoningCore(nn.Module):
    """Dedicated Latent Recurrent Reasoning Core (Node 2L).
    
    Weight-tied BitLinear SwiGLU denoiser operating in latent representation space.
    Performs internal contraction iterations towards a stable solution attractor.
    """
    def __init__(self, config: NaviTritScaleConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.sub_steps = config.reasoning_sub_steps
        self.eps_exit = config.reasoning_contraction_eps

        # Norms
        self.ln_in = BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ln_sub = BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps)

        # Weight-tied BitLinear SwiGLU denoiser
        self.gate_proj = BitLinear(config.hidden_size, config.intermediate_size, block_size=config.block_size)
        self.up_proj = BitLinear(config.hidden_size, config.intermediate_size, block_size=config.block_size)
        self.down_proj = BitLinear(config.intermediate_size, config.hidden_size, block_size=config.block_size)

        # Attractor carry state projection
        self.state_proj = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.eye_(self.state_proj.weight)
        nn.init.zeros_(self.state_proj.bias)

    def step(self, s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """One internal recurrent velocity step: ds/dt = D(s | c)."""
        x_norm = self.ln_sub(s + c)
        gate = F.silu(self.gate_proj(x_norm))
        up = self.up_proj(x_norm)
        v = self.down_proj(gate * up)
        return self.state_proj(v)

    def forward(self, h_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Runs up to `sub_steps` internal iterations with early contraction halting."""
        c = self.ln_in(h_in)
        s = torch.zeros_like(h_in)
        sub_loss_fpf = torch.tensor(0.0, device=h_in.device)
        steps_taken = 0

        for k in range(self.sub_steps):
            v_k = self.step(s, c)
            s_next = s + 0.5 * v_k
            step_disp = s_next - s
            sub_loss_fpf = sub_loss_fpf + torch.mean(step_disp ** 2)
            steps_taken += 1

            if not self.training:
                max_delta = torch.max(torch.abs(step_disp)).item()
                if max_delta < self.eps_exit:
                    s = s_next
                    break
            s = s_next

        h_out = h_in + s
        return h_out, sub_loss_fpf, steps_taken


class EnhancedFlowNavigationController(nn.Module):
    """Continuous Flow Navigation Controller across all 2L + 2 nodes."""
    def __init__(self, config: NaviTritScaleConfig, num_nodes: int):
        super().__init__()
        self.config = config
        self.num_nodes = num_nodes
        self.d_route = config.d_nav_route

        # Context projection
        self.ctx_proj = nn.Linear(config.hidden_size, self.d_route)
        self.node_embed = nn.Embedding(num_nodes + 1, self.d_route)

        # Velocity network v_phi(r | h, node_prev)
        self.v_net = nn.Sequential(
            nn.Linear(self.d_route * 3, self.d_route * 2),
            nn.SiLU(),
            nn.Linear(self.d_route * 2, self.d_route),
        )

        # Transition logit projection
        self.node_head = nn.Linear(self.d_route, num_nodes)
        self.ln_route = nn.LayerNorm(self.d_route)

    def forward(
        self,
        h_token: torch.Tensor,
        prev_node: int,
        r_prev: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = h_token.size(0)
        h_ctx = self.ctx_proj(h_token)  # (B, d_route)

        prev_node_tensor = torch.full((B,), prev_node, dtype=torch.long, device=h_token.device)
        node_emb = self.node_embed(prev_node_tensor)  # (B, d_route)

        if r_prev is None:
            r = torch.tanh(h_ctx + node_emb)
        else:
            r = r_prev

        # One velocity integration step
        v_in = torch.cat([r, h_ctx, node_emb], dim=-1)
        dr = self.v_net(v_in)
        r_next = self.ln_route(r + self.config.flow_dt * dr)

        logits = self.node_head(r_next)  # (B, num_nodes)

        # Topological Anti-Gravity Shield:
        # If previous node was an FFN tile (odd index < 2*num_layers),
        # apply scale-adaptive logit damping alpha_damp to candidate FFN tiles
        if self.config.use_scale_adaptive and prev_node < 2 * self.config.num_layers:
            is_prev_ffn = (prev_node % 2 == 1)
            if is_prev_ffn:
                alpha = self.config.scale_adaptive_ffn_damping
                ffn_indices = [2 * l + 1 for l in range(self.config.num_layers)]
                logits = logits.clone()
                logits[:, ffn_indices] = logits[:, ffn_indices] - alpha

        if self.training:
            action_soft = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        else:
            idx = torch.argmax(logits, dim=-1)
            action_soft = F.one_hot(idx, num_classes=self.num_nodes).float()

        return action_soft, logits, r_next


class NaviTritScaleForCausalLM(nn.Module):
    """Complete Scalable NaviTrit Causal Language Model."""
    def __init__(self, config: NaviTritScaleConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.max_hops = config.max_hops

        # Token & position embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Stationary Attention tiles (Nodes 2l)
        # Adapt BitRouteConfig for component construction
        sub_cfg = BitRouteConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            block_size=config.block_size,
        )
        self.attn_tiles = nn.ModuleList([BitRouteAttention(sub_cfg) for _ in range(config.num_layers)])
        self.attn_norms = nn.ModuleList([BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps) for _ in range(config.num_layers)])

        # Stationary FFN tiles (Nodes 2l + 1)
        self.ffn_tiles = nn.ModuleList([BitRouteFFN(sub_cfg) for _ in range(config.num_layers)])
        self.ffn_norms = nn.ModuleList([BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps) for _ in range(config.num_layers)])

        # Graph node mapping:
        # 0..2L-1 : Language tiles
        # 2L      : Reasoning Core
        # 2L+1    : Exit
        self.node_reasoning = 2 * config.num_layers
        self.node_exit = 2 * config.num_layers + 1
        self.num_nodes = 2 * config.num_layers + 2

        # Hop-Conditioned Tile Modulation (FiLM)
        if config.use_hop_modulation:
            self.hop_mod = HopConditionedModulation(config.max_hops, config.d_hop_embed, config.hidden_size)
        else:
            self.hop_mod = None

        # Dedicated Recurrent Reasoning Core
        if config.use_reasoning_core:
            self.reasoning_core = RecurrentReasoningCore(config)
        else:
            self.reasoning_core = None

        # Enhanced Flow Navigation Controller
        self.controller = EnhancedFlowNavigationController(config, self.num_nodes)

        # Readout head
        self.final_norm = BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Weight tying to eliminate redundant embedding parameters
        self.lm_head.weight = self.embed_tokens.weight

        # Coherence certification head
        self.coherence_head = nn.Sequential(
            BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total

    def execute_tile(self, node_idx: int, h: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes the specific stationary tile indexed by node_idx at hop t."""
        device = h.device
        loss_fpf = torch.tensor(0.0, device=device)

        # Apply Hop Modulation if active
        if self.hop_mod is not None:
            gamma_t, beta_t = self.hop_mod(t, device)
        else:
            gamma_t, beta_t = None, None

        if node_idx < 2 * self.num_layers:
            layer_idx = node_idx // 2
            is_attn = (node_idx % 2 == 0)

            if is_attn:
                norm_mod = self.attn_norms[layer_idx](h)
                if gamma_t is not None:
                    norm_mod = (1.0 + gamma_t) * norm_mod + beta_t
                attn_out = self.attn_tiles[layer_idx](norm_mod)
                h_next = h + attn_out
            else:
                norm_mod = self.ffn_norms[layer_idx](h)
                if gamma_t is not None:
                    norm_mod = (1.0 + gamma_t) * norm_mod + beta_t
                ffn_out = self.ffn_tiles[layer_idx](norm_mod)
                h_next = h + ffn_out
            return h_next, loss_fpf

        elif node_idx == self.node_reasoning:
            if self.reasoning_core is not None:
                h_next, core_fpf, _ = self.reasoning_core(h)
                return h_next, core_fpf
            else:
                return h, loss_fpf

        else:
            # Exit node: pass-through
            return h, loss_fpf

    def forward(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, S = input_ids.size()
        device = input_ids.device

        # Embeddings
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        h = self.embed_tokens(input_ids) + self.embed_positions(pos)

        # State tracking
        r_state = None
        prev_node = self.num_nodes  # Start node
        total_loss_fpf = torch.tensor(0.0, device=device)
        total_loss_state_fpf = torch.tensor(0.0, device=device)

        history_nodes = []
        history_actions = []
        history_coherence = []
        early_exit_triggered = False

        for t in range(self.max_hops):
            # Token context summary for routing
            h_pool = h.mean(dim=1)  # (B, hidden_size)

            action_soft, logits, r_state = self.controller(
                h_token=h_pool,
                prev_node=prev_node,
                r_prev=r_state,
                temperature=temperature,
                hard=True,
            )

            chosen_node = int(torch.argmax(action_soft.mean(dim=0)).item())
            history_nodes.append(chosen_node)
            history_actions.append(action_soft)

            # Check early exit
            if chosen_node == self.node_exit:
                early_exit_triggered = True
                break

            # Execute tile
            h_next, tile_fpf = self.execute_tile(chosen_node, h, t)
            total_loss_fpf = total_loss_fpf + tile_fpf

            # State contraction loss on self-loops
            if chosen_node == prev_node and prev_node < 2 * self.num_layers:
                state_disp = torch.mean((h_next - h) ** 2)
                total_loss_state_fpf = total_loss_state_fpf + state_disp

            # Monitor coherence
            coherence = self.coherence_head(h_next.mean(dim=1))
            history_coherence.append(coherence.mean().item())

            if not self.training and coherence.mean().item() > self.config.tau_cohere:
                h = h_next
                early_exit_triggered = True
                break

            h = h_next
            prev_node = chosen_node

        # Readout head
        h_norm = self.final_norm(h)
        logits = self.lm_head(h_norm)

        # Compute trajectory regularization losses
        actions_cat = torch.stack(history_actions, dim=1) if len(history_actions) > 0 else torch.zeros(B, 1, self.num_nodes, device=device)
        
        # Attention diversity with scale-adaptive weighting
        eff_lambda_attn = (
            self.config.scale_adaptive_lambda_attn_div
            if self.config.use_scale_adaptive
            else self.config.lambda_attn_div
        )
        attn_indices = [2 * l for l in range(self.num_layers)]
        attn_visitation = actions_cat[..., attn_indices].sum(dim=-1).mean()
        loss_attn_div = eff_lambda_attn * F.relu(self.config.min_attn_ratio - attn_visitation) ** 2

        # Scale-invariant Normalized Graph Entropy
        eff_lambda_entropy = (
            self.config.scale_adaptive_lambda_entropy
            if self.config.use_scale_adaptive
            else self.config.lambda_layer_entropy
        )
        layer_prob = []
        for l in range(self.num_layers):
            p_l = actions_cat[..., [2 * l, 2 * l + 1]].sum(dim=-1).mean()
            layer_prob.append(p_l + 1e-6)
        layer_prob_tensor = torch.stack(layer_prob)
        norm_factor = math.log(max(2, self.num_layers))
        norm_entropy = (layer_prob_tensor * torch.log(layer_prob_tensor)).sum() / norm_factor
        loss_entropy = eff_lambda_entropy * norm_entropy

        loss_hop_budget = self.config.lambda_hop * float(len(history_nodes))

        out = {
            "logits": logits,
            "loss_fpf": total_loss_fpf,
            "loss_state_fpf": total_loss_state_fpf * self.config.lambda_state_fpf,
            "loss_attn_div": loss_attn_div,
            "loss_entropy": loss_entropy,
            "loss_hop_budget": torch.tensor(loss_hop_budget, device=device),
            "hops_taken": len(history_nodes),
            "early_exit": early_exit_triggered,
            "trajectory": history_nodes,
            "attn_ratio": float(attn_visitation.item()),
        }
        return out
