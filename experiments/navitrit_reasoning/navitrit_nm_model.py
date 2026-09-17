"""
experiments/navitrit_reasoning/navitrit_nm_model.py: NaviTrit-NM Architecture.

Gate 14 (Track D-3): Non-Monotonic Token Navigation with Adaptive Weight Modulation
and Integrated Recurrent Reasoning Core.

Key Innovations:
1. Stationary Language Graph G = (V, E):
     Nodes 2l     : Attn_l (Hop 1) for l in 0..L-1
     Nodes 2l + 1 : FFN_l  (Hop 2) for l in 0..L-1
     Node 2L      : REASONING_CORE (Latent Flow Denoiser with FPF)
     Node 2L + 1  : EXIT (Dynamic Termination & Logit Emission)
2. Hop-Conditioned Tile Modulation (FiLM / Hop Affine Modulation):
     (gamma_t, beta_t) = MLP_hop(Embedding_hop(t)) in R^{2 x d}
     h_tilde^{(t)} = (1 + gamma_t) * LN(h^{(t)}) + beta_t
     Enables stationary ternary weights to adopt stage-specific processing modes.
3. Dedicated Recurrent Reasoning Core (v_reason):
     Weight-tied continuous attractor denoiser executing internal relaxation:
     s^{(k+1)} = s^{(k)} + dt * D_reason(s^{(k)} | h^{(t)})
     with certified contraction stopping ||s^{(k+1)} - s^{(k)}||_inf < eps_reason.
4. Hidden-State Contraction Regularization (L_state_fpf):
     Directly penalizes token representation divergence on self-loops:
     L_state_fpf = ||h^{(t+1)} - h^{(t)}||_2^2 * 1_{v_{t+1} == v_t}.
5. Enhanced Flow Navigation Controller:
     Routes tokens dynamically across all 2L + 2 nodes.
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


# Transition types
TRANSITION_FORWARD = "FORWARD_HOP"
TRANSITION_BACKWARD = "BACKWARD_HOP"
TRANSITION_SELF_LOOP = "SELF_LOOP"
TRANSITION_INTRA_LAYER = "INTRA_LAYER_HOP"
TRANSITION_REASONING = "REASONING_CORE"
TRANSITION_EXIT = "EXIT"


@dataclass
class NaviTritNMConfig:
    """Configuration for NaviTrit-NM model."""
    vocab_size: int = 50257
    hidden_size: int = 384
    intermediate_size: int = 1024
    num_layers: int = 3               # 3 Attn + 3 FFN = 6 language tiles
    num_attention_heads: int = 6
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    block_size: int = 256
    max_hops: int = 6                 # Max trajectory length T_max
    d_nav_route: int = 128            # Routing state vector dimension
    flow_dt: float = 0.5              # Velocity integration step
    eps_contraction: float = 1e-3     # Halting threshold for router
    
    # Hop Modulation
    use_hop_modulation: bool = True   # Enable FiLM hop affine conditioning
    d_hop_embed: int = 64
    
    # Dedicated Reasoning Core
    use_reasoning_core: bool = True   # Enable dedicated latent reasoning node
    reasoning_sub_steps: int = 3      # Internal recurrent steps in v_reason
    reasoning_dt: float = 0.5         # Step size for internal relaxation
    eps_reason_exit: float = 0.01     # Early contraction threshold in reasoning core
    
    # Gumbel Softmax & Policy
    gumbel_tau: float = 1.0           # Straight-through temperature
    gumbel_hard: bool = False         # Soft relaxation during training, discrete in eval
    
    # Loss Penalties
    lambda_hop: float = 0.02          # Economic hop cost penalty
    lambda_fpf: float = 0.01          # Attractor velocity penalty on router
    lambda_state_fpf: float = 0.05    # Contraction penalty on hidden-state self-loops
    lambda_attn_div: float = 0.5      # Attention diversity penalty weight
    lambda_layer_entropy: float = 0.2 # Layer hierarchy entropy penalty weight
    lambda_cohere: float = 0.1        # Semantic coherence prediction loss weight
    min_attn_ratio: float = 0.40      # Minimum desired attention utilization ratio
    tau_cohere: float = 0.85          # Coherence threshold for certified early exit
    min_hops_exit: int = 2            # Minimum hops before early exit allowed
    
    ternary: bool = True              # Native ternary BitLinear weights
    tie_word_embeddings: bool = False


@dataclass
class NaviTritNMTrajectoryStep:
    """Telemetry recorded for a single hop in the trajectory."""
    hop_idx: int
    selected_node: int                # 0 .. 2L + 1
    node_name: str                    # "ATTN_l", "FFN_l", "REASONING_CORE", "EXIT"
    transition_type: str              # FORWARD, BACKWARD, SELF_LOOP, REASONING, EXIT
    step_norm: float                  # ||h_{t+1} - h_t||_inf
    logits: torch.Tensor              # [B, num_nodes]
    coherence_score: float = 0.0      # Semantic coherence prediction c_t in [0, 1]
    reasoning_steps_taken: int = 0    # Sub-steps executed if REASONING_CORE


@dataclass
class NaviTritNMOutput:
    """Output and execution telemetry from NaviTrit-NM forward pass."""
    logits: torch.Tensor
    total_hops_taken: int
    forward_hops_count: int
    backward_hops_count: int
    self_loops_count: int
    reasoning_hops_count: int
    early_exit_taken: bool
    exit_hop: int
    trajectory: List[NaviTritNMTrajectoryStep]
    loss_hop: torch.Tensor            # Economic cost penalty
    loss_fpf: torch.Tensor            # Router contraction penalty
    loss_state_fpf: torch.Tensor      # Hidden state contraction penalty on self-loops
    loss_attn_div: torch.Tensor       # Attention diversity hinge penalty
    loss_entropy: torch.Tensor        # Layer hierarchy entropy penalty
    loss_cohere: torch.Tensor         # Coherence prediction supervision loss
    terminal_coherence: float         # Certified coherence score at termination


class HopConditionedModulation(nn.Module):
    """
    FiLM-style Hop Affine Modulation:
    Maps hop step index t in {0, ..., T_max - 1} to per-channel affine parameters
    gamma_t, beta_t in R^d such that:
        h_tilde = (1 + gamma_t) * LN(h) + beta_t
    Allows stationary ternary weights to adopt stage-specific representations.
    """
    def __init__(self, max_hops: int, d_model: int, d_embed: int = 64):
        super().__init__()
        self.hop_embedding = nn.Embedding(max_hops + 1, d_embed)
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.SiLU(),
            nn.Linear(d_embed, 2 * d_model),
        )
        # Initialize to near-identity modulation (gamma=0, beta=0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, hop_idx: int) -> torch.Tensor:
        # x: [B, S, d]
        B = x.shape[0]
        device = x.device
        idx_tensor = torch.tensor([min(hop_idx, self.hop_embedding.num_embeddings - 1)], device=device)
        emb = self.hop_embedding(idx_tensor)  # [1, d_embed]
        params = self.mlp(emb)                # [1, 2 * d_model]
        gamma, beta = params.chunk(2, dim=-1) # each [1, d_model]
        gamma = gamma.unsqueeze(1)            # [1, 1, d_model]
        beta = beta.unsqueeze(1)              # [1, 1, d_model]
        return (1.0 + gamma) * x + beta


class RecurrentReasoningCore(nn.Module):
    """
    Dedicated Latent Recurrent Reasoning Core (Node v_reason).
    A weight-tied continuous flow denoiser built from BitLinear ternary weights.
    Executes internal relaxation steps:
        s^{(k+1)} = s^{(k)} + dt * D(s^{(k)} | context)
    with certified contraction stopping ||s^{(k+1)} - s^{(k)}||_inf < eps.
    """
    def __init__(self, config: NaviTritNMConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        d_ff = config.intermediate_size
        
        self.input_norm = BitRouteRMSNorm(d, eps=config.rms_norm_eps)
        self.state_norm = BitRouteRMSNorm(d, eps=config.rms_norm_eps)
        
        # Ternary weight-tied denoiser projection
        self.gate_proj = BitLinear(d, d_ff, bias=False, block_size=config.block_size)
        self.up_proj = BitLinear(d, d_ff, bias=False, block_size=config.block_size)
        self.down_proj = BitLinear(d_ff, d, bias=False, block_size=config.block_size)
        
        # Context mixer
        self.context_proj = BitLinear(d, d, bias=False, block_size=config.block_size)
        self.out_norm = BitRouteRMSNorm(d, eps=config.rms_norm_eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Input: h in R^{[B, S, d]}
        Returns:
            h_out: Updated representation after recurrent reasoning
            steps_taken: Number of internal relaxation steps executed
            contraction_delta: Final state delta norm
        """
        B, S, d = h.shape
        c = self.context_proj(self.input_norm(h)) # [B, S, d]
        s = h.clone() # Initial carry state
        
        steps_taken = 0
        final_delta = torch.tensor(0.0, device=h.device)
        
        for k in range(self.config.reasoning_sub_steps):
            s_norm = self.state_norm(s) + c
            # SwiGLU denoiser step
            gate = F.silu(self.gate_proj(s_norm))
            up = self.up_proj(s_norm)
            velocity = self.down_proj(gate * up)
            
            delta_s = self.config.reasoning_dt * velocity
            s = s + delta_s
            steps_taken = k + 1
            
            delta_inf = torch.max(torch.abs(delta_s))
            final_delta = delta_inf
            if delta_inf < self.config.eps_reason_exit:
                break
                
        h_out = self.out_norm(s)
        return h_out, steps_taken, final_delta


class EnhancedFlowNavigationController(nn.Module):
    """
    Continuous Flow Navigation Controller routing across all 2L + 2 nodes:
    Nodes 0 .. 2L - 1 : Language Attention & FFN tiles
    Node 2L           : Dedicated Recurrent Reasoning Core
    Node 2L + 1       : EXIT (LM Head)
    """
    def __init__(self, config: NaviTritNMConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        d_route = config.d_nav_route
        self.num_nodes = 2 * config.num_layers + 2  # 2L language + 1 reasoning + 1 exit
        self.reasoning_node_idx = 2 * config.num_layers
        self.exit_node_idx = self.num_nodes - 1
        
        # Conditioning encoder: [h_mean (d), one_hot(prev_node) (num_nodes), normalized_time (1)]
        cond_in_dim = d + self.num_nodes + 1
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_in_dim, d_route),
            nn.SiLU(),
            nn.Linear(d_route, d_route),
        )
        
        # Velocity field v_phi(r | c)
        self.velocity_net = nn.Sequential(
            nn.Linear(d_route * 2, d_route),
            nn.SiLU(),
            nn.Linear(d_route, d_route),
        )
        
        # Destination projection head
        self.dest_head = nn.Linear(d_route, self.num_nodes)
        
        # Coherence certification head c_t in [0, 1]
        self.coherence_head = nn.Sequential(
            nn.Linear(d_route, d_route // 2),
            nn.SiLU(),
            nn.Linear(d_route // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h: torch.Tensor,
        r_t: torch.Tensor,
        prev_node: int,
        hop_idx: int,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Returns:
            r_{t+1}: Updated routing state
            node_logits: [B, num_nodes]
            action_mask: [B, num_nodes] (discrete in eval, soft/hard in training)
            fpf_norm: Contraction velocity norm
            coherence_score: Mean c_t in [0, 1]
        """
        B, S, d = h.shape
        device = h.device
        
        h_pool = h.mean(dim=1)  # [B, d]
        prev_node_oh = F.one_hot(torch.full((B,), prev_node, dtype=torch.long, device=device), self.num_nodes).float()
        time_norm = torch.full((B, 1), float(hop_idx) / max(1, self.config.max_hops), device=device)
        
        cond_input = torch.cat([h_pool, prev_node_oh, time_norm], dim=-1)
        c_t = self.cond_encoder(cond_input)  # [B, d_route]
        
        # Flow velocity relaxation
        flow_input = torch.cat([r_t, c_t], dim=-1)
        v = self.velocity_net(flow_input)
        r_next = r_t + self.config.flow_dt * v
        fpf_norm = (v ** 2).mean()
        
        # Destination logits & coherence
        node_logits = self.dest_head(r_next)  # [B, num_nodes]
        coherence_pred = self.coherence_head(r_next).squeeze(-1)  # [B]
        mean_coherence = coherence_pred.mean().item()
        
        # Mask emission
        if training:
            action_mask = F.gumbel_softmax(
                node_logits,
                tau=self.config.gumbel_tau,
                hard=self.config.gumbel_hard,
                dim=-1,
            )
        else:
            best_node = node_logits.argmax(dim=-1)
            action_mask = F.one_hot(best_node, self.num_nodes).float()
            
        return r_next, node_logits, action_mask, fpf_norm, mean_coherence


class NaviTritNMForCausalLM(nn.Module):
    """
    Complete NaviTrit-NM Architecture:
    - 2L Stationary Language Tiles (BitRouteAttention + BitRouteFFN)
    - HopConditionedModulation for stage-aware feature conditioning
    - RecurrentReasoningCore (Latent Attractor Node)
    - EnhancedFlowNavigationController
    """
    def __init__(self, config: NaviTritNMConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Stationary Language Tiles (2L tiles: L Attention, L FFN)
        self.num_layers = config.num_layers
        self.attn_tiles = nn.ModuleList()
        self.ffn_tiles = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        
        for l in range(config.num_layers):
            block_cfg = BitRouteConfig(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_layers,
                num_attention_heads=config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                block_size=config.block_size,
            )
            self.attn_norms.append(BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps))
            self.attn_tiles.append(BitRouteAttention(block_cfg))
            self.ffn_norms.append(BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps))
            self.ffn_tiles.append(BitRouteFFN(block_cfg))
            
        # Hop-Conditioned Affine Modulation
        self.hop_modulation = HopConditionedModulation(
            max_hops=config.max_hops,
            d_model=config.hidden_size,
            d_embed=config.d_hop_embed,
        ) if config.use_hop_modulation else None
        
        # Dedicated Recurrent Reasoning Core (Node 2L)
        self.reasoning_core = RecurrentReasoningCore(config) if config.use_reasoning_core else None
        
        # Final Norm and Language Model Head
        self.final_norm = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
            
        # Flow Navigation Controller
        self.controller = EnhancedFlowNavigationController(config)

    def _execute_tile(self, node_idx: int, h: torch.Tensor, hop_idx: int) -> Tuple[torch.Tensor, int]:
        """
        Executes a single node in the graph:
        - Even nodes 0..2L-2: Attn_l
        - Odd nodes 1..2L-1 : FFN_l
        - Node 2L           : Reasoning Core
        - Node 2L+1         : EXIT (No-op in residual stream)
        """
        sub_steps = 0
        if node_idx < 2 * self.num_layers:
            layer_idx = node_idx // 2
            is_ffn = (node_idx % 2) == 1
            
            # Apply Hop-Conditioned Modulation if enabled
            if self.hop_modulation is not None:
                h_mod = self.hop_modulation(h, hop_idx)
            else:
                h_mod = h
                
            if not is_ffn:
                normed = self.attn_norms[layer_idx](h_mod)
                delta = self.attn_tiles[layer_idx](normed)
            else:
                normed = self.ffn_norms[layer_idx](h_mod)
                delta = self.ffn_tiles[layer_idx](normed)
            return h + delta, 0
            
        elif node_idx == 2 * self.num_layers and self.reasoning_core is not None:
            # Dedicated Recurrent Reasoning Core
            h_reasoned, sub_steps, _ = self.reasoning_core(h)
            return h_reasoned, sub_steps
            
        else:
            # EXIT node
            return h, 0

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        force_nodes: Optional[List[int]] = None,
    ) -> NaviTritNMOutput:
        """
        Forward pass with non-monotonic token graph navigation.
        """
        B, S = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        pos_ids = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        h = self.embed_tokens(input_ids) + self.embed_positions(pos_ids)
        h_init = h.clone()
        
        # Initial routing state
        r_t = torch.zeros(B, self.config.d_nav_route, device=device)
        prev_node = 0
        
        # Telemetry & Counters
        trajectory: List[NaviTritNMTrajectoryStep] = []
        forward_hops = 0
        backward_hops = 0
        self_loops = 0
        reasoning_hops = 0
        early_exit_taken = False
        exit_hop = self.config.max_hops
        
        total_loss_fpf = torch.tensor(0.0, device=device)
        total_loss_state_fpf = torch.tensor(0.0, device=device)
        total_coherence_loss = torch.tensor(0.0, device=device)
        
        visited_nodes: List[int] = []
        terminal_coherence = 0.0
        
        # Track intermediate states for coherence supervision
        hidden_states_per_hop: List[torch.Tensor] = []
        coherence_preds_per_hop: List[torch.Tensor] = []
        
        reasoning_node_idx = 2 * self.num_layers
        exit_node_idx = self.controller.exit_node_idx
        
        for hop_idx in range(self.config.max_hops):
            hidden_states_per_hop.append(h)
            
            # Query Controller
            r_next, node_logits, action_mask, fpf_norm, mean_cohere = self.controller(
                h=h,
                r_t=r_t,
                prev_node=prev_node,
                hop_idx=hop_idx,
                training=self.training,
            )
            total_loss_fpf = total_loss_fpf + fpf_norm
            r_t = r_next
            
            # Select destination node
            if force_nodes is not None and hop_idx < len(force_nodes):
                selected_node = force_nodes[hop_idx]
            else:
                if self.training and not self.config.gumbel_hard:
                    # In soft training, pick the highest weight for telemetry logging
                    selected_node = node_logits.argmax(dim=-1)[0].item()
                else:
                    selected_node = action_mask.argmax(dim=-1)[0].item()
                    
            # Certified Early Exit Check
            if (
                not self.training
                and hop_idx >= self.config.min_hops_exit
                and (selected_node == exit_node_idx or mean_cohere >= self.config.tau_cohere)
            ):
                early_exit_taken = True
                exit_hop = hop_idx
                terminal_coherence = mean_cohere
                trajectory.append(
                    NaviTritNMTrajectoryStep(
                        hop_idx=hop_idx,
                        selected_node=exit_node_idx,
                        node_name="EXIT",
                        transition_type=TRANSITION_EXIT,
                        step_norm=0.0,
                        logits=node_logits.detach(),
                        coherence_score=mean_cohere,
                    )
                )
                break
                
            # Classify transition
            if selected_node == exit_node_idx:
                trans_type = TRANSITION_EXIT
            elif selected_node == reasoning_node_idx:
                trans_type = TRANSITION_REASONING
                reasoning_hops += 1
            elif selected_node == prev_node:
                trans_type = TRANSITION_SELF_LOOP
                self_loops += 1
            elif selected_node > prev_node:
                trans_type = TRANSITION_FORWARD
                forward_hops += 1
            else:
                trans_type = TRANSITION_BACKWARD
                backward_hops += 1
                
            # Determine Node Name
            if selected_node == reasoning_node_idx:
                node_name = "REASONING_CORE"
            elif selected_node == exit_node_idx:
                node_name = "EXIT"
            else:
                layer_num = selected_node // 2
                sub_type = "FFN" if (selected_node % 2) == 1 else "ATTN"
                node_name = f"{sub_type}_{layer_num}"
                
            visited_nodes.append(selected_node)
            h_before = h
            
            # Execute transition
            if self.training and not self.config.gumbel_hard:
                # Differentiable soft mixture across nodes
                h_candidates = []
                for n_idx in range(self.controller.num_nodes):
                    h_cand, _ = self._execute_tile(n_idx, h, hop_idx)
                    h_candidates.append(h_cand)
                # Stack: [num_nodes, B, S, d] -> permute to [B, S, num_nodes, d]
                h_stack = torch.stack(h_candidates, dim=2)
                weights = action_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, num_nodes, 1]
                h_next = (h_stack * weights).sum(dim=2)
                sub_steps = 0
            else:
                h_next, sub_steps = self._execute_tile(selected_node, h, hop_idx)
                
            # Contraction Regularization on Self-Loops
            step_delta = h_next - h_before
            step_norm = torch.max(torch.abs(step_delta)).item()
            if trans_type == TRANSITION_SELF_LOOP:
                total_loss_state_fpf = total_loss_state_fpf + (step_delta ** 2).mean()
                
            h = h_next
            prev_node = selected_node
            terminal_coherence = mean_cohere
            
            trajectory.append(
                NaviTritNMTrajectoryStep(
                    hop_idx=hop_idx,
                    selected_node=selected_node,
                    node_name=node_name,
                    transition_type=trans_type,
                    step_norm=step_norm,
                    logits=node_logits.detach(),
                    coherence_score=mean_cohere,
                    reasoning_steps_taken=sub_steps,
                )
            )
            
        # Final LayerNorm & LM Head Logits
        h_normed = self.final_norm(h)
        logits = self.lm_head(h_normed)
        
        # Loss Terms Formulation
        # 1. Economic Hop Penalty
        hops_taken = len(trajectory)
        loss_hop = torch.tensor(float(hops_taken) * self.config.lambda_hop, device=device)
        
        # 2. Router FPF Loss
        loss_fpf = self.config.lambda_fpf * (total_loss_fpf / max(1, hops_taken))
        
        # 3. State FPF Contraction Loss on Self-Loops
        loss_state_fpf = self.config.lambda_state_fpf * total_loss_state_fpf
        
        # 4. Attention Diversity Regularization
        # Count attention visits (even nodes < 2L)
        attn_count = sum(1 for n in visited_nodes if n < 2 * self.num_layers and (n % 2) == 0)
        actual_attn_ratio = float(attn_count) / max(1, hops_taken)
        loss_attn_div = self.config.lambda_attn_div * F.relu(
            torch.tensor(self.config.min_attn_ratio - actual_attn_ratio, device=device)
        )
        
        # 5. Layer Hierarchy Entropy Regularization
        layer_counts = torch.zeros(self.num_layers, device=device)
        for n in visited_nodes:
            if n < 2 * self.num_layers:
                layer_counts[n // 2] += 1.0
        total_layer_visits = layer_counts.sum()
        if total_layer_visits > 0:
            probs = (layer_counts / total_layer_visits) + 1e-8
            entropy = -(probs * torch.log(probs)).sum()
            target_entropy = 0.70 * math.log(self.num_layers)
            loss_entropy = self.config.lambda_layer_entropy * F.relu(
                torch.tensor(target_entropy, device=device) - entropy
            )
        else:
            loss_entropy = torch.tensor(0.0, device=device)
            
        # 6. Coherence Supervised Loss
        if len(hidden_states_per_hop) > 0 and self.training:
            h_final_detached = h.detach()
            cos_sims = []
            for h_t in hidden_states_per_hop:
                sim = F.cosine_similarity(h_t.mean(dim=1), h_final_detached.mean(dim=1), dim=-1)
                cos_sims.append(sim.mean())
            target_cohere = torch.stack(cos_sims)
            cohere_preds = torch.tensor(
                [step.coherence_score for step in trajectory if step.selected_node != exit_node_idx],
                device=device,
            )
            if len(cohere_preds) == len(target_cohere):
                loss_cohere = self.config.lambda_cohere * F.mse_loss(cohere_preds, target_cohere)
            else:
                loss_cohere = torch.tensor(0.0, device=device)
        else:
            loss_cohere = torch.tensor(0.0, device=device)
            
        return NaviTritNMOutput(
            logits=logits,
            total_hops_taken=hops_taken,
            forward_hops_count=forward_hops,
            backward_hops_count=backward_hops,
            self_loops_count=self_loops,
            reasoning_hops_count=reasoning_hops,
            early_exit_taken=early_exit_taken,
            exit_hop=exit_hop,
            trajectory=trajectory,
            loss_hop=loss_hop,
            loss_fpf=loss_fpf,
            loss_state_fpf=loss_state_fpf,
            loss_attn_div=loss_attn_div,
            loss_entropy=loss_entropy,
            loss_cohere=loss_cohere,
            terminal_coherence=terminal_coherence,
        )
