"""
experiments/frontier_scaling/navitrit_dual_model.py: Hierarchical Dual-Controller NaviTrit Architecture.

Combines Strategy B (Outcome-Driven GRPO) and Strategy D (Hierarchical Flow Planner + Local Dynamic Navigation):
1. Global Flow Planner (Level 1):
     - Predicts dynamic compute budget T_budget in [2, max_hops]
     - Computes domain routing intent g_domain = sigmoid(W_gate * h_pool)
     - Projects node prior bias, prioritizing Node 24 (Reasoning Core) on arithmetic/reasoning tasks.
2. Local Dynamic Flow Controller (Level 2):
     - Continuous Neural ODE velocity field dr/dt = v_phi(r | h_t, prev_node, g_domain)
     - Dynamically traverses stationary Attention, FFN, and Reasoning tiles.
3. Persistent Entity Registers (Slot-Variable Memory):
     - M=4 latent register slots R in R^{M x d} bound via cross-attention to prevent entity drift.
4. Dedicated Recurrent Reasoning Core (Node 24):
     - Weight-tied BitLinear SwiGLU denoiser with internal contraction halting.
     - Detached FPF loss: internal relaxation does NOT penalize router policy logits.
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

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
from experiments.frontier_scaling.navitrit_scale_model import (
    NaviTritScaleConfig,
    HopConditionedModulation,
    RecurrentReasoningCore,
    get_scale_config,
)


class PersistentEntityRegisters(nn.Module):
    """
    Persistent Entity Registers (Slot-Variable Memory).
    
    Binds M=4 persistent latent slots R in R^{B x M x d} using slot attention over the context.
    Provides persistent working memory across all hops to prevent entity hallucination
    (e.g., 'Olivia' mutating into 'Mia', '15 apples' mutating into 'crayons').
    """
    def __init__(self, hidden_size: int, num_slots: int = 4, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Learnable slot query vectors
        self.slot_queries = nn.Parameter(torch.randn(num_slots, hidden_size) * 0.02)

        # Context key/value projections
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.slot_norm = BitRouteRMSNorm(hidden_size)

        # Slot-to-token cross-attention injection
        self.q_inject = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_inject = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_inject = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_inject = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_inject = nn.Parameter(torch.zeros(1))  # Starts at 0 for identity warm-start

    def bind_slots(self, h_ctx: torch.Tensor) -> torch.Tensor:
        """
        Binds prompt tokens into M persistent register slots.
        h_ctx: [B, S, d] -> returns slots R: [B, M, d]
        """
        B, S, d = h_ctx.shape
        # Queries: [B, M, d]
        q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)
        k = self.k_proj(h_ctx)  # [B, S, d]
        v = self.v_proj(h_ctx)  # [B, S, d]

        # Multi-head attention across context
        q_heads = q.view(B, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, M, D]
        k_heads = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)                # [B, H, S, D]
        v_heads = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)                # [B, H, S, D]

        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        slots_heads = torch.matmul(attn, v_heads)  # [B, H, M, D]
        slots = slots_heads.transpose(1, 2).contiguous().view(B, self.num_slots, d)
        return self.slot_norm(slots)

    def inject_registers(self, h_token: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """
        Injects slot memory into current token state at hop t.
        h_token: [B, S, d], slots: [B, M, d]
        """
        B, S, d = h_token.shape
        q = self.q_inject(h_token).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_inject(slots).view(B, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_inject(slots).view(B, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out_heads = torch.matmul(attn, v)
        out = out_heads.transpose(1, 2).contiguous().view(B, S, d)
        
        # Gated residual connection: starts at identity when gate_inject=0
        return h_token + torch.tanh(self.gate_inject) * self.out_inject(out)


class GlobalFlowPlanner(nn.Module):
    """
    Level 1: Global Flow Routing Planner.
    
    Given the sequence context summary h_pool:
    1. Computes domain routing intent g_domain in [0, 1] (0 = general language, 1 = math/reasoning).
    2. Predicts dynamic compute budget T_budget in [min_hops, max_hops].
    3. Projects node prior bias vector: boosts Node 24 (Reasoning Core) when math is detected.
    """
    def __init__(self, hidden_size: int, d_route: int, num_nodes: int, max_hops: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_route = d_route
        self.num_nodes = num_nodes
        self.max_hops = max_hops

        # Domain intent projection
        self.domain_mlp = nn.Sequential(
            nn.Linear(hidden_size, d_route),
            nn.SiLU(),
            nn.Linear(d_route, 1),
            nn.Sigmoid(),
        )

        # Dynamic budget projection (predicts discrete hop budget 2..max_hops)
        self.budget_head = nn.Sequential(
            nn.Linear(hidden_size, d_route),
            nn.SiLU(),
            nn.Linear(d_route, max_hops - 1),  # classes: 2, 3, ..., max_hops
        )

        # Node prior projection (boosts reasoning node when domain intent is high)
        self.prior_proj = nn.Linear(1, num_nodes)
        with torch.no_grad():
            nn.init.zeros_(self.prior_proj.weight)
            nn.init.zeros_(self.prior_proj.bias)
            self.node_reasoning_idx = num_nodes - 2
            self.prior_proj.weight[self.node_reasoning_idx, 0] = 2.0  # +2.0 logit boost on math

    def forward(self, h_pool: torch.Tensor, is_math: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        budget_logits = self.budget_head(h_pool)

        if is_math:
            g_domain = torch.ones((h_pool.size(0), 1), device=h_pool.device)
            node_prior_bias = self.prior_proj(g_domain)
        else:
            # Scale down domain signal on non-math tasks so Node 24 is not artificially boosted
            g_domain = self.domain_mlp(h_pool) * 0.01
            node_prior_bias = self.prior_proj(g_domain)

        return g_domain, budget_logits, node_prior_bias


class LocalFlowController(nn.Module):
    """
    Level 2: Local Dynamic Flow Navigation Controller.
    
    Continuous Neural ODE velocity field: dr/dt = v_phi(r | h_ctx, node_prev, g_domain).
    Transitions dynamically across physical tiles (Nodes 0..2L-1), Reasoning Core (Node 2L), and Exit (Node 2L+1).
    """
    def __init__(self, config: NaviTritScaleConfig, num_nodes: int):
        super().__init__()
        self.config = config
        self.num_nodes = num_nodes
        self.d_route = config.d_nav_route

        # Context projection
        self.ctx_proj = nn.Linear(config.hidden_size, self.d_route)
        self.node_embed = nn.Embedding(num_nodes + 1, self.d_route)
        self.domain_proj = nn.Linear(1, self.d_route)

        # Velocity network v_phi(r | h_ctx, node_prev, g_domain)
        self.v_net = nn.Sequential(
            nn.Linear(self.d_route * 4, self.d_route * 2),
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
        g_domain: torch.Tensor,
        node_prior_bias: torch.Tensor,
        r_prev: Optional[torch.Tensor] = None,
        hop_step: int = 0,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = h_token.size(0)
        h_ctx = self.ctx_proj(h_token)
        domain_emb = self.domain_proj(g_domain)

        prev_node_tensor = torch.full((B,), prev_node, dtype=torch.long, device=h_token.device)
        node_emb = self.node_embed(prev_node_tensor)

        if r_prev is None:
            r = torch.tanh(h_ctx + node_emb + domain_emb)
        else:
            r = r_prev

        # One velocity integration step
        v_in = torch.cat([r, h_ctx, node_emb, domain_emb], dim=-1)
        dr = self.v_net(v_in)
        r_next = self.ln_route(r + self.config.flow_dt * dr)

        raw_logits = self.node_head(r_next)
        logits = raw_logits + node_prior_bias

        # Disallow exit on early hops (t < 3)
        if hop_step < 3:
            node_exit = self.num_nodes - 1
            logits = logits.clone()
            logits[:, node_exit] = -1e4

        # Topological Anti-Gravity Shield (FFN damping)
        if self.config.use_scale_adaptive and prev_node < 2 * self.config.num_layers:
            is_prev_ffn = (prev_node % 2 == 1)
            if is_prev_ffn:
                alpha = self.config.scale_adaptive_ffn_damping
                ffn_indices = [2 * l + 1 for l in range(self.config.num_layers)]
                logits = logits.clone()
                logits[:, ffn_indices] = logits[:, ffn_indices] - alpha

        # Damping consecutive reasoning core self-loops:
        # Prevents getting stuck in Node 24 for multiple consecutive hops
        node_reasoning = self.num_nodes - 2
        if prev_node == node_reasoning:
            logits = logits.clone()
            logits[:, node_reasoning] = logits[:, node_reasoning] - 3.5

        if self.training:
            action_soft = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        else:
            idx = torch.argmax(logits, dim=-1)
            action_soft = F.one_hot(idx, num_classes=self.num_nodes).float()

        return action_soft, logits, r_next


class NaviTritDualForCausalLM(nn.Module):
    """
    Complete Hierarchical Dual-Controller NaviTrit Causal Language Model.
    
    Integrates GlobalFlowPlanner, LocalFlowController, PersistentEntityRegisters,
    and stationary BitLinear backbone with 100% warm-start parameter compatibility.
    """
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
        # 2L      : Reasoning Core (Node 24 in 100M)
        # 2L+1    : Exit (Node 25 in 100M)
        self.node_reasoning = 2 * config.num_layers
        self.node_exit = 2 * config.num_layers + 1
        self.num_nodes = 2 * config.num_layers + 2

        # Hop-Conditioned Tile Modulation (FiLM)
        self.hop_mod = HopConditionedModulation(config.max_hops, config.d_hop_embed, config.hidden_size)

        # Dedicated Recurrent Reasoning Core
        self.reasoning_core = RecurrentReasoningCore(config)

        # Level 1: Global Flow Planner
        self.global_planner = GlobalFlowPlanner(
            hidden_size=config.hidden_size,
            d_route=config.d_nav_route,
            num_nodes=self.num_nodes,
            max_hops=config.max_hops,
        )

        # Level 2: Local Dynamic Flow Controller
        self.controller = LocalFlowController(config, self.num_nodes)

        # Level 3: Persistent Entity Registers
        self.entity_registers = PersistentEntityRegisters(config.hidden_size, num_slots=4)

        # Readout head
        self.final_norm = BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        # Coherence head
        self.coherence_head = nn.Sequential(
            BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid(),
        )

    def load_from_pretrained_backbone(self, ckpt_path: str, device: torch.device):
        """Warm-starts stationary backbone weights while gracefully initializing new dual components."""
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model_dict = self.state_dict()
        
        loaded_keys = []
        skipped_keys = []

        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)

        self.load_state_dict(model_dict)
        print(f"Warm-start complete from {ckpt_path}: {len(loaded_keys)} tensors loaded ({len(skipped_keys)} unmapped/new).")

    def count_parameters(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total

    def execute_tile(self, node_idx: int, h: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes the specific tile indexed by node_idx at hop t."""
        device = h.device
        loss_fpf = torch.tensor(0.0, device=device)

        gamma_t, beta_t = self.hop_mod(t, device)

        if node_idx < 2 * self.num_layers:
            layer_idx = node_idx // 2
            is_attn = (node_idx % 2 == 0)

            if is_attn:
                norm_mod = self.attn_norms[layer_idx](h)
                norm_mod = (1.0 + gamma_t) * norm_mod + beta_t
                attn_out = self.attn_tiles[layer_idx](norm_mod)
                h_next = h + attn_out
            else:
                norm_mod = self.ffn_norms[layer_idx](h)
                norm_mod = (1.0 + gamma_t) * norm_mod + beta_t
                ffn_out = self.ffn_tiles[layer_idx](norm_mod)
                h_next = h + ffn_out
            return h_next, loss_fpf

        elif node_idx == self.node_reasoning:
            # Node 24: Recurrent Reasoning Core
            h_next, core_fpf, _ = self.reasoning_core(h)
            # Internal relaxation loss is computed for the reasoning core,
            # but returned separately so it does NOT penalize router policy logits.
            return h_next, core_fpf

        else:
            # Exit node: pass-through
            return h, loss_fpf

    def forward(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        fixed_hop_budget: Optional[int] = None,
        enforce_reasoning_node: bool = False,
    ) -> Dict[str, Any]:
        B, S = input_ids.size()
        device = input_ids.device

        # Embeddings
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        h = self.embed_tokens(input_ids) + self.embed_positions(pos)

        # Step 1: Bind Persistent Entity Registers (Slot-Variable Memory)
        slots = self.entity_registers.bind_slots(h)  # [B, 4, d]

        # Step 2: Global Flow Planning (Level 1)
        h_pool = h.mean(dim=1)  # [B, d]
        g_domain, budget_logits, node_prior_bias = self.global_planner(h_pool)

        if enforce_reasoning_node:
            node_prior_bias = node_prior_bias.clone()
            node_prior_bias[:, self.node_reasoning] += 5.0

        # Determine dynamic compute budget
        if fixed_hop_budget is not None:
            active_hops = fixed_hop_budget
        else:
            pred_budget_idx = int(torch.argmax(budget_logits.mean(dim=0), dim=-1).item())
            active_hops = min(self.max_hops, max(4, pred_budget_idx + 2))

        # Local Dynamic Trajectory Tracking
        r_state = None
        prev_node = self.num_nodes
        total_loss_fpf = torch.tensor(0.0, device=device)
        total_loss_state_fpf = torch.tensor(0.0, device=device)

        history_nodes = []
        history_actions = []
        history_log_probs = []
        early_exit_triggered = False

        for t in range(active_hops):
            # Inject persistent entity registers into current state
            h_injected = self.entity_registers.inject_registers(h, slots)
            h_pool_t = h_injected.mean(dim=1)

            # Local Controller step
            action_soft, logits, r_state = self.controller(
                h_token=h_pool_t,
                prev_node=prev_node,
                g_domain=g_domain,
                node_prior_bias=node_prior_bias,
                r_prev=r_state,
                hop_step=t,
                temperature=temperature,
                hard=True,
            )

            chosen_node = int(torch.argmax(action_soft.mean(dim=0)).item())
            history_nodes.append(chosen_node)
            history_actions.append(action_soft)

            # Record log prob of chosen action for GRPO policy gradient
            log_probs = F.log_softmax(logits, dim=-1)
            chosen_log_prob = log_probs.gather(dim=-1, index=torch.tensor([[chosen_node]], device=device).expand(B, 1)).squeeze(-1)
            history_log_probs.append(chosen_log_prob)

            if chosen_node == self.node_exit:
                early_exit_triggered = True
                break

            # Execute tile
            h_next, tile_fpf = self.execute_tile(chosen_node, h_injected, t)
            total_loss_fpf = total_loss_fpf + tile_fpf

            # Self-loop contraction
            if chosen_node == prev_node and prev_node < 2 * self.num_layers:
                state_disp = torch.mean((h_next - h_injected) ** 2)
                total_loss_state_fpf = total_loss_state_fpf + state_disp

            h = h_next
            prev_node = chosen_node

        # Readout head
        h_norm = self.final_norm(h)
        logits = self.lm_head(h_norm)

        # Trajectory Regularization Metrics
        actions_cat = torch.stack(history_actions, dim=1) if len(history_actions) > 0 else torch.zeros(B, 1, self.num_nodes, device=device)
        attn_indices = [2 * l for l in range(self.num_layers)]
        attn_visitation = actions_cat[..., attn_indices].sum(dim=-1).mean()
        
        eff_lambda_attn = self.config.scale_adaptive_lambda_attn_div if self.config.use_scale_adaptive else self.config.lambda_attn_div
        loss_attn_div = eff_lambda_attn * F.relu(self.config.min_attn_ratio - attn_visitation) ** 2

        trajectory_log_prob = torch.stack(history_log_probs, dim=1).sum(dim=1) if len(history_log_probs) > 0 else torch.zeros(B, device=device)

        out = {
            "logits": logits,
            "loss_fpf": total_loss_fpf,
            "loss_state_fpf": total_loss_state_fpf * self.config.lambda_state_fpf,
            "loss_attn_div": loss_attn_div,
            "hops_taken": len(history_nodes),
            "early_exit": early_exit_triggered,
            "trajectory": history_nodes,
            "trajectory_log_prob": trajectory_log_prob,
            "attn_ratio": float(attn_visitation.item()),
            "g_domain": g_domain.mean().item(),
            "reasoning_visited": (self.node_reasoning in history_nodes),
        }
        return out

    def plan_trajectory(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        is_math: bool = False,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Plans a global trajectory and records its log-probability in a single pass.
        Returns:
            trajectory: List of visited node indices
            trajectory_log_prob: Scalar log-probability of this trajectory
            slots: Bound persistent entity registers [B, M, d]
        """
        B, S = input_ids.size()
        device = input_ids.device

        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        h = self.embed_tokens(input_ids) + self.embed_positions(pos)
        slots = self.entity_registers.bind_slots(h)
        h_pool = h.mean(dim=1)

        g_domain, budget_logits, node_prior_bias = self.global_planner(h_pool, is_math=is_math)
        pred_budget_idx = int(torch.argmax(budget_logits.mean(dim=0), dim=-1).item())
        active_hops = min(self.max_hops, max(4, pred_budget_idx + 2))

        r_state = None
        prev_node = self.num_nodes
        trajectory = []
        log_probs_list = []

        for t in range(active_hops):
            h_injected = self.entity_registers.inject_registers(h, slots)
            h_pool_t = h_injected.mean(dim=1)

            action_soft, logits, r_state = self.controller(
                h_token=h_pool_t,
                prev_node=prev_node,
                g_domain=g_domain,
                node_prior_bias=node_prior_bias,
                r_prev=r_state,
                hop_step=t,
                temperature=temperature,
                hard=True,
            )

            chosen_node = int(torch.argmax(action_soft.mean(dim=0)).item())
            trajectory.append(chosen_node)

            log_probs = F.log_softmax(logits, dim=-1)
            chosen_lp = log_probs.gather(dim=-1, index=torch.tensor([[chosen_node]], device=device).expand(B, 1)).squeeze(-1)
            log_probs_list.append(chosen_lp)

            if chosen_node == self.node_exit:
                break
            prev_node = chosen_node

        traj_log_prob = torch.stack(log_probs_list, dim=1).sum(dim=1) if len(log_probs_list) > 0 else torch.zeros(B, device=device)
        return trajectory, traj_log_prob, slots

    def forward_trajectory(
        self,
        input_ids: torch.Tensor,
        trajectory: List[int],
        slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fast forward execution across a pre-planned trajectory."""
        B, S = input_ids.size()
        device = input_ids.device

        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        h = self.embed_tokens(input_ids) + self.embed_positions(pos)

        if slots is not None:
            h = self.entity_registers.inject_registers(h, slots)

        for t, node_idx in enumerate(trajectory):
            h, _ = self.execute_tile(node_idx, h, t)

        h_norm = self.final_norm(h)
        logits = self.lm_head(h_norm)
        return logits

