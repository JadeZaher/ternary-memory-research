"""
experiments/navitrit/navitrit_model.py: NaviTrit Architecture.

Implements Non-Monotonic Token Navigation across Stationary Ternary Module Graphs:
1. Stationary Module Graph G = (V, E) with |V| = 2L + 1 nodes:
     Node 2l     : Attn_l (Hop 1)
     Node 2l + 1 : FFN_l  (Hop 2)
     Node 2L     : EXIT   (Dynamic Termination)
2. Non-Monotonic Dynamic Transitions:
     - Hop Forward:      l_{t+1} > l_t
     - Hop Backward:     l_{t+1} < l_t (re-query lower-level features)
     - Self-Loop:        l_{t+1} == l_t, v_{t+1} == v_t (repeated activation of same physical tile)
     - Early Exit:       v_{t+1} == EXIT or ||h_{t+1} - h_t||_inf < eps
3. Flow-Reasoned Navigation Controller (S_nav):
     - Internal routing state r_t in R^{d_r}
     - Continuous ODE velocity relaxation dr/dt = v_phi(r | h, v_{t-1})
     - Straight-through Gumbel-Softmax over all 2L + 1 destinations.

Reference: research/layer-navigation-flow-theory.md
"""

import sys
import os
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
TRANSITION_EXIT = "EXIT"


@dataclass
class NaviTritConfig:
    """Configuration for NaviTrit model."""
    vocab_size: int = 50257
    hidden_size: int = 384
    intermediate_size: int = 1024
    num_layers: int = 4            # 4 Attention + 4 FFN = 8 stationary nodes + 1 EXIT = 9 nodes
    num_attention_heads: int = 6
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    block_size: int = 256
    max_hops: int = 8              # Max non-monotonic trajectory length T_max
    d_nav_route: int = 128         # Routing state vector dimension
    flow_dt: float = 0.5           # Velocity integration step
    eps_contraction: float = 1e-3  # Dynamic contraction halting threshold
    gumbel_tau: float = 1.0        # Straight-through temperature
    lambda_hop: float = 0.02       # Economic hop cost penalty
    lambda_fpf: float = 0.01       # Attractor velocity penalty
    ternary: bool = True           # Native ternary BitLinear weights
    tie_word_embeddings: bool = False


@dataclass
class NaviTritTrajectoryStep:
    """Telemetry recorded for a single hop in the trajectory."""
    hop_idx: int
    selected_node: int             # 0 .. 2L
    layer_idx: int                 # 0 .. L-1, or L for EXIT
    sublayer_name: str             # "ATTN", "FFN", or "EXIT"
    transition_type: str           # FORWARD, BACKWARD, SELF_LOOP, INTRA_LAYER, EXIT
    step_norm: float               # ||h_{t+1} - h_t||_inf
    logits: torch.Tensor           # [B, 2L + 1]


@dataclass
class NaviTritOutput:
    """Output and execution telemetry from NaviTrit forward pass."""
    logits: torch.Tensor
    total_hops_taken: int
    forward_hops_count: int
    backward_hops_count: int
    self_loops_count: int
    early_exits_count: int
    trajectory_nodes: List[int]
    trajectory_steps: List[NaviTritTrajectoryStep]
    nav_budget_loss: torch.Tensor
    fpf_loss: torch.Tensor
    bytes_streamed: int
    full_pipeline_bytes: int


class FlowNavigationController(nn.Module):
    """
    Global Flow-Reasoned Navigation Controller.
    
    Maintains internal routing state r_t in R^{d_r}.
    At each step t, takes current representation h^{(t)} and previous node v^{(t-1)},
    relaxes r_t via continuous velocity field v_phi, and projects logits over
    all 2L + 1 candidate destination nodes in G.
    """
    def __init__(self, config: NaviTritConfig):
        super().__init__()
        self.num_layers = config.num_layers
        self.total_nodes = 2 * config.num_layers + 1  # 2L modules + 1 EXIT
        self.d_route = config.d_nav_route
        self.d_model = config.hidden_size
        self.flow_dt = config.flow_dt
        self.gumbel_tau = config.gumbel_tau

        # Embedding for previous node index (0 .. 2L)
        self.node_embed = nn.Embedding(self.total_nodes + 1, self.d_route)

        # Context condition encoder: pools h [B, S, d_model] -> condition [B, d_route]
        self.cond_encoder = nn.Sequential(
            nn.Linear(self.d_model + self.d_route, self.d_route),
            nn.LayerNorm(self.d_route),
            nn.GELU(),
            nn.Linear(self.d_route, self.d_route),
        )

        # Initial state projection from first embedding
        self.r_init_proj = nn.Linear(self.d_route, self.d_route)

        # Flow velocity network v_phi(r | c)
        self.velocity_net = nn.Sequential(
            nn.LayerNorm(self.d_route),
            nn.Linear(self.d_route, self.d_route * 2),
            nn.GELU(),
            nn.Linear(self.d_route * 2, self.d_route),
        )

        # Destination node selector head: r* -> 2L + 1 logits
        self.dest_head = nn.Sequential(
            nn.LayerNorm(self.d_route),
            nn.Linear(self.d_route, self.total_nodes),
        )

        # Matched random policy control (for unbiased baseline testing)
        self.random_policy: Optional[bool] = None

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Default initialization: slight bias towards forward flow
        with torch.no_grad():
            last_linear = self.dest_head[1]
            bias = torch.zeros(self.total_nodes)
            # Default forward chain bias
            bias[:2] = 1.0
            last_linear.bias.copy_(bias)

    def pool_representation(
        self,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if h.dim() == 2:
            return h
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            return (h * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        return h.mean(dim=1)

    def init_routing_state(
        self,
        h0: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Initializes r_0 from prompt embedding."""
        h_pool = self.pool_representation(h0, attention_mask)
        dummy_node = torch.full((h0.shape[0],), self.total_nodes, dtype=torch.long, device=h0.device)
        node_ctx = self.node_embed(dummy_node)
        ctx = torch.cat([h_pool, node_ctx], dim=-1)
        c0 = self.cond_encoder(ctx)
        return self.r_init_proj(c0)

    def forward_step(
        self,
        r: torch.Tensor,
        h: torch.Tensor,
        prev_node: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Takes single navigation step:
        Returns:
            r_next: updated routing state [B, d_route]
            logits: destination logits [B, 2L + 1]
            gumbel_weights: straight-through weights [B, 2L + 1]
            fpf_step_loss: terminal velocity penalty
        """
        B = h.shape[0]
        h_pool = self.pool_representation(h, attention_mask)
        node_ctx = self.node_embed(prev_node)
        condition = self.cond_encoder(torch.cat([h_pool, node_ctx], dim=-1))

        # Continuous relaxation step: dr/dt = v_phi(r | condition)
        v = self.velocity_net(r + condition)
        r_next = r + self.flow_dt * v

        # Fixed-point forcing loss on velocity
        fpf_step_loss = torch.mean(torch.sum(v ** 2, dim=-1))

        # Destination logits
        logits = self.dest_head(r_next)  # [B, 2L + 1]

        # Action selection
        if self.random_policy is not None and not self.training:
            # Matched random walk control: sample random node uniformly
            rand_ids = torch.randint(0, self.total_nodes, (B,), device=logits.device)
            gumbel_weights = F.one_hot(rand_ids, num_classes=self.total_nodes).to(logits.dtype)
        elif self.training:
            gumbel_weights = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True, dim=-1)
        else:
            argmax_ids = torch.argmax(logits, dim=-1)
            gumbel_weights = F.one_hot(argmax_ids, num_classes=self.total_nodes).to(logits.dtype)

        return r_next, logits, gumbel_weights, fpf_step_loss


class StationaryModuleGraph(nn.Module):
    """
    Container of stationary ternary modules.
    
    Contains:
      - L Attention modules (nodes 0, 2, 4, ...)
      - L FFN modules (nodes 1, 3, 5, ...)
      - LayerNorm modules for each stationary tile
    """
    def __init__(self, config: BitRouteConfig):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.total_nodes = 2 * self.num_layers + 1
        self.exit_node_idx = 2 * self.num_layers

        # Stationary Attention tiles
        self.attn_norms = nn.ModuleList([
            BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            for _ in range(self.num_layers)
        ])
        self.attn_tiles = nn.ModuleList([
            BitRouteAttention(config) for _ in range(self.num_layers)
        ])

        # Stationary FFN tiles
        self.ffn_norms = nn.ModuleList([
            BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            for _ in range(self.num_layers)
        ])
        self.ffn_tiles = nn.ModuleList([
            BitRouteFFN(config) for _ in range(self.num_layers)
        ])

        # Byte accounting (TQ1_0 format: 1.6875 bpw)
        self.attn_bytes = int(4 * (config.hidden_size ** 2) * 1.6875 / 8)
        self.ffn_bytes = int(3 * (config.hidden_size * config.intermediate_size) * 1.6875 / 8)

    def execute_node(
        self,
        node_idx: int,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Executes the specific module indexed by node_idx in {0 .. 2L}."""
        if node_idx == self.exit_node_idx:
            # EXIT node: identity
            return h

        layer_idx = node_idx // 2
        is_ffn = (node_idx % 2 == 1)

        if not is_ffn:
            # Attention node
            norm_h = self.attn_norms[layer_idx](h)
            attn_out = self.attn_tiles[layer_idx](norm_h, attention_mask=attention_mask)
            return h + attn_out
        else:
            # FFN node
            norm_h = self.ffn_norms[layer_idx](h)
            ffn_out = self.ffn_tiles[layer_idx](norm_h)
            return h + ffn_out


class NaviTritForCausalLM(nn.Module):
    """
    NaviTrit: Non-Monotonic Token Navigation Language Model.
    
    Navigates a stationary module graph G = (V, E) via a Global Flow-Reasoned
    Navigation Controller S_nav, permitting forward hops, backward hops,
    self-loops, and early exits.
    """
    def __init__(self, config: Optional[NaviTritConfig] = None):
        super().__init__()
        self.config = config or NaviTritConfig()

        # Shared base config for BitLinear modules
        self.bit_config = BitRouteConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.num_layers,
            num_attention_heads=self.config.num_attention_heads,
            max_position_embeddings=self.config.max_position_embeddings,
            rms_norm_eps=self.config.rms_norm_eps,
            block_size=self.config.block_size,
            ternary=self.config.ternary,
        )

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.graph = StationaryModuleGraph(self.bit_config)
        self.controller = FlowNavigationController(self.config)

        self.norm = BitRouteRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Benchmark reference bytes (standard monotonic pipeline)
        self.full_pipeline_bytes = self.config.num_layers * (self.graph.attn_bytes + self.graph.ffn_bytes)

    def _classify_transition(self, prev_node: int, curr_node: int) -> Tuple[int, str, str]:
        """Classifies the transition between nodes into forward, backward, self-loop, etc."""
        if curr_node == self.graph.exit_node_idx:
            return self.config.num_layers, "EXIT", TRANSITION_EXIT

        layer_curr = curr_node // 2
        sublayer_name = "FFN" if (curr_node % 2 == 1) else "ATTN"

        if prev_node == self.graph.total_nodes:
            # First hop
            return layer_curr, sublayer_name, TRANSITION_FORWARD

        if curr_node == prev_node:
            return layer_curr, sublayer_name, TRANSITION_SELF_LOOP

        layer_prev = prev_node // 2
        if layer_curr > layer_prev:
            trans = TRANSITION_FORWARD
        elif layer_curr < layer_prev:
            trans = TRANSITION_BACKWARD
        else:
            trans = TRANSITION_INTRA_LAYER

        return layer_curr, sublayer_name, trans

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        force_monotonic: bool = False,
    ) -> NaviTritOutput:
        """
        Forward pass executing the non-monotonic graph walk.
        """
        B = input_ids.shape[0]
        device = input_ids.device
        h = self.embed_tokens(input_ids)

        # 1. Initialize Controller State
        r = self.controller.init_routing_state(h, attention_mask)
        prev_node = torch.full((B,), self.graph.total_nodes, dtype=torch.long, device=device)

        trajectory_steps: List[NaviTritTrajectoryStep] = []
        trajectory_nodes: List[int] = []

        total_fpf_loss = torch.tensor(0.0, device=device)
        total_hops = 0
        forward_hops = 0
        backward_hops = 0
        self_loops = 0
        early_exits = 0
        bytes_streamed = 0

        halted_mask = torch.zeros(B, dtype=torch.bool, device=device)

        # -------------------------------------------------------------
        # Non-Monotonic Graph Navigation Walk Loop
        # -------------------------------------------------------------
        for hop in range(self.config.max_hops):
            # Monotonic baseline override
            if force_monotonic:
                # Force node sequence 0, 1, 2, 3, ... 2L-1
                if hop < 2 * self.config.num_layers:
                    curr_node_id = hop
                else:
                    break
                h = self.graph.execute_node(curr_node_id, h, attention_mask=attention_mask)
                total_hops += 1
                forward_hops += 1
                trajectory_nodes.append(curr_node_id)
                continue

            # Navigation controller step
            r, logits, gumbel_weights, fpf_step_loss = self.controller.forward_step(
                r, h, prev_node, attention_mask=attention_mask
            )
            total_fpf_loss = total_fpf_loss + fpf_step_loss

            if self.training:
                # Differentiable forward blend over all candidate nodes
                h_candidates = []
                for node_id in range(self.graph.total_nodes):
                    h_c = self.graph.execute_node(node_id, h, attention_mask=attention_mask)
                    h_candidates.append(h_c)
                # Stack [B, 2L+1, S, d]
                h_stack = torch.stack(h_candidates, dim=1)
                # Weighted combination via straight-through gumbel weights [B, 2L+1, 1, 1]
                gw = gumbel_weights.unsqueeze(-1).unsqueeze(-1)
                h_next = torch.sum(gw * h_stack, dim=1)

                # Track dominant node for telemetry
                curr_node_id = int(torch.argmax(gumbel_weights[0]).item())
            else:
                # Discrete branch execution (zero compute for unselected nodes)
                curr_node_id = int(torch.argmax(gumbel_weights[0]).item())
                if curr_node_id == self.graph.exit_node_idx:
                    # Early exit triggered
                    early_exits += 1
                    break
                h_next = self.graph.execute_node(curr_node_id, h, attention_mask=attention_mask)

            # Classify transition
            prev_id = int(prev_node[0].item())
            layer_idx, sublayer_name, trans_type = self._classify_transition(prev_id, curr_node_id)

            if trans_type == TRANSITION_FORWARD:
                forward_hops += 1
            elif trans_type == TRANSITION_BACKWARD:
                backward_hops += 1
            elif trans_type == TRANSITION_SELF_LOOP:
                self_loops += 1
            elif trans_type == TRANSITION_EXIT:
                early_exits += 1

            # Contraction check: ||h_{t+1} - h_t||_inf
            step_delta = (h_next - h).detach()
            step_norm = float(torch.norm(step_delta, p=float('inf')).item())

            trajectory_steps.append(NaviTritTrajectoryStep(
                hop_idx=hop,
                selected_node=curr_node_id,
                layer_idx=layer_idx,
                sublayer_name=sublayer_name,
                transition_type=trans_type,
                step_norm=step_norm,
                logits=logits.detach(),
            ))
            trajectory_nodes.append(curr_node_id)

            # Byte accounting
            if curr_node_id != self.graph.exit_node_idx:
                if curr_node_id % 2 == 0:
                    bytes_streamed += self.graph.attn_bytes
                else:
                    bytes_streamed += self.graph.ffn_bytes

            h = h_next
            prev_node = torch.full((B,), curr_node_id, dtype=torch.long, device=device)
            total_hops += 1

            # Dynamic Contraction Exit: if representation stops changing significantly
            if not self.training and step_norm < self.config.eps_contraction and hop >= 2:
                break

        # Final projection
        h_norm = self.norm(h)
        logits = self.lm_head(h_norm)

        # Navigation budget loss: penalizes total hops taken
        hop_ratio = total_hops / max(1, self.config.max_hops)
        nav_budget_loss = torch.tensor(self.config.lambda_hop * hop_ratio, device=device)

        return NaviTritOutput(
            logits=logits,
            total_hops_taken=total_hops,
            forward_hops_count=forward_hops,
            backward_hops_count=backward_hops,
            self_loops_count=self_loops,
            early_exits_count=early_exits,
            trajectory_nodes=trajectory_nodes,
            trajectory_steps=trajectory_steps,
            nav_budget_loss=nav_budget_loss,
            fpf_loss=self.config.lambda_fpf * total_fpf_loss,
            bytes_streamed=bytes_streamed,
            full_pipeline_bytes=self.full_pipeline_bytes,
        )
