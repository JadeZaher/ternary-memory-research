"""
experiments/frontier_scaling/navitrit_graph_model.py: Gate 17 Graph-of-Traversals Architecture.

Evolves the tree structure into a dynamic Graph of Traversals (Traversal Graph Routing / TGR):
1. TraversalGraphController:
     - Given router state r_t, context h_t, domain gating g_domain, and visitation history:
       Computes Neural ODE integration dr/dt = v_phi(...)
       Evaluates transition affinities across all graph nodes
       Dispatches traversals along complementary graph edges with explicit, continuous weight scores w_{u -> v}
2. GraphCollapseOperator:
     - Aggregates representations across all active traversal edges weighted by their traversal scores:
       h_{d+1} = RMSNorm( h_d + sum_{v in V_active} w_v * (Tile_v(h_d) - h_d) )
3. NaviTritGraphForCausalLM:
     - 100% parameter-compatible with navitrit-100m-tree-grpo.pt (the 9.5/10 Python code checkpoint)
     - Builds, logs, and outputs the complete Graph of Traversals:
       * Directed edges with continuous weight scores
       * Cumulative adjacency / traversal flow matrix
       * Per-node routing contribution scores
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
from experiments.frontier_scaling.navitrit_dual_model import (
    PersistentEntityRegisters,
    GlobalFlowPlanner,
)


class TraversalGraphController(nn.Module):
    """
    Level 2: Graph-of-Traversals Navigation Controller.
    
    Given router latent state r_t, token context h_token, domain gating g_domain,
    and the pass history v_vis:
    1. Computes continuous Neural ODE velocity: dr/dt = v_phi(r | h_ctx + domain_emb, prev_emb).
    2. Projects transition logits across all graph tiles V = {0, ..., 24}.
    3. Traverses complementary paths across Sequence-Mixing and Channel-Mixing sub-graphs.
    4. Computes normalized, continuous traversal weight scores w_{u -> v} for all active edges.
    5. Returns chosen traversal nodes, directed edges, explicit weight scores, and updated latent state.
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
        with torch.no_grad():
            nn.init.zeros_(self.domain_proj.weight)
            nn.init.zeros_(self.domain_proj.bias)

        # Velocity network v_phi: 3 * d_route inputs for 100% warm-start checkpoint compatibility
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
        g_domain: torch.Tensor,
        node_prior_bias: torch.Tensor,
        v_vis: torch.Tensor,
        r_prev: Optional[torch.Tensor] = None,
        tree_level: int = 0,
        temperature: float = 1.0,
    ) -> Tuple[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor], List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
        """
        Dispatches traversals along complementary graph edges with scored edge weights.
        Returns:
            chosen_nodes: (node_a, node_b)
            gates_pair: (w_a, w_b) as autograd tensors
            edge_records: List of directed edges with exact scalar weight scores
            logits: Full transition logit vector
            r_next: Updated router latent state
        """
        B = h_token.size(0)
        h_ctx = self.ctx_proj(h_token)
        domain_emb = self.domain_proj(g_domain)

        prev_node_tensor = torch.full((B,), prev_node, dtype=torch.long, device=h_token.device)
        node_emb = self.node_embed(prev_node_tensor)

        if r_prev is None:
            r = torch.tanh(h_ctx + node_emb + domain_emb)
        else:
            r = r_prev

        # One velocity integration step: dr/dt = v_phi(r, h_ctx + domain_emb, node_emb)
        v_in = torch.cat([r, h_ctx + domain_emb, node_emb], dim=-1)
        dr = self.v_net(v_in)
        r_next = self.ln_route(r + self.config.flow_dt * dr)

        raw_logits = self.node_head(r_next)
        logits = raw_logits + node_prior_bias

        # 1. Mask Exit Node: Graph traversal executes for a certified depth D
        node_exit = self.num_nodes - 1
        logits = logits.clone()
        logits[:, node_exit] = -1e4

        # 2. History Visitation Damping: Penalize tiles already visited to encourage exploration
        alpha_hist = 1.5
        logits = logits - alpha_hist * v_vis

        # Complementary Functional Graph Partition:
        # Sequence-Mixing pool: Attention tiles (0, 2, ..., 22) + Reasoning Core (Node 24)
        # Channel-Mixing pool: FFN tiles (1, 3, ..., 23)
        attn_indices = [2 * l for l in range(self.config.num_layers)]
        ffn_indices = [2 * l + 1 for l in range(self.config.num_layers)]
        node_reasoning = self.num_nodes - 2
        seq_indices = attn_indices + [node_reasoning]

        # Branch 1 (Sequence Mixing): Neutral competition across Attn + Reasoning Core
        logits_seq = logits[:, seq_indices]
        if self.training:
            dist_seq = torch.distributions.Categorical(probs=F.softmax(logits_seq / temperature, dim=-1))
            idx_s = dist_seq.sample()[0].item()
        else:
            idx_s = torch.argmax(logits_seq, dim=-1)[0].item()
        chosen_1 = seq_indices[idx_s]

        # Branch 2 (Channel Mixing): Neutral competition across all FFN tiles
        logits_chan = logits[:, ffn_indices]
        if self.training:
            dist_chan = torch.distributions.Categorical(probs=F.softmax(logits_chan / temperature, dim=-1))
            idx_c = dist_chan.sample()[0].item()
        else:
            idx_c = torch.argmax(logits_chan, dim=-1)[0].item()
        chosen_2 = ffn_indices[idx_c]

        # Continuous Traversal Edge Weight Scores w_{u -> v}
        z1 = logits[:, chosen_1].unsqueeze(-1)
        z2 = logits[:, chosen_2].unsqueeze(-1)
        branch_gates = F.softmax(torch.cat([z1, z2], dim=-1), dim=-1)  # [B, 2]
        w1 = branch_gates[:, 0:1]
        w2 = branch_gates[:, 1:2]

        # Explicit Traversal Edge Records with weight scores
        edge_records = [
            {"src": prev_node, "dst": chosen_1, "weight": round(w1.mean().item(), 4), "type": "SEQ_MIXING"},
            {"src": prev_node, "dst": chosen_2, "weight": round(w2.mean().item(), 4), "type": "CHAN_MIXING"},
        ]

        return (chosen_1, chosen_2), (w1, w2), edge_records, logits, r_next


class GraphCollapseOperator(nn.Module):
    """
    Level 3: Graph Collapse Operator & Multi-Path Aggregator.
    
    Aggregates representations along all active traversal edges weighted by their
    continuous routing scores:
        h_{t+1} = RMSNorm( h_t + sum_{v in Active} w_v * (Tile_v(h_t) - h_t) )
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_norm = BitRouteRMSNorm(hidden_size)

    def forward(
        self,
        h_orig: torch.Tensor,
        h_1: torch.Tensor,
        h_2: torch.Tensor,
        w_1: torch.Tensor,
        w_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gated residual graph collapse across scored traversal paths.
        """
        # Broadcast gates [B, 1, 1] across sequence length
        w1_b = w_1.unsqueeze(-1)
        w2_b = w_2.unsqueeze(-1)

        delta_1 = h_1 - h_orig
        delta_2 = h_2 - h_orig

        h_collapsed = h_orig + w1_b * delta_1 + w2_b * delta_2
        return self.out_norm(h_collapsed)


class NaviTritGraphForCausalLM(nn.Module):
    """
    Gate 17 In-Place Evolution: NaviTrit Graph-of-Traversals Causal Language Model.
    
    100% parameter-compatible with navitrit-100m-tree-grpo.pt.
    Builds, executes, and outputs the dynamic Graph of Traversals with edge weight scores.
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

        self.ffn_tiles = nn.ModuleList([BitRouteFFN(sub_cfg) for _ in range(config.num_layers)])
        self.ffn_norms = nn.ModuleList([BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps) for _ in range(config.num_layers)])

        # Graph node mapping:
        # 0..2L-1 : Language tiles
        # 2L      : Reasoning Core
        # 2L+1    : Exit
        self.node_reasoning = 2 * config.num_layers
        self.node_exit = 2 * config.num_layers + 1
        self.num_nodes = 2 * config.num_layers + 2

        self.hop_mod = HopConditionedModulation(config.max_hops, config.d_hop_embed, config.hidden_size)
        self.reasoning_core = RecurrentReasoningCore(config)

        # Persistent Entity Registers & Global Flow Planner
        self.entity_registers = PersistentEntityRegisters(config.hidden_size, num_slots=4)
        self.global_planner = GlobalFlowPlanner(
            hidden_size=config.hidden_size,
            d_route=config.d_nav_route,
            num_nodes=self.num_nodes,
            max_hops=config.max_hops,
        )

        # Graph Traversal Controller & Collapse Operator (named tree_controller and collapse_operator for 100% key compatibility)
        self.tree_controller = TraversalGraphController(config, self.num_nodes)
        self.collapse_operator = GraphCollapseOperator(config.hidden_size)

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

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def execute_tile(self, node_idx: int, h: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes the specific tile indexed by node_idx at traversal step t."""
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
            h_next, core_fpf, _ = self.reasoning_core(h)
            return h_next, core_fpf

        else:
            return h, loss_fpf

    def forward(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        traversal_depth: int = 3,
        is_math: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Executes a dynamic Graph of Traversals with edge weight scores.
        """
        B, S = input_ids.size()
        device = input_ids.device

        # Embeddings
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        h = self.embed_tokens(input_ids) + self.embed_positions(pos)

        # Step 1: Bind Persistent Entity Registers
        slots = self.entity_registers.bind_slots(h)

        # Step 2: Global Flow Planning
        h_pool = h.mean(dim=1)
        g_domain, budget_logits, node_prior_bias = self.global_planner(h_pool, is_math=is_math)

        r_state = None
        prev_node = self.num_nodes  # Root start node
        v_vis = torch.zeros((B, self.num_nodes), device=device)
        total_loss_fpf = torch.tensor(0.0, device=device)

        history_pairs = []
        all_edges = []
        node_scores: Dict[int, float] = {}
        adjacency_matrix = torch.zeros((self.num_nodes + 1, self.num_nodes + 1), device=device)

        for d in range(traversal_depth):
            h_pool_d = h.mean(dim=1)

            # Controller: Dispatches traversals along complementary edges with scored weights
            (c1, c2), (w1, w2), edge_records, logits, r_state = self.tree_controller(
                h_token=h_pool_d,
                prev_node=prev_node,
                g_domain=g_domain,
                node_prior_bias=node_prior_bias,
                v_vis=v_vis,
                r_prev=r_state,
                tree_level=d,
                temperature=temperature,
            )

            history_pairs.append((c1, c2))
            all_edges.extend(edge_records)

            # Accumulate node scores and adjacency matrix
            w1_val = float(w1.mean().item())
            w2_val = float(w2.mean().item())
            node_scores[c1] = round(node_scores.get(c1, 0.0) + w1_val, 4)
            node_scores[c2] = round(node_scores.get(c2, 0.0) + w2_val, 4)

            p_src = min(prev_node, self.num_nodes)
            adjacency_matrix[p_src, c1] += w1.mean()
            adjacency_matrix[p_src, c2] += w2.mean()

            # Concurrent Tile Execution along traversed edges
            h_b1, fpf_1 = self.execute_tile(c1, h, d)
            h_b2, fpf_2 = self.execute_tile(c2, h, d)
            total_loss_fpf = total_loss_fpf + fpf_1 + fpf_2

            # Gated Graph Collapse Operator
            h = self.collapse_operator(h, h_b1, h_b2, w1, w2)

            # Update pass history
            v_vis = v_vis.clone()
            v_vis[:, c1] += 1.0
            v_vis[:, c2] += 1.0
            prev_node = c2

        # Readout head
        h_norm = self.final_norm(h)
        logits = self.lm_head(h_norm)

        flat_trajectory = []
        for p in history_pairs:
            flat_trajectory.extend([p[0], p[1]])

        # Traversal graph metadata structure
        traversal_graph = {
            "depth": traversal_depth,
            "traversed_nodes": flat_trajectory,
            "node_scores": node_scores,
            "directed_edges": all_edges,
            "cumulative_adjacency": adjacency_matrix[:self.num_nodes, :self.num_nodes].cpu().tolist(),
        }

        return {
            "logits": logits,
            "loss_fpf": total_loss_fpf,
            "tree_pairs": history_pairs,
            "trajectory": flat_trajectory,
            "traversal_graph": traversal_graph,
            "g_domain": g_domain.mean().item(),
            "reasoning_visited": (self.node_reasoning in flat_trajectory),
        }
