"""
experiments/frontier_scaling/navitrit_token_graph_model.py: NaviTrit Dynamic Token-Level Graph Routing Model (DTRNet).

Gate 18 (Track H) Implementation:
1. Token-Level Fan-Out & Return: Replaces sequence-pooled routing with per-token gate distributions [B, S, 2].
2. Selective Attention Gating (DTRNet): Tokens dynamically partition into sequence mixing (Attention / Reasoning)
   and channel mixing (FFN) with selective attention bypass on low-saliency syntax tokens.
3. TokenGraphCollapseOperator: Re-converges representations per token:
   h_{t+1, i} = RMSNorm(h_{t, i} + w1_i * Delta_1,i + w2_i * Delta_2,i).
4. Full Traversal Graph & Token Saliency Telemetry: Directed edge logging, cumulative adjacency, and bypass metrics.
5. 100% Parameter-compatible base with frozen 124.15M pre-trained backbone.
"""

import os
import sys
import math
from typing import Dict, List, Tuple, Any, Optional

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


class TokenTraversalController(nn.Module):
    """
    Token-Level Traversal Controller & Dynamic Saliency Gating Head (DTRNet).
    
    1. Dispatches global tile candidates (c1_seq, c2_chan) via neural ODE velocity integration.
    2. Evaluates per-token routing weights w^{(1)}_{b, i} and w^{(2)}_{b, i} conditioned on token state h_{b, i},
       global trajectory latent r, and domain representation g_domain.
    3. Computes selective attention bypass masks for syntax/filler tokens.
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

        # Flow routing latent velocity network: dr/dt = v_phi(r, h_ctx + domain_emb, node_emb)
        self.v_net = nn.Sequential(
            nn.Linear(self.d_route * 3, self.d_route * 2),
            nn.SiLU(),
            nn.Linear(self.d_route * 2, self.d_route),
        )

        # Transition logit projection
        self.node_head = nn.Linear(self.d_route, self.num_nodes)
        self.ln_route = nn.LayerNorm(self.d_route)

        # Token-Level Saliency & Attention Gating Head (DTRNet)
        # Maps [h_token (hidden_size), r_latent (d_route), domain (1)] -> scalar attention saliency s_i
        self.token_gate_head = nn.Sequential(
            nn.Linear(config.hidden_size + self.d_route + 1, self.d_route),
            nn.SiLU(),
            nn.Linear(self.d_route, 1),
        )
        # Warm-bias toward active attention exploration: sigmoid(2.0) ~ 0.88
        with torch.no_grad():
            nn.init.constant_(self.token_gate_head[-1].bias, 2.0)
            self.token_gate_head[-1].weight.data.mul_(0.01)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        h_seq: torch.Tensor,
        prev_node: int,
        g_domain: torch.Tensor,
        node_prior_bias: torch.Tensor,
        v_vis: torch.Tensor,
        r_prev: Optional[torch.Tensor] = None,
        tree_level: int = 0,
        temperature: float = 1.0,
        bypass_thresh: float = 0.25,
    ) -> Tuple[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
        """
        Dispatches token-level traversals with selective gating.
        
        Args:
            h_seq: [B, S, hidden_size] sequence representations.
            prev_node: scalar index of origin tile.
            g_domain: [B, 1] task/domain scalar embedding.
            node_prior_bias: [B, num_nodes] inductive domain bias.
            v_vis: [B, num_nodes] visitation counts.
            r_prev: [B, d_route] previous velocity latent.
            temperature: sampling temperature.
            bypass_thresh: attention bypass threshold s_tok < thresh.
            
        Returns:
            chosen_pair: (c1_seq, c2_chan)
            token_gates: (w1_seq [B, S, 1], w2_chan [B, S, 1])
            bypass_mask: [B, S] boolean mask (True if attention bypassed)
            edge_records: list of directed graph edges with average weights
            logits: tile selection logits [B, num_nodes]
            r_next: updated router velocity latent [B, d_route]
        """
        B, S, d = h_seq.size()
        device = h_seq.device

        # 1. Global Trajectory Velocity Integration
        h_mean = h_seq.mean(dim=1)
        h_ctx = self.ctx_proj(h_mean)
        domain_emb = self.domain_proj(g_domain)

        prev_node_tensor = torch.full((B,), prev_node, dtype=torch.long, device=device)
        node_emb = self.node_embed(prev_node_tensor)

        if r_prev is None:
            r = torch.tanh(h_ctx + node_emb + domain_emb)
        else:
            r = r_prev

        v_in = torch.cat([r, h_ctx + domain_emb, node_emb], dim=-1)
        dr = self.v_net(v_in)
        r_next = self.ln_route(r + self.config.flow_dt * dr)

        raw_logits = self.node_head(r_next)
        logits = raw_logits + node_prior_bias

        # Mask Exit Node and apply history visitation damping
        node_exit = self.num_nodes - 1
        logits = logits.clone()
        logits[:, node_exit] = -1e4
        alpha_hist = 1.5
        logits = logits - alpha_hist * v_vis

        # 2. Tile Candidate Selection (Sequence Mixing vs Channel Mixing)
        attn_indices = [2 * l for l in range(self.config.num_layers)]
        ffn_indices = [2 * l + 1 for l in range(self.config.num_layers)]
        node_reasoning = self.num_nodes - 2
        seq_indices = attn_indices + [node_reasoning]

        logits_seq = logits[:, seq_indices]
        if self.training:
            dist_seq = torch.distributions.Categorical(probs=F.softmax(logits_seq / temperature, dim=-1))
            idx_s = dist_seq.sample()[0].item()
        else:
            idx_s = torch.argmax(logits_seq, dim=-1)[0].item()
        chosen_1 = seq_indices[idx_s]

        logits_chan = logits[:, ffn_indices]
        if self.training:
            dist_chan = torch.distributions.Categorical(probs=F.softmax(logits_chan / temperature, dim=-1))
            idx_c = dist_chan.sample()[0].item()
        else:
            idx_c = torch.argmax(logits_chan, dim=-1)[0].item()
        chosen_2 = ffn_indices[idx_c]

        # Balanced Branch Gates from Neural ODE Logits (preserves Sequence/Channel Invariant)
        z1 = logits[:, chosen_1].unsqueeze(-1)
        z2 = logits[:, chosen_2].unsqueeze(-1)
        branch_gates = F.softmax(torch.cat([z1, z2], dim=-1), dim=-1)
        w1_branch = branch_gates[:, 0:1].unsqueeze(1)  # [B, 1, 1]
        w2_branch = branch_gates[:, 1:2].unsqueeze(1)  # [B, 1, 1]

        # 3. Token-Level Saliency Gating (DTRNet)
        # Condition per-token routing on (h_seq, r_next, g_domain)
        r_expanded = r_next.unsqueeze(1).expand(B, S, self.d_route)
        dom_expanded = g_domain.unsqueeze(1).expand(B, S, 1)
        token_input = torch.cat([h_seq, r_expanded, dom_expanded], dim=-1)

        s_logits = self.token_gate_head(token_input)  # [B, S, 1]
        s_tok = torch.sigmoid(s_logits / max(0.5, temperature))  # [B, S, 1]

        # Branch 1 (Sequence Mixing) is modulated by token attention saliency
        w1_tok = w1_branch * s_tok
        # Branch 2 (Channel Mixing) is ALWAYS fully active (prevents zero-FFN collapse)
        w2_tok = w2_branch.expand(B, S, 1)

        # Selective Attention Bypass: if s_tok < bypass_thresh
        bypass_mask = (s_tok.squeeze(-1) < bypass_thresh)  # [B, S]

        # Directed Edge Telemetry (mean continuous weights)
        mean_w1 = round(float(w1_tok.mean().item()), 4)
        mean_w2 = round(float(w2_tok.mean().item()), 4)
        edge_records = [
            {"src": prev_node, "dst": chosen_1, "weight": mean_w1, "type": "SEQ_MIXING"},
            {"src": prev_node, "dst": chosen_2, "weight": mean_w2, "type": "CHAN_MIXING"},
        ]

        return (chosen_1, chosen_2), (w1_tok, w2_tok), bypass_mask, edge_records, logits, r_next


class TokenGraphCollapseOperator(nn.Module):
    """
    Per-Token Graph Collapse Operator (Scatter-Gather Aggregation).
    
    Re-converges representations along active graph edges per individual token:
        h_{t+1, i} = RMSNorm( h_{t, i} + w^{(1)}_i * Delta_{1, i} + w^{(2)}_i * Delta_{2, i} )
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
        w1_tok: torch.Tensor,
        w2_tok: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_orig: [B, S, d] original token representations.
            h_1: [B, S, d] Branch 1 output (Sequence Mixing).
            h_2: [B, S, d] Branch 2 output (Channel Mixing).
            w1_tok: [B, S, 1] per-token Branch 1 gate.
            w2_tok: [B, S, 1] per-token Branch 2 gate.
        Returns:
            [B, S, d] re-converged collapsed representations.
        """
        delta_1 = h_1 - h_orig
        delta_2 = h_2 - h_orig

        h_collapsed = h_orig + w1_tok * delta_1 + w2_tok * delta_2
        return self.out_norm(h_collapsed)


class NaviTritTokenGraphForCausalLM(nn.Module):
    """
    Gate 18 (Track H): NaviTrit Dynamic Token-Level Graph Routing Causal LM (DTRNet).
    
    Implements:
    - Per-token fan-out across complementary sequence/channel graph edges.
    - Selective attention bypass on syntax/filler tokens.
    - Per-token collapse operator.
    - Full backward compatibility with pre-trained 124.15M ternary backbone.
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

        # Stationary Parameter Tiles
        self.attn_tiles = nn.ModuleList([BitRouteAttention(sub_cfg) for _ in range(config.num_layers)])
        self.attn_norms = nn.ModuleList([BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps) for _ in range(config.num_layers)])
        self.ffn_tiles = nn.ModuleList([BitRouteFFN(sub_cfg) for _ in range(config.num_layers)])
        self.ffn_norms = nn.ModuleList([BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps) for _ in range(config.num_layers)])

        # Graph node mapping
        self.node_reasoning = 2 * config.num_layers
        self.node_exit = 2 * config.num_layers + 1
        self.num_nodes = 2 * config.num_layers + 2

        # Dedicated Reasoning Core (Node 24)
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

        # Token Traversal Controller & Token Collapse Operator
        self.token_controller = TokenTraversalController(config, self.num_nodes)
        self.token_collapse = TokenGraphCollapseOperator(config.hidden_size)

        # Readout Head
        self.final_norm = BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        # Coherence Head
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

    def execute_tile(
        self,
        node_idx: int,
        h: torch.Tensor,
        t: int,
        bypass_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes tile indexed by node_idx with optional attention bypass.
        
        If bypass_mask is provided and node is an Attention tile:
        Bypassed tokens skip the attention perturbation (delta = 0).
        """
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

                if bypass_mask is not None:
                    # Zero out attention update for bypassed tokens
                    mask_expanded = bypass_mask.unsqueeze(-1)  # [B, S, 1]
                    attn_out = torch.where(mask_expanded, torch.zeros_like(attn_out), attn_out)

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
        bypass_thresh: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Forward pass with Dynamic Token Graph Routing (DTRNet).
        """
        B, S = input_ids.size()
        device = input_ids.device

        # 1. Embeddings & Entity Slots
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        h = self.embed_tokens(input_ids) + self.embed_positions(pos)
        slots = self.entity_registers.bind_slots(h)

        # 2. Global Flow Planning
        h_pool = h.mean(dim=1)
        g_domain, budget_logits, node_prior_bias = self.global_planner(h_pool, is_math=is_math)

        r_state = None
        prev_node = self.num_nodes  # Root start node
        v_vis = torch.zeros((B, self.num_nodes), device=device)
        total_loss_fpf = torch.tensor(0.0, device=device)

        history_pairs = []
        all_edges = []
        node_scores: Dict[int, float] = {}
        total_bypassed_tokens = 0
        total_possible_tokens = 0
        adjacency_matrix = torch.zeros((self.num_nodes + 1, self.num_nodes + 1), device=device)

        # 3. Dynamic Token Traversal Loop
        for d in range(traversal_depth):
            (c1, c2), (w1_tok, w2_tok), bypass_mask, edge_records, logits, r_state = self.token_controller(
                h_seq=h,
                prev_node=prev_node,
                g_domain=g_domain,
                node_prior_bias=node_prior_bias,
                v_vis=v_vis,
                r_prev=r_state,
                tree_level=d,
                temperature=temperature,
                bypass_thresh=bypass_thresh,
            )

            history_pairs.append((c1, c2))
            all_edges.extend(edge_records)

            # Telemetry: Bypass statistics
            num_bypassed = int(bypass_mask.sum().item())
            total_bypassed_tokens += num_bypassed
            total_possible_tokens += (B * S)

            # Accumulate scores and adjacency
            mean_w1 = float(w1_tok.mean().item())
            mean_w2 = float(w2_tok.mean().item())
            node_scores[c1] = round(node_scores.get(c1, 0.0) + mean_w1, 4)
            node_scores[c2] = round(node_scores.get(c2, 0.0) + mean_w2, 4)

            p_src = min(prev_node, self.num_nodes)
            adjacency_matrix[p_src, c1] += w1_tok.mean()
            adjacency_matrix[p_src, c2] += w2_tok.mean()

            # Concurrent Tile Execution with selective attention bypass
            h_b1, fpf_1 = self.execute_tile(c1, h, d, bypass_mask=bypass_mask)
            h_b2, fpf_2 = self.execute_tile(c2, h, d, bypass_mask=None)
            total_loss_fpf = total_loss_fpf + fpf_1 + fpf_2

            # Gated Token Collapse Operator (Scatter-Gather)
            h = self.token_collapse(h, h_b1, h_b2, w1_tok, w2_tok)

            # Update pass history
            v_vis = v_vis.clone()
            v_vis[:, c1] += 1.0
            v_vis[:, c2] += 1.0
            prev_node = c2

        # 4. Readout Head
        h_norm = self.final_norm(h)
        logits = self.lm_head(h_norm)

        flat_trajectory = []
        for p in history_pairs:
            flat_trajectory.extend([p[0], p[1]])

        overall_bypass_rate = (total_bypassed_tokens / max(1, total_possible_tokens))

        traversal_graph = {
            "depth": traversal_depth,
            "traversed_nodes": flat_trajectory,
            "node_scores": node_scores,
            "directed_edges": all_edges,
            "cumulative_adjacency": adjacency_matrix[:self.num_nodes, :self.num_nodes].cpu().tolist(),
            "attention_bypass_rate": round(overall_bypass_rate, 4),
        }

        return {
            "logits": logits,
            "loss_fpf": total_loss_fpf,
            "tree_pairs": history_pairs,
            "trajectory": flat_trajectory,
            "traversal_graph": traversal_graph,
            "attention_bypass_rate": overall_bypass_rate,
            "g_domain": g_domain.mean().item(),
            "reasoning_visited": (self.node_reasoning in flat_trajectory),
        }

    def load_from_pretrained_backbone(self, ckpt_path: str, device: torch.device):
        """Warm-starts weights from tree-grpo or scale pre-trained checkpoint."""
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model_dict = self.state_dict()

        loaded_keys = []
        skipped_keys = []

        for k, v in state_dict.items():
            target_k = k
            if k.startswith("tree_controller."):
                target_k = "token_controller." + k[len("tree_controller."):]
            elif k.startswith("collapse_operator."):
                target_k = "token_collapse." + k[len("collapse_operator."):]
            elif k.startswith("controller."):
                target_k = "token_controller." + k[len("controller."):]

            if target_k in model_dict and model_dict[target_k].shape == v.shape:
                model_dict[target_k] = v
                loaded_keys.append(target_k)
            else:
                skipped_keys.append(k)

        self.load_state_dict(model_dict)
        print(f"Warm-start complete from {ckpt_path}: {len(loaded_keys)} tensors loaded ({len(skipped_keys)} unmapped/new).")

    def plan_token_trajectory(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        traversal_depth: int = 3,
        is_math: Optional[bool] = None,
        ref_controller: Optional[nn.Module] = None,
        return_diagnostics: bool = False,
        bypass_thresh: float = 0.20,
    ) -> Any:
        """Plans token graph trajectory and records log-probs for fast autoregressive generation."""
        B, S = input_ids.size()
        device = input_ids.device

        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        h = self.embed_tokens(input_ids) + self.embed_positions(pos)
        slots = self.entity_registers.bind_slots(h)
        h_pool = h.mean(dim=1)

        g_domain, budget_logits, node_prior_bias = self.global_planner(h_pool, is_math=is_math)

        r_state = None
        prev_node = self.num_nodes
        v_vis = torch.zeros((B, self.num_nodes), device=device)

        history_pairs = []
        history_gates = []
        history_masks = []
        history_log_probs = []
        total_kl = torch.tensor(0.0, device=device)
        total_entropy = torch.tensor(0.0, device=device)

        for d in range(traversal_depth):
            (c1, c2), (w1_tok, w2_tok), bypass_mask, edge_records, logits, r_state = self.token_controller(
                h_seq=h,
                prev_node=prev_node,
                g_domain=g_domain,
                node_prior_bias=node_prior_bias,
                v_vis=v_vis,
                r_prev=r_state,
                tree_level=d,
                temperature=temperature,
                bypass_thresh=bypass_thresh,
            )

            history_pairs.append((c1, c2))
            history_gates.append((w1_tok.detach(), w2_tok.detach()))
            history_masks.append(bypass_mask.detach())

            candidate_logits = logits[:, :self.num_nodes - 1]
            probs = F.softmax(candidate_logits, dim=-1)
            log_p = F.log_softmax(candidate_logits, dim=-1)

            # Entropy
            entropy_d = -(probs * log_p).sum(dim=-1).mean()
            total_entropy = total_entropy + entropy_d

            # KL divergence with reference controller
            if ref_controller is not None:
                with torch.no_grad():
                    _, _, _, _, ref_logits, _ = ref_controller(
                        h_seq=h,
                        prev_node=prev_node,
                        g_domain=torch.zeros_like(g_domain),
                        node_prior_bias=torch.zeros_like(node_prior_bias),
                        v_vis=v_vis,
                        r_prev=r_state,
                        tree_level=d,
                        temperature=1.0,
                    )
                    ref_log_p = F.log_softmax(ref_logits[:, :self.num_nodes - 1], dim=-1)
                kl_d = (probs * (log_p - ref_log_p)).sum(dim=-1).mean()
                total_kl = total_kl + kl_d

            log_probs = F.log_softmax(logits, dim=-1)
            lp_1 = log_probs.gather(dim=-1, index=torch.tensor([[c1]], device=device).expand(B, 1)).squeeze(-1)
            lp_2 = log_probs.gather(dim=-1, index=torch.tensor([[c2]], device=device).expand(B, 1)).squeeze(-1)

            # Token gate exploration Bernoulli log-prob & entropy
            r_exp = r_state.unsqueeze(1).expand(B, S, self.token_controller.d_route)
            dom_exp = g_domain.unsqueeze(1).expand(B, S, 1)
            tok_in = torch.cat([h, r_exp, dom_exp], dim=-1)
            tok_logits = self.token_controller.token_gate_head(tok_in)  # [B, S, 1]

            log_p_active = F.logsigmoid(tok_logits)
            log_p_bypass = F.logsigmoid(-tok_logits)
            tok_log_p = torch.where(bypass_mask.unsqueeze(-1), log_p_bypass, log_p_active)
            tok_lp = tok_log_p.mean(dim=(1, 2))

            s_prob = torch.sigmoid(tok_logits)
            tok_ent = -(s_prob * log_p_active + (1.0 - s_prob) * log_p_bypass).mean()
            total_entropy = total_entropy + 0.05 * tok_ent

            history_log_probs.append(lp_1 + lp_2 + 0.1 * tok_lp)

            # Advance representation for state tracking
            h_b1, _ = self.execute_tile(c1, h, d, bypass_mask=bypass_mask)
            h_b2, _ = self.execute_tile(c2, h, d, bypass_mask=None)
            h = self.token_collapse(h, h_b1, h_b2, w1_tok, w2_tok)

            v_vis = v_vis.clone()
            v_vis[:, c1] += 1.0
            v_vis[:, c2] += 1.0
            prev_node = c2

        traj_log_prob = torch.stack(history_log_probs, dim=1).sum(dim=1) if len(history_log_probs) > 0 else torch.zeros(B, device=device)

        if return_diagnostics:
            return history_pairs, history_gates, history_masks, traj_log_prob, slots, total_kl, total_entropy
        return history_pairs, history_gates, history_masks, traj_log_prob, slots

    def forward_token_trajectory(
        self,
        input_ids: torch.Tensor,
        tree_pairs: List[Tuple[int, int]],
        tree_gates: List[Tuple[torch.Tensor, torch.Tensor]],
        tree_masks: Optional[List[torch.Tensor]] = None,
        slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fast feedforward execution across a pre-planned token trajectory."""
        B, S = input_ids.size()
        device = input_ids.device

        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        h = self.embed_tokens(input_ids) + self.embed_positions(pos)

        for d, (c1, c2) in enumerate(tree_pairs):
            w1_tok, w2_tok = tree_gates[d]
            mask = tree_masks[d] if tree_masks is not None else None

            # Handle sequence growth during autoregressive generation
            if w1_tok.size(1) < S:
                pad_len = S - w1_tok.size(1)
                last_w1 = w1_tok[:, -1:, :]
                last_w2 = w2_tok[:, -1:, :]
                w1_tok = torch.cat([w1_tok, last_w1.expand(B, pad_len, 1)], dim=1)
                w2_tok = torch.cat([w2_tok, last_w2.expand(B, pad_len, 1)], dim=1)
                if mask is not None:
                    last_mask = mask[:, -1:]
                    mask = torch.cat([mask, last_mask.expand(B, pad_len)], dim=1)
            elif w1_tok.size(1) > S:
                w1_tok = w1_tok[:, :S, :]
                w2_tok = w2_tok[:, :S, :]
                if mask is not None:
                    mask = mask[:, :S]

            h_b1, _ = self.execute_tile(c1, h, d, bypass_mask=mask)
            h_b2, _ = self.execute_tile(c2, h, d, bypass_mask=None)

            h = self.token_collapse(h, h_b1, h_b2, w1_tok, w2_tok)

        h_norm = self.final_norm(h)
        logits = self.lm_head(h_norm)
        return logits
