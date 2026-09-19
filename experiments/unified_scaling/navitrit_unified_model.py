"""
experiments/unified_scaling/navitrit_unified_model.py: NaviTrit-Unified-Graph-DWP-Max (hardened 2026-09-18).

A parameter-shared (looped) ternary transformer super-block with per-loop dynamic weight parameterisation.
Rationale, ablation matrix and the evaluation protocol live in research/hardening-2026-09-18-heldout.md.

What the model is, in one paragraph:
  4 macro-layers (Mamba SSM -> causal attention -> dual-expert SwiGLU) are applied T=`max_loops` times to
  the residual stream, giving 4*T virtual layers from one set of weights. A ContextHyperNet conditions the
  shared weights on (loop index, role) through zero-initialised FiLM and LoRA adapters ("dynamic weight
  parameterisation"). A per-token router decides how much of each loop's sequence-mixing / channel-mixing
  work a token receives.

Routing modes (config.routing_mode):
  "dense"  no router; every token runs every tile every loop. The fair baseline for every routing claim.
  "soft"   per-token scalar gates (w_seq, w_chan, w_skip) blend branch outputs. All tiles still execute,
           so this mode saves NO compute; it exists to measure whether learned gating helps accuracy.
  "mod"    Mixture-of-Depths style capacity routing: attention + FFN run only on the top-`mod_capacity`
           fraction of tokens per sequence (Mamba, O(S), always runs). This is the only mode whose
           efficiency claim is real. Top-k selection over the sequence is non-causal at training time;
           pass `causal_threshold=True` at generation time to use the per-token sigmoid score instead.

Invariants that changed versus the 2026-09-17 version (and why):
  * `w_chan >= min_chan_weight` was a hard clamp, which zeroes the router gradient whenever the raw
    channel probability is below the floor (i.e. always, at init). It is now the smooth map
    w_chan = m + (1 - m) * p_chan, which satisfies the same floor everywhere and stays differentiable.
  * FiLM now modulates all three sub-block inputs, not just the Mamba input.
  * LoRA adapters now exist for the FFN down projections; the previous model looked up
    `mod_lora["down_{expert}"]` keys that were never generated (dead path).
  * Routing statistics are returned as detached tensors; the previous per-layer `.item()` calls forced
    24 GPU synchronisations per forward pass.
"""

import os
import sys
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.bitlinear import BitLinear
from experiments.bitroute_model import BitRouteRMSNorm
from experiments.mamba.ternary_mamba_block import TernaryMambaBlock, MambaConfig


ROLE_FORWARD_BIND = 0
ROLE_FORWARD_SOLVE = 1
ROLE_BACKWARD_VERIFY = 2
ROLE_SKIP = 3


@dataclass
class NaviTritUnifiedConfig:
    """Configuration for NaviTrit-Unified-Graph-DWP-Max."""
    vocab_size: int = 50257
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    num_macro_layers: int = 4
    max_loops: int = 6                  # virtual depth = num_macro_layers * max_loops
    max_position_embeddings: int = 1024

    # Mamba SSM (use_mamba=False drops the SSM tile: attention-only sequence mixing; see hardening report §7)
    use_mamba: bool = True
    d_state: int = 32
    mamba_expand: int = 2
    dt_rank: int = 64

    # Dual-expert SwiGLU (num_experts=1 gives a plain SwiGLU tile)
    num_experts: int = 2
    balance_loss_weight: float = 0.01

    # Dynamic weight parameterisation
    use_dwp: bool = True
    lora_rank: int = 64
    num_roles: int = 4
    role_sequence: Optional[List[int]] = None   # per-loop role; default schedule built in `roles_for_loops`

    # Routing
    routing_mode: str = "soft"          # "dense" | "soft" | "mod"
    min_chan_weight: float = 0.50       # soft mode floor on channel-mixing weight
    mod_capacity: float = 0.50          # mod mode: fraction of tokens that receive attention + FFN

    # Stage-1 backbone for the traversal experiment (hardening report section 13):
    loop_order_random: bool = False     # permute macro-layer order each loop while training -> order-agnostic tiles
    tile_drop: float = 0.0              # drop each macro-layer with this prob while training (>=1 kept per loop)
    deep_sup_weight: float = 0.0        # per-hop readout loss weight (0 = off)
    deep_sup_frac: float = 0.125        # fraction of positions read out at each intermediate hop
    grad_checkpoint: bool = False       # checkpoint each macro-layer call (training-time memory)
    ternary: bool = True
    ternary_embed: bool = False         # ternarise the tied embedding/head matrix too (arm O experiment)
    tie_word_embeddings: bool = True
    rms_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def virtual_depth(self) -> int:
        return self.num_macro_layers * self.max_loops

    def roles_for_loops(self) -> List[int]:
        """Fixed role curriculum: BIND, SOLVE..., VERIFY, SKIP. An assumption, not a learned schedule."""
        if self.role_sequence is not None:
            assert len(self.role_sequence) == self.max_loops
            return list(self.role_sequence)
        roles = []
        for k in range(self.max_loops):
            if k == 0:
                roles.append(ROLE_FORWARD_BIND)
            elif k < self.max_loops - 2:
                roles.append(ROLE_FORWARD_SOLVE)
            elif k == self.max_loops - 2:
                roles.append(ROLE_BACKWARD_VERIFY)
            else:
                roles.append(ROLE_SKIP)
        return roles


def _apply_lora(x: torch.Tensor, mod_lora: Optional[Dict[str, Any]], key: str) -> torch.Tensor:
    """Parallel low-rank adapter: (x @ A^T) @ B^T, zero at init because B = 0.
    A 3-tuple (A_all [K*r,d], B_all [d,K*r], alpha [..,K*r]) is a per-token soft mixture over a bank of K
    adapters (path-conditioned interpretation, see navitrit_traverse.py)."""
    if mod_lora is None or key not in mod_lora:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    entry = mod_lora[key]
    if len(entry) == 3:
        A, B, alpha = entry
        return ((x @ A.t().to(x.dtype)) * alpha.to(x.dtype)) @ B.t().to(x.dtype)
    A, B = entry
    return (x @ A.t().to(x.dtype)) @ B.t().to(x.dtype)


def _film(u: torch.Tensor, mod_film: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]], key: str) -> torch.Tensor:
    if mod_film is None or key not in mod_film:
        return u
    gamma, beta = mod_film[key]
    return u * (1.0 + gamma.to(u.dtype)) + beta.to(u.dtype)


def embedding_weight(module: nn.Module) -> torch.Tensor:
    """Tied embedding / LM-head matrix. With config.ternary_embed the rows are absmean-ternarised with STE
    (same quantiser as the tiles); otherwise the fp master weight. Section 11 of the hardening report:
    this is an experiment (arm O), not a default; BitNet keeps embeddings in higher precision."""
    w = module.tok_embeddings.weight
    if getattr(module.config, "ternary_embed", False):
        from experiments.bitlinear import absmean_quantize_weights
        w_eff, _, _ = absmean_quantize_weights(w, block_size=256)
        return w_eff
    return w


class FlashAttentionTile(nn.Module):
    """Ternary multi-head causal attention on the fused SDPA kernel."""
    def __init__(self, config: NaviTritUnifiedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.q_proj = BitLinear(self.hidden_size, self.hidden_size, bias=False, ternary=config.ternary)
        self.k_proj = BitLinear(self.hidden_size, self.hidden_size, bias=False, ternary=config.ternary)
        self.v_proj = BitLinear(self.hidden_size, self.hidden_size, bias=False, ternary=config.ternary)
        self.o_proj = BitLinear(self.hidden_size, self.hidden_size, bias=False, ternary=config.ternary)

    def forward(self, x: torch.Tensor, mod_lora: Optional[Dict[str, Any]] = None, attn_mask: Optional[torch.Tensor] = None,
                return_received: bool = False):
        """attn_mask: optional boolean [B,1,S,S] (True = may attend); replaces the causal flag for gathered subsets.
        return_received: also return attention-received mass per key [B,S] (liveness signal, section 13)."""
        B, S, D = x.shape
        self.last_received = None
        q = self.q_proj(x) + _apply_lora(x, mod_lora, "q")
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        if return_received:
            scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)      # [B,h,S,S]
            if attn_mask is None:
                attn_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device)).view(1, 1, S, S)
            scores = scores.masked_fill(~attn_mask, float("-inf"))
            w = torch.softmax(scores, dim=-1)
            out = (w.to(v.dtype) @ v)
            received = w.mean(dim=1).sum(dim=-2)                                                  # [B,S]: mass each key received
        elif attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out) + _apply_lora(x, mod_lora, "o")
        if return_received:
            return out, received
        return out


class DualExpertSwiGLUTile(nn.Module):
    """SwiGLU FFN with `num_experts` softly-mixed experts. Both experts always execute (no MoE saving)."""
    def __init__(self, config: NaviTritUnifiedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.w_gate = nn.ModuleList()
        self.w_up = nn.ModuleList()
        self.w_down = nn.ModuleList()
        for _ in range(self.num_experts):
            self.w_gate.append(BitLinear(self.hidden_size, self.intermediate_size, bias=False, ternary=config.ternary))
            self.w_up.append(BitLinear(self.hidden_size, self.intermediate_size, bias=False, ternary=config.ternary))
            self.w_down.append(BitLinear(self.intermediate_size, self.hidden_size, bias=False, ternary=config.ternary))
        if self.num_experts > 1:
            self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
            nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def _expert_forward(self, x: torch.Tensor, i: int, mod_lora: Optional[Dict[str, Any]], gate_act=F.silu) -> torch.Tensor:
        out = self.w_down[i](gate_act(self.w_gate[i](x)) * self.w_up[i](x))
        return out + _apply_lora(x, mod_lora, f"down_{i}")

    def forward(self, x: torch.Tensor, mod_lora: Optional[Dict[str, Any]] = None, gate_act=F.silu) -> Tuple[torch.Tensor, torch.Tensor]:
        """gate_act: activation on the SwiGLU gate; the traverse model's strategy selector passes a per-token mixture."""
        if self.num_experts == 1:
            return self._expert_forward(x, 0, mod_lora, gate_act), torch.zeros((), device=x.device, dtype=x.dtype)
        probs = F.softmax(self.router(x), dim=-1)
        self.last_expert_probs = probs.detach()
        combined = sum(probs[..., i:i + 1] * self._expert_forward(x, i, mod_lora, gate_act) for i in range(self.num_experts))
        avg_p = probs.mean(dim=[0, 1])
        balance_loss = self.num_experts * torch.sum(avg_p ** 2) - 1.0
        return combined, balance_loss


class UnifiedMacroLayer(nn.Module):
    """
    One macro-layer of the recurrent super-block: Mamba SSM -> causal attention -> SwiGLU, with a
    per-token router (see module docstring for the three routing modes).
    """
    def __init__(self, config: NaviTritUnifiedConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mamba = None
        if config.use_mamba:
            self.norm_mamba = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mamba = TernaryMambaBlock(MambaConfig(
                d_model=config.hidden_size, d_state=config.d_state, expand=config.mamba_expand,
                dt_rank=config.dt_rank, ternary=config.ternary,
            ))
        self.norm_attn = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = FlashAttentionTile(config)
        self.norm_ffn = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dual_ffn = DualExpertSwiGLUTile(config)

        self.norm_graph = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        router_out = {"dense": 0, "soft": 3, "mod": 1}[config.routing_mode]
        if router_out:
            self.graph_router = nn.Linear(config.hidden_size, router_out, bias=False)
            nn.init.normal_(self.graph_router.weight, mean=0.0, std=0.02)

    # ---- routing weights ------------------------------------------------------------------
    def _soft_weights(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = F.softmax(self.graph_router(self.norm_graph(h)).float(), dim=-1)  # [B,S,3] (seq, chan, skip)
        m = self.config.min_chan_weight
        w_chan = m + (1.0 - m) * p[..., 1:2]                # smooth floor, differentiable everywhere
        rest = 1.0 - w_chan
        denom = (p[..., 0:1] + p[..., 2:3]).clamp(min=1e-6)
        w_seq = rest * p[..., 0:1] / denom
        w_skip = rest * p[..., 2:3] / denom
        return w_seq.to(h.dtype), w_chan.to(h.dtype), w_skip.to(h.dtype)

    # ---- forward --------------------------------------------------------------------------
    def forward(
        self,
        h: torch.Tensor,
        mod_film: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        mod_lora: Optional[Dict[str, Any]] = None,
        causal_threshold: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        mode = self.config.routing_mode
        B, S, D = h.shape

        # Sequence mixing part 1: Mamba always runs (O(S)) when present.
        h_mamba = self.mamba(_film(self.norm_mamba(h), mod_film, "mamba")) if self.mamba is not None else torch.zeros_like(h)

        if mode == "mod":
            score = torch.sigmoid(self.graph_router(self.norm_graph(h)).float()).squeeze(-1)  # [B,S]
            if causal_threshold:
                sel_mask = score > 0.5
                k = int(sel_mask.sum(dim=1).max().item()) if S > 0 else 0
                k = max(k, 1)
                idx = torch.topk(score.masked_fill(~sel_mask, -1.0), k, dim=1).indices
            else:
                k = max(1, int(math.ceil(self.config.mod_capacity * S)))
                idx = torch.topk(score, k, dim=1).indices
            idx, _ = idx.sort(dim=1)                                   # keep original order -> causal within subset
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, D)
            h_base = h + h_mamba
            h_sel = torch.gather(h_base, 1, gather_idx)                 # [B,k,D]
            attn_sel = self.attn(_film(self.norm_attn(h_sel), mod_film, "attn"), mod_lora)
            h_sel2 = h_sel + attn_sel
            ffn_sel, balance_loss = self.dual_ffn(_film(self.norm_ffn(h_sel2), mod_film, "ffn"), mod_lora)
            gate = torch.gather(score, 1, idx).unsqueeze(-1).to(h.dtype)  # router gradient path
            delta_sel = gate * (attn_sel + ffn_sel)
            h_next = h_base.scatter_add(1, gather_idx, delta_sel)
            stats = {
                "w_seq": torch.tensor(1.0, device=h.device),
                "w_chan": torch.tensor(1.0, device=h.device),
                "w_skip": torch.tensor(1.0 - k / S, device=h.device),
                "tokens_processed_frac": torch.tensor(k / S, device=h.device),
                "router_score_mean": score.detach().mean(),
            }
            return h_next, balance_loss, stats

        # dense / soft: every tile runs on every token
        h_attn = self.attn(_film(self.norm_attn(h + h_mamba), mod_film, "attn"), mod_lora)
        branch_seq = h_mamba + h_attn
        branch_chan, balance_loss = self.dual_ffn(_film(self.norm_ffn(h + branch_seq), mod_film, "ffn"), mod_lora)

        if mode == "dense":
            h_next = h + branch_seq + branch_chan
            one = torch.tensor(1.0, device=h.device)
            stats = {"w_seq": one, "w_chan": one, "w_skip": torch.tensor(0.0, device=h.device)}
            return h_next, balance_loss, stats

        w_seq, w_chan, w_skip = self._soft_weights(h)
        h_next = h + (1.0 - w_skip) * (w_seq * branch_seq + w_chan * branch_chan)
        stats = {"w_seq": w_seq.detach().mean(), "w_chan": w_chan.detach().mean(), "w_skip": w_skip.detach().mean()}
        return h_next, balance_loss, stats


class ContextHyperNet(nn.Module):
    """
    Dynamic weight parameteriser: (loop index, role) -> FiLM (mamba/attn/ffn inputs) + LoRA (q, o, down_i).
    Zero-drift initialisation: FiLM heads and every LoRA B are exactly zero, so at init the looped model
    is a plain weight-tied transformer.
    """
    def __init__(self, config: NaviTritUnifiedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.lora_rank = config.lora_rank
        self.max_loops = config.max_loops
        self.num_roles = config.num_roles
        self.num_experts = config.num_experts
        self.step_embed = nn.Embedding(self.max_loops, 64)
        self.role_embed = nn.Embedding(self.num_roles, 64)
        self.film_targets = ("mamba", "attn", "ffn")
        self.film_mlp = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 2 * len(self.film_targets) * self.hidden_size),
        )
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        self.lora_targets = ["q", "o"] + [f"down_{i}" for i in range(self.num_experts)]
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for k in range(self.max_loops):
            for r in range(self.num_roles):
                for t in self.lora_targets:
                    key = f"{t}_k{k}_r{r}"
                    self.lora_A[f"A_{key}"] = nn.Parameter(torch.randn(self.lora_rank, self.hidden_size) * 0.01)
                    self.lora_B[f"B_{key}"] = nn.Parameter(torch.zeros(self.hidden_size, self.lora_rank))

    def get_modulations(self, loop_idx: int, role_idx: int) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
        cond = torch.cat([self.step_embed.weight[loop_idx:loop_idx + 1], self.role_embed.weight[role_idx:role_idx + 1]], dim=-1)
        film_raw = self.film_mlp(cond).view(len(self.film_targets), 2, self.hidden_size)
        mod_film = {t: (film_raw[i, 0:1], film_raw[i, 1:2]) for i, t in enumerate(self.film_targets)}
        mod_lora = {t: (self.lora_A[f"A_{t}_k{loop_idx}_r{role_idx}"], self.lora_B[f"B_{t}_k{loop_idx}_r{role_idx}"]) for t in self.lora_targets}
        return mod_film, mod_lora


class NaviTritUnifiedForCausalLM(nn.Module):
    """Looped ternary causal LM with dynamic weight parameterisation and per-token routing."""
    def __init__(self, config: NaviTritUnifiedConfig):
        super().__init__()
        assert config.routing_mode in ("dense", "soft", "mod"), config.routing_mode
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_macro_layers = config.num_macro_layers
        self.max_loops = config.max_loops

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)
        self.macro_layers = nn.ModuleList([UnifiedMacroLayer(config, i) for i in range(config.num_macro_layers)])
        self.hypernet = ContextHyperNet(config) if config.use_dwp else None
        self.final_norm = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.roles = config.roles_for_loops()

    def count_parameters(self) -> Dict[str, Any]:
        embed = sum(p.numel() for p in self.tok_embeddings.parameters()) + sum(p.numel() for p in self.pos_embeddings.parameters())
        macro = sum(p.numel() for p in self.macro_layers.parameters())
        hyper = sum(p.numel() for p in self.hypernet.parameters()) if self.hypernet is not None else 0
        norm = sum(p.numel() for p in self.final_norm.parameters())
        total = embed + macro + hyper + norm
        # ternary tiles at 1.58 bit; hypernet adapters and embeddings are not ternary (16 bit)
        packed_mb = (macro * 1.58 + (hyper + embed) * 16.0) / (8 * 1024 * 1024)
        return {
            "total_physical_parameters": total, "total_millions": round(total / 1e6, 2),
            "embeddings_params": embed, "superblock_macro_params": macro, "hypernet_dwp_params": hyper,
            "packed_ternary_mb": round(packed_mb, 2), "superblock_packed_mb": round(macro * 1.58 / (8 * 1024 * 1024), 2),
            "virtual_layers_count": self.config.virtual_depth, "routing_mode": self.config.routing_mode,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        max_loops: Optional[int] = None,
        causal_threshold: bool = False,
    ) -> Dict[str, Any]:
        B, S = input_ids.shape
        device = input_ids.device
        T = max_loops if max_loops is not None else self.max_loops
        positions = torch.arange(0, S, device=device).unsqueeze(0)
        h = F.embedding(input_ids, embedding_weight(self)) + self.pos_embeddings(positions)

        total_balance = torch.zeros((), device=device)
        acc = {"w_seq": torch.zeros((), device=device), "w_chan": torch.zeros((), device=device), "w_skip": torch.zeros((), device=device)}
        n_evals = 0
        deep_sup = torch.zeros((), device=device)
        n_deep = 0
        cfg = self.config
        randomise = self.training and (cfg.loop_order_random or cfg.tile_drop > 0.0)
        for k in range(T):
            mod_film, mod_lora = (None, None)
            if self.hypernet is not None:
                mod_film, mod_lora = self.hypernet.get_modulations(k, self.roles[k])
            order = list(range(self.num_macro_layers))
            if randomise:
                if cfg.loop_order_random:
                    random.shuffle(order)
                if cfg.tile_drop > 0.0:
                    kept = [i for i in order if random.random() >= cfg.tile_drop]
                    order = kept if kept else [order[0]]
            for idx in order:
                layer = self.macro_layers[idx]
                if cfg.grad_checkpoint and self.training:
                    from torch.utils.checkpoint import checkpoint as _ckpt
                    # Modulations are passed as explicit arguments, never captured by closure: closure-captured
                    # hypernet tensors gave ~300x inflated gradients under recomputation (found 2026-09-19, C-cont).
                    h, bal, st = _ckpt(lambda h_in, mf, ml, layer=layer: layer(h_in, mod_film=mf, mod_lora=ml, causal_threshold=causal_threshold),
                                       h, mod_film, mod_lora, use_reentrant=False)
                else:
                    h, bal, st = layer(h, mod_film=mod_film, mod_lora=mod_lora, causal_threshold=causal_threshold)
                total_balance = total_balance + bal
                for key in acc:
                    acc[key] = acc[key] + st[key]
                n_evals += 1
                # Deep supervision: tied-head readout on a random subset of positions after every intermediate hop
                is_last = (k == T - 1) and (idx == order[-1])
                if self.training and cfg.deep_sup_weight > 0.0 and labels is not None and not is_last:
                    n_pos = max(1, int(cfg.deep_sup_frac * (S - 1)))
                    pos = torch.randperm(S - 1, device=device)[:n_pos]
                    h_sub = self.final_norm(h[:, pos])
                    lg = F.linear(h_sub, embedding_weight(self)).float()
                    tgt = labels[:, pos + 1]
                    deep_sup = deep_sup + F.cross_entropy(lg.reshape(-1, self.vocab_size), tgt.reshape(-1), ignore_index=-100)
                    n_deep += 1

        h = self.final_norm(h)
        logits = F.linear(h, embedding_weight(self))

        loss, ce_loss = None, None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size).float(), shift_labels.view(-1), ignore_index=-100)
            loss = ce_loss + self.config.balance_loss_weight * total_balance
            if n_deep > 0:
                loss = loss + self.config.deep_sup_weight * deep_sup / n_deep

        n = max(1, n_evals)
        return {
            "logits": logits, "loss": loss,
            "ce_loss": ce_loss if ce_loss is not None else torch.zeros((), device=device),
            "balance_loss": total_balance, "deep_sup_loss": (deep_sup / max(1, n_deep)).item(),
            "avg_w_seq": (acc["w_seq"] / n).item(), "avg_w_chan": (acc["w_chan"] / n).item(), "avg_w_skip": (acc["w_skip"] / n).item(),
            "virtual_layers": self.config.num_macro_layers * T,
        }
