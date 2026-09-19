"""
experiments/general_model/general_model.py: GeneralRoutedLM, a hybrid Mamba/attention ternary tile stack with
per-token adaptive depth.

The objective of this track is the smallest STORED byte footprint that reaches a loss/BPB target, so every
structural choice is exposed as a config toggle and the parameter accounting is in bytes, not parameters.
Rationale, ablation motivation and the report sections behind each toggle live in
`experiments/general_model/AGENTS.md`.

Everything reusable is imported, never copied:
  * `BitLinear` / `absmean_quantize_weights`  ternary STE            (experiments/bitlinear.py)
  * `BitRouteRMSNorm`                                                (experiments/bitroute_model.py)
  * `TernaryMambaBlock` / `MambaConfig`       O(S) sequence mixer    (experiments/mamba/ternary_mamba_block.py)
  * `FlashAttentionTile`, `DualExpertSwiGLUTile`, `_apply_lora`, `_film`, `embedding_weight`,
    `NaviTritUnifiedConfig`                                          (experiments/unified_scaling/navitrit_unified_model.py)
  * `PathAdapterBank`, `NaviTritTraverseForCausalLM._gather_subset`  (experiments/unified_scaling/navitrit_traverse.py)
"""

import os
import sys
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.bitlinear import BitLinear
from experiments.bitroute_model import BitRouteRMSNorm
from experiments.mamba.ternary_mamba_block import MambaConfig, TernaryMambaBlock
from experiments.unified_scaling.navitrit_unified_model import (
    DualExpertSwiGLUTile,
    FlashAttentionTile,
    NaviTritUnifiedConfig,
    _apply_lora,
    _film,
    embedding_weight,
)
from experiments.unified_scaling.navitrit_traverse import (
    NaviTritTraverseForCausalLM,
    PathAdapterBank,
)

# Gathered-subset helper: sel [B,S] bool -> (positions [B,k] in sorted causal order, valid [B,k], k).
gather_subset = NaviTritTraverseForCausalLM._gather_subset

TERNARY_BITS = 1.58
DENSE_BITS = 16.0


@dataclass
class GeneralConfig:
    """Geometry, routing policy and ablation toggles for `GeneralRoutedLM`."""

    # --- geometry ---
    vocab_size: int = 16384
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    max_position_embeddings: int = 1024
    tile_kinds: List[str] = field(default_factory=lambda: ["mamba", "mamba", "mamba", "attn"])
    num_experts: int = 2
    d_state: int = 16
    mamba_expand: int = 2
    dt_rank: int = 32
    d_conv: int = 4
    ternary: bool = True
    ternary_embed: bool = False
    tie_word_embeddings: bool = True
    rms_norm_eps: float = 1e-5

    # --- routing ---
    routing: str = "free"           # "free": per-token choice of any tile at every hop (non-monotonic)
                                    # "fixed": hop h always uses tile h % M (value exit still per token)
    max_hops: int = 8
    min_hops: int = 2
    exit_mode: str = "value"        # "value": continue iff predicted gain > exit_lambda | "none": always max_hops
    exit_lambda: float = 0.02
    explore_prob: float = 0.3       # training only: continue past a value-rule exit with this probability
    exit_warmup_steps: int = 500    # trainer flips `model.allow_exit` once this many steps are done
    gate_ste: bool = True           # tiles run at full magnitude, router still receives gradient
    router_temperature: float = 1.0

    # --- conditioning ---
    router_cond: str = "state"      # "state": RMSNorm(h) projection + hop embedding | "path": also a GRU path state
    adapter_cond: str = "hop"       # "hop": per-hop mixture over the adapter bank | "path": path-conditioned
    path_dim: int = 64
    adapter_bank: int = 4
    adapter_rank: int = 16
    hop_feature_dropout: float = 0.0

    # --- inside a tile ---
    tile_soft_mix: bool = True      # per-token 2-way softmax weighting mixer branch vs FFN branch
    min_chan_weight: float = 0.5    # smooth floor on the FFN branch weight

    # --- losses ---
    balance_loss_weight: float = 0.01
    hop_cost_weight: float = 0.0
    value_loss_weight: float = 1.0
    deep_sup_weight: float = 0.1
    deep_sup_frac: float = 0.125

    # --- runtime ---
    grad_checkpoint: bool = False

    # --- eval-time exit policy (swept by the trainer read-out; training always uses "threshold") ---
    exit_policy: str = "threshold"  # "threshold" | "random" | "capacity"
    random_continue_prob: float = 0.5
    capacity_frac: float = 0.5

    @property
    def num_tiles(self) -> int:
        return len(self.tile_kinds)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_unified(self) -> NaviTritUnifiedConfig:
        """Geometry view consumed by the imported attention / SwiGLU tiles."""
        return NaviTritUnifiedConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_macro_layers=self.num_tiles,
            max_loops=self.max_hops,
            max_position_embeddings=self.max_position_embeddings,
            use_mamba=False,
            d_state=self.d_state,
            mamba_expand=self.mamba_expand,
            dt_rank=self.dt_rank,
            num_experts=self.num_experts,
            balance_loss_weight=self.balance_loss_weight,
            routing_mode="dense",
            lora_rank=self.adapter_rank,
            ternary=self.ternary,
            ternary_embed=self.ternary_embed,
            tie_word_embeddings=self.tie_word_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            grad_checkpoint=self.grad_checkpoint,
        )


class HybridTile(nn.Module):
    """One routed tile: RMSNorm -> sequence mixer (Mamba or attention) -> residual -> RMSNorm -> dual-expert
    SwiGLU -> residual. Runs on a gathered subset [B,k,d] whose rows are in original causal order with
    padding at the tail (see AGENTS.md "Why Mamba on a gathered subset is causal")."""

    def __init__(self, config: GeneralConfig, unified: NaviTritUnifiedConfig, kind: str):
        super().__init__()
        assert kind in ("mamba", "attn"), kind
        self.kind = kind
        self.config = config
        self.norm_mixer = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if kind == "mamba":
            self.mixer = TernaryMambaBlock(MambaConfig(
                d_model=config.hidden_size, d_state=config.d_state, expand=config.mamba_expand,
                dt_rank=config.dt_rank, d_conv=config.d_conv, ternary=config.ternary,
            ))
        else:
            self.mixer = FlashAttentionTile(unified)
        self.norm_ffn = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = DualExpertSwiGLUTile(unified)
        if config.tile_soft_mix:
            self.branch_router = nn.Linear(config.hidden_size, 2, bias=False)
            nn.init.normal_(self.branch_router.weight, mean=0.0, std=0.02)

    def _branch_weights(self, normed: torch.Tensor):
        """Per-token (w_mixer, w_ffn) with a smooth differentiable floor on w_ffn; scalars when disabled."""
        if not self.config.tile_soft_mix:
            return 1.0, 1.0
        probs = F.softmax(self.branch_router(normed).float(), dim=-1)
        floor = self.config.min_chan_weight
        w_ffn = floor + (1.0 - floor) * probs[..., 1:2]
        w_mixer = 1.0 - w_ffn
        return w_mixer.to(normed.dtype), w_ffn.to(normed.dtype)

    @staticmethod
    def _subset_attention_mask(valid: torch.Tensor) -> torch.Tensor:
        """Causal + padding mask [B,1,k,k]; padded queries attend to themselves so softmax never sees all -inf."""
        B, k = valid.shape
        device = valid.device
        causal = torch.tril(torch.ones(k, k, dtype=torch.bool, device=device))
        mask = causal.unsqueeze(0) & valid.unsqueeze(1)
        mask = mask | torch.eye(k, dtype=torch.bool, device=device).unsqueeze(0)
        return mask.unsqueeze(1)

    def forward(
        self,
        h_sub: torch.Tensor,
        valid: torch.Tensor,
        mod_film: Optional[Dict[str, Any]] = None,
        mod_lora: Optional[Dict[str, Any]] = None,
        need_received: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        normed = self.norm_mixer(h_sub)
        w_mixer, w_ffn = self._branch_weights(normed)
        mixer_in = _film(normed, mod_film, "attn")          # the bank's sequence-mixer FiLM slot
        received = None
        if self.kind == "attn":
            mask = self._subset_attention_mask(valid)
            if need_received:
                mixer_out, received = self.mixer(mixer_in, mod_lora, attn_mask=mask, return_received=True)
            else:
                mixer_out = self.mixer(mixer_in, mod_lora, attn_mask=mask)
        else:
            # Mamba takes no adapter dict, so the bank's q/o slots become pre- and post-adapters.
            pre = mixer_in + _apply_lora(mixer_in, mod_lora, "q")
            mixer_out = self.mixer(pre) + _apply_lora(mixer_in, mod_lora, "o")
        mixer_delta = w_mixer * mixer_out
        h_mid = h_sub + mixer_delta
        ffn_out, balance_loss = self.ffn(_film(self.norm_ffn(h_mid), mod_film, "ffn"), mod_lora)
        delta = mixer_delta + w_ffn * ffn_out
        return delta, balance_loss, {"received": received}


class GeneralRoutedLM(nn.Module):
    """Ternary tile stack with per-token tile routing and a value-based per-token exit."""

    def __init__(self, config: GeneralConfig):
        super().__init__()
        assert config.routing in ("free", "fixed"), config.routing
        assert config.exit_mode in ("value", "none"), config.exit_mode
        assert config.router_cond in ("state", "path"), config.router_cond
        assert config.adapter_cond in ("hop", "path"), config.adapter_cond
        self.config = config
        self.unified = config.to_unified()
        self.hidden = config.hidden_size
        self.vocab_size = config.vocab_size
        self.M = config.num_tiles
        self.EXIT = self.M
        self.use_value = config.exit_mode == "value"
        self.use_path = config.router_cond == "path" or config.adapter_cond == "path"
        self.allow_exit = config.exit_warmup_steps == 0
        self.force_random_tiles = False     # diversity probe: ignore the tile router, pick tiles at random
        self.force_tile = None              # test/diagnostic hook: send every token to one tile
        self.collect_received = False       # attention-received mass per key (off by default: forces the dense softmax path)

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)

        self.tiles = nn.ModuleList([HybridTile(config, self.unified, k) for k in config.tile_kinds])
        self.banks = nn.ModuleList([
            PathAdapterBank(config.hidden_size, config.num_experts, config.adapter_bank, config.adapter_rank)
            for _ in range(self.M)
        ])
        self.final_norm = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # controller
        self.hop_embed = nn.Embedding(config.max_hops + 1, config.path_dim)
        self.norm_route = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.state_proj = nn.Linear(config.hidden_size, config.path_dim)
        if self.use_path:
            self.tile_embed = nn.Embedding(self.M + 1, config.path_dim)     # +1: START / EXIT slot
            self.path_cell = nn.GRUCell(config.path_dim, config.path_dim)
        if config.routing == "free":
            self.tile_router = nn.Sequential(
                nn.Linear(2 * config.path_dim, config.path_dim), nn.GELU(),
                nn.Linear(config.path_dim, self.M),
            )
            # small but non-zero: exactly-zero logits tie and argmax sends every token to tile 0
            nn.init.normal_(self.tile_router[-1].weight, std=0.02)
            nn.init.zeros_(self.tile_router[-1].bias)
        self.adapter_router = nn.Linear(config.path_dim, config.adapter_bank)
        nn.init.normal_(self.adapter_router.weight, std=0.02)
        nn.init.zeros_(self.adapter_router.bias)
        if self.use_value:
            self.value_head = nn.Sequential(
                nn.Linear(2 * config.path_dim, config.path_dim // 2), nn.GELU(),
                nn.Linear(config.path_dim // 2, 1),
            )

    # ------------------------------------------------------------------ footprint ----------------
    def count_parameters(self) -> Dict[str, Any]:
        """Stored-byte accounting: ternary tile weights at 1.58 bit, everything else at 16 bit.
        `stored_mb_total` is the objective of this track."""
        tiles_all = sum(p.numel() for p in self.tiles.parameters())
        tiles_ternary = 0
        for module in self.tiles.modules():
            if isinstance(module, BitLinear) and module.ternary:
                tiles_ternary += module.weight.numel()
        params = {
            "tiles_ternary": tiles_ternary,
            "tiles_dense": tiles_all - tiles_ternary,     # Mamba conv/x_proj/dt_proj/A/D, norms, tile routers
            "token_embed": self.tok_embeddings.weight.numel(),
            "position_embed": self.pos_embeddings.weight.numel(),
            "adapters": sum(p.numel() for p in self.banks.parameters()),
            "controller": sum(p.numel() for p in self._controller_parameters()),
        }
        token_embed_bits = TERNARY_BITS if self.config.ternary_embed else DENSE_BITS
        bits_per_group = {
            "tiles_ternary": TERNARY_BITS if self.config.ternary else DENSE_BITS,
            "tiles_dense": DENSE_BITS,
            "token_embed": token_embed_bits,
            "position_embed": DENSE_BITS,
            "adapters": DENSE_BITS,
            "controller": DENSE_BITS,
        }
        stored_bytes = {k: params[k] * bits_per_group[k] / 8.0 for k in params}
        stored_mb = {k: v / (1024 * 1024) for k, v in stored_bytes.items()}
        total = sum(params.values())    # controller already carries final_norm
        return {
            "total_physical_parameters": int(sum(p.numel() for p in self.parameters())),
            "total_millions": round(sum(p.numel() for p in self.parameters()) / 1e6, 3),
            "params": params,
            "bits_per_group": bits_per_group,
            "stored_bytes": stored_bytes,       # exact: the ledger and the tests compare ratios
            "stored_mb": {k: round(v, 4) for k, v in stored_mb.items()},
            "stored_mb_total": round(sum(stored_mb.values()), 4),
            "tiles_packed_mb": round(stored_mb["tiles_ternary"] + stored_mb["tiles_dense"], 4),
            "embed_packed_mb": round(stored_mb["token_embed"] + stored_mb["position_embed"], 4),
            "accounted_params": int(total),
            "tile_kinds": list(self.config.tile_kinds),
            "routing": self.config.routing,
            "exit_mode": self.config.exit_mode,
            "virtual_layers_count": self.config.max_hops,
        }

    def _controller_parameters(self):
        """Router / value / path parameters plus the final norm: everything outside tiles, banks, embeddings."""
        names = ("hop_embed", "norm_route", "state_proj", "tile_embed", "path_cell",
                 "tile_router", "adapter_router", "value_head", "final_norm")
        for name, param in self.named_parameters():
            if name.split(".")[0] in names:
                yield param

    def load_dense_checkpoint(self, path: str) -> None:
        """Not implemented: this track trains from scratch, and no dense checkpoint shares this tile layout
        (hybrid Mamba/attention tiles with a soft branch mix). Kept so the trainer interface matches."""
        raise NotImplementedError(
            f"GeneralRoutedLM has no dense counterpart to warm-start from (asked for {path}); "
            "train from scratch or add an explicit key mapping first."
        )

    # ------------------------------------------------------------------ helpers ------------------
    def _sample_loss(self, h: torch.Tensor, labels: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Per-token CE at sampled positions [B,n] through the tied head (deep supervision + gain targets)."""
        logits = F.linear(self.final_norm(h[:, pos]), embedding_weight(self)).float()
        target = labels[:, pos + 1]
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), target.reshape(-1),
                               reduction="none", ignore_index=-100).view(h.size(0), -1)

    def _cond(self, h: torch.Tensor, path: Optional[torch.Tensor], hop: int) -> torch.Tensor:
        """Router / value input: projected residual state concatenated with path history or the hop index."""
        state = self.state_proj(self.norm_route(h))
        if self.config.router_cond == "path" and path is not None:
            context = path
        else:
            context = self._hop_vector(hop).to(state.dtype).expand_as(state)
        return torch.cat([state, context], dim=-1)

    def _hop_vector(self, hop: int) -> torch.Tensor:
        """Hop embedding row, clamped so a shortened hop budget never indexes past the table."""
        return self.hop_embed.weight[min(hop, self.config.max_hops)]

    def _alpha(self, path: Optional[torch.Tensor], hop: int, shape: Tuple[int, int]) -> torch.Tensor:
        """Mixture weights [B,S,K] over the adapter bank."""
        if self.config.adapter_cond == "path" and path is not None:
            source = path
        else:
            source = self._hop_vector(hop).view(1, 1, -1).expand(shape[0], shape[1], -1)
        return F.softmax(self.adapter_router(source), dim=-1)

    def _run_tile(self, tile_index: int, h_sub: torch.Tensor, valid: torch.Tensor, alpha: torch.Tensor,
                  need_received: bool):
        """Per-tile gradient checkpointing: the SwiGLU intermediates are d_ff-wide per hop and dominate
        activation memory once several hops are live."""
        if self.config.grad_checkpoint and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint as _ckpt
            return _ckpt(self._run_tile_impl, tile_index, h_sub, valid, alpha, need_received, use_reentrant=False)
        return self._run_tile_impl(tile_index, h_sub, valid, alpha, need_received)

    def _run_tile_impl(self, tile_index: int, h_sub: torch.Tensor, valid: torch.Tensor, alpha: torch.Tensor,
                       need_received: bool):
        mod_film, mod_lora = self.banks[tile_index].modulations(alpha)
        return self.tiles[tile_index](h_sub, valid, mod_film, mod_lora, need_received=need_received)

    # ------------------------------------------------------------------ forward ------------------
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                max_loops: Optional[int] = None, probe: bool = False, **_) -> Dict[str, Any]:
        """probe=True additionally returns per-token loss at min_hops and at the end, plus the final hidden state."""
        cfg = self.config
        B, S = input_ids.shape
        device = input_ids.device
        # a batch of seq_len tokens is fed as seq_len+1 (input and shifted labels share the tensor)
        assert S <= cfg.max_position_embeddings, (
            f"sequence of {S} exceeds max_position_embeddings={cfg.max_position_embeddings}")
        H = max_loops if max_loops is not None else cfg.max_hops

        positions = torch.arange(S, device=device).unsqueeze(0)
        h = F.embedding(input_ids, embedding_weight(self)) + self.pos_embeddings(positions)
        path = None
        if self.use_path:
            start = self.tile_embed.weight[self.M].expand(B * S, -1)
            path = self.path_cell(start, torch.zeros(B * S, cfg.path_dim, device=device)).view(B, S, -1)

        active = torch.ones(B, S, dtype=torch.bool, device=device)
        balance = torch.zeros((), device=device)
        hops_executed = torch.zeros((), device=device)
        tile_usage = torch.zeros(self.M + 1, device=device)
        deep_sup = torch.zeros((), device=device)
        value_loss = torch.zeros((), device=device)
        gain_sum = torch.zeros((), device=device)
        gain_n = torch.zeros((), device=device)
        pred_gain_sum = torch.zeros((), device=device)
        n_deep = 0
        token_loss_min = None

        use_ds = (cfg.deep_sup_weight > 0.0 or self.use_value) and labels is not None
        if use_ds:
            n_pos = max(1, int(cfg.deep_sup_frac * (S - 1)))
            if self.training:
                ds_pos = torch.randperm(S - 1, device=device)[:n_pos]
            else:
                ds_pos = torch.arange(0, S - 1, max(1, (S - 1) // n_pos), device=device)[:n_pos]
            with torch.no_grad():
                loss_prev = self._sample_loss(h, labels, ds_pos)

        for hop in range(H):
            if probe and hop == cfg.min_hops and labels is not None:
                with torch.no_grad():
                    token_loss_min = self._sample_loss(h, labels, torch.arange(S - 1, device=device))

            cond = self._cond(h, path, hop)
            pred_gain = None
            if self.use_value:
                pred_gain = self.value_head(cond).float().squeeze(-1)
                pred_gain_sum = pred_gain_sum + (pred_gain.detach() * active.float()).sum()

            # --- tile choice ---
            if cfg.routing == "free":
                logits = self.tile_router(cond).float() / cfg.router_temperature
                if self.training:                       # Gumbel noise on the hard choice; probs stay clean for the gate
                    logits = logits - torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-9)))
                probs = F.softmax(logits, dim=-1)       # [B,S,M]
                choice = probs.argmax(dim=-1)
            else:                                       # fixed: hop h uses tile h % M, no router at all
                probs = None
                choice = torch.full((B, S), hop % self.M, dtype=torch.long, device=device)
            if self.force_random_tiles:
                choice = torch.randint(0, self.M, (B, S), device=device)
            if self.force_tile is not None:
                choice = torch.full((B, S), int(self.force_tile), dtype=torch.long, device=device)

            # --- value-based exit ---
            if self.use_value and hop >= cfg.min_hops and self.allow_exit:
                if cfg.exit_policy == "threshold":
                    cont = pred_gain.detach() > cfg.exit_lambda
                    if self.training and cfg.explore_prob > 0.0:
                        cont = cont | (torch.rand(B, S, device=device) < cfg.explore_prob)
                elif cfg.exit_policy == "random":
                    cont = torch.rand(B, S, device=device) < cfg.random_continue_prob
                else:                                   # capacity: top fraction of active tokens by predicted gain
                    score = pred_gain.detach().masked_fill(~active, float("-inf"))
                    k_keep = max(1, int(cfg.capacity_frac * S))
                    threshold = score.topk(k_keep, dim=1).values[:, -1:]
                    cont = score >= threshold
                choice = torch.where(cont, choice, torch.full_like(choice, self.EXIT))
            choice = torch.where(active, choice, torch.full_like(choice, self.EXIT))

            # --- Switch-style balance over tiles (free routing only; fixed routing has no tile decision) ---
            active_f = active.float()
            n_active = active_f.sum().clamp(min=1.0)
            if probs is not None:
                frac = torch.stack([((choice == m).float() * active_f).sum() for m in range(self.M)]) / n_active
                p_mean = (probs * active_f.unsqueeze(-1)).sum(dim=(0, 1)) / n_active
                balance = balance + self.M * (frac * p_mean).sum()
            tile_usage = tile_usage + torch.bincount(choice[active].flatten(), minlength=self.M + 1).float()

            alpha = self._alpha(path, hop, (B, S))
            h_new = h
            for m in range(self.M):
                sel = (choice == m) & active
                if not bool(sel.any()):
                    continue
                order, valid, k = gather_subset(sel)
                gather_index = order.unsqueeze(-1).expand(-1, -1, self.hidden)
                h_sub = torch.gather(h, 1, gather_index)
                alpha_sub = torch.gather(alpha, 1, order.unsqueeze(-1).expand(-1, -1, alpha.size(-1)))
                need_received = self.collect_received and self.tiles[m].kind == "attn"
                delta, tile_balance, _extras = self._run_tile(m, h_sub, valid, alpha_sub, need_received)
                if probs is not None:
                    gate = torch.gather(probs[..., m], 1, order).unsqueeze(-1).to(delta.dtype)
                    if cfg.gate_ste:
                        gate = gate / gate.detach().clamp(min=1e-6)
                    delta = delta * gate
                delta = delta * valid.unsqueeze(-1).to(delta.dtype)
                h_new = h_new.scatter_add(1, gather_index, delta)
                balance = balance + tile_balance

            moved_f = (choice != self.EXIT).float()
            h = h_new
            hops_executed = hops_executed + moved_f.sum()

            if use_ds and bool(moved_f.any()):
                loss_now = self._sample_loss(h, labels, ds_pos)          # with grad: deep supervision
                moved_ds = moved_f[:, ds_pos]
                gain = loss_prev - loss_now.detach()                     # measured gain of this hop
                if hop < H - 1 and cfg.deep_sup_weight > 0.0:
                    deep_sup = deep_sup + (loss_now * moved_ds).sum() / moved_ds.sum().clamp(min=1.0)
                    n_deep += 1
                if self.use_value:
                    pg = pred_gain[:, ds_pos]
                    value_loss = value_loss + (F.huber_loss(pg, gain, reduction="none", delta=0.5) * moved_ds).sum() / moved_ds.sum().clamp(min=1.0)
                gain_sum = gain_sum + (gain * moved_ds).sum()
                gain_n = gain_n + moved_ds.sum()
                loss_prev = torch.where(moved_ds > 0, loss_now.detach(), loss_prev)

            if self.use_path:
                hop_e = self._hop_vector(hop)
                if self.training and cfg.hop_feature_dropout > 0.0:
                    keep = (torch.rand(B, S, 1, device=device) >= cfg.hop_feature_dropout).float()
                    hop_e = hop_e * keep
                step_in = self.tile_embed(choice.clamp(max=self.M)) + hop_e
                path_next = self.path_cell(step_in.reshape(B * S, -1), path.reshape(B * S, -1)).view(B, S, -1)
                path = torch.where((choice != self.EXIT).unsqueeze(-1), path_next, path)

            active = active & (choice != self.EXIT)
            if not bool(active.any()):
                break

        h = self.final_norm(h)
        logits = F.linear(h, embedding_weight(self))
        mean_hops = hops_executed / (B * S)

        loss, ce_loss = None, None
        if labels is not None:
            ce_loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, self.vocab_size).float(),
                labels[..., 1:].reshape(-1), ignore_index=-100,
            )
            loss = ce_loss + cfg.balance_loss_weight * balance + cfg.hop_cost_weight * mean_hops / H
            if n_deep > 0:
                loss = loss + cfg.deep_sup_weight * deep_sup / n_deep
            if self.use_value:
                loss = loss + cfg.value_loss_weight * value_loss / max(1, H)

        extra: Dict[str, Any] = {}
        if probe and labels is not None:
            with torch.no_grad():
                extra["token_loss_final"] = F.cross_entropy(
                    logits[..., :-1, :].reshape(-1, self.vocab_size).float(),
                    labels[..., 1:].reshape(-1), reduction="none").view(B, S - 1)
                extra["token_loss_min"] = token_loss_min
                extra["final_hidden"] = h.detach()

        return {
            **extra,
            "logits": logits,
            "loss": loss,
            "ce_loss": ce_loss if ce_loss is not None else torch.zeros((), device=device),
            "balance_loss": balance,
            "deep_sup_loss": (deep_sup / max(1, n_deep)).item(),
            "value_loss": (value_loss / max(1, H)).item(),
            "mean_measured_gain": (gain_sum / gain_n.clamp(min=1.0)).item(),
            "mean_pred_gain": (pred_gain_sum / hops_executed.clamp(min=1.0)).item(),
            "mean_hops": mean_hops.item(),
            "tile_usage": (tile_usage / tile_usage.sum().clamp(min=1)).tolist(),
            "virtual_layers": H,
        }


__all__ = ["GeneralConfig", "HybridTile", "GeneralRoutedLM", "TERNARY_BITS", "DENSE_BITS"]
