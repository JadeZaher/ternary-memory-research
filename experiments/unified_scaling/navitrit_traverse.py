"""
experiments/unified_scaling/navitrit_traverse.py: NaviTrit-Traverse, per-token non-monotonic traversal with
continuation-style path state (the thesis of this project, implemented so it can be falsified).

Hypothesis under test (research/hardening-2026-09-18-heldout.md §8):
  Each token carries its own traversal history p_t (a small recurrent "continuation" state updated with an
  embedding of every tile it passed through). That state conditions
    (a) ROUTING: at every hop the token picks ONE tile of the shared ternary block, or EXIT. Easy tokens
        leave early; hard tokens revisit tiles in any order (non-monotonic). Tile evaluations per token,
        not depth, is the compute unit.
    (b) INTERPRETATION: the tile's shared ternary weights are modulated by a per-token soft mixture over a
        bank of K low-rank adapters + FiLM vectors selected from p_t. The same weights do different work on
        different paths ("dynamic weight parameterisation by history").

Ablation switches that isolate the two halves (both are config fields):
  router_cond  "path" (state + path history)  vs  "state" (residual state only)
  adapter_cond "path" (mixture picked by history) vs "hop" (mixture picked by hop index only)
If "path" beats its control on held-out data at matched tile evaluations, the non-monotonic advantage is
real. If not, the honest architecture is the tied block with Mixture-of-Depths (navitrit_unified_model.py).

Dispatch: tokens routed to tile m are gathered per sequence (padded to the batch max), attention is run over
that subset with a causal+padding mask (original order is kept, so causality holds), the FFN runs on the
same gathered tensor, and the update is scattered back scaled by the router probability (straight-through
style gradient to the router). A Switch-style balance loss keeps tiles in use; a hop-cost term gives the
router a reason to exit.

Warm start: the tiles are `UnifiedMacroLayer`s from navitrit_unified_model.py (use_mamba=False), so a dense
arm checkpoint loads into `tiles.*` directly (`load_dense_checkpoint`).
"""

import os
import sys
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.bitroute_model import BitRouteRMSNorm
from experiments.unified_scaling.navitrit_unified_model import (
    NaviTritUnifiedConfig,
    UnifiedMacroLayer,
    _film,
    embedding_weight,
)


@dataclass
class TraverseConfig:
    """Traversal-specific knobs; the tile geometry comes from NaviTritUnifiedConfig."""
    max_hops: int = 8               # hop budget H (= max tile evaluations per token)
    min_hops: int = 4               # EXIT is masked before this many hops (arms H-K used 1 and collapsed to 1 hop)
    exit_warmup_steps: int = 1000   # EXIT is masked entirely until the trainer has taken this many steps
    gate_ste: bool = True           # scatter delta at full magnitude, gradient through the router prob (MoD-style)
    path_dim: int = 128             # continuation state size
    adapter_bank: int = 4           # K adapters per tile
    adapter_rank: int = 16
    router_cond: str = "path"       # "path" | "state"
    adapter_cond: str = "path"      # "path" | "hop"
    balance_loss_weight: float = 0.01
    hop_cost_weight: float = 0.01   # penalty on mean executed hops / max_hops
    router_temperature: float = 1.0
    # Strain track (research/hardening-2026-09-18-heldout.md section 10): a per-token latent measuring how
    # unresolved the token still is. strain_mode:
    #   "off"      no strain machinery (arms H-K)
    #   "observe"  strain features (this hop's relative delta norm, router entropy, hop/H) feed the path GRU,
    #              and a head predicts the NEXT hop's delta norm (auxiliary loss); exit stays the router's call
    #   "gate"     as observe, plus predicted strain shifts the EXIT logit: low predicted strain -> exit
    strain_mode: str = "off"
    strain_loss_weight: float = 0.1
    # Activation strategy selector (report section 12, arm P): the router also forwards an activation for
    # the tile it sends the token to. A per-token softmax over ACT_BANK, conditioned on the path state and
    # the EDGE the token takes (signed tile offset v-u: backward / forward-by-how-much / how many tiles are
    # skipped, plus hop index), mixes the FFN gate nonlinearity. Initialised to pure SiLU (= arm H).
    act_strategy: bool = False
    # Stage 2 of the condensed experiment (report section 13): value-based exit and richer path inputs.
    exit_mode: str = "router"            # "router" (arms H-Q) | "value": continue iff predicted gain > exit_lambda
    exit_lambda: float = 0.02            # compute price in nats per hop (training); swept at eval
    value_loss_weight: float = 1.0       # Huber on measured per-token gain (from the deep-sup sample)
    deep_sup_weight: float = 0.0         # tied-head readout on a sampled subset after every hop (also yields gain targets)
    deep_sup_frac: float = 0.125
    use_liveness: bool = False           # attention-received mass -> path state
    use_gain_proxy: bool = False         # change of proxy-head top-2 margin -> path state (needs set_proxy_vocab)
    use_coord: bool = False              # learned 2-D coordinate per (tile, expert) + activation stats -> path state
    hop_feature_dropout: float = 0.0     # zero the hop embedding in the path input with this prob (training)
    coord_reg: float = 1e-3
    # eval-time exit policies on the same model (trainer sweeps these)
    exit_policy: str = "threshold"       # "threshold" | "random" | "capacity"
    random_continue_prob: float = 0.5
    capacity_frac: float = 0.5


ACT_BANK = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
    "identity": lambda g: g,
    "tanh": torch.tanh,
    "sign_sqrt": lambda g: torch.sign(g) * torch.sqrt(g.abs() + 1e-6),   # expansive: amplifies small gate values
}
ACT_NAMES = list(ACT_BANK)


class MixedGateActivation:
    """Callable: gate_act(g) = sum_a w_a * act_a(g) with per-token weights w [B,k,A] (broadcast over d_ff)."""
    def __init__(self, weights: torch.Tensor):
        self.w = weights
    def __call__(self, g: torch.Tensor) -> torch.Tensor:
        out = None
        for a, name in enumerate(ACT_NAMES):
            term = ACT_BANK[name](g) * self.w[..., a: a + 1].to(g.dtype)
            out = term if out is None else out + term
        # activation statistics for the coordinate record: fraction of active units, mean magnitude
        self.stats = torch.stack([(out > 0).float().mean(-1), out.float().abs().mean(-1)], dim=-1).detach()  # [B,k,2]
        return out


class PathAdapterBank(nn.Module):
    """K zero-initialised LoRA (q, o, down_i) + FiLM (attn, ffn) adapters for one tile, mixed per token."""
    def __init__(self, hidden: int, num_experts: int, K: int, rank: int):
        super().__init__()
        self.K, self.rank, self.hidden = K, rank, hidden
        self.targets = ["q", "o"] + [f"down_{i}" for i in range(num_experts)]
        self.A = nn.ParameterDict({t: nn.Parameter(torch.randn(K * rank, hidden) * 0.01) for t in self.targets})
        self.B = nn.ParameterDict({t: nn.Parameter(torch.zeros(hidden, K * rank)) for t in self.targets})
        self.film = nn.Parameter(torch.zeros(K, 2, 2, hidden))  # [K, (attn, ffn), (gamma, beta), d]

    def modulations(self, alpha: torch.Tensor) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """alpha: [..., K] mixture weights -> mod_film (per-token gamma/beta), mod_lora (bank triples)."""
        alpha_rep = alpha.repeat_interleave(self.rank, dim=-1)          # [..., K*r]
        mod_lora = {t: (self.A[t], self.B[t], alpha_rep) for t in self.targets}
        film = torch.einsum("...k,kabd->...abd", alpha, self.film)      # [..., 2, 2, d]
        mod_film = {"attn": (film[..., 0, 0, :], film[..., 0, 1, :]), "ffn": (film[..., 1, 0, :], film[..., 1, 1, :])}
        return mod_film, mod_lora


class NaviTritTraverseForCausalLM(nn.Module):
    def __init__(self, config: NaviTritUnifiedConfig, tcfg: TraverseConfig):
        super().__init__()
        assert not config.use_mamba, "traverse tiles are attention+FFN (Mamba cannot run on gathered subsets)"
        self.config, self.tcfg = config, tcfg
        self.hidden = config.hidden_size
        self.vocab_size = config.vocab_size
        self.M = config.num_macro_layers                    # number of tiles
        self.EXIT = self.M
        self.allow_exit = tcfg.exit_warmup_steps == 0   # trainer flips this once warmup is over

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        nn.init.normal_(self.tok_embeddings.weight, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, std=0.02)
        dense_cfg = NaviTritUnifiedConfig(**{**config.__dict__, "routing_mode": "dense"})
        self.tiles = nn.ModuleList([UnifiedMacroLayer(dense_cfg, i) for i in range(self.M)])
        self.final_norm = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # continuation state: p <- GRU(p, tile_embed + hop_embed)
        self.tile_embed = nn.Embedding(self.M + 1, tcfg.path_dim)   # +1: START
        self.hop_embed = nn.Embedding(tcfg.max_hops + 1, tcfg.path_dim)
        self.path_cell = nn.GRUCell(tcfg.path_dim, tcfg.path_dim)
        if tcfg.act_strategy:
            # edge offset v-u in [-(M-1), M-1] -> index offset+(M-1); index 2M-1 = first hop (from START)
            self.offset_embed = nn.Embedding(2 * self.M, tcfg.path_dim)
            self.strategy_router = nn.Linear(2 * tcfg.path_dim, len(ACT_NAMES))
            nn.init.zeros_(self.strategy_router.weight); nn.init.zeros_(self.strategy_router.bias)
            with torch.no_grad():
                self.strategy_router.bias[ACT_NAMES.index("silu")] = 4.0   # softmax ~ 0.97 SiLU at init
            nn.init.normal_(self.strategy_router.weight, std=0.02)
        t = tcfg
        self.use_value = t.exit_mode == "value"
        if self.use_value:
            self.value_head = nn.Sequential(nn.Linear(2 * t.path_dim, t.path_dim // 2), nn.GELU(), nn.Linear(t.path_dim // 2, 1))
        if t.use_liveness or t.use_gain_proxy:
            self.sig_in = nn.Linear(2, t.path_dim)
        if t.use_coord:
            self.coord = nn.Parameter(torch.randn(self.M, config.num_experts, 2) * 0.1)   # learned 2-D point per (tile, expert)
            self.coord_in = nn.Linear(2 + 1 + len(ACT_NAMES) + 2 + config.num_experts, t.path_dim)
        self.register_buffer("proxy_ids", torch.arange(0), persistent=False)
        self.force_random_tiles = False   # diversity probe: ignore the tile router, pick tiles at random
        self.use_strain = tcfg.strain_mode != "off"
        if self.use_strain:
            self.strain_in = nn.Linear(3, tcfg.path_dim)                      # (log delta norm, router entropy, hop/H)
            self.strain_head = nn.Sequential(nn.Linear(2 * tcfg.path_dim, tcfg.path_dim // 2), nn.GELU(), nn.Linear(tcfg.path_dim // 2, 1))
            self.strain_exit_scale = nn.Parameter(torch.zeros(()))            # gate mode: exit_logit -= scale * predicted strain

        # routers
        self.norm_route = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.state_proj = nn.Linear(config.hidden_size, tcfg.path_dim)
        self.tile_router = nn.Sequential(nn.Linear(2 * tcfg.path_dim, tcfg.path_dim), nn.GELU(), nn.Linear(tcfg.path_dim, self.M + 1))
        nn.init.normal_(self.tile_router[-1].weight, std=0.02); nn.init.zeros_(self.tile_router[-1].bias)  # small, not zero: zero logits tie -> every token to tile 0
        self.banks = nn.ModuleList([PathAdapterBank(config.hidden_size, config.num_experts, tcfg.adapter_bank, tcfg.adapter_rank) for _ in range(self.M)])
        self.adapter_router = nn.Linear(tcfg.path_dim, tcfg.adapter_bank)
        nn.init.normal_(self.adapter_router.weight, std=0.02); nn.init.zeros_(self.adapter_router.bias)  # zero weight would block gradient into the path state

    # ------------------------------------------------------------------ helpers -------------
    def count_parameters(self) -> Dict[str, Any]:
        tiles = sum(p.numel() for p in self.tiles.parameters())
        banks = sum(p.numel() for p in self.banks.parameters())
        ctrl = sum(p.numel() for n, p in self.named_parameters() if n.split(".")[0] in ("tile_embed", "hop_embed", "path_cell", "state_proj", "tile_router", "adapter_router", "norm_route"))
        embed = self.tok_embeddings.weight.numel() + self.pos_embeddings.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        return {"total_physical_parameters": total, "total_millions": round(total / 1e6, 2), "embeddings_params": embed,
                "superblock_macro_params": tiles, "adapter_bank_params": banks, "controller_params": ctrl,
                "superblock_packed_mb": round(tiles * 1.58 / (8 * 1024 * 1024), 2), "packed_ternary_mb": round((tiles * 1.58 + (banks + ctrl + embed) * 16) / (8 * 1024 * 1024), 2),
                "virtual_layers_count": self.tcfg.max_hops, "routing_mode": "traverse"}

    def load_dense_checkpoint(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        own = self.state_dict()
        loaded = 0
        for k, v in sd.items():
            k2 = k.replace("macro_layers.", "tiles.")
            if k2 in own and own[k2].shape == v.shape:
                own[k2] = v; loaded += 1
        self.load_state_dict(own)
        print(f"[traverse] warm-started {loaded} tensors from {path}")

    def set_proxy_vocab(self, ids: torch.Tensor) -> None:
        self.proxy_ids = ids.to(self.tok_embeddings.weight.device)

    def _proxy_margin(self, h: torch.Tensor) -> torch.Tensor:
        """top-1 minus top-2 logit over the proxy vocabulary rows of the tied head: a cheap confidence per token."""
        if self.proxy_ids.numel() == 0:
            return torch.zeros(h.shape[:2], device=h.device)
        w = embedding_weight(self)[self.proxy_ids]
        lg = F.linear(self.final_norm(h), w).float()
        top2 = lg.topk(2, dim=-1).values
        return (top2[..., 0] - top2[..., 1]).detach()

    def _sample_loss(self, h: torch.Tensor, labels: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """per-token CE at sampled positions [B,n] using the full tied head (deep supervision / gain targets)."""
        lg = F.linear(self.final_norm(h[:, pos]), embedding_weight(self)).float()
        tgt = labels[:, pos + 1]
        return F.cross_entropy(lg.reshape(-1, self.vocab_size), tgt.reshape(-1), reduction="none", ignore_index=-100).view(h.size(0), -1)

    def _cond(self, h: torch.Tensor, p: torch.Tensor, hop: int) -> torch.Tensor:
        s = self.state_proj(self.norm_route(h))
        if self.tcfg.router_cond == "state":
            p = torch.zeros_like(p) + self.hop_embed.weight[hop]  # hop index only, no history
        return torch.cat([s, p], dim=-1)

    def _alpha(self, p: torch.Tensor, hop: int) -> torch.Tensor:
        if self.tcfg.adapter_cond == "hop":
            p = torch.zeros_like(p) + self.hop_embed.weight[hop]
        return F.softmax(self.adapter_router(p), dim=-1)

    @staticmethod
    def _gather_subset(sel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """sel [B,S] bool -> positions [B,k] (original order), valid [B,k], k."""
        B, S = sel.shape
        k = int(sel.sum(dim=1).max().item())
        pos = torch.arange(S, device=sel.device).unsqueeze(0).expand(B, S)
        order = torch.where(sel, pos, torch.full_like(pos, S)).sort(dim=1).values[:, :k]
        valid = order < S
        return order.clamp(max=S - 1), valid, k

    def _strategy(self, p: torch.Tensor, prev: torch.Tensor, m: int, hop: int) -> torch.Tensor:
        """Per-token activation mixture [B,S,A] for tokens about to enter tile m from tile `prev` (M = START)."""
        offset = torch.where(prev == self.M, torch.full_like(prev, 2 * self.M - 1), (m - prev) + (self.M - 1))
        e = self.offset_embed(offset) + self.hop_embed.weight[hop]
        return F.softmax(self.strategy_router(torch.cat([p, e], dim=-1)).float(), dim=-1)

    def _run_tile(self, m: int, h_sub: torch.Tensor, valid: torch.Tensor, alpha: torch.Tensor, act_w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Attention over the gathered subset (causal + padding mask) then FFN; returns the residual delta."""
        tile = self.tiles[m]
        B, k, _ = h_sub.shape
        mod_film, mod_lora = self.banks[m].modulations(alpha)
        causal = torch.tril(torch.ones(k, k, dtype=torch.bool, device=h_sub.device))
        mask = causal.unsqueeze(0) & valid.unsqueeze(1)                       # [B,k,k] queries may see valid keys <= them
        mask = mask | torch.eye(k, dtype=torch.bool, device=h_sub.device).unsqueeze(0)  # padded queries attend to themselves (no NaN)
        if self.tcfg.use_liveness:
            attn, received = tile.attn(_film(tile.norm_attn(h_sub), mod_film, "attn"), mod_lora, attn_mask=mask.unsqueeze(1), return_received=True)
        else:
            attn = tile.attn(_film(tile.norm_attn(h_sub), mod_film, "attn"), mod_lora, attn_mask=mask.unsqueeze(1)); received = None
        h2 = h_sub + attn
        gate_act = MixedGateActivation(act_w) if act_w is not None else F.silu
        ffn, bal = tile.dual_ffn(_film(tile.norm_ffn(h2), mod_film, "ffn"), mod_lora, gate_act=gate_act)
        extras = {"received": received}
        if self.tcfg.use_coord:
            A = len(ACT_NAMES)
            act_stats = gate_act.stats if isinstance(gate_act, MixedGateActivation) else torch.zeros(B, k, 2, device=h_sub.device)
            act_mix = act_w if act_w is not None else F.one_hot(torch.tensor(ACT_NAMES.index("silu"), device=h_sub.device), A).float().expand(B, k, A)
            eprobs = getattr(tile.dual_ffn, "last_expert_probs", None)
            if eprobs is None:
                eprobs = torch.ones(B, k, 1, device=h_sub.device)
            coord_tok = torch.einsum("bke,ed->bkd", eprobs.float(), self.coord[m].float())      # expert-weighted 2-D location
            extras["record"] = torch.cat([coord_tok, act_mix.float(), act_stats.float(), eprobs.float()], dim=-1)   # hop/H added by caller
        return attn + ffn, bal, extras

    # ------------------------------------------------------------------ forward -------------
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, max_loops: Optional[int] = None,
                probe: bool = False, **_) -> Dict[str, Any]:
        """probe=True additionally returns per-token loss at min_hops and at the end (all positions; eval-time diversity probe)."""
        B, S = input_ids.shape
        dev = input_ids.device
        H = max_loops if max_loops is not None else self.tcfg.max_hops
        t = self.tcfg
        rec_dim = 2 + len(ACT_NAMES) + 2 + self.config.num_experts
        positions = torch.arange(S, device=dev).unsqueeze(0)
        h = F.embedding(input_ids, embedding_weight(self)) + self.pos_embeddings(positions)
        p = self.path_cell(self.tile_embed.weight[self.M].expand(B * S, -1), torch.zeros(B * S, self.tcfg.path_dim, device=dev)).view(B, S, -1)
        active = torch.ones(B, S, dtype=torch.bool, device=dev)

        balance = torch.zeros((), device=dev)
        hops_executed = torch.zeros((), device=dev)
        tile_usage = torch.zeros(self.M + 1, device=dev)
        strain_loss = torch.zeros((), device=dev)
        strain_sum = torch.zeros((), device=dev)
        prev_choice = torch.full((B, S), self.M, dtype=torch.long, device=dev)   # START
        act_usage = torch.zeros(len(ACT_NAMES), device=dev)
        # deep supervision sample + measured gains
        use_ds = (t.deep_sup_weight > 0.0 or self.use_value) and labels is not None
        deep_sup = torch.zeros((), device=dev); value_loss = torch.zeros((), device=dev); n_ds = 0
        if use_ds:
            n_pos = max(1, int(t.deep_sup_frac * (S - 1)))
            ds_pos = torch.randperm(S - 1, device=dev)[:n_pos] if self.training else torch.arange(0, S - 1, max(1, (S - 1) // n_pos), device=dev)[:n_pos]
            with torch.no_grad():
                loss_prev = self._sample_loss(h, labels, ds_pos)          # [B,n] loss before any hop
        gain_sum = torch.zeros((), device=dev); gain_n = torch.zeros((), device=dev)
        pred_gain_sum = torch.zeros((), device=dev)
        margin_prev = self._proxy_margin(h) if t.use_gain_proxy else None
        liveness = torch.zeros(B, S, device=dev)
        token_loss_min = None
        for hop in range(H):
            if probe and hop == t.min_hops and labels is not None:
                with torch.no_grad():
                    token_loss_min = self._sample_loss(h, labels, torch.arange(S - 1, device=dev))
            cond = self._cond(h, p, hop)
            logits = self.tile_router(cond).float() / self.tcfg.router_temperature
            pred_gain = None
            if self.use_value:
                pred_gain = self.value_head(cond).float().squeeze(-1)             # predicted gain (nats) of this hop, per token
                pred_gain_sum = pred_gain_sum + (pred_gain.detach() * active.float()).sum()
            pred_strain = None
            if self.use_strain:
                pred_strain = F.softplus(self.strain_head(cond).float().squeeze(-1))   # predicted relative delta norm of this hop
                strain_sum = strain_sum + (pred_strain * active.float()).sum()
                if self.tcfg.strain_mode == "gate":
                    logits = logits.clone()
                    logits[..., self.EXIT] = logits[..., self.EXIT] - self.strain_exit_scale * pred_strain
            if self.training:  # Gumbel exploration noise on the hard choice (probabilities stay clean for the gate)
                logits = logits - torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-9)))
            if hop < self.tcfg.min_hops or not self.allow_exit:
                logits = logits.clone(); logits[..., self.EXIT] = -1e4
            if self.use_value:
                logits = logits.clone(); logits[..., self.EXIT] = -1e4        # exit is decided by value, not by the tile router
            probs = F.softmax(logits, dim=-1)                                  # [B,S,M+1]
            entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(-1)     # [B,S]
            choice = probs.argmax(dim=-1)
            if self.force_random_tiles:
                choice = torch.randint(0, self.M, (B, S), device=dev)
            if self.use_value and hop >= t.min_hops and self.allow_exit:
                if t.exit_policy == "threshold":
                    cont = pred_gain.detach() > t.exit_lambda
                elif t.exit_policy == "random":
                    cont = torch.rand(B, S, device=dev) < t.random_continue_prob
                else:  # capacity: top fraction of active tokens by predicted gain, per sequence
                    score = pred_gain.detach().masked_fill(~active, float("-inf"))
                    k_keep = max(1, int(t.capacity_frac * S))
                    thresh = score.topk(k_keep, dim=1).values[:, -1:]
                    cont = score >= thresh
                choice = torch.where(cont, choice, torch.full_like(choice, self.EXIT))
            choice = torch.where(active, choice, torch.full_like(choice, self.EXIT))
            # Switch balance loss over tiles (exit excluded), computed on active tokens
            act_f = active.float()
            n_act = act_f.sum().clamp(min=1.0)
            frac = torch.stack([((choice == m).float() * act_f).sum() for m in range(self.M)]) / n_act
            pmean = (probs[..., : self.M] * act_f.unsqueeze(-1)).sum(dim=(0, 1)) / n_act
            balance = balance + self.M * (frac * pmean).sum()
            tile_usage = tile_usage + torch.bincount(choice[active].flatten(), minlength=self.M + 1).float()
            alpha = self._alpha(p, hop)                                        # [B,S,K]

            h_new = h
            record = torch.zeros(B, S, rec_dim, device=dev) if t.use_coord else None
            for m in range(self.M):
                sel = (choice == m) & active
                if not bool(sel.any()):
                    continue
                order, valid, k = self._gather_subset(sel)
                gi = order.unsqueeze(-1).expand(-1, -1, self.hidden)
                h_sub = torch.gather(h, 1, gi)
                a_sub = torch.gather(alpha, 1, order.unsqueeze(-1).expand(-1, -1, alpha.size(-1)))
                act_w = None
                if self.tcfg.act_strategy:
                    act_all = self._strategy(p, prev_choice, m, hop)                       # [B,S,A]
                    act_w = torch.gather(act_all, 1, order.unsqueeze(-1).expand(-1, -1, act_all.size(-1)))
                    act_usage = act_usage + (act_w.detach() * valid.unsqueeze(-1)).sum(dim=(0, 1))
                delta, bal, extras = self._run_tile(m, h_sub, valid, a_sub, act_w)
                if extras.get("received") is not None:
                    liveness = liveness.scatter(1, order, extras["received"].to(liveness.dtype) * valid.float())
                if t.use_coord:
                    record = record.scatter(1, order.unsqueeze(-1).expand(-1, -1, rec_dim), extras["record"] * valid.unsqueeze(-1).float())
                gate = torch.gather(probs[..., m], 1, order).unsqueeze(-1).to(delta.dtype)
                if self.tcfg.gate_ste:
                    gate = gate / gate.detach().clamp(min=1e-6)   # value 1, gradient d/dp: tiles run at full magnitude like the dense model
                delta = delta * gate * valid.unsqueeze(-1).to(delta.dtype)
                h_new = h_new.scatter_add(1, gi, delta)
                balance = balance + bal
            moved_f = (choice != self.EXIT).float()
            rel_delta = (h_new - h).float().norm(dim=-1) / h.float().norm(dim=-1).clamp(min=1e-6)   # measured strain this hop
            log_delta = torch.log1p(rel_delta)
            if self.use_strain:
                # the head predicted this hop's strain before it happened; supervise on tokens that moved
                strain_loss = strain_loss + (((pred_strain - log_delta.detach()) ** 2) * moved_f).sum() / moved_f.sum().clamp(min=1.0)
            h = h_new
            hops_executed = hops_executed + moved_f.sum()
            if use_ds and bool(moved_f.any()):
                loss_now = self._sample_loss(h, labels, ds_pos)                 # [B,n] (with grad: deep supervision)
                moved_ds = moved_f[:, ds_pos]
                gain = (loss_prev - loss_now.detach())                           # measured gain of this hop at sampled positions
                if hop < H - 1:
                    deep_sup = deep_sup + (loss_now * moved_ds).sum() / moved_ds.sum().clamp(min=1.0); n_ds += 1
                if self.use_value:
                    pg = pred_gain[:, ds_pos]
                    value_loss = value_loss + (F.huber_loss(pg, gain, reduction="none", delta=0.5) * moved_ds).sum() / moved_ds.sum().clamp(min=1.0)
                gain_sum = gain_sum + (gain * moved_ds).sum(); gain_n = gain_n + moved_ds.sum()
                loss_prev = torch.where(moved_ds > 0, loss_now.detach(), loss_prev)
            # continuation update for tokens that executed a tile
            hop_e = self.hop_embed.weight[hop]
            if self.training and t.hop_feature_dropout > 0.0:
                hop_e = hop_e * (torch.rand(B, S, 1, device=dev) >= t.hop_feature_dropout).float()
            step_in = self.tile_embed(choice.clamp(max=self.M)) + hop_e
            if t.use_liveness or t.use_gain_proxy:
                dmargin = torch.zeros(B, S, device=dev)
                if t.use_gain_proxy:
                    margin_now = self._proxy_margin(h); dmargin = margin_now - margin_prev; margin_prev = margin_now
                step_in = step_in + self.sig_in(torch.stack([liveness if t.use_liveness else torch.zeros_like(dmargin), dmargin], dim=-1))
            if t.use_coord:
                rec_in = torch.cat([record[..., :2], torch.full((B, S, 1), hop / H, device=dev), record[..., 2:]], dim=-1)
                step_in = step_in + self.coord_in(rec_in)
            if self.use_strain:
                feats = torch.stack([log_delta.detach(), entropy.detach(), torch.full_like(log_delta, hop / H)], dim=-1)
                step_in = step_in + self.strain_in(feats)
            p_next = self.path_cell(step_in.view(B * S, -1), p.view(B * S, -1)).view(B, S, -1)
            moved = (choice != self.EXIT).unsqueeze(-1)
            p = torch.where(moved, p_next, p)
            prev_choice = torch.where(choice != self.EXIT, choice, prev_choice)
            active = active & (choice != self.EXIT)
            if not bool(active.any()):
                break

        h = self.final_norm(h)
        logits = F.linear(h, embedding_weight(self))
        mean_hops = hops_executed / (B * S)
        loss = ce = None
        if labels is not None:
            ce = F.cross_entropy(logits[..., :-1, :].reshape(-1, self.vocab_size).float(), labels[..., 1:].reshape(-1), ignore_index=-100)
            loss = ce + self.tcfg.balance_loss_weight * balance + self.tcfg.hop_cost_weight * mean_hops / H
            if self.use_strain:
                loss = loss + self.tcfg.strain_loss_weight * strain_loss
            if n_ds > 0:
                loss = loss + t.deep_sup_weight * deep_sup / n_ds
            if self.use_value:
                loss = loss + t.value_loss_weight * value_loss / max(1, H)
            if t.use_coord:
                loss = loss + t.coord_reg * self.coord.pow(2).mean()
        out_extra = {}
        if probe and labels is not None:
            with torch.no_grad():
                out_extra["token_loss_final"] = F.cross_entropy(logits[..., :-1, :].reshape(-1, self.vocab_size).float(), labels[..., 1:].reshape(-1), reduction="none").view(B, S - 1)
                out_extra["token_loss_min"] = token_loss_min
                out_extra["final_hidden"] = h.detach()
        return {
            **out_extra,
            "deep_sup_loss": (deep_sup / max(1, n_ds)).item(), "value_loss": (value_loss / max(1, H)).item(),
            "mean_measured_gain": (gain_sum / gain_n.clamp(min=1.0)).item(), "mean_pred_gain": (pred_gain_sum / hops_executed.clamp(min=1.0)).item(),
            "strain_loss": strain_loss.item(), "mean_pred_strain": (strain_sum / hops_executed.clamp(min=1.0)).item(),
            "act_usage": (act_usage / act_usage.sum().clamp(min=1e-6)).tolist(), "act_names": ACT_NAMES,
            "logits": logits, "loss": loss, "ce_loss": ce if ce is not None else torch.zeros((), device=dev),
            "balance_loss": balance, "mean_hops": mean_hops.item(), "tile_usage": (tile_usage / tile_usage.sum().clamp(min=1)).tolist(),
            "avg_w_seq": 1.0, "avg_w_chan": 1.0, "avg_w_skip": 1.0 - mean_hops.item() / H,
            "virtual_layers": H,
        }
