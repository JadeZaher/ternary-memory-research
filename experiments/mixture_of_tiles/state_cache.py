"""
experiments/mixture_of_tiles/state_cache.py: per-(hop, tile) sequence state so a generated token costs one
pass over its own hops instead of a full recompute of the prefix.

Every tile in `GeneralRoutedLM` runs on a gathered subset of the tokens that chose it at one hop, in causal
order. That subset is a separate sequence per (hop, tile), so the Mamba convolution/SSM state and the
attention keys/values are cached per (hop, tile) stream, never per tile. A token that skips a stream must not
advance it. Rationale and the exactness caveats live in `experiments/mixture_of_tiles/AGENTS.md`.

Nothing here owns model weights: the tile modules are called piecewise with their own parameters, and the
stream state lives in the decoder, so evicting a tile's weights never touches its sequence state.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard rule: this folder never touches the GPU

import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn.functional as F

torch.set_num_threads(4)
CPU_DEVICE = torch.device("cpu")

from experiments.mamba.ternary_mamba_block import TernaryMambaBlock
from experiments.mixture_of_tiles.tile_store import TileNotResidentError
from experiments.unified_scaling.navitrit_unified_model import FlashAttentionTile, _apply_lora, _film


@dataclass
class MambaStreamState:
    """Causal-conv history [1, d_inner, d_conv-1] and SSM state [1, d_inner, d_state] of one stream."""
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    tokens: int = 0


@dataclass
class AttentionStreamState:
    """Keys and values [1, heads, tokens, head_dim] of every token that passed through one stream."""
    key: torch.Tensor
    value: torch.Tensor
    tokens: int = 0


StreamState = Union[MambaStreamState, AttentionStreamState]


class StreamStateCache:
    """Sequence state keyed by (hop, tile). Skipped tokens never advance a stream; weight eviction never clears one."""

    def __init__(self):
        self._streams: Dict[Tuple[int, int], StreamState] = {}

    def get(self, hop: int, tile: int) -> Optional[StreamState]:
        return self._streams.get((hop, tile))

    def put(self, hop: int, tile: int, state: StreamState) -> None:
        self._streams[(hop, tile)] = state

    def clear(self) -> None:
        self._streams.clear()

    def num_streams(self) -> int:
        return len(self._streams)

    def stream_lengths(self) -> Dict[str, int]:
        """Tokens seen per stream, keyed "hop:tile" (JSON-friendly)."""
        return {f"{hop}:{tile}": state.tokens for (hop, tile), state in sorted(self._streams.items())}

    def bytes(self) -> int:
        """RAM held by the state tensors (what a KV/SSM cache costs on top of the resident tiles)."""
        total = 0
        for state in self._streams.values():
            for tensor in _state_tensors(state):
                total += tensor.numel() * tensor.element_size()
        return total


def _state_tensors(state: StreamState):
    if isinstance(state, MambaStreamState):
        return (state.conv_state, state.ssm_state)
    return (state.key, state.value)


# ------------------------------------------------------------------ mamba -----------------------
def selective_scan_chunk(block: TernaryMambaBlock, x_act: torch.Tensor, A_bar: torch.Tensor, B_bar: torch.Tensor,
                         C: torch.Tensor, previous_state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """(y, final_state) of the selective scan over a chunk that starts from `previous_state` (None = zeros).

    x_act [B,n,d], A_bar/B_bar [B,n,d,N] with A_bar in (0,1), C [B,n,N]; state [B,d,N].
    Linear recurrence, so the chunk splits into the block's own zero-state scan plus the carried term
    A_t * s_0, A_t = prod_{r<=t} a_r. A single token is the plain step recurrence."""
    n = x_act.size(1)
    if n == 1:
        # Mamba step recurrence: s_t = a_t * s_{t-1} + b_t * x_t, y_t = <s_t, c_t>
        state = B_bar[:, 0] * x_act[:, 0].unsqueeze(-1)
        if previous_state is not None:
            state = A_bar[:, 0] * previous_state + state
        y = (state * C[:, 0].unsqueeze(1)).sum(dim=-1).unsqueeze(1)
        return y.to(x_act.dtype), state

    y = block._selective_scan(x_act, A_bar, B_bar, C)                  # the block's arithmetic from a zero state
    # Final state of the zero-state scan in signed log space, the same formulation as the block's parallel
    # scan (its `state` tensor is internal, so its last row is recomputed here for the last position only).
    A_bar_f = A_bar.float()
    log_acum = torch.cumsum(torch.log(A_bar_f.clamp(min=1e-30)), dim=1)  # [B,n,d,N], <= 0
    u = B_bar.float() * x_act.float().unsqueeze(-1)
    eps = 1e-30
    u_pos = torch.where(u >= 0, u, torch.zeros_like(u))
    u_neg = torch.where(u < 0, -u, torch.zeros_like(u))
    log_pos = torch.logsumexp(torch.log(u_pos + eps) - log_acum, dim=1)  # [B,d,N]
    log_neg = torch.logsumexp(torch.log(u_neg + eps) - log_acum, dim=1)
    final_state = torch.exp(log_acum[:, -1] + log_pos) - torch.exp(log_acum[:, -1] + log_neg)
    if previous_state is not None:
        carried = torch.exp(log_acum) * previous_state.float().unsqueeze(1)   # A_t * s_0 at every position
        y = y + (carried * C.float().unsqueeze(2)).sum(dim=-1).to(y.dtype)
        final_state = final_state + carried[:, -1]
    return y, final_state.to(x_act.dtype)


def mamba_forward_with_state(block: TernaryMambaBlock, x: torch.Tensor,
                             state: Optional[MambaStreamState]) -> Tuple[torch.Tensor, MambaStreamState]:
    """The block's forward over a chunk [1,n,d_model] continuing a stream; returns (out, new state)."""
    batch, n, _ = x.shape
    d_inner, d_conv = block.d_inner, block.d_conv
    in_proj_out = block.in_proj(x)
    u, z = in_proj_out.chunk(2, dim=-1)                                # [1,n,d_inner] each

    u_t = u.transpose(1, 2)                                            # [1,d_inner,n]
    if state is None:
        conv_prefix = torch.zeros(batch, d_inner, d_conv - 1, dtype=u.dtype, device=u.device)
    else:
        conv_prefix = state.conv_state
    conv_in = torch.cat([conv_prefix, u_t], dim=-1)                    # [1,d_inner,d_conv-1+n]
    # Explicit zero/history prefix and no padding equals the block's padding=d_conv-1 + causal crop.
    u_conv = F.conv1d(conv_in, block.conv1d.weight, block.conv1d.bias, groups=d_inner)
    new_conv_state = conv_in[:, :, conv_in.size(-1) - (d_conv - 1):]
    x_act = F.silu(u_conv.transpose(1, 2))                             # [1,n,d_inner]

    ssm_params = block.x_proj(x_act)
    dt_input, B_proj, C_proj = torch.split(ssm_params, [block.dt_rank, block.d_state, block.d_state], dim=-1)
    delta = F.softplus(block.dt_proj(dt_input))                        # [1,n,d_inner]
    A = block.A
    A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    B_bar = delta.unsqueeze(-1) * B_proj.unsqueeze(2)
    previous = None if state is None else state.ssm_state
    y, ssm_state = selective_scan_chunk(block, x_act, A_bar, B_bar, C_proj, previous)

    y = y + x_act * block.D.unsqueeze(0).unsqueeze(0)
    y = y * F.silu(z)
    out = block.out_proj(y)
    seen = (0 if state is None else state.tokens) + n
    return out, MambaStreamState(conv_state=new_conv_state.contiguous(), ssm_state=ssm_state, tokens=seen)


# ------------------------------------------------------------------ attention -------------------
def attention_forward_with_state(mixer: FlashAttentionTile, x: torch.Tensor, mod_lora: Optional[Dict[str, Any]],
                                 state: Optional[AttentionStreamState]) -> Tuple[torch.Tensor, AttentionStreamState]:
    """Causal attention of a chunk [1,n,d] over the cached keys/values of its stream plus itself."""
    batch, n, width = x.shape
    heads, head_dim = mixer.num_heads, mixer.head_dim
    q = mixer.q_proj(x) + _apply_lora(x, mod_lora, "q")
    k = mixer.k_proj(x)
    v = mixer.v_proj(x)
    q = q.view(batch, n, heads, head_dim).transpose(1, 2)
    k = k.view(batch, n, heads, head_dim).transpose(1, 2)
    v = v.view(batch, n, heads, head_dim).transpose(1, 2)
    if state is not None:
        k = torch.cat([state.key, k], dim=2)
        v = torch.cat([state.value, v], dim=2)
    past = k.size(2) - n
    # query i may attend key j iff j <= past + i: the causal mask of the stream, restricted to the new rows
    query_pos = torch.arange(n, device=x.device).unsqueeze(1) + past
    key_pos = torch.arange(k.size(2), device=x.device).unsqueeze(0)
    mask = (key_pos <= query_pos).view(1, 1, n, k.size(2))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = out.transpose(1, 2).contiguous().view(batch, n, width)
    out = mixer.o_proj(out) + _apply_lora(x, mod_lora, "o")
    return out, AttentionStreamState(key=k, value=v, tokens=k.size(2))


# ------------------------------------------------------------------ tile ------------------------
def hybrid_tile_forward_with_state(tile, h_sub: torch.Tensor, mod_film: Optional[Dict[str, Any]],
                                   mod_lora: Optional[Dict[str, Any]],
                                   state: Optional[StreamState]) -> Tuple[torch.Tensor, StreamState]:
    """`HybridTile.forward` for a chunk of one stream, with the sequence mixer continuing from `state`.

    Everything except the mixer is per token (norms, branch mix, FiLM, LoRA, SwiGLU), so only the mixer needs
    state. Raises `TileNotResidentError` like the residency guard does for the module call."""
    if not getattr(tile, "tile_is_resident", True):
        raise TileNotResidentError(
            f"tile {getattr(tile, 'tile_index', '?')} executed while evicted; it never came through the cache")
    normed = tile.norm_mixer(h_sub)
    w_mixer, w_ffn = tile._branch_weights(normed)
    mixer_in = _film(normed, mod_film, "attn")
    if tile.kind == "attn":
        mixer_out, new_state = attention_forward_with_state(tile.mixer, mixer_in, mod_lora, state)
    else:
        pre = mixer_in + _apply_lora(mixer_in, mod_lora, "q")
        mixer_out, new_state = mamba_forward_with_state(tile.mixer, pre, state)
        mixer_out = mixer_out + _apply_lora(mixer_in, mod_lora, "o")
    mixer_delta = w_mixer * mixer_out
    h_mid = h_sub + mixer_delta
    ffn_out, _balance = tile.ffn(_film(tile.norm_ffn(h_mid), mod_film, "ffn"), mod_lora)
    delta = mixer_delta + w_ffn * ffn_out
    return delta, new_state


__all__ = [
    "AttentionStreamState", "MambaStreamState", "StreamState", "StreamStateCache",
    "attention_forward_with_state", "hybrid_tile_forward_with_state", "mamba_forward_with_state",
    "selective_scan_chunk",
]
