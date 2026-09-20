"""
experiments/mixture_of_tiles/cached_decoder.py: greedy CPU decoding with per-(hop, tile) sequence state, so a
generated token costs one pass over its own hops instead of a full recompute of the prefix.

`CachedStreamingDecoder` re-runs the model's hop loop lane-locally (batch 1, eval-mode branches only) and
dispatches every tile through the same paging funnel as `StreamingDecoder` (`_page_in`), so bytes per
generated token stay a measured quantity. The full-recompute `StreamingDecoder` is untouched and remains
the reference the exactness test compares against. Why the state is keyed by (hop, tile), what is and is
not supported, and how to read the timings: `experiments/mixture_of_tiles/AGENTS.md`.

Protocol: greedy, batch 1, one prefill pass over the prompt, then one incremental pass per generated token.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard rule: this folder never touches the GPU

import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn.functional as F

torch.set_num_threads(4)
CPU_DEVICE = torch.device("cpu")

from experiments.general_model.general_model import gather_subset
from experiments.mixture_of_tiles.cpu_decoder import MEGABYTE, StreamingDecoder, usage_entropy_bits
from experiments.mixture_of_tiles.state_cache import StreamStateCache, hybrid_tile_forward_with_state
from experiments.mixture_of_tiles.tile_store import TileCache
from experiments.unified_scaling.navitrit_unified_model import embedding_weight

CACHED_PROTOCOL = "greedy batch-1, cached (hop, tile) sequence state, one prefill pass then one pass per token"


class CachedStreamingDecoder(StreamingDecoder):
    """Greedy batch-1 CPU decoder: paged tile weights plus cached (hop, tile) sequence state."""

    def __init__(self, model, cache: TileCache, verbose: bool = False, **decoder_kwargs):
        super().__init__(model, cache, verbose=verbose, **decoder_kwargs)
        self.states = StreamStateCache()
        self.verbose = bool(verbose)
        self.tile_token_counts: Counter = Counter()     # (token, hop) assignments per tile, not calls

    # -------------------------------------------------------------- policy ---------------------
    def _check_supported(self) -> None:
        """Refuse configurations whose full-recompute answer is not a causal per-token function."""
        cfg = self.model.config
        if self.model.collect_received:
            raise ValueError("collect_received uses a diagnostic attention path unsupported by cached decoding")
        if self.model.use_value and self.model.allow_exit and cfg.exit_policy != "threshold":
            raise ValueError(
                f"exit_policy={cfg.exit_policy!r}: 'capacity' ranks a token against the whole sequence (later "
                "tokens change earlier exits) and 'random' is stochastic; neither has a causal per-token "
                "answer, so the cached decoder only supports 'threshold'")
        if self.model.force_random_tiles:
            raise ValueError("force_random_tiles draws fresh tiles every forward; nothing to cache against")

    # -------------------------------------------------------------- generation ------------------
    def reset(self) -> None:
        super().reset()
        self.states.clear()
        self.tile_token_counts.clear()

    def _begin_token(self) -> None:
        self._touched_this_token = set()
        self._tile_calls_this_token = 0

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, keep_logits: bool = False) -> Dict[str, Any]:
        """Prefill the prompt once, then one incremental pass per generated token.

        `keep_logits=True` also returns the prefill logits [S,V] and the last-position logits of every step
        (tensors, for the exactness test; not written to ledgers)."""
        self.reset()
        self._check_supported()
        ids = prompt_ids.to(CPU_DEVICE).long()
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.size(0) != 1:
            raise ValueError("CachedStreamingDecoder is batch-1 by construction")
        prompt_tokens = int(ids.size(1))
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        records: List[Dict[str, Any]] = []
        generated: List[int] = []
        last_logits: List[torch.Tensor] = []
        prefill_logits: Optional[torch.Tensor] = None
        previous = self.cache.stats()

        for step in range(max_new_tokens):
            if self.safety_check is not None:
                self.safety_check()
            if step == 0:
                chunk, position, phase = ids, 0, "prefill"
            else:
                chunk = torch.tensor([[generated[-1]]], dtype=torch.long)
                position, phase = prompt_tokens + step - 1, "decode"
            self._begin_token()
            started = time.perf_counter()
            with torch.no_grad():
                logits, mean_hops = self._forward_chunk(chunk, position)
            compute_seconds = time.perf_counter() - started
            next_id = int(logits[0, -1].argmax().item())
            generated.append(next_id)
            if keep_logits:
                last_logits.append(logits[0, -1].clone())
                if step == 0:
                    prefill_logits = logits[0].clone()

            current = self.cache.stats()
            record = self._token_record(step, next_id, prompt_tokens + step, mean_hops, compute_seconds,
                                        previous, current, phase)
            records.append(record)
            previous = current
            if self.verbose:
                print(f"  [{phase:7s}] token {step + 1:>3}/{max_new_tokens} id {next_id:>6} "
                      f"hops {mean_hops:4.1f} tiles {record['tiles_touched']} compute {record['compute_ms']:8.1f} ms "
                      f"misses {record['misses']} read {record['bytes_read_total'] / MEGABYTE:6.2f} MB", flush=True)

        summary = self._summarise_cached(records, generated, prompt_tokens)
        if keep_logits:
            summary["last_logits"] = last_logits
            summary["prefill_logits"] = prefill_logits
        return summary

    def _token_record(self, step: int, token_id: int, context_len: int, mean_hops: float, compute_seconds: float,
                      previous: Dict[str, Any], current: Dict[str, Any], phase: str) -> Dict[str, Any]:
        injected_delta = ((current["injected_seconds"] - previous["injected_seconds"])
                          + (current["prefetch_injected_seconds"] - previous["prefetch_injected_seconds"]))
        accounted = 0.0 if self.cache.sleep_injected else injected_delta
        return {
            "step": step,
            "phase": phase,
            "token_id": token_id,
            "context_len": context_len,
            "tiles_touched": sorted(self._touched_this_token),
            "unique_tiles": len(self._touched_this_token),
            "tile_calls": self._tile_calls_this_token,
            "mean_hops": float(mean_hops),
            "hits": current["hits"] - previous["hits"],
            "misses": current["misses"] - previous["misses"],
            "evictions": current["evictions"] - previous["evictions"],
            "prefetches": current["prefetches"] - previous["prefetches"],
            "bytes_read": current["bytes_read"] - previous["bytes_read"],
            "bytes_read_total": current["bytes_read_total"] - previous["bytes_read_total"],
            "read_ms": (current["read_seconds"] - previous["read_seconds"]) * 1000.0,
            "injected_ms": injected_delta * 1000.0,
            "compute_ms": compute_seconds * 1000.0,
            "wall_ms": (compute_seconds + accounted) * 1000.0,
        }

    def _summarise_cached(self, records: List[Dict[str, Any]], generated: List[int],
                          prompt_tokens: int) -> Dict[str, Any]:
        """The reference summary (all-in tokens/s, comparable with full recompute) plus prefill/decode split."""
        summary = self._summarise(records, generated, prompt_tokens)
        prefill = records[0]
        decode = records[1:]
        decode_wall_ms = sum(record["wall_ms"] for record in decode)
        decode_bytes = sum(record["bytes_read_total"] for record in decode)
        token_histogram = {int(index): int(count) for index, count in sorted(self.tile_token_counts.items())}
        summary.update({
            "protocol": CACHED_PROTOCOL,
            "prefill_ms": prefill["wall_ms"],
            "prefill_compute_ms": prefill["compute_ms"],
            "prefill_bytes_read": prefill["bytes_read_total"],
            "prefill_tiles_touched": prefill["tiles_touched"],
            "prefill_mean_hops": prefill["mean_hops"],
            "decode_tokens": len(decode),
            "decode_tokens_per_second": (len(decode) / (decode_wall_ms / 1000.0)) if decode_wall_ms > 0 else 0.0,
            "decode_mean_wall_ms_per_token": (decode_wall_ms / len(decode)) if decode else 0.0,
            "decode_mean_mb_per_token": (decode_bytes / MEGABYTE / len(decode)) if decode else 0.0,
            "decode_mean_hops": (sum(record["mean_hops"] for record in decode) / len(decode)) if decode else 0.0,
            "state_bytes": self.states.bytes(),
            "state_mb": self.states.bytes() / MEGABYTE,
            "num_streams": self.states.num_streams(),
            "stream_lengths": self.states.stream_lengths(),
            "tile_token_histogram": token_histogram,
            "tile_token_entropy_bits": usage_entropy_bits(token_histogram),
        })
        return summary

    # -------------------------------------------------------------- the hop loop ----------------
    def _run_tile_with_state(self, hop: int, tile_index: int, h_sub: torch.Tensor, valid: torch.Tensor,
                             alpha_sub: torch.Tensor) -> torch.Tensor:
        """Page the tile in, run it on the chunk's subset continuing the (hop, tile) stream, store the state."""
        self._page_in(tile_index)
        self.tile_token_counts[tile_index] += int(valid.sum().item())
        mod_film, mod_lora = self.model.banks[tile_index].modulations(alpha_sub)
        state = self.states.get(hop, tile_index)
        delta, new_state = hybrid_tile_forward_with_state(self.model.tiles[tile_index], h_sub, mod_film, mod_lora, state)
        self.states.put(hop, tile_index, new_state)
        return delta

    def _forward_chunk(self, ids: torch.Tensor, position_offset: int) -> Tuple[torch.Tensor, float]:
        """`GeneralRoutedLM.forward` (eval, no labels) for tokens at positions offset..offset+n-1, with every
        tile continuing its (hop, tile) stream. Returns (logits [1,n,V], mean hops per token)."""
        model = self.model
        cfg = model.config
        M, EXIT = model.M, model.EXIT
        n = ids.size(1)
        assert position_offset + n <= cfg.max_position_embeddings, (
            f"position {position_offset + n} exceeds max_position_embeddings={cfg.max_position_embeddings}")
        H = cfg.max_hops

        positions = torch.arange(position_offset, position_offset + n, device=ids.device).unsqueeze(0)
        h = F.embedding(ids, embedding_weight(model)) + model.pos_embeddings(positions)
        path = None
        if model.use_path:
            start = model.tile_embed.weight[M].expand(n, -1)
            path = model.path_cell(start, torch.zeros(n, cfg.path_dim, device=ids.device)).view(1, n, -1)

        active = torch.ones(1, n, dtype=torch.bool, device=ids.device)
        hops_executed = 0.0
        for hop in range(H):
            cond = model._cond(h, path, hop)
            pred_gain = model.value_head(cond).float().squeeze(-1) if model.use_value else None

            if cfg.routing == "free":
                logits = model.tile_router(cond).float() / cfg.router_temperature
                probs = F.softmax(logits, dim=-1)                     # [1,n,M]
                choice = probs.argmax(dim=-1)
            else:                                                     # fixed: hop h uses tile h % M
                probs = None
                choice = torch.full((1, n), hop % M, dtype=torch.long, device=ids.device)
            if model.force_tile is not None:
                choice = torch.full((1, n), int(model.force_tile), dtype=torch.long, device=ids.device)

            if model.use_value and hop >= cfg.min_hops and model.allow_exit:
                cont = pred_gain > cfg.exit_lambda                    # threshold policy (the only causal one)
                choice = torch.where(cont, choice, torch.full_like(choice, EXIT))
            choice = torch.where(active, choice, torch.full_like(choice, EXIT))

            alpha = model._alpha(path, hop, (1, n))
            h_new = h
            for m in range(M):
                sel = (choice == m) & active
                if not bool(sel.any()):
                    continue
                order, valid, _k = gather_subset(sel)
                assert bool(valid.all()), "batch 1 gathers no padding rows"
                gather_index = order.unsqueeze(-1).expand(-1, -1, model.hidden)
                h_sub = torch.gather(h, 1, gather_index)
                alpha_sub = torch.gather(alpha, 1, order.unsqueeze(-1).expand(-1, -1, alpha.size(-1)))
                delta = self._run_tile_with_state(hop, m, h_sub, valid, alpha_sub)
                if probs is not None:
                    gate = torch.gather(probs[..., m], 1, order).unsqueeze(-1).to(delta.dtype)
                    if cfg.gate_ste:
                        gate = gate / gate.detach().clamp(min=1e-6)
                    delta = delta * gate
                delta = delta * valid.unsqueeze(-1).to(delta.dtype)
                h_new = h_new.scatter_add(1, gather_index, delta)

            moved = choice != EXIT
            h = h_new
            hops_executed += float(moved.sum().item())

            if model.use_path:
                step_in = model.tile_embed(choice.clamp(max=M)) + model._hop_vector(hop)
                path_next = model.path_cell(step_in.reshape(n, -1), path.reshape(n, -1)).view(1, n, -1)
                path = torch.where(moved.unsqueeze(-1), path_next, path)

            active = active & moved
            if not bool(active.any()):
                break

        h = model.final_norm(h)
        logits = F.linear(h, embedding_weight(model))
        return logits, hops_executed / n


__all__ = ["CachedStreamingDecoder", "CACHED_PROTOCOL"]
