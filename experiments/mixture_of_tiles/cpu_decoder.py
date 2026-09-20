"""
experiments/mixture_of_tiles/cpu_decoder.py: greedy CPU decoding where every tile must be paged in first.

The model's single tile-dispatch funnel (`GeneralRoutedLM._run_tile`) is wrapped so that no tile can execute
unless it came through the `TileCache`; an eviction is mirrored straight back into the model. That makes
bytes-read-per-generated-token a measured quantity rather than an estimate. See AGENTS.md for why that is
the metric this folder cares about.

Protocol: batch 1, greedy, FULL RECOMPUTE of the prefix per generated token (no KV cache), the same
protocol as deep dive 02.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard rule: this folder never touches the GPU

import sys
import math
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

torch.set_num_threads(4)
CPU_DEVICE = torch.device("cpu")

from experiments.mixture_of_tiles.tile_store import (
    MEGABYTE,
    TileCache,
    enable_residency_guard,
    install_tile,
    tile_is_resident,
    uninstall_tile,
)


class StreamingDecoder:
    """Greedy batch-1 CPU decoder with a paged tile library behind it."""

    def __init__(self, model, cache: TileCache, prefetch: Optional[bool] = None, prefetch_top_k: int = 2,
                 build_masks: Optional[bool] = None, scrub_on_evict: bool = False,
                 verbose: bool = False, safety_check: Optional[Callable[[], None]] = None):
        self.model = model
        self.model.eval()
        self.cache = cache
        self.prefetch_enabled = cache.prefetch_enabled if prefetch is None else bool(prefetch)
        self.prefetch_top_k = int(prefetch_top_k)
        # "prefetch the two most-used tiles so far": the router's next-hop distribution is only available
        # from inside the hop loop, which `_run_tile` does not see, so this folder uses the usage-history
        # policy the spec allows. Documented in AGENTS.md.
        self.prefetch_policy = "most_used_so_far"
        self.build_masks = build_masks
        self.scrub_on_evict = bool(scrub_on_evict)
        self.verbose = bool(verbose)
        self.safety_check = safety_check

        enable_residency_guard(self.model)
        for index in range(len(self.model.tiles)):
            uninstall_tile(self.model, index, scrub=False)      # start cold: nothing is paged in
        self.cache.on_evict = self._on_evict
        self._original_run_tile = self.model._run_tile
        self.model._run_tile = self._run_tile                   # instance attribute shadows the bound method

        self.tile_use_counts: Counter = Counter()
        self._touched_this_token: set = set()
        self._tile_calls_this_token = 0

    # -------------------------------------------------------------- dispatch --------------------
    def detach(self) -> None:
        """Put the model's own dispatch back (so the same model can be reused outside the decoder)."""
        self.model._run_tile = self._original_run_tile
        self.cache.on_evict = None

    def _on_evict(self, index: int) -> None:
        uninstall_tile(self.model, index, scrub=self.scrub_on_evict)

    def _prefetch_candidates(self, current: int) -> List[int]:
        ranked = [index for index, _count in self.tile_use_counts.most_common()
                  if index != current and not self.cache.contains(index)]
        return ranked[: self.prefetch_top_k]

    def _page_in(self, tile_index: int) -> None:
        """The one funnel every tile execution goes through: cache lookup, install if evicted, counters, prefetch."""
        tensors = self.cache.get(tile_index)
        if not tile_is_resident(self.model, tile_index):
            install_tile(self.model, tile_index, tensors, build_masks=self.build_masks)
        self.tile_use_counts[tile_index] += 1
        self._touched_this_token.add(tile_index)
        self._tile_calls_this_token += 1
        if self.prefetch_enabled:
            self.cache.prefetch(self._prefetch_candidates(tile_index), protect=(tile_index,))

    def _run_tile(self, tile_index: int, h_sub, valid, alpha, need_received):
        """Page tile `tile_index` in through the cache, then run the model's own tile call."""
        self._page_in(tile_index)
        return self._original_run_tile(tile_index, h_sub, valid, alpha, need_received)

    # -------------------------------------------------------------- generation ------------------
    def reset(self) -> None:
        """Cold start: evict everything and zero the counters."""
        self.cache.clear()
        self.cache.reset_stats()
        self.tile_use_counts.clear()

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> Dict[str, Any]:
        """Greedy decode with full recompute per token; returns per-token paging records plus totals."""
        self.reset()
        ids = prompt_ids.to(CPU_DEVICE).long()
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.size(0) != 1:
            raise ValueError("StreamingDecoder is batch-1 by construction")

        records: List[Dict[str, Any]] = []
        generated: List[int] = []
        previous = self.cache.stats()
        for step in range(max_new_tokens):
            if self.safety_check is not None:
                self.safety_check()
            self._touched_this_token = set()
            self._tile_calls_this_token = 0
            started = time.perf_counter()
            with torch.no_grad():
                out = self.model(ids)
            compute_seconds = time.perf_counter() - started
            next_id = int(out["logits"][0, -1].argmax().item())
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long)], dim=1)

            current = self.cache.stats()
            injected_delta = ((current["injected_seconds"] - previous["injected_seconds"])
                              + (current["prefetch_injected_seconds"] - previous["prefetch_injected_seconds"]))
            accounted = 0.0 if self.cache.sleep_injected else injected_delta
            records.append({
                "step": step,
                "token_id": next_id,
                "context_len": int(ids.size(1) - 1),
                "tiles_touched": sorted(self._touched_this_token),
                "unique_tiles": len(self._touched_this_token),
                "tile_calls": self._tile_calls_this_token,
                "mean_hops": float(out["mean_hops"]),
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
            })
            previous = current
            if self.verbose:
                record = records[-1]
                print(f"  [full   ] token {step + 1:>3}/{max_new_tokens} id {next_id:>6} "
                      f"hops {record['mean_hops']:4.1f} tiles {record['tiles_touched']} "
                      f"compute {record['compute_ms']:8.1f} ms misses {record['misses']} "
                      f"read {record['bytes_read_total'] / MEGABYTE:6.2f} MB", flush=True)

        return self._summarise(records, generated, int(prompt_ids.numel()))

    def _summarise(self, records: List[Dict[str, Any]], generated: List[int], prompt_tokens: int) -> Dict[str, Any]:
        total_wall_ms = sum(record["wall_ms"] for record in records) or 1e-9
        total_bytes = sum(record["bytes_read_total"] for record in records)
        total_hits = sum(record["hits"] for record in records)
        total_misses = sum(record["misses"] for record in records)
        requests = total_hits + total_misses
        working_set = sorted({index for record in records for index in record["tiles_touched"]})
        histogram = {int(index): int(count) for index, count in sorted(self.tile_use_counts.items())}
        stats = self.cache.stats()
        return {
            "protocol": "greedy batch-1, full recompute per token, no KV cache",
            "prompt_tokens": prompt_tokens,
            "new_tokens": len(records),
            "generated_ids": generated,
            "per_token": records,
            "tokens_per_second": len(records) / (total_wall_ms / 1000.0),
            "total_wall_ms": total_wall_ms,
            "mean_wall_ms_per_token": total_wall_ms / max(1, len(records)),
            "mean_bytes_per_token": total_bytes / max(1, len(records)),
            "mean_mb_per_token": total_bytes / MEGABYTE / max(1, len(records)),
            "mean_tiles_per_token": sum(record["unique_tiles"] for record in records) / max(1, len(records)),
            "mean_tile_calls_per_token": sum(record["tile_calls"] for record in records) / max(1, len(records)),
            "mean_hops": sum(record["mean_hops"] for record in records) / max(1, len(records)),
            "hits": total_hits,
            "misses": total_misses,
            "hit_rate": (total_hits / requests) if requests else 0.0,
            "mean_miss_latency_ms": stats["mean_miss_latency_ms"],
            "total_read_ms": (stats["read_seconds"] + stats["prefetch_read_seconds"]) * 1000.0,
            "total_injected_ms": (stats["injected_seconds"] + stats["prefetch_injected_seconds"]) * 1000.0,
            "total_bytes_read": total_bytes,
            "working_set_tiles": working_set,
            "working_set_size": len(working_set),
            "tile_usage_histogram": histogram,
            "tile_usage_entropy_bits": usage_entropy_bits(histogram),
            "tile_usage_entropy_max_bits": math.log2(len(self.model.tiles)) if len(self.model.tiles) > 1 else 0.0,
            "prefetch": self.prefetch_enabled,
            "prefetch_policy": self.prefetch_policy if self.prefetch_enabled else None,
            "cache_stats": stats,
        }


def usage_entropy_bits(histogram: Dict[int, int]) -> float:
    """Shannon entropy of the tile-usage distribution, in bits (log2 M for a uniform router)."""
    total = sum(histogram.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in histogram.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


__all__ = ["StreamingDecoder", "usage_entropy_bits"]
