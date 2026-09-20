---
type: track
status: active
started: 2026-09-19
---

# Disk-resident, GPU-free inference tier

A dedicated lane for validating the hardware half of the thesis on this machine's CPU: the ternary tile
library lives on disk, tiles are paged into RAM as the router asks for them, and the machine has no GPU
in the loop. Code: `experiments/mixture_of_tiles/`. Everything in this tier runs CPU-only, never alongside
a spilling GPU job, and under an explicit thread and memory cap (see section 4).

## 1. What "usable" means here

| level | tokens/s at batch 1 | what it proves |
|---|---:|---|
| floor (measured 2026-09-19, random router, fp32 tiles, full recompute, 4 threads) | ~1 | the mechanics work; nothing else |
| interactive draft | >= 10 | state caching alone must get here |
| usable assistant | >= 30 | needs a ternary kernel path |
| headroom target | >= 100 | ternary kernel + paged library with a small working set |

The metric that matters for the disk story is **bytes read per generated token**, not tokens/s: tokens/s
is bounded by compute until the library exceeds RAM, and by bytes/token after that.

## 2. Why the floor is ~1 token/s and what removes each factor

1. **Full recompute per token.** Every generated token re-runs the whole prompt through every hop. A
   per-tile state cache (Mamba: 64 KB per tile per sequence; attention: the KV of the one attention tile)
   makes each new token cost one pass over its own hops. Expected gain: the prompt length, 50-100x at
   64-128 tokens.
2. **fp32 unpacked tiles.** The store unpacks ternary to float32 for `torch.matmul`. A lookup-table
   ternary kernel (the BitNet b1.58 CPU path: 2-bit weights, add/subtract only) reads 8x fewer bytes and
   skips the multiplies. Expected gain: 3-6x on x86 with AVX2, per published numbers for 1.58-bit kernels.
3. **Eight hops of four tiles with a random router.** A trained router stops at ~3 hops (section 13.1)
   and concentrates on a hot set. Expected gain: 2-3x, measured only when the trained checkpoint exists.

Multiplied, the floor becomes hundreds of tokens/s for a 40M-parameter model, which is where the ternary
CPU literature already sits for models of this size.

## 3. How it scales: bigger than RAM

The tile store is a library of independent blobs; the cache pages them. Scaling beyond RAM has three
conditions, each a measurable number:

- **Working set per sequence** (distinct tiles a sequence touches) must be a small fraction of the
  library. Measured by `bench_cpu_decode.py`; needs the trained router. With the random router it is
  the whole library, which is a floor, not a result.
- **Bytes per token** must stay under the disk's sustained rate divided by the target tokens/s: at
  3 GB/s NVMe and 30 tokens/s that is 100 MB per token, i.e. up to ~50 tile misses per token at 1.9 MB
  each, so the miss budget is generous once the working set is small.
- **A tile design that can be small.** The fixed-bytes planner showed the current tile (mixer + FFN)
  cannot be split past 8 tiles at width 512, because the mixer's size does not shrink with the FFN.
  A many-tile library needs a shared mixer with tiled FFN experts (mixture-of-experts-of-tiles). This
  is the design change the tier exists to test, after the trained working set is measured.

Resident bytes are the other half: embeddings (17 MB fp16) are 2.8x the pageable library today. Ternary
embeddings (1.6 MB) are a precondition, not an option, for the disk-resident design.

## 4. Machine-safety rules for this tier (the 2026-09-19 freeze)

The freeze happened while a CPU benchmark ran next to a GPU trainer that had spilled into shared system
memory; the spill, not the benchmark alone, saturated RAM bandwidth. Rules from here:

- CPU-tier jobs run only when no GPU trainer is spilling (`nvidia-smi` memory under ~7 GB) or when the
  GPU is idle.
- `torch.set_num_threads` at most half the physical cores; process memory cap via `--max-resident-mb`
  in the benchmark (fail fast rather than page).
- Benchmarks use short prompts by default (32 tokens, 16 new) and print per-token timings as they go.

## 5. Ladder (each rung is one test plus one benchmark row)

1. state caching (Mamba state + attention KV) with exactness test against full recompute;
2. ternary lookup-table matmul on CPU (the repo's `experiments/bitlinear.py` additive GEMM as reference),
   exactness test, bytes/token and tokens/s;
3. trained-router working set and hit rate from the night-1 checkpoint;
4. shared-mixer tiled-FFN design in `general_model.py` behind a toggle, fixed-bytes sweep to 64 tiles;
5. cold-disk measurement with unbuffered reads (`FILE_FLAG_NO_BUFFERING`) instead of injected latency.
