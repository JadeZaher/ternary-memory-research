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
| interactive draft (target) | >= 10 | a latency target to test after state caching |
| usable assistant (target) | >= 30 | a latency target, separate from language quality |
| headroom (target) | >= 100 | a stretch target, not a forecast |

**Bytes read per generated token** is a central disk-streaming metric, alongside tokens/s. Computation,
DRAM bandwidth, unpacking and storage can each limit throughput; library size alone does not identify
the bottleneck. Measure the costs under matched settings.

## 2. Why the floor is ~1 token/s and what removes each factor

1. **Full recompute per token.** Every generated token re-runs the whole prompt through every hop.
   State keyed by **(hop, tile)** preserves each gathered stream's Mamba convolution/SSM state or
   attention KV. The implemented cached decoder processes the prompt once and then only each new
   token's hops. This removes repeated prefix work; its throughput improvement is still unmeasured.
2. **fp32 unpacked tiles.** The store packs five ternary weights per byte, then expands them to
   floating-point matrices for computation. A packed ternary CPU kernel is a separate proposed
   implementation. Its storage format, scale handling, exactness and speed need their own test.
3. **Random routing.** The existing benchmark uses random initialization with exits disabled.
   A trained router might reduce hops or concentrate accesses, but neither is guaranteed. Its
   incremental access trace and hit rate versus capacity must be measured at useful model quality.

These are opportunities, not measured speedup factors. They cannot be multiplied into a throughput
prediction: changing one bottleneck can expose another, and prompt prefill still costs time.

## 3. How it scales: bigger than RAM

The tile store is a library of independent blobs; the cache pages them. Scaling beyond RAM has three
conditions, each a measurable number:

- **Reuse within the cache capacity.** Distinct tiles over a sequence describe coverage, but access
  order and reuse distance determine cache hits. A sequence can gradually visit a large library without
  thrashing; a smaller repeatedly evicted set can still perform poorly. Measure incremental traces,
  token-assignment entropy and hit rate versus capacity together.
- **Bytes per token** must fit the effective disk-bandwidth budget at the target tokens/s. That is
  an upper bound: read latency, unpacking and computation also take time. Current mmap counters measure
  logical bytes requested from the store, not physical NVMe transfers; cold-disk evidence is pending.
- **A tile design that can be small.** The fixed-bytes planner showed the current tile (mixer + FFN)
  cannot be split past 8 tiles at width 512, because the mixer's size does not shrink with the FFN.
  A many-tile library needs a shared mixer with tiled FFN experts (mixture-of-experts-of-tiles). This
  is the design change the tier exists to test, after the trained working set is measured.

Resident bytes are the other half. The existing ledger reports 20,083,218 bytes for the stored resident
tensors, versus 7,183,998 bytes for the tile library. These are artifact sizes, not process RAM.
The implementation retains master tile parameters after eviction, and installed tiles also have derived
compute tensors. A bounded-RAM deployment must remove those retained masters and account for actual
allocations, embeddings, routing/adapters, sequence states and temporary buffers. It is not yet demonstrated.

## 4. Machine-safety rules for this tier (the 2026-09-19 freeze)

The freeze happened while a CPU benchmark ran next to a GPU trainer that had spilled into shared system
memory; the spill, not the benchmark alone, saturated RAM bandwidth. Rules from here:

- CPU-tier jobs run only when no GPU trainer is spilling (`nvidia-smi` memory under ~7 GB) or when the
  GPU is idle.
- `torch.set_num_threads(4)` at most; process memory cap via `--max-rss-gb` in the benchmark.
  Missing machine telemetry must refuse execution. Guards at stage and token boundaries are checks,
  not an operating-system hard allocation cap; a single operation can transiently exceed the limit.
- Benchmarks use short prompts by default (32 tokens, 16 new) and print per-token timings as they go.

## 5. Ladder (each rung is one test plus one benchmark row)

1. state caching (Mamba state + attention KV) with exactness test against full recompute;
   **status 2026-09-19:** implemented (`experiments/mixture_of_tiles/cached_decoder.py`, state keyed by
   (hop, tile)); exactness test passes (ids identical, logits within ~3e-7 of full recompute, exits mixed,
   state survives eviction); tokens/s row not yet measured — the GPU trainer was running (section 4);
   **review update 2026-09-20 UTC:** runner safety/accounting defects fixed; one integrated run passes
   all ten tests, including dense-GEMM fixed-routing checks at 32/64/128-token prompts on tiny geometry
   (maximum generated-step logit error 1.788e-07 across those cases, IDs identical). Evidence:
   `outputs/mixture-of-tiles-rung1-validation-2026-09-20.json`. Full-size throughput remains unmeasured;
2. ternary lookup-table matmul on CPU (the repo's `experiments/bitlinear.py` additive GEMM as reference),
   exactness test, bytes/token and tokens/s;
3. trained-router working set and hit rate from the night-1 checkpoint;
4. shared-mixer tiled-FFN design note and fixed-bytes plan to 64 tiles; no changes to `general_model.py`;
5. cold-disk measurement with unbuffered reads (`FILE_FLAG_NO_BUFFERING`) instead of injected latency.
