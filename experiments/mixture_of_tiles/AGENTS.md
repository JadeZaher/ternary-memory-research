# `experiments/mixture_of_tiles` — module notes

## Why this folder exists

The hardware thesis of this project is **GPU-free inference**: the ternary weight library lives on disk
(NVMe), tiles are paged into RAM as the per-token router asks for them — the flash-resident /
expert-offloading pattern — and only a small hot set is ever resident. Everywhere else in the repo that
thesis is an argument. Here it is a measurement.

The mapping this folder makes concrete:

| storage tier | what lives there | represented by |
|---|---|---|
| **NVMe** | the whole tile library at ~1.6 bits/weight | `tiles.bin` + `index.json` (`TileStore`) |
| **DRAM (hot set)** | the `capacity_tiles` most recently routed tiles, unpacked to fp32 | `TileCache` (LRU) |
| **"SRAM slots"** | the `model.tiles[i]` modules that actually execute | `install_tile` / `uninstall_tile` |
| **always pinned** | token/position embeddings, adapter banks, controller, final norm | `resident.pt` |

`GeneralRoutedLM` already funnels every tile execution through one method, `_run_tile`. `StreamingDecoder`
wraps that funnel so a tile **cannot** execute unless it came through the cache, and mirrors every eviction
back into the model (`TileCache.on_evict` → `uninstall_tile`). A residency guard
(`enable_residency_guard`, a forward pre-hook raising `TileNotResidentError`) makes the invariant
falsifiable rather than assumed: if the mirroring were ever wrong, the model would raise instead of quietly
computing with stale weights.

### Why bytes per token is the metric

Parameter count and FLOPs both mis-describe an offloaded model. What decides whether a 41M-parameter tile
stack decodes at a usable rate on a laptop with no GPU is **how many bytes cross the NVMe→DRAM boundary per
generated token**, because that boundary is two to three orders of magnitude slower than DRAM→cache. So the
measured quantities are: bytes read per generated token, tiles touched per token, cache hit rate, miss
latency, and tokens/s on CPU — plus the per-sequence working set (distinct tiles over a whole generation),
which is the honest answer to "how much RAM does this actually need".

This is the same argument the parent track makes about stored megabytes
(`experiments/general_model/AGENTS.md`), pushed one step further: stored bytes decide whether the model
*fits*; bytes per token decide whether it *runs*.

## The 1.6 bits/weight packing

`pack_ternary` / `unpack_ternary` are base-3: five weights in {-1,0,+1} per byte, because 3⁵ = 243 < 256.
That is **8/5 = 1.6 bits per weight** exactly, against 1.58 bits as the information-theoretic floor
(log₂3) and 2 bits for the naive two-bits-per-trit packing — a 20% saving over the naive form for four
lines of arithmetic. The tail is zero-padded, so short tensors pay more (a 1-element tensor costs 8
bits/weight); at real layer sizes the measured figure is 1.6006.

Block scales (`gamma`, one fp16 per 256 weights, BitLinear's `block_size`) add 0.0625 bits/weight, so the
honest end-to-end ternary figure is **1.663 bits/weight**. Both numbers are printed by the byte table and
both are in the JSON; quoting only 1.6 would under-report the artifact.

Everything in a tile that is *not* a ternary `BitLinear` weight is stored fp16: RMSNorm scales, the Mamba
`conv1d` / `x_proj` / `dt_proj` / `A_log` / `D`, the expert router and the soft-mix branch gate. That
follows the parent track's accounting (`count_parameters`'s `tiles_dense` group) and is not a rounding
detail — see the sweep result below, where the dense part is what ends the sweep.

## `build()` canonicalises the model to store precision

`TileStore.build(model, path, canonicalize=True)` rounds every float parameter and buffer of the source
model to fp16 and back, **in place**, and rounds each BitLinear's `gamma` to fp16 inside its eval cache.
Without this the store is an fp16 *approximation* of the model and a round trip could only be checked to
~1e-3; with it, the deployed artifact and the reference model are the same model and the round-trip test
asserts an exact match (measured: `max |logit delta| = 0.000e+00`). The ternary snapshot is taken from
BitLinear's own eval cache, so the quantisation math is never re-implemented here; `install_tile` writes
the stored snapshot straight back into `_eval_cache` rather than re-quantising a master weight — which is
the point, since in a deployed store the fp32 master weights do not exist at all.

Consequence to know: `canonicalize=True` mutates the model you pass it, and calling `.train()` afterwards
drops the eval caches and re-derives `gamma` from the (fp16-rounded) masters. Build stores from a model you
are finished training.

## What the tile-count sweep tests, and where it stops

`sweep_tile_count.py` plans a **fixed-bytes** sweep: hold `tiles_ternary + tiles_dense` at the 4-tile
reference budget, hold hidden size and the 3:1 mamba:attn ratio, and solve for `intermediate_size` at tile
counts 4/8/16/32/64. The question is whether quality survives splitting one block into many small tiles —
which is what would make paging pay, since more and smaller tiles means a smaller hot set and fewer bytes
per token.

The planner's own result is that **at hidden 512 the sweep runs out of budget after 8 tiles**. The reason is
structural: the sequence mixer (Mamba `in_proj`/`out_proj`, attention q/k/v/o) and the dense Mamba block
are *per-tile constants* — they do not shrink with `intermediate_size`. Only the SwiGLU does. So the fixed
part grows linearly with tile count and at 16 tiles it already exceeds the whole reference budget on its
own. Infeasible counts are emitted with `feasible: false`, the deviation at the minimum legal
`intermediate_size`, and that reason, rather than being silently dropped. Fixing it needs a different lever
(smaller `hidden_size`, smaller `mamba_expand`/`d_state`, or mixer sharing across tiles), not a smaller FFN.

Known shape constraints, carried in the JSON:

* `train_general.py` has **no `--intermediate-size` flag** — `intermediate_size` comes from `PRESETS`. Each
  non-reference plan therefore ships a one-line `preset_patch` that must be added to `PRESETS` before its
  command will run. The commands are emitted, never executed; this script runs no training.
* `intermediate_size` is searched on a multiple-of-8 grid. BitLinear needs `(intermediate × hidden) % 256 == 0`
  for block-scaled gamma, which multiples of 8 satisfy at both hidden 512 and hidden 64.
* Tile counts must be multiples of 4 to hold the 3:1 ratio exactly.

## What a random-init router does and does not show

`bench_cpu_decode.py` falls back to a random-init `GeneralRoutedLM` when no checkpoint is given, and says so
loudly. What that measures is **real**: bytes actually read off disk, real unpack time, real LRU behaviour,
real CPU tokens/s. What it does **not** show is a learned routing pattern. A random tile router routes close
to uniformly, so the usage entropy sits near its log₂M maximum and the working set is the whole tile set —
which is the *worst case* for paging. A trained router that concentrates on a hot subset can only do better
on hit rate and bytes/token. Read the random-init numbers as an upper bound on traffic, never as the
efficiency claim.

For the same reason the benchmark defaults to `allow_exit=False` (every token runs the full hop budget): a
random-init value head makes the adaptive-depth policy meaningless, and the full budget is the comparable,
deterministic worst case.

## Known limitations

* **Full recompute per generated token, no KV cache.** Same protocol as deep dive 02. Tokens/s here is a
  *relative* number for comparing cache capacities, not the throughput a real deployment would see; a KV
  cache (or Mamba's O(1) `step()` path, which this decoder does not use) changes it by orders of magnitude.
* **"Cold" is simulated.** The Windows page cache keeps `tiles.bin` resident after the first pass, so a
  genuinely cold NVMe read is not measurable in-process. `injected_miss_latency_ms_per_mb` models it
  (0.33 ms/MB ≈ 3 GB/s) and is **accounted, not slept**, so the numbers are deterministic and the benchmark
  does not idle. `TileCache(sleep_injected=True)` makes it a real sleep if wall-clock realism matters more
  than run time. Any table quoting an injected column is a model, not a measurement, and is labelled so.
* **Prefetch policy is usage-history, not router lookahead.** The spec's preferred policy — ask the router
  for the next hop's top-2 candidates — needs the hop loop's conditioning vector, which `_run_tile` does not
  receive. The allowed fallback is implemented instead: prefetch the two most-used tiles so far
  (`StreamingDecoder.prefetch_policy == "most_used_so_far"`). It never evicts the tile currently executing
  (`prefetch(..., protect=(current,))`), and it is a no-op at capacity 1.
* **RAM held ≫ bytes read.** A tile is 1.6 bits/weight on disk and fp32 in RAM, a ~20× expansion. The byte
  table reports both (`disk MB` and `ram MB (fp32)`); the cache capacity is in *tiles*, so the DRAM figure to
  budget against is `capacity × ram_bytes`, not `capacity × disk_bytes`. An int8 or fp16 compute path would
  close most of that gap and is the obvious next step.
* **`uninstall_tile` does not scrub by default.** Zeroing a p512 tile costs a full pass over ~8M weights per
  eviction, which would dominate the timing, so eviction sets the residency flag and drops the eval cache;
  the guard is what makes a bypass detectable. `scrub=True` (used by the tests) additionally zeroes the
  tensors.
* **BitLinear's additive GEMM is off by default in the benchmark.** On CPU the mask path is two dense GEMMs
  plus two fp32 mask matrices per layer — double the RAM and roughly double the time, for hardware that
  cannot skip zeros (the same finding as `experiments/bench_gemm_paths.py`). `--additive-gemm` restores it.
  The tests run with it **on**, because that is the model's default eval path and the exactness claim has to
  hold there.
* **Single process, four threads, batch 1.** `torch.set_num_threads(4)` is set in every file in this folder
  and `CUDA_VISIBLE_DEVICES=""` at the top of every script: this folder must never contend with a training
  job for the GPU.

## Measured, 2026-09-19 (CPU, 4 threads, random-init p512, batch 1, 64-token prompt, 32 new tokens)

Byte table, p512 / 4 tiles / `mamba,mamba,mamba,attn` / intermediate 2048 / vocab 16384:

| | bytes | MB |
|---|---:|---:|
| tile 0–2 (mamba), each | 1,884,166 | 1.797 (30.477 MB fp32 in RAM) |
| tile 3 (attn) | 1,531,500 | 1.461 (28.012 MB fp32 in RAM) |
| **tiles.bin total** | **7,183,998** | **6.851** |
| ternary packed (30,932,992 weights) | 6,186,622 | 5.900 |
| gamma, fp16 | 241,664 | 0.230 |
| tile dense, fp16 | 755,712 | 0.721 |
| **resident.pt** | **20,083,218** | **19.153** |

**1.6000 bits/weight packed, 1.6625 with gamma.** The headline here is the last row: *the pinned
embeddings are 2.8× the entire pageable tile library*. Paging optimises the small half of the footprint
until `ternary_embed` or a smaller vocabulary is in play — the same trap section 11 of
`research/hardening-2026-09-18-heldout.md` documents for the parent track.

| capacity | inj ms/MB | tok/s | MB/token | tiles/token | hit rate | working set | mean miss ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.531 | 36.538 | 4.00 | 0.0% | 4 | 53.58 |
| 1 | 0.33 | 0.874 | 36.538 | 4.00 | 0.0% | 4 | 32.42 |
| 2 | 0.00 | 0.945 | 31.044 | 4.00 | 16.4% | 4 | 33.52 |
| 2 | 0.33 | 0.731 | 31.044 | 4.00 | 16.4% | 4 | 43.70 |
| 4 | 0.00 | 1.401 | 0.214 | 4.00 | 99.4% | 4 | 113.83 |
| 4 | 0.33 | 1.704 | 0.214 | 4.00 | 99.4% | 4 | 114.65 |

Read these as: **capacity is everything, and the cliff is at "all tiles fit"**. Going from capacity 2 to 4
cuts traffic 145× (31.0 → 0.214 MB/token) because the whole working set becomes resident and the library is
read exactly once for the whole sequence. Capacity 1 thrashes: eight hops × four tiles is 32 demand misses
per token against a one-slot cache. Tile-usage entropy is 1.976 bits against a log₂4 = 2.0 maximum — a
random-init router is very nearly uniform, so the working set is the whole tile set and these are worst-case
traffic figures.

The `tok/s` column does **not** order the way the injected column implies: the injected cost is accounted,
not slept, so it can only *add* time, yet capacity 1 at 0.33 ms/MB looks faster than at 0.00. The cause is
the Windows page cache — the first row executed is the one that actually pulls `tiles.bin` off disk (mean
miss 53.6 ms against 32.4 ms for the identical run afterwards). This is the "cold is simulated" limitation
above, showing up in the data. Compare capacities within a run, not tok/s across injection settings.

Sweep plan, fixed at the 4-tile budget (6,864,978 B of tile bytes = 6.547 MB by `count_parameters`, 6.851 MB
as actually packed):

| tiles | intermediate | tile MB | packed MB | byte deviation | param deviation | feasible |
|---:|---:|---:|---:|---:|---:|:--|
| 4 (reference) | 2048 | 6.547 | 6.851 | +0.00% | +0.00% | yes |
| 8 | 632 | 6.539 | 6.806 | −0.12% | −11.14% | yes |
| 16 | 8 (floor) | 7.302 | 7.533 | +11.53% | −20.24% | **no** |
| 32 | 8 (floor) | 14.604 | 15.065 | +123.06% | +59.52% | **no** |
| 64 | 8 (floor) | 29.207 | 30.131 | +346.12% | +219.04% | **no** |

So the runnable sweep is **{4, 8}**, and 16 misses by 11.5% even with the FFN driven to nothing. The byte
and parameter deviations move in opposite directions at 8 and 16 tiles because more tiles shift the mix from
1.58-bit ternary toward 16-bit dense: matching bytes costs you parameters.

## Files

| file | what it is |
|---|---|
| `tile_store.py` | base-3 packing, `TileStore` (build/open/read), `TileCache` (LRU + prefetch + injected latency), `install_tile`/`uninstall_tile`, residency guard |
| `cpu_decoder.py` | `StreamingDecoder`: greedy batch-1 CPU decode where every tile is paged in through the cache; per-token paging records |
| `bench_cpu_decode.py` | CLI benchmark → `outputs/mixture-of-tiles-bench.json` |
| `sweep_tile_count.py` | fixed-bytes tile-count planner → `outputs/mixture-of-tiles-sweep-plan.json` (plans only, no training) |
| `test_mixture_of_tiles.py` | CPU-only suite: packing, store round trip, cache semantics, decode invariance, planner |

## Running

```bash
python experiments/mixture_of_tiles/test_mixture_of_tiles.py
python experiments/mixture_of_tiles/bench_cpu_decode.py --capacity 1,2,4 --prompt-tokens 64 --new-tokens 32
python experiments/mixture_of_tiles/bench_cpu_decode.py --checkpoint outputs/checkpoints/general-G1-full-best.pt
python experiments/mixture_of_tiles/sweep_tile_count.py
```
