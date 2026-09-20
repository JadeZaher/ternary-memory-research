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

Parameter count and FLOPs alone do not describe an offloaded model. Disk traffic per generated token
can limit its throughput, alongside CPU computation and loading overhead. The measured quantities are
logical bytes requested from the store, tiles touched, cache hit rate, page-in latency (including
unpacking), and CPU tokens/s. Current mmap reads may be served from the operating-system page cache;
they are not measurements of physical NVMe transfers. Distinct tiles over a sequence describe coverage;
access order and reuse distance determine how much cache capacity avoids repeated loading.

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
the intended deployment path. This prototype still allocates the original module parameters, including
fp32 master weights; removing them from a deployed runtime is outstanding work.

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
loudly. It measures logical store reads, unpack time, LRU behaviour and CPU tokens/s. It does not show a
learned routing pattern or language quality. The original call histogram counts dispatches over full
prefixes, so its near-maximal entropy does not establish uniform token assignments. Random initialization
is a baseline, not a worst-case traffic bound: training can change both tile frequency and reuse distance.
Use incremental token assignments and hit rate versus capacity to assess learned locality.

For the same reason the benchmark defaults to `allow_exit=False` (every token runs the full hop budget): a
random-init value head makes the adaptive-depth policy meaningless, and the full budget is the comparable,
deterministic worst case.

## State caching (ladder rung 1): why the state is keyed by (hop, tile)

Every hop of `GeneralRoutedLM.forward` gathers its own subset — the tokens that chose tile `m` at hop `h`
and were still active — in causal order, and runs the tile on that subset as *one sequence*. Mamba's
convolution/SSM state and attention's keys/values therefore belong to the **(hop, tile) stream**, not to
the tile. One state per tile would splice the sequence histories of different hops together and change what
the model computes; a token that skips a stream (it exited earlier, or chose another tile at that hop)
must not advance it. `StreamStateCache` (`state_cache.py`) is exactly that map.

`CachedStreamingDecoder` (`cached_decoder.py`) re-runs the eval-mode hop loop lane-locally for a **chunk**
of tokens at a position offset: the prompt is one chunk (prefill), every generated token a chunk of one.
Tiles are called piecewise through `hybrid_tile_forward_with_state`: norms, branch mix, FiLM, LoRA and the
SwiGLU are per token, so only the sequence mixer continues from stream state.

* **Mamba chunk.** The selective scan is a linear recurrence, so a chunk that starts from state `s_0` is
  the block's own zero-state scan (its arithmetic, `_selective_scan`) plus the carried term
  `C_t · A_t s_0`, `A_t = prod_{r<=t} a_r` over the chunk. The final state is recomputed in the same
  signed-log-space form for the last position only; a single-token chunk is the plain step recurrence
  (the arithmetic of `TernaryMambaBlock.step`). The causal conv runs on `[history ‖ chunk]` with no
  padding, which for a zero history is the block's `padding=d_conv-1` + crop.
* **Attention chunk.** Cached K,V plus the new rows; mask `key <= past + query`, i.e. the stream's causal
  mask restricted to the new rows. At prefill (`past = 0`) that is the reference `causal & valid | eye`,
  because batch 1 gathers no padding rows.
* **State lives in the decoder, not in the tile.** `uninstall_tile` never touches sequence state, so a
  stream survives any number of weight evictions and re-installs (tested at capacity 1). `reset()`
  clears the streams between generations.
* **The stateful path keeps the paging invariant.** Every tile call goes through the one funnel
  (`_page_in`) and re-checks residency itself: an evicted tile raises `TileNotResidentError` from
  `hybrid_tile_forward_with_state`, because the module pre-hook only fires on `tile.forward`, which the
  piecewise path does not call.

### What the cached decoder refuses, and why

* `exit_policy="capacity"` ranks a token against the whole sequence (top-k over S), so a later token can
  flip an earlier token's exit; with full recompute that decision changes as the prefix grows, and there is
  no causal per-token answer to cache. `"random"` is stochastic. Both raise `ValueError`; `"threshold"`
  (the trained policy) is the only supported one.
* `force_random_tiles` (stochastic), `collect_received` (a dense-softmax diagnostic path) and the
  training-only branches (labels, deep supervision, `probe`) are not reproduced. Batch 1 only.

### What "exact" means here

Test *f* asserts identical greedy ids, prefill logits within 1e-4 of `model(prompt)`, and per-token logits
within 1e-4 of a fresh full recompute, with exits on and off, at capacity 1 and "all tiles", and with the
path-conditioned controller. Test *e* isolates the mixer math: a 12-token sequence fed as chunks of 5, 1
and 6 tokens equals one whole-sequence tile call within 1e-5. Expect ~1e-6, not 0.0: the reference
computes each position's SSM state with a `logcumsumexp` over the whole prefix, the cached path carries a
state; both are fp32 roundings of the same recurrence. Attention differs only in where SDPA splits the
softmax.

### Reading the cached rows

* `tokens_per_second` includes **prefill + decode forwards**, comparable with the full-recompute row
  whose first step also pays a prompt forward. It excludes token selection, bookkeeping, monitoring and
  progress printing; it is not end-to-end request latency. `decode_tokens_per_second` and `prefill_ms`
  separate the two phases.
  `mean_mb_per_token` includes prefill traffic; `decode_mean_mb_per_token` covers decode-only paging
  traffic. A short post-prefill sample does not establish steady state.
* `tile_token_histogram` counts (token, hop) assignments. The older `tile_usage_histogram` counts tile
  *calls*: in the full-recompute decoder one call covers a whole recomputed prefix, so its entropy is not a
  per-token routing statistic — the token histogram is the one to quote for rung 3.
* `state_bytes` is the RAM the streams hold: per Mamba stream `d_inner × (d_state + d_conv − 1)` floats,
  per attention stream `2 × tokens × hidden` floats. At p512 that is ~76 KB per Mamba stream against a
  30 MB fp32 tile — the cache is not where the RAM goes.

## Known limitations

* **The full-recompute `StreamingDecoder` is kept unchanged as the reference** (deep dive 02 protocol);
  `CachedStreamingDecoder` is the rung-1 path. Its tokens/s is not yet in a ledger: the benchmark must
  not run next to the GPU trainer (machine rule below), so the rung-1 row is pending.
* **The benchmark guards itself.** `bench_cpu_decode.py` refuses missing GPU/RAM/RSS telemetry and checks
  the configured limits before and after construction and between rows. RAM/RSS are checked before each
  token; GPU polling is throttled between those token checks. Defaults are 7000 MiB GPU usage, 4 GiB free
  RAM and 6 GiB process RSS. These are boundary checks, not an OS allocation cap: one operation can
  transiently exceed a threshold. Both modes print each token. `--check-only` inspects readiness without
  allocating a model; it still imports torch. There is no guard override; flags may tighten the limits.
* **Evidence files are append-by-new-file.** The default is `outputs/mixture-of-tiles-state-cache.json`;
  an existing path is refused. Choose a new ledger filename for each run. Failed or interrupted runs
  save completed rows with `completed: false` and a stop reason, then exit nonzero. Paired full/cached
  rows must generate identical IDs. Temporary stores are cleaned up on success and failure.
* **"Cold" is simulated.** The Windows page cache keeps `tiles.bin` resident after the first pass, so a
  genuinely cold NVMe read is not measurable in-process. `injected_miss_latency_ms_per_mb` models it
  (0.33 ms/MB ≈ 3 GB/s) and is **accounted, not slept**, so the numbers are deterministic and the benchmark
  does not idle. `TileCache(sleep_injected=True)` makes it a real sleep if wall-clock realism matters more
  than run time. Any table quoting an injected column is a model, not a measurement, and is labelled so.
  Prefetch measured read time and injected time have separate counters; only unslept injected time is
  added to measured forward duration. The legacy `prefetch_seconds` field is their sum.
* **Prefetch policy is usage-history, not router lookahead.** The spec's preferred policy — ask the router
  for the next hop's top-2 candidates — needs the hop loop's conditioning vector, which `_run_tile` does not
  receive. The allowed fallback is implemented instead: prefetch the two most-used tiles so far
  (`StreamingDecoder.prefetch_policy == "most_used_so_far"`). It never evicts the tile currently executing
  (`prefetch(..., protect=(current,))`), and it is a no-op at capacity 1.
* **Artifact bytes are not process RAM.** `ram_bytes` counts an unpacked ternary tensor and dense tile
  tensors; it omits scales, effective compute weights, optional masks, retained master parameters and
  temporary allocations. `capacity × ram_bytes` is not a complete RAM budget. The resident artifact is
  stored at fp16 precision but loaded to fp32. RSS checks are needed in addition to logical accounting;
  a runtime that actually discards master parameters remains future work.
* **`uninstall_tile` does not scrub by default.** Zeroing a p512 tile costs a full pass over ~8M weights per
  eviction, which would dominate the timing, so eviction sets the residency flag and drops the eval cache;
  the guard is what makes a bypass detectable. `scrub=True` (used by the tests) additionally zeroes the
  tensors without releasing their storage.
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
per token against a one-slot cache. Tile-call entropy is 1.976 bits against a log₂4 = 2.0 maximum.
That describes dispatch frequency over recomputed prefixes, not token-assignment uniformity or a
worst-case traffic bound. The recorded sequence working set covers the whole library.

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

## Verified, 2026-09-19 (rung 1 exactness; CPU, 4 threads, tiny test geometry, `test_mixture_of_tiles.py`)

```
  tile 0 (mamba): chunks [(0, 5), (5, 6), (6, 12)] max |delta| 1.863e-08 stream tokens 12
  free routing, allow_exit=True, capacity 1: ids [39, 39, 39, 351, 505, 505] prefill max |delta| 3.576e-07 step max |delta| 1.788e-07 mean hops ref 2.59 cached 2.40 misses 13 evictions 12 streams 7 state 32,256 B
  path-conditioned routing, capacity 2: ids [148, 215, 215, 215] prefill max |delta| 1.825e-07 step max |delta| 2.161e-07 mean hops ref 4.00 cached 4.00 misses 14 evictions 12 streams 11 state 47,104 B
```

The rung-1 **benchmark row is pending** (see `HANDOFF-2026-09-20.md`): it must not run beside the GPU trainer.

## Reviewed and validated, 2026-09-20 UTC

The independent adversarial review found no core caching defect for the supported batch-one fp32 eval
path. It identified an unsupported diagnostic-mode omission and runner defects: fail-open telemetry,
infrequent checks, double-counted prefetch reads, missing full-mode progress, failure-path scratch leaks,
and possible ledger overwrite. These are fixed. The runner has no guard bypass, and threshold flags can
only tighten the machine limits. The display labels dispatch entropy as `call H`.

One integrated CPU run passed all ten tests. Evidence and source hashes are in
`outputs/mixture-of-tiles-rung1-validation-2026-09-20.json`; raw output is in
`work/rung1-integrated-tests-2026-09-20.log`. The added longer-prompt checks use a tiny fixed-routing model,
dense GEMM (the benchmark compute mode), capacity one, and three generated tokens. They are correctness
experiments, not throughput measurements or evidence of trained language quality.

| prompt tokens | maximum prefill logit error | maximum generated-step logit error | greedy IDs |
|---:|---:|---:|---|
| 32 | 0.000e+00 | 8.941e-08 | identical |
| 64 | 0.000e+00 | 6.706e-08 | identical |
| 128 | 0.000e+00 | 1.788e-07 | identical |

Other added cases cover decoder reuse, a unit-width causal convolution, all unsupported routing/attention
policies, mocked prefetch timing, and stopping at the next token boundary when a guard fails. Mocked
timings validate accounting only. Long generated sequences, trained-checkpoint exactness and p512
throughput remain outstanding. Read-only trainer/memory observations are recorded separately in
`outputs/mixture-of-tiles-monitor-2026-09-20.json`; they are not performance results.

## Files

| file | what it is |
|---|---|
| `tile_store.py` | base-3 packing, `TileStore` (build/open/read), `TileCache` (LRU + prefetch + injected latency), `install_tile`/`uninstall_tile`, residency guard |
| `cpu_decoder.py` | `StreamingDecoder`: greedy batch-1 CPU decode, full recompute per token, every tile paged in through the cache (the reference); per-token paging records |
| `state_cache.py` | `StreamStateCache` keyed by (hop, tile); chunked Mamba / attention / tile forwards that continue a stream |
| `cached_decoder.py` | `CachedStreamingDecoder`: one prefill pass then one pass per generated token, same paging funnel, prefill/decode timings split |
| `bench_cpu_decode.py` | CLI benchmark (`--mode full|cached|both`, prompt-length list, machine guards) → a new `outputs/mixture-of-tiles-state-cache.json` |
| `sweep_tile_count.py` | fixed-bytes tile-count planner → `outputs/mixture-of-tiles-sweep-plan.json` (plans only, no training) |
| `test_mixture_of_tiles.py` | CPU-only suite: packing, store round trip, cache semantics, decode invariance, planner |

## Running

```bash
python experiments/mixture_of_tiles/test_mixture_of_tiles.py
# rung 1: full recompute vs cached state at 32/64/128-token prompts (only when the GPU trainer is idle)
python experiments/mixture_of_tiles/bench_cpu_decode.py --mode both --prompt-tokens 32,64,128 --new-tokens 16 --capacity 2,4 --no-cold --output outputs/mixture-of-tiles-state-cache.json
python experiments/mixture_of_tiles/bench_cpu_decode.py --mode full --capacity 1,2,4 --prompt-tokens 64 --new-tokens 32
python experiments/mixture_of_tiles/bench_cpu_decode.py --checkpoint outputs/checkpoints/general-G1-full-best.pt
python experiments/mixture_of_tiles/sweep_tile_count.py
```
