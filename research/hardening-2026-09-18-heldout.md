---
type: hardening-report
date: 2026-09-18
scope: NaviTrit-Unified-Graph-DWP-Max (Gate 23) and the Gate 15-22 evidence it rests on
status: findings verified; clean protocol + rebuilt model in place; pilot ablations queued
---

# Hardening 2026-09-18: the unified looped-graph-DWP path, re-validated on held-out data

**Verdict in one line.** The architecture direction (weight-tied looped ternary block + per-loop low-rank
weight modulation + per-token depth routing) is sound and has independent support in the literature, but
**none of the internal accuracy evidence from Gates 15-23 (working results, never published) survives a held-out test**: every reported perplexity is a
memorisation score on a templated corpus, the Gemini benchmark prompts are training templates, and the
Gate 23 model had never been trained (its training script stopped at "Training Engine Ready"). The
efficiency claims are partly real (looping, packed footprint, Mamba O(S)) and partly not implemented
(the "graph routing" in the unified model executes every tile on every token and skips nothing).

Everything below is reproducible from the ledgers named in each section.

---

## 1. Evidence: the corpus is templated and the val split is the train set

`data/multicorpus_10m.pt` (Gates 15-19), `multicorpus_unified_25m.pt` and `multicorpus_agro_environmental_25m.pt`
(Gates 21-23) are built by `experiments/frontier_scaling/prepare_multicorpus.py` and
`experiments/data/prepare_agro_environmental_corpus.py`: a handful of hand-written templates, re-sampled with
`random.choice(...)` and integer "salts", then interleaved in 2048-token chunks and split by *position*
(`tokens[:0.9N]` / `tokens[0.9N:]`). Measured with `scratchpad/corpus_dup_audit.py` (16-gram rolling hash):

| corpus | tokens | unique 16-gram ratio in train | val 16-grams found verbatim in train |
|---|---:|---:|---:|
| multicorpus_10m.pt | 7.85M | 0.504 | **99.4%** (60/60 val chunks > 90%) |
| multicorpus_unified_25m.pt | 25.0M | 0.193 | **96.0%** |
| multicorpus_agro_environmental_25m.pt | 25.0M | 0.155 | **93.6%** |
| tinystories_tokens.pt (real text) | **2.0M** | 0.988 | 1.4% |

Two further facts compound this:

- The only real language in the project is **2.0M tokens** of TinyStories. `prepare_multicorpus.py` asks for
  6M (`ts_all[:0.60 * total]`) but the cache holds 2M, so the 100M models saw 2M real tokens and ~5.9M template
  tokens, then 20M tokens of that same stream on repeat.
- `data/tinystories_tokens.pt` was tokenised from `data/tinystories_valid.txt` (TinyStoriesV2 **validation**
  file): 99.8% of its 16-grams are in the training slice (`outputs/heldout-clean-eval.json:leakage`). The
  "validation" file is therefore not held-out for any checkpoint in `outputs/checkpoints/`.

## 2. Evidence: headline checkpoints re-scored on data they cannot have memorised

`experiments/frontier_scaling/eval_heldout_clean.py` scores each checkpoint on (A) the corpus tail the deep
dives used, (B) the TinyStories validation file, (C) eight hand-written Python functions absent from every
template (0.7% 16-gram overlap), (D) eight hand-written arithmetic problems (0.0% overlap). Ledger:
`outputs/heldout-clean-eval.json`.

| checkpoint | claimed PPL | A corpus tail | B TinyStories valid | C clean Python | D clean math |
|---|---:|---:|---:|---:|---:|
| navitrit-100m-step10000 (Gate 15) | 1.15 | 5.12 | 38.5 | 1214 | 12458 |
| navitrit-100m-loopformer (Gate 19) | 1.12 | 5.20 | 39.8 | 1775 | 8623 |
| navitrit-100m-looped-dwp (Gate 19-D2) | 1.12 | **1.12** | 23.0 | 730 | 4320 |

On the `data/clean` held-out sets that the new pilot arms are scored on (same script, `E_dataclean_*`):

| checkpoint | TinyStories val | CodeSearchNet test | GSM8K test |
|---|---:|---:|---:|
| (per-checkpoint rows pending: the first run mislabelled them; re-queued after the traverse arms) | | | |
| navitrit-100m-looped-dwp (last row of that run) | 26.6 | 2156 | 4164 |

For scale: the pilot dense baseline (arm A, 62M params, attention-only) is at TinyStories 83.7 / code 66 /
math 1037 after its first 2M clean tokens, i.e. already below every legacy 130M checkpoint on code and math.

Looped-DWP reproduces its claimed 1.12 exactly, on the tail; on unseen Python it is at PPL 730. A model with
any general code ability would sit below 20 on probe C. The Gate 16-19 Gemini code score of 9.0/10 is explained
by `gemini_benchmark.py:180`: the prompt `def binary_search(arr, target): low = 0 ...` is training template #1
(`prepare_multicorpus.py:66-83`) verbatim, and the math prompt's names/items (`Olivia`, `Liam`, `apples`, `gave`)
are drawn from the generator's own lists (`:23-26`). The benchmark has 5 prompts, 45 sampled tokens each,
one sample, one judge: its story/math scores are noise on n = 2.

## 3. What the code does versus what the documents say

Read against `experiments/unified_scaling/navitrit_unified_model.py` (2026-09-17 version) and
`experiments/frontier_scaling/navitrit_graph_model.py` (independent read-only review recorded in this session):

| claim | reality |
|---|---|
| "Non-monotonic graph of traversals" (Gate 23) | A fixed `for k in loops: for layer in macro_layers` pipeline. The "router" emits three scalars that blend branch outputs; all tiles execute on all tokens every loop. Nothing is skipped and no node choice exists. |
| "Anti-gravity invariant w_chan >= 0.50" | `torch.clamp(raw_chan, min=0.5)`: at init softmax gives 1/3, so the clamp is active and the channel logit gets **zero gradient**. It also hard-codes the FFN-heavy bias that Gate 15 diagnosed as the collapse mode. |
| DWP modulates the FFN ("down_{expert}" adapters) | `ContextHyperNet` only generated `q` and `o` adapters; the FFN lookup path was dead code. FiLM touched only the Mamba input. |
| Graph-of-Traversals routes per token (Gate 17) | `h_pool = h.mean(dim=1)` (`navitrit_graph_model.py:358`): one route per sequence per hop. The "discovered" bipartite Attn+FFN invariant is hard-coded index partitioning (`:141-162`). |
| DWP roles BIND/SOLVE/VERIFY/EMIT are learned | A fixed lookup table over the loop index (`looped_dwp_model.py:392-402`, same in the unified model). |
| "24 GPU syncs are fine" (not claimed, but true) | `routing_stats = {"w_seq": w.mean().item(), ...}` in every macro-layer call: 24 device syncs per forward. |
| Gate 23 "training engine primed and verified" | `train()` built the optimizer and returned. Zero steps were ever run. |
| `train_navitrit_max.py` loss | `get_batch` returns `y = tokens[i+1 : i+S+1]`, and the model shifts again (`navitrit_max_model.py:345-346`): the Gate 21 log trained the model to predict token *i+2* from position *i*. |

## 4. What changed in this pass

### 4.1 Decontaminated corpus: `experiments/data/prepare_clean_corpus.py` -> `data/clean/`
Real sources with author-provided held-out splits: TinyStoriesV2-GPT4 train (first 150 MB, range-downloaded
to `data/raw/`) vs. its valid file; CodeSearchNet Python `validation` (train) vs. `test` (val); GSM8K `train`
vs. `test` (val + `gsm8k_test_probe.json` for generation accuracy). Split is by **document**, exact duplicates
removed, any val document sharing >= 20% of its 16-grams with the train stream dropped, residual leak written to
`data/clean/manifest.json`. Mix 70/22/8, 40M tokens (see manifest for the achieved counts).

### 4.2 Held-out scorer for legacy checkpoints: `experiments/frontier_scaling/eval_heldout_clean.py`
Section 2 above. Run `python experiments/frontier_scaling/eval_heldout_clean.py`.

### 4.3 Model rewrite: `experiments/unified_scaling/navitrit_unified_model.py`
Public names unchanged (`NaviTritUnifiedConfig`, `NaviTritUnifiedForCausalLM`, `UnifiedMacroLayer`,
`ContextHyperNet`); the original 5 tests still pass, plus a sixth.

- `routing_mode = "dense" | "soft" | "mod"`. `dense` is the mandatory baseline. `mod` is Mixture-of-Depths
  capacity routing: attention + FFN run only on the top-`mod_capacity` tokens per sequence (`torch.topk` on a
  sigmoid score, gathered in original order so causality holds inside the subset), Mamba runs on all tokens.
  This is the first mode in the project whose compute saving is real. `causal_threshold=True` gives the
  generation-time per-token rule.
- Smooth invariant: `w_chan = m + (1 - m) * p_chan` replaces the clamp; same floor, gradient everywhere
  (TEST 6 asserts the router gradient is non-zero in soft and mod).
- `ContextHyperNet` now emits FiLM for mamba/attn/ffn inputs and LoRA for `q`, `o`, `down_0`, `down_1`;
  `role_sequence` is a config field and documented as an assumption.
- `num_experts=1` gives a plain SwiGLU tile; `use_dwp=False` removes the hypernet: both are ablation switches.
- Routing stats are detached tensors summed once per forward (one sync instead of 24).
- `count_parameters` no longer counts the FP hypernet as 1.58-bit.

### 4.4 Training engine: `experiments/unified_scaling/train_navitrit_unified.py`
A real loop: presets `pilot` (d=512, 4x4, ~50M) and `max` (d=1024, 4x6, ~210M, `--grad-checkpoint`),
bf16 autocast, AdamW + cosine, fixed seeded held-out batches per domain (tinystories/code/math PPL comparable
across steps and runs), best/latest checkpoints, JSON ledger, and a final loop-budget sweep T=1..max on
held-out data (the LoopFormer test-time-compute claim, measured cleanly). Labels are passed unshifted; the
model shifts once.

## 5. Protocol from here (the only way a routing or DWP claim gets into a deep dive)

1. Same preset, same `data/clean`, same token budget, same seed for every arm.
2. Baseline arm is always `--routing-mode dense --no-dwp`. Report deltas, not absolute "records".
3. Efficiency arms are compared at **matched FLOPs**: `mod` at capacity 0.5 against `dense` with the loop
   count that gives the same attention+FFN work, not against `dense` at full depth.
4. Three seeds before any number is called a result; report mean and spread.
5. Generation accuracy on `data/clean/gsm8k_test_probe.json` (exact-match of the final number) replaces the
   5-prompt LLM-judge benchmark. If an LLM judge is used, >= 50 prompts per domain, none from any template.
6. Never evaluate on `data/multicorpus_*.pt`, `data/tinystories_tokens.pt` or `data/tinystories_valid.txt`.

Ablation matrix (pilot preset, 3000 steps x 8 x 2 x 512 = 24.6M tokens each, ~1-2 h per arm on the 4060):

| arm | flags | question it answers |
|---|---|---|
| A dense | `--routing-mode dense --no-dwp` | baseline |
| B dwp | `--routing-mode dense` | does per-loop FiLM/LoRA help a tied block? |
| C soft | `--routing-mode soft` | does learned soft gating help accuracy (no compute saving)? |
| D mod | `--routing-mode mod --mod-capacity 0.5` | does skipping half the tokens cost accuracy? |
| E dense-2 | `--routing-mode dense --no-dwp --max-loops 2` | FLOP-matched control for D |
| F fp | `--routing-mode dense --no-dwp --fp-control` | the ternary tax at this scale |
| G single-expert | `--routing-mode dense --no-dwp --num-experts 1` | is the "agro expert" doing anything? |
| H mamba | `--routing-mode dense --no-dwp --mamba` | does the SSM tile earn its ~15x throughput cost? |

```powershell
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode dense --no-dwp --tag A-dense --max-steps 3000
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode dense --tag B-dwp --max-steps 3000
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode soft --tag C-soft --max-steps 3000
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode mod --tag D-mod --max-steps 3000
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode dense --no-dwp --max-loops 2 --tag E-dense2 --max-steps 3000
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode dense --no-dwp --fp-control --tag F-fp --max-steps 3000
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode dense --no-dwp --num-experts 1 --tag G-single --max-steps 3000
```

Expected honest numbers at this budget: TinyStories held-out PPL in the 4-9 range, CodeSearchNet 10-30,
GSM8K 8-20. A pilot arm reporting PPL < 2 on any held-out set is a bug, not a result.

## 6. Is the path worth pursuing generally?

Yes, with the claims cut down to what the components are known to do:

- **Looped / weight-tied blocks** (Universal Transformer 2018; ALBERT 2019; recurrent-depth scaling,
  Geiping et al. 2025; Ouro looped LMs 2025) reliably trade parameters for compute and show test-time
  loop-count scaling. Our `loop_budget_sweep` on held-out data is the direct test.
- **Per-loop low-rank relaxation of tied weights** is exactly Relaxed Recursive Transformers (Bae et al. 2024):
  LoRA per loop recovers most of the gap to untied layers. That is the DWP idea; arm B measures it here.
- **Per-token depth routing** with capacity top-k is Mixture-of-Depths (Raposo et al. 2024) and, combined with
  recursion, Mixture-of-Recursions (Bae et al. 2025). Arm D vs E measures it here.
- **Ternary weights** at 2B+ scale are established (BitNet b1.58, 2024; BitNet-b1.58-2B-4T, 2025). Arm F
  measures the tax at 50M, where it is usually larger.
- **Mamba/attention hybrids** (Jamba, Samba 2024) are the standard way to get O(S) context with recall.

What is genuinely open, and where this project can contribute if it gets clean numbers: (i) whether ternary
tied blocks keep the loop-count scaling behaviour (quantisation noise compounds through recursion; the
contraction analyses in Gates 19-C/19-D are the right tool and should be re-run on the pilot checkpoints),
(ii) whether per-loop LoRA relaxation still helps when the base is ternary and the adapters are not, and
(iii) whether MoD-style token skipping interacts badly with a recurrent Mamba path that cannot skip.
Those are publishable questions; "PPL 1.12" and "9.0/10 code" are not.

What to stop doing: templated corpora, positional splits, template-derived judge prompts, single-run
single-seed "all-time records", and calling hard-coded structure (bipartite pools, fixed role tables) a
discovery.

## 7. Smoke runs, profiling, and the real efficiency drop-off

Smoke runs (10 steps, `--smoke`) of the rebuilt trainer on `data/clean`, RTX 4060 8 GB, bf16:

| arm | tok/s | peak VRAM | note |
|---|---:|---:|---|
| pilot dense, no DWP, Mamba on, grad-ckpt | 518 | 7.2 GB | loss 10.94 -> 10.55 in 10 steps; loop sweep T=1..4 already monotone on held-out |
| pilot mod, Mamba on, grad-ckpt | 501 | 7.2 GB | router trains; `w_skip` = 0.50 as configured |
| pilot dense, Mamba on, no grad-ckpt | OOM | 22 GB requested | Mamba-1 scan state tensors [B,S,d_inner,d_state] saved for backward |
| max mod, Mamba on, grad-ckpt, 2x8x512 | 47 | 7.9 GB | not usable on this card |

Profile of one pilot step (torch.profiler, CUDA time 6.2 s): `aten::mm` 0.22 s. Everything else is
elementwise work on the Mamba scan's 4-D tensors (`logcumsumexp` + backward 1.46 s, `pow` backward 1.45 s,
mul/add/sub/where/exp/log ~3 s). Before this pass the scan was a Python loop over S tokens
(`ternary_mamba_block.py:_selective_scan_sequential`), which is why Gate 21 logged ~1k tok/s; the new parallel
log-space scan is 17x faster on the block benchmark and numerically identical (1.5e-8 forward, 7e-15 grad,
`test_ternary_mamba.py::TestParallelScan`), but Mamba-1's per-(channel, state) decay is inherently
memory-bound without a fused kernel, and none is available for Windows/CUDA 12.1 here.

Decision: the pilot preset runs **attention-only sequence mixing** (`use_mamba=False`, `--mamba` to force on).
Measured after the switch (pilot, 8x512):

| arm | tok/s | peak VRAM |
|---|---:|---:|
| dense, no grad-ckpt | 4,175 | 8.4 GB (paging) |
| dense, grad-ckpt | 8,200 | 3.3 GB |
| mod (capacity 0.5), no grad-ckpt | 10,030 | 6.9 GB |

So a 3000-step arm (24.6M tokens) takes ~50 min and the seven-arm matrix fits in an evening. The Mamba tile
becomes its own arm (H: `--mamba`, ~15x slower) rather than a tax on every question. The efficiency story
therefore splits cleanly: looping and packed ternary weights are real memory wins; `mod` routing is a real
compute win (2.4x tokens/s over dense at the same depth, before accounting for accuracy); the Mamba path is
a throughput loss on this hardware until it has a kernel.

Full pilot results (arms A-G, 3000 steps each) are written to `outputs/navitrit-unified-<tag>-log.json` by the
matrix launched at the end of this session; each log carries the fixed-batch held-out PPL trajectory and the
final loop-budget sweep.


## 8. The thesis itself, implemented: NaviTrit-Traverse (`experiments/unified_scaling/navitrit_traverse.py`)

The claim this project exists to test is not "looping" or "MoD"; it is that a token's own traversal history,
carried forward continuation-style, can (a) route the token to earlier exits and non-monotonic revisits and
(b) reinterpret the shared ternary weights, so that one small stationary block yields more function per byte
and fewer tile evaluations per token at once. None of the previous models implemented it: routing was per
sequence, modulation depended only on the loop counter, and skip did not skip. `NaviTritTraverseForCausalLM`
does, and exposes the two halves as independent switches so each can be falsified.

Mechanism (per token, per hop, H hops max):
1. path state `p` (GRU cell, 128-d) is updated with an embedding of the tile just executed plus the hop index;
   START is a learned embedding.
2. `tile_router(norm(h) ⊕ p)` -> logits over {tile_0..tile_3, EXIT}; hard argmax (Gumbel noise while
   training), EXIT masked for the first `min_hops`. Exited tokens are frozen.
3. tokens choosing tile m are gathered per sequence in original order and padded; attention runs on the
   subset with a causal + padding mask (`FlashAttentionTile(attn_mask=...)`), then the FFN; the residual delta
   is scattered back scaled by the router probability (gradient path to the router).
4. interpretation: `alpha = softmax(adapter_router(p))` mixes a bank of K=4 rank-16 adapters (q, o, down_i)
   and FiLM vectors per tile (`PathAdapterBank`); `_apply_lora` takes the bank triple and applies the per-token
   mixture in one matmul pair. Zero-initialised, so the model starts as a plain tied block.
5. losses: CE + Switch balance over tiles + `hop_cost` * mean executed hops / H.

Switches: `router_cond = path | state` (history vs residual state only), `adapter_cond = path | hop`
(history vs hop index only). Reported per eval: mean hops per token (the compute unit) and tile usage.

Arms (pilot preset, 3000 steps, from scratch, queued after A-G):

| arm | flags | question |
|---|---|---|
| H traverse | `--routing-mode traverse --router-cond path --adapter-cond path` | the full thesis |
| I control | `--router-cond state --adapter-cond hop` | same machinery, no history anywhere |
| J | `--router-cond path --adapter-cond hop` | does history help routing alone? |
| K | `--router-cond state --adapter-cond path` | does history help interpretation alone? |

Reading the result: compare H, I, J, K on held-out PPL **at their measured mean hops** and against A (dense,
16 tile evaluations per token) and D (mod). A win for H over I at equal or fewer hops is the non-monotonic
advantage. J minus I and K minus I attribute it. If H does not beat I, the honest statement is that path
history is not informative beyond the hop counter at this scale, and the efficiency gains belong to plain
depth routing.

Throughput was measured only under GPU contention with the running matrix (dense 351 vs traverse 292 tok/s,
step-for-step comparable) and will be re-measured sequentially; per the user's instruction, all GPU work now
runs one job at a time.


## 9. Results so far and how to resume (machine restart 2026-09-18)

Completed arms (pilot, 3000 steps = 24.6M clean tokens, seed 0, ~14.5k tok/s uncontended):

| arm | TinyStories PPL | Code PPL | Math PPL | mean loss | note |
|---|---:|---:|---:|---:|---|
| A dense 4x4, no DWP | **9.93** | **22.03** | **84.9** | 3.276 | baseline; ledger `outputs/navitrit-unified-A-dense-log.json` |
| D mod, capacity 0.5 | 11.40 | 23.83 | 107.5 | 3.428 | half the attention+FFN token-work of A; ledger `...-D-mod-log.json` |
| E dense, 2 loops (FLOP-matched to D) | **9.58** | **21.34** | **80.6** | **3.237** | half the compute of A and better on every domain; ledger `...-E-dense2-log.json` |

Reading: at 24.6M tokens, **looping less beats both looping more and token skipping**. E (2 loops) is 0.04
nats better than A (4 loops) at half the compute and 0.19 nats better than D (mod) at equal compute. The extra
loops are undertrained at this budget, so "more virtual depth" is not a free win and per-token skipping does
not recover what fewer loops give for free. Consequences: (i) E, not A, is the baseline every traverse /
strain / activation arm is read against at matched mean hops; (ii) arm E1 (1 loop) is queued to find where
the depth curve turns; (iii) the loop-budget sweeps in each ledger show whether a model trained at T loops
degrades gracefully at fewer, which is the test-time-compute claim measured cleanly. Legacy checkpoints on the same sets: TinyStories 26.6 / code 2156 / math 4164, so one clean
25M-token pilot arm already beats every previous 100M+ model on every domain.

Resume (one GPU job at a time; each arm writes its own ledger and best/latest checkpoint):
```powershell
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --grad-checkpoint --max-steps 3000 --routing-mode dense --no-dwp --max-loops 2 --tag E-dense2
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --grad-checkpoint --max-steps 3000 --routing-mode dense --tag B-dwp
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --grad-checkpoint --max-steps 3000 --routing-mode soft --tag C-soft
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --grad-checkpoint --max-steps 3000 --routing-mode dense --no-dwp --fp-control --tag F-fp
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --grad-checkpoint --max-steps 3000 --routing-mode dense --no-dwp --num-experts 1 --tag G-single
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode traverse --max-hops 8 --max-steps 3000 --router-cond path  --adapter-cond path --tag H-traverse-path
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode traverse --max-hops 8 --max-steps 3000 --router-cond state --adapter-cond hop  --tag I-traverse-control
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode traverse --max-hops 8 --max-steps 3000 --router-cond path  --adapter-cond hop  --tag J-traverse-pathroute
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode traverse --max-hops 8 --max-steps 3000 --router-cond state --adapter-cond path --tag K-traverse-pathadapt
python experiments/frontier_scaling/eval_heldout_clean.py
```
Tests before launching: `python experiments/unified_scaling/test_navitrit_unified.py`,
`python experiments/unified_scaling/test_navitrit_traverse.py`, `python -m unittest experiments.mamba.test_ternary_mamba`.

## 10. Strain track: a latent for "how unresolved is this token" (arms L, M, N)

Idea (user, 2026-09-18): give each token a stress/strain latent so the router can free history it no longer
needs and, later, decide which past passes to keep reachable for long-context reasoning. Long context is
parked for a separate bench; this track only asks whether strain is a useful signal for routing and exit.

Definition. Strain at a hop is the relative residual displacement the tile produced,
`||h_{t+1} - h_t|| / ||h_t||`, stored as `log1p`. A converged token has strain near zero (the fixed-point idea
of the FlowTrit track, per token). Implementation in `navitrit_traverse.py`, `TraverseConfig.strain_mode`:

- `observe`: three features (this hop's measured strain, router entropy, hop/H) enter the path GRU through
  `strain_in`, and `strain_head(cond)` predicts the coming hop's strain before it runs, supervised with an
  MSE auxiliary loss on tokens that moved. The router still decides EXIT on its own.
- `gate`: as observe, plus the predicted strain shifts the EXIT logit by `-strain_exit_scale * pred_strain`
  (learned scalar, zero-init). Low predicted strain makes exit cheaper; high strain keeps the token moving.
- `off`: arms H to K.

Arms (pilot, 3000 steps, queued after K):

| arm | flags | question |
|---|---|---|
| L | `--strain observe` (path/path) | does a strain-aware path state route better than H? |
| M | `--strain gate` (path/path) | does strain-driven exit beat the hop-cost exit at equal or fewer hops? |
| N | `--strain gate` (state/hop) | does strain help even without history? (isolates strain from path) |

Read against H (same machinery, no strain) at their measured mean hops. Per-eval logs carry the auxiliary
loss and mean predicted strain; a strain head that stays at chance (aux loss not falling) means strain is
not predictable from the token's state and the track stops there. Tests: `test_navitrit_traverse.py::test_strain_modes`.

Deferred to the long-context bench: using strain as a learned KV-cache eviction score (two-tier cache:
full keys for high-strain tokens, compressed summary for the rest), evaluated on needle-in-haystack and
multi-hop retrieval past training length against full cache and H2O-style eviction at equal cache size.

## 11. Parameter accounting for a looped, routed, ternary model

Physical parameters, pilot preset (`count_parameters`):

| block | params | note |
|---|---:|---|
| token + position embeddings | 26.3M | fp16, tied to the LM head; **46% of the model** at this width |
| 4 macro-layer tiles (attn + dual SwiGLU) | 29.4M | ternary: 5.53 MB packed, the only weights that stay on-chip |
| traverse adapter banks (4 tiles x 4 adapters x rank 16) | 1.08M | fp16 |
| controller (path GRU, routers, strain head) | 0.20M | fp16 |
| total | 56.9M | |

Max preset: embeddings 52.5M, tiles 117.5M (22.1 MB packed), adapters 4.3M, controller 0.3M, total 174.5M.

Three different numbers get called "parameter count" here, and the docs previously mixed them:

1. **Physical / stored** parameters: the table above. This sets the footprint. Ternary applies only to the
   tiles; embeddings and adapters are full precision, so at pilot width the packed model is dominated by
   the fp16 embedding matrix (51 MB) rather than the 5.5 MB of tiles. Shrinking the vocabulary or
   quantising embeddings matters more for footprint than anything the router does.
2. **Virtual / effective depth** parameters: what a token actually passes through. Dense 4x4 applies the
   29.4M tile set four times, comparable in compute to a 117M untied model. Traverse at 8 hops applies one
   7.3M tile per hop, so a token that uses all 8 hops sees ~59M of tile-work and one that exits after 2 sees
   ~15M. This is the number the routing trades: mean hops times tile size is the per-token compute, and the
   loop-budget sweep in every ledger is the accuracy-versus-virtual-parameters curve.
3. **Function-class size** ("expressivity"): how many distinct functions the stored weights can realise
   across paths and adapter mixtures. It is bounded above by stored bits and is only observable through
   held-out loss at matched virtual depth. That is exactly what arms H vs I, J, K measure. A claim that
   routing "represents more parameters" is a claim about (3) and must be evidenced by (2)-matched
   comparisons; it can never lower (1).

### 11.1 Ternary embeddings, adapters, controller (arm O)

- **Embeddings**: worth one experiment, not a default. BitNet b1.58 keeps embeddings and the head in higher
  precision because a tied ternary row has 3 values per dimension and rare tokens lose separability. The
  footprint argument is also weaker than it looks: the embedding is a lookup, not a bandwidth-bound matmul,
  and the head runs once per token rather than once per hop. `--ternary-embed` (config `ternary_embed`,
  helper `embedding_weight`) ternarises the tied matrix with the tile quantiser; arm O = arm A + that flag,
  queued after the strain arms. Decision rule: gap to A under 0.02 nats mean loss -> free at this scale;
  otherwise int8 embeddings are the answer and the question closes.
- **Adapter banks and controller**: not ternarised, deliberately. They are ~2% of the model; the adapter B
  matrices start at exactly zero (ternarising breaks the zero-drift start and rank-16 rows are too short for
  absmean scaling); and the router / strain head produce the decisions everything else depends on, where
  quantisation noise is far more costly than in a weight. Every routed model in the literature keeps its
  router in full precision.

## 12. Activation strategy selector (arms P, Q)

Idea (user, 2026-09-18): the router should also forward an activation function to the tile it sends the
token to, chosen by a trained selector from the edge being taken (backward, forward by how much, how many
tiles are skipped). Rationale: the same ternary tile with a different gate nonlinearity is a different
function, so this is another axis of path-conditioned reinterpretation on top of the adapter bank.

Implementation (`navitrit_traverse.py`, `TraverseConfig.act_strategy`):
- `ACT_BANK` = {silu, gelu, relu, identity, tanh, sign_sqrt}; the last is expansive (amplifies small gate
  values) so the bank spans contractive, linear and expansive choices.
- Edge features: signed tile offset `v - u` embedded over 2M-1 offsets (a 2M-th index marks the first hop
  from START) plus the hop embedding; concatenated with the path state `p` and fed to `strategy_router`.
- The resulting per-token softmax mixes the bank on the FFN gate: `gate = sum_a w_a act_a(W_gate x)`
  (`MixedGateActivation`, passed as `gate_act` into `DualExpertSwiGLUTile`). Cost: A extra elementwise passes
  over d_ff, no extra matmuls.
- Initialised with the SiLU logit at +4 (97% SiLU), so arm P starts as arm H (test: logit gap 0.013).
- Per-eval logs report the mean mixture (`act_usage`); a selector that stays at 97% SiLU has found nothing.

Arms (pilot, 3000 steps, queued after O): P = H + `--act-strategy`; Q = P + `--strain gate`, i.e. the full
stack (history routing, history interpretation, strain-gated exit, edge-conditioned activation). Read P
against H and Q against M at matched mean hops. A win for P that comes with the mixture actually moving
off SiLU on backward or long-skip edges is the interesting outcome; a win with the mixture unchanged is
noise and should be re-run on a second seed before it is believed.
