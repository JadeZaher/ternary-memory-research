# `experiments/general_model` — module notes

Track objective: **the smallest stored-byte footprint that reaches a held-out loss / BPB target.** Not the
smallest parameter count, not the fewest FLOPs — stored megabytes, because that is the number a ternary model
is supposed to win on and the number every previous ledger in this repo under-reported (section 11 of
`research/hardening-2026-09-18-heldout.md`: at pilot width the fp16 embedding matrix was 51 MB against 5.5 MB
of ternary tiles, i.e. the routing was optimising the small half of the footprint).

Files: `general_model.py` (model), `train_general.py` (training + read-out), `test_general_model.py` (tests).
Nothing here reimplements a mechanism that already exists in the repo; `BitLinear`, `BitRouteRMSNorm`,
`TernaryMambaBlock`, `FlashAttentionTile`, `DualExpertSwiGLUTile`, `_apply_lora`, `_film`, `embedding_weight`,
`PathAdapterBank` and `_gather_subset` are all imported.

---

## The tile abstraction

A **tile** is one self-contained residual block: `RMSNorm -> sequence mixer -> residual -> RMSNorm ->
dual-expert SwiGLU -> residual`. The only thing that varies between tiles is the mixer:

| `tile_kinds` entry | mixer | cost in sequence length | why it is here |
|---|---|---|---|
| `"mamba"` | `TernaryMambaBlock` | O(S), O(1) state at generation | carries long-range state cheaply and has no KV cache |
| `"attn"` | `FlashAttentionTile` | O(S²) | exact content-addressed lookup, which an SSM approximates poorly |

`tile_kinds` is a list, so the mix is a config decision, not an architecture decision. The default
`["mamba","mamba","mamba","attn"]` is the Samba/Jamba ratio: one attention tile is enough for retrieval when
the other three tiles keep a recurrent state, and it is the only tile whose cost grows with sequence length.
`tile_kinds=["attn"]*4` is the reference arm every efficiency claim must be read against.

Tiles are **not** weight-tied to each other; the same tile can be re-entered at several hops (that is the
looping), and tile *m*'s weights are reinterpreted per hop by its own adapter bank.

All tiles return `(delta, balance_loss, extras)` — the residual **delta**, never the updated stream — because
the caller scatters the delta back into full-length positions and must be able to zero it on padded rows.
`extras["received"]` (attention-received mass per key) is only produced by attention tiles and only when
`model.collect_received` is set: it forces the materialised-softmax path in `FlashAttentionTile`, which costs
an S×S tensor per head, so it is off during training.

### `tile_soft_mix`

Inside a tile, a per-token 2-way softmax weights the mixer branch against the FFN branch, with a **smooth**
floor on the FFN weight (`w_ffn = m + (1-m)·p_ffn`, `m = min_chan_weight = 0.5`). Motivation: arm C in
section 9 — soft gating on top of per-loop modulation was worth **0.07 nats** and was the best fixed arm at
16 tile applications. The floor is the smooth map, not a `clamp`: a hard clamp zeroes the router gradient
whenever the raw probability is under the floor, which at initialisation is always (the bug fixed in the
2026-09-18 rewrite of `navitrit_unified_model.py`). `--no-soft-mix` is the ablation.

---

## Why running Mamba on a gathered subset is causal (and why the traverse model refused to)

`navitrit_traverse.py` asserts `not config.use_mamba` with the comment "Mamba cannot run on gathered subsets".
That is over-cautious, and the caution costs the whole O(S) half of the design space. The argument that it is
in fact safe, and the test that enforces it:

1. `_gather_subset(sel)` returns positions **sorted ascending** (`torch.where(sel, pos, S).sort()`), so the
   gathered tensor `h_sub[b, i]` holds original position `order[b, i]` with `order[b, i] < order[b, i+1]`
   for every valid row. The subset is a *subsequence* of the original sequence, in the original order.
2. Rows beyond `sel.sum(dim=1)` are padding, clamped to index `S-1`, and they are all at the **tail**
   (`sort` puts the sentinel `S` last).
3. `TernaryMambaBlock` is causal in its input index: the depthwise `conv1d` is left-padded and cropped, and
   the selective scan (`logcumsumexp` over `dim=1`) accumulates strictly forward. Output row `i` therefore
   depends only on input rows `<= i`.
4. (1)+(2)+(3): a valid output row depends only on valid rows at earlier *original* positions. Padding, being
   at the tail, can never reach a valid row. The delta is multiplied by `valid` before the scatter, so the
   padded rows (which alias position `S-1`) contribute nothing.

This is a recurrence over *the selected tokens only* — the skipped tokens are genuinely skipped, not zeroed —
which is the correct semantics for depth routing and is exactly what attention with a causal+padding mask
does in the traverse model.

Enforced by `test_causality` (a future token cannot move a past logit, in both routing modes with exit live)
and by `test_subset_equivalence_and_padding` (the all-selected dispatch is bit-identical to a full-sequence
tile call; and randomising the padded tail leaves the valid rows unchanged).

---

## The two routing modes

`routing="fixed"` — hop *h* applies tile `h % M` to every token. No tile router exists, no balance loss is
accumulated, the gate is a constant 1. This is the honest baseline: section 13.1's verdict was that *"the
honest architecture remains the modulated tied block with fixed order (arm C), now with one addition that this
run does justify: a per-token value-based exit."* `fixed` + `exit_mode="value"` **is** that architecture.

`routing="free"` — at every hop each token picks any tile via `tile_router(cond)`, hard argmax with Gumbel
noise while training. Non-monotonic: a token may revisit a tile or take them out of order. This is the thesis
that section 13.1 declined to support (path history moved held-out loss by ≤ 0.03 nats, and per-token routing
at matched compute trailed fixed order by 0.16 nats), kept here because the hybrid tile set changes the
question: with heterogeneous tiles (SSM vs attention) there is a real reason for a token to prefer one mixer,
which there was not when all four tiles were attention+FFN.

Dispatch is identical in both modes: gather the selected tokens per sequence, run the tile once on the
subset, scatter the delta back. In `free` mode the delta is scaled by the router probability through a
**straight-through gate** `p / p.detach()` (`gate_ste`), so the tile runs at full magnitude while the router
still receives gradient through `p`. Without it (section 9.1) routed tiles ran at ~1/M of the dense model's
magnitude and the router was rewarded for concentrating probability rather than for choosing well — arms H–K
collapsed to one hop per token.

A Switch-style balance loss (`balance_loss_weight`) over active tokens keeps every tile in use in `free` mode.

---

## Adaptive depth: the value-based exit

`exit_mode="value"` is the one mechanism section 13.1 endorsed. A value head reads the same conditioning
vector the router reads and predicts, in nats, the gain of the **coming** hop. A token continues iff
`predicted_gain > exit_lambda` (the compute price). The head is trained by Huber loss against the **measured**
gain — the drop in per-token CE through the tied head between consecutive hops, taken on the deep-supervision
sample — so it is supervised by the thing it is supposed to predict, not by a proxy.

Three failure modes are designed around, each from a specific collapse in the record:

* **Early-exit collapse** (sections 9.1, 13.1 §3): once every token exits at the floor, later hops receive no
  training data and can never become worth taking. Guards: `min_hops` (exit masked below it),
  `exit_warmup_steps` (exit masked entirely for the first N steps; the trainer flips `model.allow_exit`), and
  `explore_prob` (training-only epsilon: a token the value rule would exit continues anyway with p=0.3 and its
  hop still trains the value head). Arms H–K died without the first two; stage 2 died without the third.
* **No readable intermediate state**: without deep supervision the hidden state at hop *k* is not a state the
  tied head can score, so "measured gain" is noise. `deep_sup_weight=0.1` on `deep_sup_frac=1/8` of positions
  is what makes the gain target meaningful; it is the same readout the value target is computed from, so the
  two costs are shared.
* **Unfalsifiable efficiency claims**: `exit_mode="none"` runs every token for `max_hops`, and the trainer's
  read-out re-evaluates the *same trained model* under a random-continuation null (per-hop continue
  probability bisected to the trained policy's mean hops) and a per-sequence capacity null at the same hops.
  A win that does not clear both nulls at matched mean hops is not a win.

`exit_policy` (`threshold` | `random` | `capacity`) is an **eval-time** switch on a trained model; training
always uses `threshold`. `capacity` is deliberately non-causal (top-k over the sequence) and exists only as a
null.

---

## Conditioning: `router_cond` and `adapter_cond`

`_cond(h, path, hop) = [state_proj(RMSNorm(h)) ‖ context]` feeds both the tile router and the value head.

* `router_cond="state"` (default): `context` is the hop embedding — residual state plus hop index, no history.
* `router_cond="path"`: `context` is a GRU path state updated with (tile just executed + hop embedding).

Default is `"state"` because section 13.1 §2 found path history worth ≤ 0.03 nats and gave the mechanistic
reason: every tile that runs writes into the residual stream, so `h` already carries a compressed path record
and an explicit recurrence re-encodes it. The GRU and tile embedding are **not constructed at all** unless
some setting needs them (`use_path`), because unused controller parameters are stored bytes and stored bytes
are the objective.

`adapter_cond="hop"` (default) mixes each tile's `PathAdapterBank` by hop index — a per-hop FiLM+LoRA
modulation of shared weights. That is the per-loop DWP that arm B bought **0.11 nats** with (section 9), the
largest single accuracy win in the record, and it is the analogue of a looped model's per-loop weights.
`adapter_cond="path"` mixes by path state instead: the ablation for whether *history* rather than *depth*
should pick the interpretation (arms J/K: no).

The Mamba block takes no adapter dict, so its tile uses the bank's `q` slot as a **pre**-adapter on the mixer
input and the `o` slot as a **post**-adapter on the mixer output. Both are zero at initialisation (LoRA `B` is
zero), so a fresh model is exactly the unmodulated block, and no slot of the bank is dead weight.

---

## Stored-byte accounting (`count_parameters`)

Six groups, each with its own bit width:

| group | bits | contents |
|---|---:|---|
| `tiles_ternary` | 1.58 | every `BitLinear` weight in the tiles: attention q/k/v/o, SwiGLU gate/up/down, Mamba in_proj/out_proj |
| `tiles_dense` | 16 | what ternarising would break: Mamba `conv1d`/`x_proj`/`dt_proj`/`A_log`/`D`, RMSNorm scales, expert and branch routers |
| `token_embed` | 16, or 1.58 with `ternary_embed` | the tied embedding / LM head |
| `position_embed` | 16 | learned positions |
| `adapters` | 16 | the per-tile `PathAdapterBank`s |
| `controller` | 16 | routers, value head, path state, `final_norm` |

`stored_mb_total` is the headline. Three points that the earlier ledgers got wrong and this one does not:

1. **Ternary applies to `BitLinear` weights only.** Lumping all tile parameters at 1.58 bit (as
   `NaviTritUnifiedForCausalLM.count_parameters` does) overstates the win, and it overstates it *more* the
   more Mamba you use, because `x_proj`/`dt_proj`/`A_log`/`conv1d` are full precision by construction.
2. **The token embedding is its own group** so `ternary_embed` scales exactly one number by 1.58/16.
   Section 11.1: this is an experiment, not a default — BitNet keeps embeddings in higher precision because a
   ternary row has three values per dimension and rare tokens lose separability. It is also the single largest
   footprint lever at these widths, which is why `vocab_size` defaults to 16384 rather than GPT-2's 50257.
3. **Adapters and the controller stay full precision, deliberately.** The adapter `B` matrices start at
   exactly zero (ternarising breaks the zero-drift start, and rank-16 rows are too short for absmean scaling),
   and the router and value head produce the decisions everything else depends on.

`virtual_layers_count` (= `max_hops`) is a *different* number: what a token passes through, not what is
stored. `mean_hops × tile_size` is the per-token compute, and that is what the hop-cap sweep trades against
accuracy. A routing claim can never lower `stored_mb_total`.

`load_dense_checkpoint` raises `NotImplementedError` on purpose: no checkpoint in this repo shares this tile
layout (hybrid mixers, soft branch mix), and silently loading a partial state dict is how a warm start becomes
an unreproducible result.

---

## Gradient checkpointing

`grad_checkpoint=True` wraps each tile call in `torch.utils.checkpoint(..., use_reentrant=False)`. Stage 2 of
the condensed experiment spilled to 13.4 GB without it and ran at 7.0 GB with it (section 13.1 §2).

**Every grad-carrying tensor the checkpointed callable uses must be an explicit positional argument**, never
captured by closure. A closure-captured non-leaf tensor was measured in this repo (2026-09-19, arm C) to
produce ~300× inflated gradients on recompute while the forward loss stayed identical — gnorm 6.8 plain
against 2212 checkpointed. `_run_tile(tile_index, h_sub, valid, alpha, need_received)` therefore passes
everything explicitly and computes the FiLM/LoRA modulations *inside* the checkpointed function from `alpha`
and the bank's leaf `nn.Parameter`s. `test_grad_checkpoint_equivalence` takes an optimizer step first so LoRA
`B` and the FiLM vectors are non-zero before it compares gradients; with a fresh zero-initialised bank the
adapters contribute no gradient and the test would pass regardless.

---

## What each toggle ablates

| toggle | ablates | motivating section |
|---|---|---|
| `tile_kinds` | SSM vs attention mix; `["attn"]*4` is the reference | §7 (use_mamba), §13.1 §5 (latency) |
| `routing` `free`\|`fixed` | per-token non-monotonic traversal vs fixed order | §8, §9.2, §13.1 |
| `exit_mode` `value`\|`none` | adaptive depth vs a fixed hop budget | §13, §13.1 §1 |
| `exit_lambda` | the compute price in nats/hop; swept at eval | §13 read-out 1 |
| `explore_prob` | starvation of hops past the exit point | §13.1 §3 |
| `min_hops`, `exit_warmup_steps` | early-exit collapse | §9.1 |
| `gate_ste` | router-probability magnitude collapse | §9.1 |
| `router_cond` | path history in the routing decision (≤ 0.03 nats) | §9.2, §13.1 §2 |
| `adapter_cond` | per-hop modulation (0.11 nats) vs per-path | §9 arm B, §9.2 arms J/K |
| `tile_soft_mix` | soft branch gating (0.07 nats) | §9 arm C |
| `ternary_embed` | the largest footprint lever, accuracy cost unknown | §11.1 arm O |
| `num_experts` | dual expert (0.04 nats) | §9 arm G |
| `deep_sup_weight` | readable intermediate states / the gain target | §13 stage 1 |
| `hop_cost_weight` | an explicit compute penalty (0 by default: `exit_lambda` already prices compute) | §8 |
| `balance_loss_weight` | tile collapse in `free` routing | §8 |

### Config fields that are currently inert

* `tie_word_embeddings` — the LM head is **always** the transposed token embedding (`embedding_weight`).
  The field is carried so the geometry view matches `NaviTritUnifiedConfig`; untying would need a separate
  head matrix, which at vocab 16384 × 512 would add 16 MB of stored bytes and is therefore against the track
  objective. Setting it to `False` changes nothing today.
* `hop_feature_dropout` — applies to the hop embedding *in the path-state update*, so it only does anything
  when `use_path` is on (`router_cond="path"` or `adapter_cond="path"`). With the defaults there is no path
  state and the flag is inert.

---

## Trainer notes (`train_general.py`)

* **Data.** Held-out sets are discovered by globbing `val_*_tokens.pt`, so a new domain needs no code change.
  `manifest.json` supplies `vocab_size`, `eos_id` and `val_bytes_per_token`; without it the trainer falls back
  to GPT-2's 50257 and reports no BPB (which is what `data/clean` does, so `--data-dir data/clean` is a valid
  smoke target). Metric names match the patched unified trainer: `<set>_loss`, `<set>_ppl`, `<set>_bpb`,
  `mean_loss`, `mean_bpb_core` over {tinystories, code, math}.
* **BPB, not perplexity, is the comparable metric** across tokenizers: `loss / ln 2 / bytes_per_token`. A 16k
  vocabulary and a 50k vocabulary produce incomparable perplexities on the same text; they produce comparable
  bits per byte. `--target-bpb` prints PASS/FAIL at every eval and records the verdict in the ledger.
* **Fixed eval batches**, seeded once, so a curve is a curve and not a resampling artifact.
* **A batch carries `seq_len + 1` tokens** (input and shifted labels share one tensor), so
  `max_position_embeddings = max(1025, seq_len + 1)`. `max(1024, seq_len)` — what
  `train_navitrit_unified.py` uses — is off by one and only survives because that trainer has never been
  run at `--seq-len 1024`; at 1024 it indexes row 1024 of a 1024-row table and dies as an opaque CUDA
  device-side assert. `GeneralRoutedLM.forward` now asserts the bound in Python instead.
* **`--time-budget-h`** trains 60 steps, measures tokens/s, and solves for the `max_steps` that fits the
  budget minus `--readout-reserve-min` (25 by default). Because the cosine schedule takes `max_steps` as its
  horizon, this must happen inside the LR warmup window (warmup is 200 steps), where the schedule does not yet
  depend on it. The decision is printed as `[budget] ...`.
* **`--resume`** restores model, optimizer, sampler generator **and the global torch RNG state**. The last of
  those is an addition over `train_navitrit_unified.py`: without it a resumed run takes different Gumbel noise
  and different deep-supervision samples, so "resume" is not reproducible and cannot be unit-tested.
  `test_resume_round_trip` asserts that the step after a save/restore reproduces the uninterrupted run exactly.
  Checkpoints are loaded with `weights_only=True`; nothing but tensors, dicts and primitives is stored.
* **Read-out** (eval only, one model, after training): lambda sweep → random-continuation null bisected to the
  trained policy's mean hops → capacity null at the same hops → hop-cap sweep 1..`max_hops`. Ledger at
  `outputs/general-<tag>-log.json`.
* Log lines are bracket-tagged for grep: `[footprint]`, `[budget]`, `[eval N]`, `[traverse N]`, `[value N]`,
  `[target N]`, `[lambda x]`, `[controls]`, `[hopcap T]`, and a final `DONE <tag>`.

## Measured throughput and memory (RTX 4060 8 GB, 2026-09-19)

Preset p512, vocab 16384, seq 1024, `--grad-checkpoint`, bf16 autocast, AdamW, tokens/step fixed at 16384,
`allow_exit=False` so every token runs all 8 hops (the worst case, and what the first `exit_warmup` steps do).
Each row is a fresh process; the script is `calibrate_general.py` in this session's scratchpad.

| routing | tiles | batch | accum | allocated peak | reserved | nvidia-smi | tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| free | mamba×3 + attn | 16 | 1 | 9.79 GB | 20.54 GB | — | 183 |
| free | mamba×3 + attn | 8 | 2 | 5.20 GB | 13.09 GB | 7903 MiB | 859 |
| **free** | **mamba×3 + attn** | **4** | **4** | **3.09 GB** | **5.92 GB** | **6644 MiB** | **1551** |
| free | mamba×3 + attn | 2 | 8 | 2.01 GB | 3.23 GB | 3898 MiB | 1077 |
| fixed | mamba×3 + attn | 4 | 4 | 6.66 GB | 6.99 GB | 7751 MiB | 1822 |
| **fixed** | **mamba×3 + attn** | **2** | **8** | **3.74 GB** | **4.03 GB** | **4743 MiB** | **1696** |
| free | attn × 4 | 8 | 2 | 3.65 GB | 4.27 GB | 4972 MiB | 5560 |

Four things that are easy to get wrong and are worth keeping:

1. **`torch.cuda.max_memory_allocated` is not the memory budget on Windows.** WDDM lets an over-budget
   allocation fall back to shared system memory instead of raising OOM, so batch 16 "succeeded" at an
   allocated peak of 9.79 GB with 20.5 GB reserved and ran at 183 tok/s — an 8× slowdown with no error.
   Size batches against `max_memory_reserved` and the nvidia-smi process figure.
2. **`free` routing uses far less memory than `fixed`, and `fixed` is slightly faster.** In `fixed` routing
   one tile receives all S tokens, and the Mamba scan materialises `[B, k, d_inner, d_state]` float32
   tensors, so its memory is linear in *k*. In `free` routing the same tokens are split across four tiles
   (k ≈ S/M each) and, with per-tile checkpointing, the peak is one tile call rather than four. The
   throughput goes the other way (fewer, larger kernels win), 1822 vs 1551 tok/s at batch 4.
3. **The Mamba scan is the bottleneck, not the routing.** The attention-only reference runs at 5560 tok/s
   against 1551 for the hybrid — 3.6×. The log-space parallel scan is what the 2026-09-18 hardening pass
   introduced to replace a per-token Python loop, and it is still bandwidth-bound in `d_state`.
4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is roughly neutral here** (5.92 → 5.60 GB reserved,
   1551 → 1512 tok/s at batch 4). The variable-*k* subsets fragment the caching allocator, but not enough
   for expandable segments to pay for themselves.

Recommended starting point for a real run: `--batch-size 4 --grad-accum 4 --seq-len 1024 --grad-checkpoint`
for `--routing free`, and `--batch-size 2 --grad-accum 8` for `--routing fixed`.

## Running

```bash
python experiments/general_model/test_general_model.py            # CPU, ~1 min
python experiments/general_model/test_general_model.py --cuda     # same checks on the GPU
python experiments/general_model/train_general.py --data-dir data/clean --smoke --batch-size 2 --seq-len 256
python experiments/general_model/train_general.py --preset p512 --tag g1 --time-budget-h 6 --grad-checkpoint
```

One GPU job at a time. `--grad-checkpoint` is required at `p512`/seq 1024 on an 8 GB card.
