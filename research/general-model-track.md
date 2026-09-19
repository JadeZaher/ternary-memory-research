---
type: track
status: active
started: 2026-09-19
---

# General model track: smallest stored footprint at a fixed held-out quality

**Decided 2026-09-19 after two grilling rounds.** This track works backwards from the results of
`hardening-2026-09-18-heldout.md` (sections 9, 13, 13.1) to the best general-purpose model this GPU
(RTX 4060, 8 GB) can train, under three fixed requirements: ternary weights, Mamba sequence mixing, and
per-token non-monotonic routing. All of its code lives in `experiments/general_model/`.

## 1. Decisions (locked)

| decision | choice | why / cost |
|---|---|---|
| objective | **smallest stored bytes that reach a loss target** | Footprint is the thesis; loss at fixed compute is reported alongside but does not decide |
| loss target | **arm C's bits-per-byte on the three existing held-out sets** (TinyStories valid, CodeSearchNet test, GSM8K test) | Continuity with every ledger so far; bits/byte is tokenizer-independent |
| run budget | **~8 h per overnight run**, one GPU job at a time | ~100M tokens at pilot width on the routed model |
| tokenizer | **16k byte-level BPE trained on the new corpus** | Embeddings were 79% of stored bytes under GPT-2's 50k vocabulary; breaks direct nats/token comparison, hence bits/byte |
| corpus | **~300M tokens: FineWeb-Edu (sample-10BT) 50%, TinyStoriesV2 25%, CodeSearchNet Python 17%, GSM8K 8%**, document-level splits, 16-gram leak filter, fourth held-out set from a disjoint FineWeb slice | Chosen over Pile (licensing/noise) and over three-domain-only (not general) |
| Mamba | **hybrid tiles, 3 Mamba + 1 attention** per block of four | Jamba/Zamba/Samba evidence; footprint is FFN-dominated so mixer choice costs little; attention keeps in-context copying for code and math |
| routing | **free (non-monotonic) tile choice per token per hop + value-based exit**, with a `fixed` toggle | Required by the track; the value exit is the one routing result section 13.1 supports |
| path history | **off by default** (`router_cond=state`), toggle kept | <= 0.03 nats in arms H-K and S2 (section 13.1); the residual stream already carries the path |
| adapters | **hop-conditioned adapter bank** (K=4, rank 16, fp16) | The per-loop DWP analogue, worth 0.11 nats in arm B |
| soft mixing | **on inside each tile** (mixer vs FFN, floor 0.5) | Worth 0.07 nats in arm C |
| FRP reasoning block | **skipped for now** | User decision: risks breaking general use; deep dive 11 remains the spec if revisited |
| first run | **full stack minus FRP**: hybrid Mamba, free routing, value exit, fp16 embeddings, width 512 | User choice; the ablation nights switch toggles off one at a time |

## 2. Assumptions (not asked; stated with reversal cost)

- Width 512, 4 tiles, 8 hops max / 2 min, sequence length 1024, ~16k tokens per step. Reversal: a preset flag.
- The anchor (target) is arm C's recipe retrained under the 16k tokenizer on the **same number of tokens
  as the candidate**, not the same wall clock (equal tokens isolates the architecture; the wall-clock
  overhead of routing is reported separately). Under GPT-2 the target is already known:
  `outputs/armC-bpb-target-gpt2.json`, mean core bits/byte **1.3909** (TinyStories 0.7444, code 1.8468,
  math 1.5815). Reversal: rerun one 3-hour anchor.
- Embeddings stay fp16 in the first run; `--ternary-embed` is the first ablation on the footprint ladder.
- Learned absolute positions (as in every prior arm), not RoPE. Reversal: one flag, one rerun.
- Exploration 0.3 during training (section 13.1 deviation 3) is now a default, not a deviation.
- Evaluation protocol unchanged: fixed seeded held-out batches per set, lambda sweep, matched random and
  capacity nulls, hop-cap sweep. bits/byte uses `val_bytes_per_token` from the corpus manifest.

## 3. Protocol

1. **Data** (`experiments/general_model/data/prepare_general_corpus.py`): download full sources, train the
   tokenizer on train text only, document-level dedupe, encode, 16-gram leak filter of every held-out
   document against the whole train stream (drop >= 20% leaked, record residual), mix by tokens, write
   `data/general/{train_tokens.pt, val_*_tokens.pt, tokenizer.json, manifest.json}`.
2. **Model** (`experiments/general_model/general_model.py`): `HybridTile` (Mamba or attention mixer + dual
   SwiGLU, soft mix, FiLM/LoRA hooks, runs on gathered causal subsets), `GeneralRoutedLM` (free/fixed
   routing, value exit, deep supervision with measured gain, exploration, STE gate, per-tile checkpointing,
   stored-byte accounting). Tests in `test_general_model.py` include causality through Mamba on subsets.
3. **Trainer** (`experiments/general_model/train_general.py`): time-budgeted cosine schedule, resume with
   optimizer state, per-set loss/ppl/bits-per-byte, target PASS/FAIL, end-of-run read-out, ledger
   `outputs/general-<tag>-log.json`.
4. **Anchor**: `train_navitrit_unified.py` patched to read any `val_*` sets, the manifest vocabulary and
   bytes-per-token, so arm C's recipe runs on the new corpus unchanged.

## 4. Run schedule (one GPU job at a time)

| night | run | purpose |
|---|---|---|
| 0 (now, idle GPU) | `C-cont`: arm C + 3000 steps, fresh cosine, GPT-2 corpus | Equal-tokens control for section 13.1 confound 4 (does S2's compute win survive?) |
| 1 | `G1-full`: hybrid Mamba, free routing, value exit, fp16 embed, p512, 8 h | The candidate; its stored MB and bits/byte are the first point on the ladder |
| 2 | `G-anchor`: arm C recipe, 16k tokenizer, same tokens as G1 | Defines the target under the new tokenizer |
| 3+ | footprint ladder: ternary embeddings, p384, fixed routing, 2 experts -> 1 | Each judged by: reaches the target? then stored MB |

## 5. Decision rule (fixed before night 1)

A configuration **qualifies** if its mean core bits/byte (TinyStories, code, math) is <= the anchor's at
equal training tokens. Among qualifying configurations the winner is the one with the **smallest stored
bytes** (`count_parameters()["stored_mb_total"]`: ternary at 1.58 bit, everything else at 16 bit). Ties
within 0.005 bits/byte go to fewer stored bytes; a configuration that qualifies only at lambda = 0 (no
exit) is reported with its hop count and does not get credit for adaptive depth. Wall-clock tokens/s is
reported for every run and is not part of the rule.
