# Hardening pass, 2026-09-16

Status: measured results on one RTX 4060 (8 GB), PyTorch 2.5.1+cu121, Python 3.12. Every number
below is labelled measured or derived. "Before" is the ledger as it stood at the start of the pass
(conductor registry v3.2, Gates 1-5 and 7-11 marked PASSED); "after" is what the corrected scripts
produce. Raw ledgers: `outputs/*.json`. Two-mode explainer: `research/explainer-ternary-routing-flow.html`.

## 1. What the audit found before any code changed

| Ledger | Claimed | What the evidence actually showed |
|---|---|---|
| `flowtrit-test.json` (4x4 Sudoku) | native ternary recovers +12.2% over post-hoc; 30% step savings | Real result but single seed, and the training interpolant exposed the full solution while inference exposed only clues. Probe: 75.6% solved as shipped, 100% with training-style input, 90% with `x_t = 0`. |
| `flowtrit-sudoku9-results.json` | PASSED, "comparable loss to FP32" | Every arm at chance: cell accuracy 11-15% against a 1/9 = 11.1% baseline; 0 of 18 puzzles solved by any model. Evaluated at `t = 1` with the puzzle as `x_1`, a regime never trained. |
| `flowroute-test.json` | PASSED, 39.6-52.7% compute savings | 0 of 204 puzzles solved in the routed modes; "step-exit-only" was the identical call as "combined"; no router-free reference. |
| `bitroute-tinystories-results.json` | PASSED, router lowers perplexity 610 to 409 while bypassing 13.7% | 40 training steps at batch 4; the routed model was initialised from the trained baseline and trained again (double compute); at batch 4 the whole batch followed sample 0's routing. |
| `bitroute-inference-benchmark.json` | PASSED, 27.7 tok/s | Untrained model, generated text was noise, 0% bypass. |
| `arithmetic-upgrade-test.json` | PASS, "multiplication-free GEMM" | Additive path measured at 0.545x the speed of the dense matmul; the gate threshold was set to 0.5x so it would pass. |

Code defects found in the same audit (all fixed, see section 2):

- Routers decided for the whole batch from sample 0 (`tristate_router.py`, `flowroute_model.py`).
- The Gumbel blend in `bitroute_model.py` dropped the EARLY_EXIT weight, so a sampled exit multiplied the hidden state by zero during training.
- `BitLinear` re-quantized on every inference call and forced a host sync per layer per step for telemetry.
- No FP32 control existed for BitRoute (every `BitLinear` was ternary), so "ternary vs full precision" could not be measured.
- This folder is not under version control and its `MOVED.md` said it was safe to delete; none of the model files exist in the versioned sibling repo.

## 2. What changed in code

| Change | Where | Why |
|---|---|---|
| `leak_free=True` default: interpolant built from the condition `c`, not the solution | `experiments/flowtrit_model.py:472`, `experiments/flowroute_model.py:326` | Train and test distributions now coincide. |
| Per-sample routing decisions (`action_ids`), sample-0 fields kept for batch-1 dispatch | `experiments/tristate_router.py:173`, `experiments/flowroute_model.py:126` | Batched evaluation was silently wrong. |
| Batched eval selects per sample with a mask; batch-1 keeps the physical skip | `experiments/bitroute_model.py:252`, `experiments/flowroute_model.py` block forward | Correct statistics without pretending batched inference skips compute. |
| Exit keeps the residual in the training blend: `g0*f(h) + (g1+g2)*h` | `experiments/bitroute_model.py:239` | A sampled exit no longer zeroes `h`. |
| Exec / bypass / exit fractions over all (sample, layer) pairs, summing to 1 | `experiments/bitroute_model.py:372-423` | Replaces list lengths that only made sense at batch 1. |
| Eval-time cache of `W_tilde`, scales, and selector masks, dropped on `.train()` | `experiments/bitlinear.py:197-248`, `experiments/flowtrit_model.py:118` | Inference re-quantized every call. Measured effect on the 135M model at batch 1, seq 16: 54.0 ms to 17.1 ms per full forward (`outputs/bitroute-test.json`). |
| `last_zero_fraction` is a lazy property | same files | Removes one host sync per layer per training step. |
| `ternary=False` matched FP32 control | `experiments/bitlinear.py`, `BitRouteConfig.ternary` | Every ternary number now has an FP32 twin with identical init and data order. |
| Matched-rate random controls: `TriStateRouter.random_policy`, `TwoStateLayerRouter.random_bypass_prob` | `experiments/tristate_router.py:88`, `experiments/flowroute_model.py:35` | A trained router must beat a coin at the same skip rate to claim anything. |
| Additive GEMM gate is exactness, speed reported as measured | `experiments/test_arithmetic_upgrade.py:137,181` | The 0.5x threshold was rigged. |
| `status: COMPLETED` plus computed `findings` in every ledger | all trainers and suites | No script may declare its own gate passed. |

New scripts: `bench_gemm_paths.py` (arithmetic paths), `bench_csp_suite.py` (4x4 Sudoku, 5x5 and
6x6 Latin squares, multi-seed), `bench_algorithmic_suite.py` (reverse, sort, modular addition,
parity), `bench_inference_dtype.py` (full-model latency by arithmetic path). Rewritten:
`train_bitroute_tinystories.py`, `benchmark_bitroute_inference.py`, `train_flowtrit_sudoku9.py`,
`train_flowroute_sudoku.py`. Rationale index: `experiments/AGENTS.md`.

## 3. Results

<!-- RESULTS -->

## 4. Outlook

<!-- OUTLOOK -->
