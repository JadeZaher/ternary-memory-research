# Experiments directory

## Finite operator checks

`verify_operators.py` checks the definitions in `research/operator-reference.md` on their full three-state domains, along with concrete matrix examples. Run it from the project root with Python 3. It uses only the standard library and writes a JSON result to `outputs/`.

The printed table constants independently transcribe the specification; algebraic laws and Boolean endpoint checks supplement table agreement. Matrix examples supplement the general algebraic identities, rather than replacing their proofs. No model, numerical library, accelerator, storage benchmark, or network service is involved.

For a future reusable evaluator, improve failure diagnostics to print the failing input and both sides of each law. The kickoff script currently identifies failed laws by name and stops immediately.

## Region encoding checks

`region_encoding.py` checks the counting claims in `research/region-memory.md`: entropy bounds, mask/sign planes with a rank directory, the zero-point identity, escape codes, activation-side lookup tables, template dictionaries on seeded random blocks, and which of the six state permutations are affine or factor into row and column signs. Same conventions as `verify_operators.py`: standard library only, run from the project root, JSON written to `outputs/region-encoding.json`. Failed checks print the failing inputs and both sides where an equality is involved. Dictionary rows are one seeded sample, so their exact counts change if the seed changes; the inequalities they support should not.

## Model line (PyTorch, CUDA)

Run every model script from the project root as `python experiments/<name>.py`; they import as `experiments.<module>`. The two exceptions, `test_flowtrit.py` and `test_arithmetic_upgrade.py`, import siblings directly and run from inside `experiments/`.

| Module | Role |
|---|---|
| `bitlinear.py` | Block-scaled (g=256) ternary linear layer with STE; `ternary=False` is the matched FP32 control. Eval mode caches the ternary snapshot and selector masks until `.train()`. |
| `tristate_router.py` | Per-layer probe emitting EXECUTE / ROUTE_AROUND / EARLY_EXIT. Decisions are per sample (`action_ids`); `random_policy` is the matched-rate random control. |
| `bitroute_model.py` | Track A causal LM. Batch size 1 physically skips layers; batched eval selects per sample (correct statistics, no skipping). |
| `flowtrit_model.py` | Track B recurrent flow denoiser with its own per-tensor BitLinear, FPF training, contraction early exit. |
| `flowroute_model.py` | Track C: FlowTrit blocks with a two-state layer router; `random_bypass_prob` is the matched-rate random control. |
| `train_bitroute_tinystories.py`, `bench_algorithmic_suite.py`, `benchmark_bitroute_inference.py` | Track A trainers and task suite. |
| `bench_csp_suite.py`, `train_flowtrit_sudoku9.py`, `train_flowroute_sudoku.py` | Track B/C constraint-satisfaction suite and trainers. |
| `bench_gemm_paths.py`, `test_arithmetic_upgrade.py` | Inference arithmetic paths: dense FP32, additive masks, cached masks, bf16, int8 tensor-core. |

### Why the 2026-09-16 hardening pass changed these files

Full account: `research/hardening-2026-09-16.md`. The short version:

- **Train/test mismatch in the flow interpolant.** Training built `x_t` from the full solution, inference from the clues. `leak_free=True` (default) builds it from the condition `c` in both. Numbers recorded before this date are not comparable with numbers after it.
- **Batch followed sample 0.** Routers decided for the whole batch from the first sample. Batched evaluation now routes per sample; physical skipping remains a batch-size-1 property, which is the deployment case the architecture note targets.
- **Sampled EARLY_EXIT zeroed the hidden state in training.** The Gumbel blend dropped the exit weight. Exit now keeps the residual, which is what exiting means for that layer.
- **Ledgers hard-coded PASSED.** Every script now writes `status: COMPLETED` plus a `findings` block computed from the data; the registry decides gates.
- **Missing controls.** Every routed model is compared with (a) the same backbone forced to execute every layer and (b) a random router at the same bypass rate. A router that does not beat (b) has learned nothing about which layers matter.
- **Inference re-quantized every call.** Eval mode now caches the ternary snapshot; the telemetry `.item()` sync per layer per step is gone from the training path.
- **The additive GEMM gate was rigged.** It reported 0.55x the speed of a dense matmul and passed a 0.5x threshold. The gate is now exactness; speed is reported as measured, and `bench_gemm_paths.py` measures the paths that can actually be faster on this GPU.

Two BitLinear implementations remain (`bitlinear.py` block-scaled for Track A, `flowtrit_model.BitLinear` per-tensor for Tracks B/C). They differ in scale granularity on purpose; merging them is a later refactor, not a correctness issue.
