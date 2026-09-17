# Independent review of the research kickoff

Reviewed on 2026-09-11 in a separate agent context from the authors. This is a static review; the final integrated executable validation belongs to the main task.

## Scope and conclusion

Reviewed the project README, project instructions, question log, operator reference, technical research assessment, and the expanded finite-domain validation script. No blocking mathematical, source-grounding, or authorization defect was found.

The review checked:

- The supplied NOT, AND, and XOR-labelled tables against sign inversion, minimum, and signed multiplication.
- The distinction between logical XOR and XNOR under the false=-1, unknown=0, true=+1 encoding.
- All three nonidentity swaps, their involutive character, the two cycles, and the counts of 27 unary maps and six permutations.
- Modulo-3 addition and subtraction, balanced ternary carries, and the distinction between ordinary and wrapped accumulation.
- The symmetric integer accumulator-width formula and the sign/permutation preservation identities for a linear layer.
- The zero-moving and ReLU counterexamples, block-storage counting, and the need to count representation metadata.
- Primary-source captures supporting the BitNet training and architecture statements, the current official implementation's CPU/GPU paths, flash windowing and row/column bundling, FlexGen's batched latency-insensitive setting, and IBM's distinct 64-core phase-change-memory chip.
- The separation between the user-adopted orientation workflow and external source instructions, and the explicit absence of publishing, spending, installation, or external messaging authorization.

The expanded validation script checks the main printed tables, independent algebraic laws, Boolean endpoints, carry behavior, and concrete preservation/counterexample cases. Its matrix checks are examples backed by algebraic identities in the text; they are not exhaustive neural-network or transformer checks.

## Limits

This review does not verify a model's accuracy, speed, storage footprint, disk performance, analog circuit behavior, or superiority over established kernels. No performance claim for the proposed system is approved by this record. Source checking used the captured primary-source excerpts available in the workspace, not an exhaustive literature search.

One nonblocking diagnostic improvement remains for a future reusable evaluator: failed finite-law checks currently identify the assertion by name, while the reference's full verification contract asks for failing inputs and both sides of an equality. This does not change the present formulas or the meaning of successful checks.
