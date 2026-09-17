# Research kickoff verification

The integrated mathematical check passed **35 named checks** on the first and only execution of the final script, at **2026-09-12 00:36:55 UTC** (2026-09-11 America/Denver), using **Python 3.12.10** and its standard library.

Run from the project root with `python experiments/verify_operators.py`. The first-run JSON is retained at `outputs/math-verification.json` in the project. The kickoff was executed from the preparation workspace using the identical staged script.

## What passed

- The three supplied image tables, the added OR and logical XOR tables, and modulo-3 addition/subtraction tables agree with their stated definitions.
- Boolean endpoint behavior, both De Morgan laws, min/max commutativity, associativity, identities, absorption and mutual distributivity pass over the full applicable finite domains.
- Both min/max constructions of unknown-preserving XOR agree with `-xy`; XOR commutativity and associativity pass.
- Modulo-3 arithmetic, inverses and balanced-digit carry reconstruction pass. Ordinary and wrapped accumulation give their deliberately different outputs.
- There are 27 unary maps, six reversible permutations, three nonidentity involutive swaps and two cyclic shifts; only sign reversal is the order-reversing bijection.
- Concrete matrix examples preserve their outputs under paired column/input sign changes and permutations. Unpaired changes, a zero-moving swap, and passing a sign change through ReLU have explicit counterexamples.

Binary equalities were checked on all nine input pairs; laws involving three operands were checked on all 27 triples. Permutation classification covers all six permutations. The matrix checks are concrete examples backed by the general algebraic identities in the operator reference; they are not exhaustive checks of networks or transformers.

## Independent review

The project setup reviewer examined the operator reference, technical assessment, project instructions and finite checker in a separate context from their authors. A second review context examined the orientation against the exact v0.3 protocol and independently reproduced the selected packet content hashes under the documented capture normalization. No blocking findings remained after the authors incorporated the review corrections.

The review records are `independent-review.md` and `orientation-review.md`. Project registration and task association have their own evidence in `project-setup.md`.

The initial deployment copied 77 project files (1,555,250 bytes) to the requested Programming folder, with every target SHA-256 matching its prepared source. The three original image files and pasted protocol were also compared directly with their user-supplied originals; all four matched. Later status-note updates do not change the verified mathematics.

## Limits and next step

These results establish consistency of the finite definitions and the stated examples. They do not measure model quality, compression on trained weights, speed, energy, disk access, or physical ternary/analog memory. No ecosystem service, identity, key, registry, funding or recipient-acceptance flow was exercised.

The checker identifies a failing law by name. A future reusable evaluator should also print the failing inputs and both sides. That diagnostic improvement is not needed to interpret this successful run.

The next mathematical experiment is to choose one representation transformation, state exactly which operation it must preserve, and prove the invariant or retain a counterexample. If a region/template encoding is pursued, separately count the template, selector and exception information it needs.
