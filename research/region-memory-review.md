# Independent review of the region-memory document

Reviewed on 2026-09-12 (America/Denver) in a separate agent context from the author, with the instruction to refute rather than confirm. Scope: `research/region-memory.md` version 0.1, `experiments/region_encoding.py`, and `outputs/region-encoding.json`, read against `operator-reference.md`, `technical-research.md`, and `verification.md`. The reviewer ran the checker once and confirmed 23 passing checks and a deterministic JSON under the fixed seed. Verdict on version 0.1: **blocking issues present**, 4 blocking, 11 should-fix, 9 nit. All 24 findings were applied by the author to produce version 0.2; the applied resolutions are listed below so that a reader can check each one.

## Blocking findings and resolutions

| Finding | Resolution in version 0.2 |
|---|---|
| The claim that an undecoded per-block sign flip *must* factor as row times column signs was asserted, not proven, and is false without the restriction to row/column compensation; a kernel can negate a block's partial sum instead. | Section 4.5 now states the hypothesis, gives the one-line proof (`r_i * f_ij * c_j = 1` for every entry), and adds the partial-sum alternative. Claim 9 is labeled "derived given that compensation rule". |
| The single-level 16-bit rank directory overflows at 65,536 nonzeros. | The script and section 4.1 use a two-level directory (32-bit superblock count per 65,536 bits, 16-bit relative block count per 256 bits), 0.063 bits per weight. A new check ranks across a superblock boundary and verifies relative counts stay under sixteen bits. |
| "No cost inside `P`" for the spare escape code contradicted the script's own accounting (2 bits versus 1.585). | Section 4.6 now says the spare code is free only when `P` already uses fixed 2-bit codes and costs 0.415 bits per weight over packed ternary; the JSON records that penalty. |
| The document said every failed check prints inputs and both sides; only six of 23 did, and none printed the offending element. | The script gained `verify_all`, which reports the first failing input with both sides, and `verify_equal` reports the first differing index in list comparisons. Section 8 describes exactly what is printed and notes that `verify_operators.py` still reports by name only. |

## Should-fix findings and resolutions

- Dictionary accounting was pessimistic for the dictionary (flat index, `log2(3)` per dictionary trit) and the prose overstated the margin; the reviewer's recomputation under favourable accounting landed within hundredths of the bound and below it on sparse samples. Section 4.5 now states the accounting, says the conclusion rests on the source-coding theorem, and stage A reports both accountings.
- `N = 100,000` and the block count were missing from the dictionary table; both were added.
- Claim 8 called input compensation "exhaustive" while the script only tries an input sign; the claim now separates the exhaustive affine and input-sign checks from the prose argument for arbitrary `phi`, and the script comments why the `a, b` search is exhaustive.
- Section 8 said the third permutation check shows failure at `w = 0`; it only verifies that non-affine permutations move zero. Renamed and reworded.
- Claim 1 dropped the balanced-sign hypothesis; restored, with the unbalanced formula.
- Consequence 1 of section 3 now says "on average" and notes finite samples can dip below the bound.
- Rank was presented as one popcount; it is up to four 64-bit popcounts plus a mask, on hardware with the instruction. Section 4.1 and the glossary were corrected.
- An unmeasured claim that unsigned digits multiply more cheaply was removed and moved to the open questions as unmeasured.
- Section 4.1 had the large mask as `D`; the mask and signs are now `P`, the directory is `D`, and section 6 notes the exception.
- Section 5's opening sentence contradicted its own table by calling all three NOTs decode-time-only; reworded.
- "Most useful" was an unsupported ranking; replaced with the affine property it stood for.

## Nits and resolutions

Rounding convention stated in section 0; "almost every block is unique" corrected to the exact count; "halving `K`" qualified to "at most half" with the index/selector cancellation noted; the actionable definition now says one streaming pass over `P` with `D` read at random; the "seventh map" sentence names its codomain; exercise 2 gives the crossover with the directory included; exercise 4 uses the shorter direct argument; the pos/neg-plane sentence no longer claims a mask plane; the ASCII diagram border was aligned.

## Verified correct by the reviewer

The entropy table and `H(d) = H_b(d) + d`; overhead `1 - H_b(d)` with equality only at `d = 0.5`; exhaustiveness of the affine search; exactly two affine permutations; the zero-point identity; all six exercise answers including the parity argument; the affine-without-wrap versus `cycle_plus`-with-wrap distinction; the escape-cost arithmetic; every table cell against the JSON; the check grouping in section 8.

## Limits

This review checked mathematics, internal consistency, label discipline, and novice-facing clarity. It did not verify any published source quotation against its origin, any model, kernel, or hardware behaviour, or the empirical statistics of trained weights. Version 0.2 has not itself been re-reviewed; the resolutions above are the author's and are open to the next reviewer.
