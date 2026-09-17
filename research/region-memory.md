# Region memory: a small region that makes the rest actionable, version 0.2

Prepared 2026-09-12 (America/Denver). Status: definitions, derivations, and one seeded finite experiment. `experiments/region_encoding.py` passed 24 named checks; its numbers are in `outputs/region-encoding.json`. Published-source quotations live in `work/region/source-findings.md`, captured 2026-09-12 between 14:10 and 14:12 UTC by direct HTTP with per-URL status in `work/region/fetch-log.json`; the GitHub pull-request body came from the GitHub REST API because the page itself is a client-rendered shell. Version 0.1 was reviewed adversarially in a separate context the same day; every finding was applied and the record is `research/region-memory-review.md`. No model accuracy, kernel timing, or hardware result is claimed anywhere in this document.

This document answers one question from the project owner: *can a region of memory be set aside so that other parts of memory contain more actionable information, especially for AI inference?* The short answer is yes, this pattern already exists in several forms, and the useful work is to name the forms, count what each costs, and say which operations each makes cheap. Section 5 shows what the "three NOT operations" from `operator-reference.md` become inside this pattern.

## 0. How to read this

Every claim carries one of four labels:

- **derived**: follows from the definitions in `operator-reference.md` by algebra or counting; checked by the script wherever the domain is finite.
- **published**: reported by a primary source, quoted verbatim in `work/region/source-findings.md`.
- **inference**: a connection this project proposes; nobody has established it.
- **proposal**: a bounded experiment to run next.

Symbols: `S = {-1, 0, +1}`; `N` weights; `d` = density = fraction of weights that are nonzero; `g` = block size; `log2(3) = 1.585`. A *bit per weight* figure counts stored bits divided by `N`. Figures are rounded to three decimals unless a table says otherwise; the JSON keeps four.

## 1. The idea in one sentence

Split memory into a small **descriptor region** `D` and a large **payload region** `P`, so that the compute unit can act on `P` through `D` without first rebuilding every value.

```
   D (small, kept close)            P (large, streamed)
  +--------------------+           +-------------------------------+
  | masks, scales,     |  guides   | packed trits, signs, indices, |
  | tables, pointers   | --------> | exceptions ...                |
  +--------------------+           +-------------------------------+
            |                                     |
            +---------- kernel: y = f(D, P, x) ---+
```

"Actionable" is given a precise meaning in section 2. Sections 3 and 4 work through the information budget and six ways to fill `D` and `P`. Section 5 revisits the three NOTs. Section 6 places regions in the memory hierarchy. Section 9 is a self-study kit with exercises whose answers the script can confirm.

## 2. Definitions

**Region.** A contiguous set of stored bits with one declared format. A representation is a list of regions plus a decode rule.

**Two-region encoding.** A triple `(D, P, decode)` where `decode(D, P) = W`, the intended weight tensor, either exactly (lossless) or with a stated error (lossy).

**Actionable for f.** An encoding is *actionable for an operation f* when `f` can be computed from `(D, P, x)` in one streaming pass over `P`, with `D` read at random as needed, operating on the stored codes directly and never writing out `W`. Here `f` is the dot product `y_i = sum_j W_ij * x_j`, the operation every linear layer performs. The point of the definition is that memory traffic and decode work then scale with the region sizes, not with a decoded copy.

**Cost ledger.** Five numbers to record for every encoding:

1. bits per weight in `P`;
2. bits per weight in `D`;
3. random access: can weight `j` be read without decoding its neighbours? (yes / by block / no);
4. decode work per weight for `f`;
5. which operations are directly actionable.

Bits and actionability are two different currencies. A region can be worth adding even when it raises the bit count, if it removes work from the inner loop. Section 4 gives the ledger for each pattern.

## 3. The information budget (derived)

No layout of regions can store `N` weights in fewer bits, on average, than their entropy. For independent trits with density `d` and equally likely signs:

```
H(d) = H_b(d) + d,   where   H_b(d) = -d*log2(d) - (1-d)*log2(1-d)
```

`H_b(d)` is the cost of answering "is this weight zero?"; the `+ d` term is one sign bit paid only for nonzeros. `H_b` is the *binary entropy*: it is 0 when the answer is certain (`d = 0` or `d = 1`) and 1 when the two answers are equally likely (`d = 0.5`). If the signs are not balanced the true entropy is lower, `H_b(d) + d * H_b(p_neg)` with `p_neg` the share of negative nonzeros; the script's entropy function carries that parameter and every figure below uses `p_neg = 0.5`.

| density `d` | 1.0 | 2/3 | 0.5 | 0.3 | 0.1 |
|---|---:|---:|---:|---:|---:|
| entropy `H(d)` bits/weight | 1.000 | 1.585 | 1.500 | 1.181 | 0.569 |

The uniform case `d = 2/3` gives `log2(3)` exactly (checked). Three consequences:

1. On data that really is independent and identically distributed, no region layout beats `H(d)` on average. A particular finite sample can be coded slightly below it, which is why sample results in section 4.5 must not be read as a theorem.
2. Regions pay off in bits only by exploiting structure (sparsity, sign imbalance, correlation, repetition) that lowers the real entropy below 1.585.
3. Even at equal bits, a region can change which operation is cheap. That is the second currency.

*Independent and identically distributed* (i.i.d.) means each weight is drawn from the same distribution with no relation to its neighbours. Trained weights are not i.i.d.; that is precisely where a region layout could gain, and stage A in section 10 measures it.

## 4. Six region patterns

Each subsection ends with its cost ledger. "Checked" means the script verified the statement on its finite or seeded domain.

### 4.1 Bit planes: a mask region and a sign region (derived; published analogues)

Store a **mask plane**, one bit per weight saying whether it is nonzero, and a **sign plane**, one bit per *nonzero* weight. Together they are the payload `P`, `1 + d` bits per weight; the mask is the part of `P` a kernel reads first, but it is not small. The overhead above entropy is exactly `1 - H_b(d)` (checked): zero at `d = 0.5`, 0.082 at `d = 2/3`, 0.119 at `d = 0.3`, 0.531 at `d = 0.1`. The mask plane is the redundant part; it is compressible whenever `d` is far from one half.

Random access into the sign plane needs **rank**: `rank(j)` = number of set mask bits before position `j`, which is the index of weight `j` inside the sign plane. The descriptor `D` is a two-level directory: one 32-bit absolute count per 65,536 mask bits and one 16-bit count per 256 mask bits relative to its superblock, 0.063 bits per weight in total. A single-level 16-bit directory would overflow at 65,536 nonzeros, which any real layer exceeds. Rank is then one superblock read, one block read, and a popcount over the partial block: up to four 64-bit popcounts and one mask, on processors that have a popcount instruction. Script example: 4,096 weights at density 0.405 needed 4,096 mask bits, 1,660 sign bits, and 288 directory bits, 1.476 bits per weight against an entropy of 1.379, and every weight was recovered through rank (checked); a second example crossed a superblock boundary with every relative count under sixteen bits (checked).

The dot product is directly actionable: `y = sum(x over mask AND positive) - sum(x over mask AND negative)`. The kernel never forms `W`; it performs two masked sums (checked on 300 random trials). If rank is unwanted, store two full planes, positive and negative, at 2 bits per weight with no directory; there is then no mask plane at all, and "nonzero" is the OR of the two planes. That is the same bit count as fixed 2-bit codes with the same information split across two planes instead of interleaved.

This is the one pattern in which the guiding information, the mask, is larger than the sign data it guides. The rule of section 6, keep `D` resident and stream `P`, applies to the directory, not to the mask.

Published analogues, quoted in `work/region/source-findings.md`. NVIDIA's 2:4 structured sparsity keeps the two nonzeros of every four values in a compressed matrix and stores separately "2-bits to encode the position of each nonzero value within the group of 4 values"; for 16-bit operands that is 36 bits in place of 64, about 44% saved, and the Sparse Tensor Cores "use the metadata that is stored with the nonzeros to pull only the necessary values from the other, uncompressed operand" (Mishra et al. 2021; NVIDIA developer blog). That is a mask region making a payload region actionable, in hardware. CHERI's tags "add one bit of memory for every 128 or 256 bits of data, with a <1% memory overhead", held in a set-aside partition of physical memory (CHERI FAQ). AddressSanitizer "maps 8 bytes of the application memory into 1 byte of the shadow memory" (sanitizers wiki). Succinct bit vectors support rank "in O(1) time and n+o(n) bits of space" (Wikipedia, *Succinct data structure*); using that rank to index a compacted sign plane is this project's construction (derived, checked), not a statement from that source.

Ledger: `P` = `1 + d` bits/weight (mask plus signs); `D` = 0.063 bits/weight (directory); random access yes via rank; decode = one directory lookup and up to four popcounts per access, none for a streaming pass; actionable: dot product, zero skipping.

### 4.2 Block scale region (published; derived)

BitNet b1.58 scales a whole ternary tensor by one absmean value; `technical-research.md` records the source. Microscaling (MX) formats make the scale a block region: an MX block "consists of a single *shared scale* *X* and *k* scalar *elements*", every concrete MX format uses a block size of 32, and all of them use "E8M0 (an 8-bit exponent) as the format for the shared scale" (Rouhani et al. 2023), which puts 0.25 bits per element into `D`. llama.cpp's ternary type TQ1_0 is the same pattern at ternary width: a 256-element block holds "240 elements encoded in 5 elements per byte, while the last 16 elements are encoded in 4 elements per byte", plus "one `float16` scale per block, so the size of a block is 54 bytes making it a `1.6875 bpw` type" (pull request 8151). In ledger terms that is 52 bytes of `P`, 1.625 bits per weight, and 2 bytes of `D`, 0.0625 bits per weight. The ideal 1.6 is not reached because a 256-element block needs `ceil(256 / 5) = 52` payload bytes, not 51.2. In region terms `D` = scales and `P` = codes. The dot product becomes `y = scale * sum(T * x)`: the scale multiplies once per block *after* an integer accumulation, so it never enters the inner loop.

This is also where the "wider sum" of `operator-reference.md` section 6 lives. The integer accumulator sits between `P` and the scale, and its width is fixed by the block length and input range, not by the number of weight states.

Ledger: `P` = 1.585 to 2 bits/weight; `D` = 0.0625 to 0.25 bits/weight; random access by block; decode = one multiply per block; actionable: dot product with deferred scaling.

### 4.3 Zero-point region: unsigned digits (derived)

Store `u = w + 1`, so `u` is in `{0, 1, 2}`. Then

```
sum_j w_j * x_j  =  sum_j u_j * x_j  -  sum_j x_j
```

The correction `sum_j x_j` is computed once per input vector and shared by every output row (checked on 300 random trials). `D` is one integer per input vector, on the activation side; `P` holds unsigned digits.

Why this matters for the three-NOT question: "add one" is the cyclic shift `cycle_plus` from `operator-reference.md` section 3 *only if the result wraps*. Here it does not wrap; the codomain is `{0, 1, 2}`, not `S`. The map is **affine over the integers** (`u = 1*w + 1`), and an affine map is exactly what lets the correction be a single linear term. This is the zero-point of ordinary asymmetric integer quantization, arrived at from the ternary side.

Ledger: `P` = 2 bits/weight (or 1.6 packed); `D` = one accumulator per input vector; random access yes; decode none; actionable: dot product with one shared correction.

### 4.4 Lookup-table region (published; derived)

For a group of `g` input values, precompute the table of all `3^g` partial dot products. Each weight group is then an index into that table, and the kernel adds `table[index]` instead of multiplying. `D` = the table, built once per input vector on the activation side, `3^g` entries per `g` inputs; `P` = packed indices. Two trits fit a 4-bit index (`9 <= 16`) and three trits fit a 5-bit index (`27 <= 32`); the script confirms the table sum equals the dot product for `g = 3` (checked).

T-MAC and the bitnet.cpp CPU kernels are the published forms of this pattern. T-MAC: "Given an activation, it can be first computed with all possible bit patterns and saved in tables. The mpGEMM of activation and one-bit matrix is then transformed to table lookup indexed by each bit pattern in weight, and addition to accumulate the looked-up results", and "The table size grows exponentially with the group size g" (Wei et al. 2024). bitnet.cpp: the TL1 kernel "transforms every two full-precision weights into 4-bit index" and "pre-computes their corresponding activations into 3^2=9 values" held in an int16 table; TL2 "compresses every three full-precision weights into a 1-bit sign (0 or 1) and a 4-bit index"; the plain I2_S kernel instead unpacks 2-bit weights and multiplies (Wang et al. 2024). The reported speedups are "2.37x to 6.17x on x86 CPUs and from 1.37x to 5.07x on ARM CPUs" against llama.cpp, measured on an Intel Core i7-13700H laptop and an Apple M2 Ultra; T-MAC reports "up to 4× increase in throughput" over llama.cpp on its edge devices. Those numbers belong to those kernels, models, and machines; nothing in this project reproduces them. The lookup table is a region that *replaces multiplication with addressing*. Its cost is table construction and table size against register or first-level cache capacity (T-MAC notes that at `g = 4` "the LUT is four times larger than the original activation"); its payoff is that the weight region becomes pure addresses.

Ledger: `P` = 2 to 1.67 bits/weight (4 bits per 2 trits, 5 bits per 3 trits); `D` = `3^g` accumulators per `g` inputs, rebuilt per input vector; random access by group; decode none; actionable: dot product by table lookup.

### 4.5 Template dictionary and transform selector (derived; the "regions of weights" idea made testable)

The project owner's original hypothesis, restated: blocks of `g` weights repeat, so store `K` templates in `D` and a per-block index in `P`, optionally with a per-block **transform selector** naming one of the six permutations of `S` (2.585 bits as an ideal fractional cost, 3 bits as a flat code) or just a sign flip (1 bit).

Counting result on random data (checked, seeded, 100,000 weights per row): a flat index of `ceil(log2 K) / g` bits per weight plus a dictionary of `K * g * log2(3) / N` bits per weight never fell below the entropy bound for any `g` in `{2, 4, 8, 16}`, for uniform blocks (`d = 2/3`) or sparse blocks (`d = 0.2`):

| density | block `g` | blocks | distinct | possible | index bits/w | dictionary bits/w | total bits/w | entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2/3 | 2 | 50,000 | 9 | 9 | 2.000 | 0.000 | 2.000 | 1.585 |
| 2/3 | 4 | 25,000 | 81 | 81 | 1.750 | 0.005 | 1.755 | 1.585 |
| 2/3 | 8 | 12,500 | 5,575 | 6,561 | 1.625 | 0.707 | 2.332 | 1.585 |
| 2/3 | 16 | 6,250 | 6,250 | 43,046,721 | 0.813 | 1.585 | 2.398 | 1.585 |
| 0.2 | 2 | 50,000 | 9 | 9 | 2.000 | 0.000 | 2.000 | 0.922 |
| 0.2 | 4 | 25,000 | 78 | 81 | 1.750 | 0.005 | 1.755 | 0.922 |
| 0.2 | 8 | 12,500 | 1,140 | 6,561 | 1.375 | 0.145 | 1.520 | 0.922 |
| 0.2 | 16 | 6,250 | 4,301 | 43,046,721 | 0.813 | 1.091 | 1.903 | 0.922 |

Reading the table: with small blocks every possible template appears, so the index alone costs as much as a fixed code; with large blocks every block is unique (6,250 distinct of 6,250 at `d = 2/3`, `g = 16`), so the dictionary is a full second copy of the data. On i.i.d. data there is no block size where both are small at once.

Two cautions about the accounting. These totals use a flat `ceil(log2 K)` index and price every dictionary trit at `log2(3)`; both are pessimistic for the dictionary. An entropy-coded index over a dictionary stored as an unordered set, priced at the source entropy, lands within a few hundredths of the bound and on a finite sample can dip just below it. The conclusion, that a dictionary cannot beat the bound on i.i.d. data on average, rests on the source-coding theorem, not on the margin in this table. Gains require *real* repetition. Stage A in section 10 measures the empirical block statistics of a trained tensor before any further investment.

The transform selector has two different uses, and they must not be confused:

- **As a decode step.** Any of the six permutations may be applied when a block is read, `w = t(template)`. The dictionary can then be canonicalized, for example under sign flip so that the first nonzero of every template is `+1`. That reduces `K` by at most half (the all-zero template is its own image, and an observed template set need not be closed under negation) for 1 bit per block; the 1 bit saved from the index is exactly cancelled by the selector bit, so the whole gain is in the dictionary. This is lossless *because the weight is decoded*, and it costs one operation per block in the kernel. bitnet.cpp's TL2 index, "a 1-bit sign (0 or 1) and a 4-bit index" for three trits, is consistent with exactly this canonicalization: 27 patterns do not fit in 4 bits, but the 13 sign-paired patterns plus the all-zero pattern, 14 in total, do. That reading is this project's inference; the report does not say it in those words.
- **As an invariant that is never decoded.** Only identity and sign flip can be pushed onto the inputs. Checked: exactly 2 of the 6 permutations are affine on `S`, and exactly those 2 are undone by an input sign `x -> c*x`. Compensation by an arbitrary function `phi(x)` is ruled out by argument, not by the script: every other permutation moves zero (checked), so a stored `t(0) != 0` would need `t(0) * phi(x) = 0` for all `x`, forcing `phi = 0`, while a stored `t(1)` needs `t(1) * phi(x) = x`. If the only compensation allowed is one scale per input column and one per output row, then `r_i * f_ij * c_j = 1` for every entry, hence `f_ij = r_i * c_j`: under that rule an undecoded sign region costs `rows + columns` bits, one per row and one per column, not one per block (checked on a 2x3 factorable pattern and a 2x2 pattern that no row/column choice reproduces). A per-block sign that does not factor can still avoid decoding weights if the kernel negates each block's partial sum instead, at one negate per block per output row; that is a decode step on the accumulator rather than on the weights.

Ledger: `P` = `ceil(log2 K)/g` plus selector bits per weight; `D` = `K * g * log2(3)` bits total; random access by block; decode = one template read plus one transform per block; actionable: dot product via per-template partial sums (a lookup-table variant), otherwise decode first.

### 4.6 Escape and exception region (derived; published analogues)

A 2-bit code has four values and a trit uses three. The fourth can be a **pointer**: "the real value lives in the exception region", indexed by the rank of an escape mask. Script example with 10,000 weights, 115 int8 outliers, and 2,996 true zeros (checked, both variants round-trip):

| escape convention | bits per weight |
|---|---:|
| spare fourth code points to the exception region | 2.159 |
| trit zero points to the exception region (true zeros must also be exceptions) | 4.141 |

This is the cleanest place to see why the meaning of zero must be chosen deliberately. The logical family's `0` means "unknown, resolve elsewhere". The numerical family's `0` means "contributes nothing". Giving `0` the pointer meaning charges eight bits for every numerical zero. The spare code keeps the two meanings apart at no cost inside `P` *when `P` already uses fixed 2-bit codes*; against a packed ternary `P` the fourth code costs `log2(4) - log2(3) = 0.415` bits per weight, which is exactly the gap between the 2-bit and 1.585-bit rows of the table in section 3.

Published analogues: CHERI's one tag bit per 128 or 256 data bits and AddressSanitizer's one shadow byte per eight application bytes (both quoted in 4.1) are small out-of-band regions that change how the main region is interpreted: whether a word is a valid pointer, whether a byte may be touched. Same mechanism, different purpose. The shadow byte is itself a small code with an escape flavour: "All 8 bytes in qword are unpoisoned (i.e. addressable). The shadow value is 0", "First k bytes are unpoisoned, the rest 8-k are poisoned. The shadow value is k", and a negative value means all eight are poisoned (sanitizers wiki).

Ledger: `P` = 2 bits/weight; `D` = 8 bits per exception plus 0.063 bits/weight directory; random access yes via rank; decode = one branch per weight; actionable: dot product with an exception branch.

## 5. The three NOTs, revisited as region transforms (derived)

`operator-reference.md` section 3 found exactly six reversible unary maps on `S`. Inside a region layout each one is a candidate entry in a transform selector. The script sorts them:

| permutation of `S` | affine over the integers | usable as an invariant compensated on inputs | usable as a decode-time transform |
|---|---|---|---|
| identity | yes | yes | yes |
| sign flip, swap `-1` and `+1` | yes (`a = -1`) | yes, with `x -> -x`, factoring as row times column signs | yes |
| swap `0` and `+1`, fix `-1` | no | no | yes |
| swap `-1` and `0`, fix `+1` | no | no | yes |
| cycle plus | no | no | yes |
| cycle minus | no | no | yes |

Conclusion: the two zero-moving "NOTs" are decode-time transforms of equal cost; the third, the sign flip, is the one that can also leave the weight region entirely and become a one-bit-per-row-and-column invariant. Storing a free choice among all six costs `log2(6) = 2.585` bits per block as an ideal fractional cost, 3 bits as a flat code (0.162 or 0.188 bits per weight at `g = 16`); storing only a sign flip costs 1 bit per block (0.0625 at `g = 16`). The unwrapped "add one" of section 4.3 is not a permutation of `S` at all but a map from `S` onto `{0, 1, 2}`; it is the only map in this section besides identity and sign flip that is affine, which is what lets the correction in 4.3 be a single shared term.

## 6. Where regions live in the memory hierarchy (inference; published analogues)

Registers and first-level cache, then second and third level caches, then DRAM, then SSD or flash: each step is larger and slower. The design rule this document proposes is **keep `D` resident, stream `P`**. The value of `D` is then measured in bytes of `P` that are *not* read (skipped zeros, prefetched windows, table hits) and in operations replaced (lookup instead of multiply). Section 4.1 is the exception where the guiding mask is itself large; there the rule applies to the directory.

`technical-research.md` already records the published analogue for the outermost level: *LLM in a flash* keeps a neuron-activity window that decides which rows to fetch from flash. In this vocabulary that window is a predictor region, `D` deciding which parts of `P` to touch. What to measure, from the same document: bytes read, number and size of requests, decode time, cache-hit rate, and, added here, the resident size of `D`.

For inference beyond weights, activations and the attention cache change every token, so a weight-side `D` cannot describe them. The lookup table of section 4.4 is the one region in this document that is built on the activation side; whether an activation-side region is the entry point for "AI inference memory" more broadly is an open question in section 11.

## 7. History and neighbours (published)

Balanced ternary hardware is not new. Setun "was a computer developed in 1958 at Moscow State University", "the first modern ternary computer, using the balanced ternary numeral system and three-valued ternary logic", with words of 18 trits; "Fifty computers were built from 1959 until 1965" (Wikipedia; the same page's infobox says released 1959, and both dates are kept rather than resolved). Brusentsov's recorded reason was component count rather than arithmetic elegance: ternary "allowed him to create very simple and reliable elements, and he needed only one seventh as many elements".

Ternary weights predate BitNet. Ternary Weight Networks "constrain the weights to be ternary-valued: +1, 0 and -1", set a weight to zero when its magnitude is below a threshold with the rule of thumb "Δ* ≈ 0.75E(|W|)", and claim "up to 16× model compression rate" against float32 (Li and Liu 2016). Trained Ternary Quantization adds "two quantization factors W^p_l and W^n_l for positive and negative weights in each layer", trained "together with other parameters" (Zhu et al. 2017): in this document's vocabulary, a per-sign scale region of two numbers per layer. Both are recorded so that the project does not rediscover them, and the 0.75 threshold is the rule stage A in section 10 uses.

## 8. Results of `experiments/region_encoding.py`

Run from the project root with `python experiments/region_encoding.py`. It uses the standard library, a fixed seed of 20260912, and writes `outputs/region-encoding.json`. Each failed check prints the inputs it failed on; equality checks also print both sides, and list comparisons print the first differing index. That is the diagnostic improvement requested in `verification.md`, applied to this script; `verify_operators.py` still reports by name only.

The 24 checks, grouped:

- **Information budget** (5): uniform entropy equals `log2(3)`; five trits fit one byte; mask+sign costs `1 + d` and never beats the bound; equality only at `d = 0.5`; overhead equals `1 - H_b(d)`.
- **Permutations** (3): exactly identity and sign flip are affine (the search over `a, b` in `-4..4` is exhaustive because `a = t(1) - t(0)` and `b = t(0)`); exactly those are undone by an input sign; every other permutation moves zero, which is the premise of the argument in section 4.5.
- **Dot-product identities** (4): unsigned digits plus zero point; digits lie in `{0, 1, 2}`; positive minus negative planes; mask and sign planes.
- **Rank access** (2): every weight of a 4,096-weight example recovered through the two-level directory and compacted sign plane; ranks across a superblock boundary agree with a brute-force count, with relative counts under sixteen bits.
- **Escape codes** (3): spare-code and zero-as-pointer round-trips; the zero-as-pointer variant costs more when zeros are common.
- **Lookup table** (2): table sum equals the dot product; two- and three-trit indices fit four and five bits.
- **Dictionary on random blocks** (2): flat-index totals never fall below the bound on uniform or sparse samples.
- **Sign factorization** (3): rank-one pattern factors; a non-rank-one pattern does not; row and column compensation reproduces a matrix-vector product.

Limitations recorded in the JSON: the entropy bounds assume i.i.d. trits with balanced signs; the dictionary rows are one seeded sample on random data under pessimistic accounting, not a proof and not a statement about trained weights; bit counts exclude scales, padding, alignment, and accumulators; no kernel was timed.

## 9. Self-study kit

### Glossary

- **Entropy**: the average number of bits needed per symbol by the best possible code; a lower bound on the average, not a recipe.
- **Binary entropy `H_b(p)`**: the entropy of a yes/no answer that is "yes" with probability `p`.
- **Density `d`**: fraction of nonzero weights.
- **Bit plane**: a region holding one bit from every weight (for example, all the signs).
- **Rank / select**: `rank(j)` counts set bits before position `j`; `select(k)` finds the position of the `k`-th set bit. Together they let a mask index a compacted array.
- **Popcount**: the number of set bits in one machine word; one instruction on processors that have it, and a rank over a 256-bit block needs up to four of them.
- **Zero point**: the constant added so that stored digits are unsigned; corrected by one shared term.
- **Lookup table**: precomputed results indexed by a code, replacing arithmetic with addressing.
- **Template / codebook**: a dictionary of reusable blocks referenced by index.
- **Affine map**: `f(w) = a*w + b` with constants `a` and `b`; the only maps a linear layer can absorb.
- **Rank-one sign pattern**: `f_ij = r_i * c_j`; the only sign patterns absorbable by inputs and outputs.
- **Out-of-band**: stored in a separate region that ordinary reads of the main region do not see.

### Exercises

Answers follow each exercise; the script confirms the general statements.

1. Compute `H(0.25)`. *Answer:* `H_b(0.25) = 0.811`, plus `0.25`, gives `1.061` bits per weight.
2. At what density does mask+sign (`1 + d`) cost the same as five trits per byte (1.6)? *Answer:* `d = 0.6`. Below it, planes win on bits; above it, packing wins. With the 0.063 rank directory included the crossover moves down to about `d = 0.54`.
3. Check the zero-point identity for `w = (-1, 0, 1, 1)` and `x = (3, -2, 5, 4)`. *Answer:* `w.x = -3 + 0 + 5 + 4 = 6`; `u = (0, 1, 2, 2)`; `u.x = 0 - 2 + 10 + 8 = 16`; `sum(x) = 10`; `16 - 10 = 6`.
4. Why can the swap of `0` and `+1` not be compensated on the input side? *Answer:* it stores a true `+1` as `0`, and `0 * phi(x) = 0` for every `phi`, so a true `+1` can never contribute `x`.
5. Give a 2x2 sign pattern that factors into row and column signs, and one that does not. *Answer:* `[[1, -1], [-1, 1]]` factors with `r = (1, -1)` and `c = (1, -1)`. `[[1, 1], [1, -1]]` cannot: for any `r_i * c_j` the product of all four entries is `(r_1 r_2 c_1 c_2)^2 = +1`, but this pattern's product is `-1`.
6. Build the nine-entry lookup table for the input block `(2, -3)` and read off the entry for weights `(-1, +1)`. *Answer:* `(-1)*2 + (+1)*(-3) = -5`. The other eight entries are the remaining sums of `{-2, 0, 2}` and `{3, 0, -3}`.

## 10. Bounded next experiments (proposal)

| Stage | Experiment | Pass / fail evidence | Cap |
|---|---|---|---|
| A. Real block statistics | Take one ternary linear-layer tensor: a published BitNet b1.58 checkpoint's weights, or a small open model ternarized with the Ternary Weight Networks threshold rule, zero below 0.75 times the mean absolute weight. Compute `d`, sign balance, `H(d)`, and distinct-block counts for `g` in `{4, 8, 16}`. Compare with the random baseline table in section 4.5. | If distinct blocks approach `min(3^g, N/g)`, the dictionary route is closed for that tensor. If far fewer, report bits per weight *including* dictionary cost, under both the flat and the entropy-coded accounting. | at most 1 GB download, 30 minutes CPU, no training |
| B. Bytes touched | Implement the mask+sign path and the 2-bit path on one matrix in plain Python; count bytes read and operations, not wall time. | Bytes read match the ledger prediction for each path. | 10 minutes |
| C. Escape rate | On a real int8 tensor, choose ternary thresholds and measure the outlier fraction and resulting bits per weight under the spare-code escape. | Under 2.5 bits per weight with under 1% escaped, else the design is not worth it for that tensor. | 30 minutes |
| D. Kernel measurement | Only after A to C: time a lookup-table kernel against bitnet.cpp TL1/TL2 at a pinned commit, with hardware recorded. | Matched shapes, matched precision, reported bytes and time separately. | separate decision |

Stop condition: if stage A shows no repetition beyond the random baseline, drop section 4.5 and keep 4.1, 4.3, 4.4, and 4.6.

## 11. Open questions

1. What are the real density and sign balance in trained ternary layers? Every number in section 3 depends on them.
2. Is the "region" in the owner's idea a block of weights (4.5), a mask (4.1), or a table (4.4)? All three fit the phrase and have different ledgers.
3. Should `D` be per tensor, per row, or per block? Row-level regions allow invariant row signs; block-level regions require decoding, or a per-block negate on the accumulator.
4. For activations and the attention cache the data changes every token. Is an activation-side region, like the lookup table, the right entry point for inference memory beyond weights? (inference)
5. Do unsigned digits actually cost less to multiply on the hardware that matters here? Unmeasured; no claim is made. (inference)

## 12. Claims ledger

| # | Claim | Label |
|---|---|---|
| 1 | `H(d) = H_b(d) + d` bounds, on average, any i.i.d. ternary layout with balanced signs; unbalanced signs lower it to `H_b(d) + d*H_b(p_neg)` | derived, checked at five densities |
| 2 | Mask+sign costs `1 + d`; overhead `1 - H_b(d)` | derived, checked |
| 3 | Rank with a two-level directory gives random access at 0.063 bits/weight | derived, checked on one 4,096-weight example and one superblock-boundary example |
| 4 | Dot product equals positive sum minus negative sum, and mask sum minus twice negative sum | derived, checked on 300 trials |
| 5 | `sum(w x) = sum(u x) - sum(x)` with `u = w + 1` | derived, checked on 300 trials |
| 6 | A `3^g` table per `g` inputs replaces multiplication | derived, checked at `g = 3` |
| 7 | Flat-index dictionary plus full dictionary never beat the bound on random blocks in one sample; the general statement is the source-coding theorem | derived, one seeded sample under pessimistic accounting |
| 8 | Exactly identity and sign flip are affine and undone by an input sign | derived, exhaustive over 6 permutations for affineness and input-sign compensation; arbitrary `phi` ruled out by the `w = 0` argument in 4.5 |
| 9 | Under row-and-column compensation, an undecoded sign flip must factor as row times column signs | derived given that compensation rule; checked on two patterns |
| 10 | Spare-code escape beats zero-as-pointer when zeros are common; the spare code itself costs 0.415 bits/weight over packed ternary | derived, one seeded sample |
| 11 | MX, TQ1_0, 2:4 sparsity, CHERI tags, ASan shadow, T-MAC, TL1/TL2, Setun, TWN, TTQ are published instances or neighbours of the pattern | published; verbatim quotes with fetch times in `work/region/source-findings.md` |
| 11a | TL2's sign bit plus 4-bit index is a sign-canonicalized dictionary | inference from the published description |
| 12 | Keep `D` resident, stream `P` | inference |
| 13 | Stages A to D | proposal |

## Sources

Primary sources are quoted in `work/region/source-findings.md` with fetch times and methods, and the raw captures sit beside it; the list below is the reading order for a learner. Five requested facts were not found in the captured pages and are listed at the end of the findings file rather than filled in from memory.

- Rouhani et al., *Microscaling Data Formats for Deep Learning*, 2023. https://arxiv.org/abs/2310.10537
- Wei et al., *T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge*, 2024. https://arxiv.org/abs/2407.00088
- Wang et al., *1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs*, 2024. https://arxiv.org/abs/2410.16144
- llama.cpp pull request 8151, ternary types TQ1_0 and TQ2_0. https://github.com/ggml-org/llama.cpp/pull/8151
- Mishra et al., *Accelerating Sparse Deep Neural Networks*, 2021. https://arxiv.org/abs/2104.08378
- CHERI project pages, University of Cambridge. https://www.cl.cam.ac.uk/research/security/ctsrd/cheri/
- AddressSanitizer algorithm wiki. https://github.com/google/sanitizers/wiki/AddressSanitizerAlgorithm
- Setun, Wikipedia. https://en.wikipedia.org/wiki/Setun
- Succinct data structure, Wikipedia. https://en.wikipedia.org/wiki/Succinct_data_structure
- Li and Liu, *Ternary Weight Networks*, 2016. https://arxiv.org/abs/1605.04711
- Zhu et al., *Trained Ternary Quantization*, 2017. https://arxiv.org/abs/1612.01064
- BitNet, LLM in a flash, FlexGen, and the IBM analog chip: see `technical-research.md`.
