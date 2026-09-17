# Ternary operator reference, version 0.1

Prepared 2026-09-11. Status: initial mathematical specification, independently reviewed and checked by the finite-domain validation pass recorded in `verification.md`. The tables below are definitions and direct finite calculations, not performance results.

The symbols `-1`, `0`, and `+1` do not determine what an operation means. First choose their meaning, then choose the operation. A useful starting point is to maintain three separate families:

| Family | Meaning of zero | What the operations do |
|---|---|---|
| Ordered three-valued logic | Unknown or undetermined truth | Extend familiar AND, OR, and NOT while preserving an unresolved state |
| Signed arithmetic | The ordinary number zero | Add, subtract, multiply, and accumulate numeric quantities |
| Arithmetic modulo 3 | One of three numerical residues | Wrap results back into three states; useful for finite algebra and encoding |

Do not silently interchange these families. In particular, an unknown logical value and a numerical zero have different meanings even when written with the same character.

## 1. The domain and truth encoding

Use the domain `S = {-1, 0, +1}`. In the logical family, order it as `-1 < 0 < +1` and interpret `-1 = false`, `0 = unknown`, and `+1 = true`. Binary truth values enter this family through `false -> -1` and `true -> +1`; they do not use the subset `{0,1}`.

Rows below are the first argument `x`; columns are the second argument `y`.

## 2. Ordered AND and OR

AND takes the smaller value: `and_min(x,y) = min(x,y)`.

| AND | -1 | 0 | +1 |
|---|---:|---:|---:|
| **-1** | -1 | -1 | -1 |
| **0** | -1 | 0 | 0 |
| **+1** | -1 | 0 | +1 |

OR takes the larger value: `or_max(x,y) = max(x,y)`.

| OR | -1 | 0 | +1 |
|---|---:|---:|---:|
| **-1** | -1 | 0 | +1 |
| **0** | 0 | 0 | +1 |
| **+1** | +1 | +1 | +1 |

These tables explain why `false AND unknown = false`, while `true AND unknown = unknown`: a known false input already settles the first result. Similarly, `true OR unknown = true`.

With the sign-reversing NOT below, these are the usual strong Kleene tables for AND, OR, and NOT. They form a distributive ordered three-state algebra with De Morgan's laws. The name is secondary to the explicit tables: alternative three-valued logics exist.

## 3. One familiar NOT, and three possible reversible swaps

The attached NOT table defines one operation, `not_sign(x) = -x`, with three input/output cases. It swaps false and true and leaves unknown unresolved.

| x | `not_sign(x)` |
|---|---:|
| -1 | +1 |
| 0 | 0 |
| +1 | -1 |

There is also a precise interpretation of the idea of **three different NOT operations**: all three nonidentity reversible swaps. Each swaps two values, fixes the third, and returns the original value when applied twice.

| x | Swap -1 and +1; fix 0 | Swap 0 and +1; fix -1 | Swap -1 and 0; fix +1 |
|---|---:|---:|---:|
| -1 | +1 | -1 | 0 |
| 0 | 0 | +1 | -1 |
| +1 | -1 | 0 | +1 |

These are three distinct **involutions**: an involution is a function `f` for which `f(f(x)) = x`. Only the first reverses the stated order and swaps the designated false and true values. The others are useful state transformations, but should not automatically be called logical negation under this truth encoding.

For a common formula, define `bal(n)` as the member of `S` congruent to integer `n` modulo 3. With a nonnegative remainder, `bal(n) = ((n + 1) mod 3) - 1`. Then the three swaps are `N_c(x) = bal(c - x)` for `c = 0, +1, -1`, respectively. Languages whose `%` can return a negative remainder need explicit normalization.

There are also two cyclic shifts:

| x | `cycle_plus(x) = bal(x+1)` | `cycle_minus(x) = bal(x-1)` |
|---|---:|---:|
| -1 | 0 | +1 |
| 0 | +1 | -1 |
| +1 | -1 | 0 |

Each shift requires three applications to return to its starting state; the two shifts undo each other. Calling a shift NOT would use a different convention from the binary idea that NOT twice restores the input.

The complete count is small enough to enumerate: `3^3 = 27` unary functions, because each of three inputs independently has three possible outputs. Exactly `3! = 6` are reversible permutations: identity, these three swaps, and these two cycles. Exactly four permutations are involutions when identity is included. There is no unique set of "three NOTs" without stating the requirements.

## 4. The image labelled XOR is signed multiplication

The attached table is exactly `multiply_signed(x,y) = x*y`:

| Signed product / logical XNOR | -1 | 0 | +1 |
|---|---:|---:|---:|
| **-1** | +1 | 0 | -1 |
| **0** | 0 | 0 | 0 |
| **+1** | -1 | 0 | +1 |

Numerically, zero removes a product: `x*0 = 0`. Logically, using the chosen truth encoding, equal known inputs return true and different known inputs return false. That is XNOR, the negation of XOR. An unresolved input returns unknown.

The corresponding strong Kleene XOR is `xor_kleene(x,y) = -x*y`:

| Logical XOR | -1 | 0 | +1 |
|---|---:|---:|---:|
| **-1** | -1 | 0 | +1 |
| **0** | 0 | 0 | 0 |
| **+1** | +1 | 0 | -1 |

It is also exactly `OR(AND(x,NOT(y)), AND(NOT(x),y))` with the specified min, max, and sign-NOT tables. This equality must be checked on all nine input pairs. These tables are appropriate for an unknown-preserving extension, but other definitions of ternary XOR are possible.

Two more familiar operations follow without guessing new semantics:

| x,y operation | Definition | Plain meaning |
|---|---|---|
| NAND | `-min(x,y)` | NOT of AND |
| NOR | `-max(x,y)` | NOT of OR |
| XNOR | `x*y` | NOT of this XOR |

Do not interpret an unknown result as proof of falsehood. At `x=0`, both `OR(x,NOT(x))` and `AND(x,NOT(x))` are `0`. Thus the classical formulas "x or not x is always true" and "x and not x is always false" do not hold at the unknown value under these tables. This is the specified behavior, not a coding error.

## 5. Arithmetic modulo 3

Use the same written values but interpret `-1` as the residue ordinarily called `2`. Define `add_mod3(x,y) = bal(x+y)` and `multiply_mod3(x,y) = bal(x*y)`.

| Addition modulo 3 | -1 | 0 | +1 |
|---|---:|---:|---:|
| **-1** | +1 | -1 | 0 |
| **0** | -1 | 0 | +1 |
| **+1** | 0 | +1 | -1 |

Multiplication modulo 3 has the same table as signed multiplication, because every product of two members of `S` is already in `S`. **Their accumulations differ.** Modulo-3 addition wraps `1+1` to `-1`. Ordinary numerical addition keeps `1+1 = 2`.

| Subtraction modulo 3, `bal(x-y)` | -1 | 0 | +1 |
|---|---:|---:|---:|
| **-1** | 0 | -1 | +1 |
| **0** | +1 | 0 | -1 |
| **+1** | -1 | +1 | 0 |

These addition and multiplication tables are the finite field commonly written `F3`. Here zero is the additive identity, each nonzero value has a multiplicative inverse, and multiplication distributes over addition. Addition modulo 3 is sometimes a natural ternary counterpart to a binary parity operation, but it is not the XOR table above. Ordinary binary XOR has `x XOR x = false`; in this arithmetic `1+1 = -1`, not zero.

For balanced ternary **numerals**, carries preserve ordinary integer values. For example, ordinary `1+1 = 2` becomes a low digit `-1` and a carry `+1`, since `2 = (-1) + 3*(+1)`. Ordinary `-1 + -1 = -2` becomes a low digit `+1` and a carry `-1`. Dropping the carry gives modulo arithmetic; keeping it gives ordinary arithmetic represented with ternary digits.

## 6. Neural-network dot products need a wider sum

For ordinary matrix multiplication with ternary weights, `y_i = sum_j(W_ij*x_j)` uses ordinary numerical addition. A ternary weight selects `-x_j`, `0`, or `+x_j`, but the sum is not generally ternary. Two unit contributions produce 2. This is where the supplied signed-product table is directly useful.

With ternary weights and ternary inputs, a dot product of length `n` lies in `[-n,n]`; an accumulator must represent all `2n+1` outcomes. With integer inputs bounded by absolute value `A`, the range is `[-nA,nA]`. A signed binary integer accumulator needs at least `1 + ceil(log2(nA+1))` bits for that symmetric range when `nA >= 1`. Scaling, rounding, or clipping back to three values is a separate, lossy operation requiring a specified rule.

Matrix multiplication using modulo-3 sums is a valid different computation. Replacing a neural network's ordinary sums with it changes the model and requires independent accuracy evidence.

## 7. A concrete meaning for transforming weight regions

For a matrix `W`, let `D = diag(s)` with every sign `s_j` equal to `-1` or `+1`. Then `D*D = I` and `(W*D)*(D*x) = W*x`. A column sign change can therefore be paired with the same input sign change to preserve a linear result. Allowing `s_j=0` destroys invertibility; it discards information unless the discarded contribution was already irrelevant.

Likewise, a permutation matrix `P` simply reorders channels, and `(W*P^-1)*(P*x) = W*x`. These are exact ways to change a representation while preserving a linear computation. They do not make arbitrary swaps of `-1`, `0`, and `+1` harmless: swaps that move zero generally do not preserve ordinary multiplication or addition.

Across multiple layers, every affected connection must be compensated. Elementwise ReLU commutes with a channel permutation, but does not commute with sign reversal: `ReLU(-1)=0`, while `-ReLU(1)=-1`. An operation that preserves one linear layer is not automatically a symmetry of a whole transformer.

## 8. Exhaustive verification contract

Use names containing the family (`and_min`, `xor_kleene`, `add_mod3`) instead of overloaded labels. Reject values outside `S` at these operator interfaces; an accumulator has a separate numeric interface. Verify all relevant inputs, not a sample.

| Check | Exhaustive domain | Required result |
|---|---|---|
| Every printed binary table | All 9 input pairs per operation | Exact agreement with its definition |
| AND/OR commutativity, identities, absorption | All 9 pairs and all 3 values as appropriate | Identities are +1 for AND and -1 for OR |
| AND/OR associativity and mutual distributivity | All 27 triples | Every equality holds |
| Sign-NOT and De Morgan | All 3 values / all 9 pairs | Double negation and both De Morgan equalities hold |
| XOR expression | All 9 pairs | Min/max definition equals `-x*y` |
| Boolean endpoint restriction | All 4 pairs in `{-1,+1}^2` | Standard AND, OR, XOR, XNOR after decoding |
| Unknown counterexamples | `x=0` | Excluded middle and contradiction expressions both return 0 |
| All unary functions | All 27 output triples | Exactly 6 bijections; exactly 4 involutions including identity; exactly 2 permutations of order 3 |
| Three `N_c` transforms | All 9 pairs of c and x | Correct fixed point and involution; only `N_0` reverses the fixed order |
| Cyclic shifts | All 3 values | Third application is identity; plus and minus undo each other |
| Modulo arithmetic laws | All 27 triples / all 9 pairs | Associativity, distributivity, identities, additive inverses, nonzero multiplicative inverses |
| Ordinary versus modulo accumulation | Two unit products | Ordinary dot product is 2; modulo dot product is -1 |
| Balanced addition with carry | All 9 digit pairs | `x+y = digit+3*carry`, with digit and carry in `S` |
| Region invariance | Small exact integer matrix examples, including zeros | Paired signs and permutations preserve output; unpaired changes can change it |

Each failed law should print a counterexample with inputs and both sides. A table matching its own implementation is insufficient by itself: check the independent laws, endpoint meaning, and deliberately different arithmetic results too. Passing these tests establishes consistency of this specification; it does not establish compression, learning quality, speed, or energy savings.
