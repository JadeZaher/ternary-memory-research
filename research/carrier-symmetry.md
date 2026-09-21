# The symmetry group of the ternary carrier, version 0.1

Contributed 2026-09-21 by Mitchell (agentprivacy.org) as a pull request; not yet reviewed by the owner. Status: definitions and one exhaustive finite-domain check. `experiments/verify_carrier_symmetry.py` passes 27 named checks; its tables are in `outputs/carrier-symmetry.json`. Labels follow `AGENTS.md`: **derived** means proved or exhaustively checked on the finite domain; **observation** means a correspondence noted, not a result; nothing here is a measured result about a model.

## 0. Why this note exists

`research/operator-reference.md` section 3 enumerates the six reversible unary maps on `S = {-1, 0, +1}`, names the three non-identity involutions `N_c(x) = bal(c - x)`, and shows that only `N_0` reverses the order. `research/region-memory.md` section 4.5 and claim 8 show that exactly identity and the sign flip can be pushed onto the inputs of a linear layer. This note fills the gap between those two facts: what group the six maps form, how the three swaps compose, and, for each of the three operator families the reference keeps distinct, which of the six maps is a symmetry, which is a dual, and which can be moved onto one input.

## 1. Domain and notation (as in the reference)

- `S = {-1, 0, +1}` with the order `-1 < 0 < +1`.
- `bal(n) = ((n + 1) mod 3) - 1`, the member of `S` congruent to `n` modulo 3.
- The six permutations: `identity`; `N_0(x) = -x` (swap `-1` and `+1`, fix `0`); `N_+1(x) = bal(1 - x)` (swap `0` and `+1`, fix `-1`); `N_-1(x) = bal(-1 - x)` (swap `-1` and `0`, fix `+1`); `cycle_plus(x) = bal(x + 1)`; `cycle_minus(x) = bal(x - 1)`.
- Three operator families, never merged: ordered logic (`min`, `max`); signed arithmetic (the product `xy`, which is the table labelled XOR in the supplied image); arithmetic modulo 3 (`bal(x + y)`).

## 2. The group (derived, exhaustive)

| fact | statement | check |
|---|---|---|
| closure | composing any two of the six maps gives one of the six; there is an identity and every map has an inverse | 36 compositions |
| orders | `identity` and the three `N_c` are involutions; the two cycles have order 3; four involutions counting identity | as the reference states |
| non-abelian | `N_0 ∘ N_-1 ≠ N_-1 ∘ N_0` | one pair |
| composition law of the swaps | `N_c ∘ N_d = (x -> bal(c - d + x))`: two swaps compose to the shift by the difference of their fixed-point labels | all 9 pairs `(c, d)` × 3 inputs |
| presentation | `N_0² = identity`, `cycle_plus³ = identity`, `N_0 ∘ cycle_plus ∘ N_0 = cycle_minus`; `N_0` and `cycle_plus` generate all six | closure of the generated set |

The group is therefore the symmetric group on three elements, which is also the dihedral group of the triangle, the smallest non-abelian group. The three swaps are its reflections and the two cycles its rotations. Two consequences the reference does not state:

1. **`N_0 ∘ N_-1 = cycle_plus`** and **`N_+1 ∘ N_0 = cycle_plus`**: the sign flip composed with the swap that fixes `+1` is the successor `x -> bal(x + 1)`, and the same successor arises from the sign flip followed by the swap that fixes `-1`. This is the special case `c - d = 1` of the composition law.
2. **There is no unique "three NOTs", but there is a unique reflection class:** all three `N_c` are conjugate in the group (each is `cycle^k ∘ N_0 ∘ cycle^-k` for some `k`), so as abstract symmetries they are indistinguishable. What distinguishes `N_0` is not group-theoretic but order-theoretic: it is the only one of the six that reverses `-1 < 0 < +1`, and the only non-identity map that fixes `0`. Every use of "negation" in the reference rests on those two properties, not on being an involution.

## 3. The commutation table (derived, exhaustive)

For each map `f` and operator `∘`, three properties over all nine pairs: **automorphism** `f(x ∘ y) = f(x) ∘ f(y)`; **anti-automorphism** (`min`/`max` only) `f(min(x, y)) = max(f(x), f(y))` and dually; **pushable onto one input** `f(x ∘ y) = f(x) ∘ y = x ∘ f(y)`.

| map | `min` / `max` | signed product `xy` | addition modulo 3 |
|---|---|---|---|
| `identity` | automorphism | automorphism, pushable | automorphism, pushable |
| `N_0` (sign flip) | **anti-automorphism** (De Morgan) | **pushable** onto one input; not an automorphism | **automorphism**; not pushable |
| `N_+1` | none | none | none |
| `N_-1` | none | none | none |
| `cycle_plus` | none | none | **pushable** onto one input; not an automorphism |
| `cycle_minus` | none | none | **pushable** onto one input; not an automorphism |

Readings, each checked rather than argued:

- Under ordered logic the only symmetry is the identity and the only duality is the sign flip; the reference's De Morgan laws are the whole story.
- Under the signed product the sign flip is the only non-trivial map that can be moved onto one input. The four zero-moving maps have no property at all. This is the algebraic form of region-memory claim 8 and of the section 4.5 argument that a stored `t(0) ≠ 0` cannot be compensated by any input function: the script asserts that the product-pushable set equals the zero-fixing set.
- Under addition modulo 3 the roles swap: the sign flip is an automorphism (a symmetry of the ring's additive group) and the two **cycles** are the maps that can be pushed onto one input, because a shift of the sum is a shift of either summand. A representation that accumulates modulo 3 therefore has a different free-transform set from one that accumulates ordinary signed integers; the reference's insistence on keeping the families distinct is exactly what makes this visible.

Practical reading for the region idea: a "NOT" stored per region is free at decode time under the signed product only if it is the sign flip; under a modulo-3 accumulator only if it is a cycle. No single map is free under both, and no zero-moving swap is free under either family that a dot product uses.

## 4. An observation from outside (observation, fenced)

The agentprivacy dual-agent harness proves, for the 64-element ring `Z/(2^6)Z`, that two involutions compose to the successor: `neg(bnot(x)) = succ(x)`. The ecosystem orientation (`research/agentprivacy-orientation.md`) correctly recorded that this is not a law of balanced ternary. Section 2 shows the triangle has an identity of the same shape with different ingredients: `N_0 ∘ N_-1 = cycle_plus`. In both cases a reflection composed with a reflection is a rotation, which is a property of every dihedral group and of nothing else here. The observation carries no claim about memory, speed or model behaviour, and the ring-side conjectures of that ecosystem stay on their own domain.

## 5. What this note does not claim

No statement about trained weights, kernels, packed bytes, cache traffic, accuracy or hardware. The six maps are symmetries of a three-element set; whether any of them is worth storing per region is the cost question of `research/region-memory.md`, which this note does not reopen.

## 6. Claims ledger

| # | claim | status |
|---|---|---|
| 1 | The six permutations of `S` form a group of order 6, non-abelian, generated by `N_0` and `cycle_plus` | derived, exhaustive |
| 2 | `N_c ∘ N_d = x -> bal(c - d + x)` for all `c, d` | derived, exhaustive over 27 cases |
| 3 | `N_0 ∘ N_-1 = cycle_plus` and `N_+1 ∘ N_0 = cycle_plus` | derived, exhaustive |
| 4 | `N_0` is the only order-reversing bijection and, with identity, the only zero-fixing map | derived, exhaustive (restates the reference) |
| 5 | Commutation table of section 3, 24 cells | derived, exhaustive over 9 pairs per cell |
| 6 | Product-pushable set = zero-fixing set = {identity, `N_0`}, agreeing with region-memory claim 8 | derived, exhaustive |
| 7 | Modulo-3 addition: automorphisms {identity, `N_0`}; pushable {identity, both cycles} | derived, exhaustive |
| 8 | The `Z/(2^6)Z` identity and claim 3 share the dihedral shape | observation only |

## 7. How to run

```
python experiments/verify_carrier_symmetry.py
```

Prints `{"status": "passed", "check_count": 27}` and rewrites `outputs/carrier-symmetry.json` with the composition table, the zero-moving flags and the full commutation table. A failing check names the assertion and, for the `verify_all` checks, the first failing input with both sides.

## Sources

- `research/operator-reference.md` version 0.1, section 3 (the six maps, `N_c`, order reversal) and section 8 (verification contract).
- `research/region-memory.md` version 0.2, section 4.5 and claim 8 (pushable onto inputs; zero-moving maps).
- `research/independent-review.md` (the request that failed checks print inputs and both sides).
- Dihedral group of the triangle: any introductory group-theory text; the presentation `⟨r, s | r² = s³ = (rs)² = 1⟩` is standard.
