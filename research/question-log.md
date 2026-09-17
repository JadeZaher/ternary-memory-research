# Questions and decisions

## Confirmed by the owner

- Project name: `ternary-memory-research`.
- Desired folder: under the existing Programming directory.
- Priority: mathematics and operator design.
- Experience: novice in this topic; add missing representations and explain them.
- Interests: −1/0/+1 memory and weights, operator representations within regions of weights, shape and representation, matrix multiplication, transformers, disk-assisted operations, analog computation, and Microsoft BitNet.

## First research question

Can we define a family of three-state operators with clear semantics and algebraic laws, then identify an exact matrix transformation or a measurable representation saving that uses those laws?

## Decisions still open

| Question | Why it matters | Current treatment |
|---|---|---|
| Is 0 unknown truth, neutral numerical weight, or one residue of a finite field? | The meaning determines valid operator laws. | Compare the alternatives; do not merge them. |
| Does “three NOTs” mean the three possible swaps or three cases of a single inversion? | Both are sensible descriptions but define different objects. | Include all three swaps; reserve ordered negation for sign inversion. |
| What is a region? | A block, channel, sparse support, codebook group, or geometric pattern carries different metadata. | Start with a small matrix block, then define more structure if useful. |
| What counts as a useful shape? | A visual pattern need not support economical decoding or a useful invariant. | Seek an explicit transformation, its inverse where applicable, and its byte cost. |
| Are weights fixed, fine-tuned, or trained under a ternary constraint? | Accuracy and storage depend on how ternary structure arises. | Unresolved; published BitNet is an application reference. |
| What is meant by “better for AI”? | A new operator, accuracy, bytes, bandwidth, latency, energy, and capacity are different objectives. | First success is a correct operator specification and a concrete derivation. |
| What hardware is available? | Later performance questions depend on RAM, storage, processor, and kernels. | Unspecified; no benchmark claims. |
| Is there an existing private City Key journey? | Needed for supported continuity operations, not for local research. | Unknown; no key requested or used. |
| Who may receive research output? | Public sources do not imply permission to share the user's work. | Local/private default; no external recipient agreed. |

## Two bounded next routes

**Primary: operator mathematics.** Work through all three swaps, ordered AND/OR/NOT/XOR, and modulo-3 arithmetic. Check closure, commutativity, associativity, identities, inverses, and De Morgan's laws where claimed. Carry one worked matrix example onward.

**Neighbour: weight representation.** Use the signed product table to derive a dot product as positive terms minus negative terms. Relate it to the BitNet sources, count packed bytes and metadata on a toy matrix, and identify which operations remain wider precision.

Neither route requires an account, a community credential, a purchase, or external publication. Disk and analog experiments remain deferred until the mathematical target makes their role explicit.

## Added 2026-09-12: region memory

**Question asked:** can a region of memory be set aside so other parts of memory contain more actionable information, especially for AI inference?

**Working answer** (`research/region-memory.md`): yes, as a descriptor region `D` plus a payload region `P`, with "actionable" defined as computing the dot product from the stored codes without materializing weights. Six concrete patterns were written down with a cost ledger each: mask+sign planes, block scales, zero-point digits, lookup tables, template dictionaries with transform selectors, and escape codes.

| Decision or finding | Status |
|---|---|
| Mask+sign planes cost `1 + density` bits per weight; overhead over entropy is `1 - H_b(density)` | derived, checked |
| Only identity and sign flip among the six permutations can be pushed onto inputs; the other four are decode-time transforms only | derived, exhaustive |
| An undecoded per-block sign flip must factor as row sign times column sign | derived, checked |
| Template dictionaries never beat the entropy bound on random blocks | derived, one seeded sample; real tensors unmeasured |
| The spare fourth 2-bit code is the right escape pointer; using the numerical zero as a pointer is expensive when zeros are common | derived, checked |
| What "region" means in the owner's idea: block (dictionary), mask (planes), or table (lookup) | open; all three now have ledgers |
| Real density and sign balance of trained ternary layers | open; stage A in `region-memory.md` section 10 |
