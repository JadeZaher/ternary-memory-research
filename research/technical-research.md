# Ternary memory research: initial technical assessment

Prepared 2026-09-11, America/Denver. Primary sources checked on 2026-09-11 local time, including a capture at 2026-09-12 00:26 UTC. Scope: mathematical orientation and a bounded research plan. No model installation, training run, hardware prototype, or runtime benchmark was performed for this assessment.

The most useful first step is to define exactly which operations should obey which rules. A system with states `-1, 0, +1` is mathematically coherent, and ternary neural-network weights already have substantial research support. It does not follow that changing logic gates or adding two alternative NOT operations improves an AI system. That proposed advantage must be separated into testable mathematical, storage, and execution claims.

The companion [operator reference](operator-reference.md) supplies complete core tables, three reversible swap operations, two cyclic shifts, and an exhaustive verification contract. It corrects the image labelled XOR: the supplied table is signed multiplication, and is logical XNOR when `-1` means false and `+1` means true. The displayed NOT is one sign-reversing function with three cases. Three distinct nonidentity involutions do exist, but only one is logical order reversal under the chosen truth encoding.

## What "memory" might mean

| Layer | Concrete meaning | What success would require |
|---|---|---|
| Mathematical values | Three symbols with explicit operations | Consistent tables and useful algebraic laws |
| Weight storage | Encoding learned numbers using three values, scales, or reusable blocks | Lower total stored bytes at acceptable model quality |
| Runtime representation | Layout and instructions used by CPU/GPU kernels | Reduced measured latency or energy, including decoding |
| Physical memory | A device reliably storing distinguishable physical states | Device-level evidence including sensing, noise, endurance, and peripheral circuits |
| Semantic memory | Facts or experiences an agent can retain and retrieve | Better task performance, retrieval accuracy, and update behavior |

These layers can interact, but success at one does not prove the next. A ternary tensor stored on an ordinary binary computer is already a useful research target. It does not require a three-state transistor or a new semantic memory architecture.

For the current project, interpret the user's "shapes" or "regions" as a hypothesis about recurring local patterns in weight blocks: perhaps the same template can appear with a sign change, a permutation, a scale, or another well-defined transform. This is a working interpretation to test, not an established property of trained weights.

## Mathematical possibilities worth testing first

Three compatible research tracks can share a value representation while retaining separate operation names:

1. **Unknown-preserving logic:** min-based AND, max-based OR, sign NOT, and XOR `-xy`. This provides interpretable truth behavior, including explicit unknowns.
2. **Ordinary signed numerical computation:** product `xy`, followed by a wider ordinary sum. This is the immediate connection to neural-network weights and matrix multiplication.
3. **Finite arithmetic:** balanced residues with addition and multiplication modulo 3. This supplies a compact closed arithmetic system, useful for exploring encodings and transformations, but changes ordinary neural-network accumulation.

One particularly concrete representation idea is to apply matching transforms to a weight block and its inputs. With a diagonal matrix `D` containing only signs, `(W D)(D x)=W x`. With a permutation matrix `P`, `(W P^-1)(P x)=W x`. These identities explain when changing a representation preserves a linear result. A swap that turns numerical zeros into nonzeros generally lacks that guarantee; a sign reversal across ReLU also fails without more changes. Full-network preservation therefore needs a separate proof for every boundary and nonlinearity.

The three nonidentity reversible swaps are **not three free storage channels**. Each is a way to transform the same three states. Selecting one of three transforms itself requires information, unless the surrounding structure determines the selection. Compression is possible only when shared structure saves more space than the transform selector, templates, scales, offsets, and exceptions cost.

## What the storage arithmetic actually promises

The following are direct counting calculations, independent of published benchmark claims:

| Representation | Payload cost before metadata | Qualification |
|---|---:|---|
| Binary sign only, two possible values | 1 bit/value | Cannot represent an independent third state |
| Ideal long blocks of arbitrary ternary values | Approaches `log2(3) = 1.5849625` bits/value | Worst-case information count; practical block sizes and decoding matter |
| Independent fixed binary slot for each trit | 2 bits/value | One of four codes is unused or reserved |
| Five trits in one byte | 1.6 bits/value | `3^5=243` valid patterns fit in 256 byte values |
| Signed 8-bit container | 8 bits/value | Simple arithmetic; it wastes representational capacity |

A single independently addressed trit needs at least two binary bits in a fixed-width slot. The approximately 1.585-bit figure comes from coding many states together. For nonuniform values, ideal entropy coding can be smaller, at the cost of additional structure and access constraints.

For `N` weights partitioned into blocks of `g` weights, suppose `K` shared templates can exactly describe those blocks. A fixed-width template identifier alone costs approximately `ceil(log2(K))/g` bits per weight. Add the entire template dictionary, transform identifiers, scales, padding, exceptions, and index structures. For ternary templates a straightforward dictionary itself costs about `K*g*log2(3)` bits ideally. A dictionary of almost-unique blocks may expand the data.

A dense ternary format already represents zero cheaply. A sparse format must also store locations. More zeros therefore do not automatically make sparse indexing smaller or a kernel faster. The fraction of zeros, their clustering, block size, and execution method determine the outcome.

## Where Microsoft's BitNet fits

The original BitNet b1.58 paper directly supports the feasibility of **trained ternary linear-layer weights**. It describes training from scratch using absmean weight quantization and 8-bit activations. The reported GPU experiments used a 2-bit kernel; embeddings remained full precision. Its quality and efficiency results belong to the tested models, training setups, and hardware, not to every ternary representation. This is evidence for the numerical-weight track, rather than evidence for the proposed three-NOT logic system. [Ma et al., *The Era of 1-bit LLMs*, 2024](https://arxiv.org/html/2402.17764v1)

The later 2B4T report describes a native model trained from scratch, ternary quantization during the forward pass of BitLinear layers, per-token 8-bit activation quantization, and squared ReLU in the feed-forward network. This provides a more concrete architectural reference for later experiments. Quantized forward-pass weights do not imply that all training state uses ternary storage; optimizer state, gradients, and trainable parameter representations must be accounted for separately. The report does not make the three-state logic tables a substitute for normal transformer computations. [Wang et al., *BitNet b1.58 2B4T Technical Report*, 2025](https://arxiv.org/html/2504.12285v1)

The current official repository provides CPU and GPU inference paths and optimized kernels. It is a useful future baseline, and its state should be pinned to a commit before reproducing results. Its runtime support and benchmark claims must not be generalized to arbitrary model conversions or a new codebook. No repository build or model download was performed here. [Microsoft BitNet repository](https://github.com/microsoft/BitNet)

A complete memory budget separates weights, activation buffers, attention KV cache, runtime scratch space, and training state. Weight compression alone does not establish that the KV cache is ternary or that long-context memory shrinks by the same ratio. Ordinary neural-network accumulation also needs a wider numerical range than one trit.

## Disk and analog computation: later application tracks

*LLM in a flash* stores parameters in flash and transfers selected data to DRAM. Its two central techniques reuse recently activated neurons and bundle rows/columns into larger contiguous reads. This establishes that access patterns and reuse can matter as much as total model size. Its reported speedups compare against naive loading in its studied configurations; they are not a promise that disk is faster than resident RAM. [Alizadeh et al., ACL 2024](https://aclanthology.org/2024.acl-long.678/)

*FlexGen* combines GPU, CPU, and disk capacity, plans tensor placement/access, and uses compression for throughput-oriented inference. Its setting explicitly includes latency-insensitive batched processing. It supports a disk-backed research direction, but does not establish fast interactive batch-one decoding for a proposed ternary design. [Sheng et al., ICML 2023](https://proceedings.mlr.press/v202/sheng23a.html)

**Inference for this project:** compact templates and predictable block requests might lower transferred bytes. The benefit disappears if lookup, decoding, random access, or mispredicted prefetching costs more than the saved transfer time. Report bytes read, number and size of I/O requests, cache-hit rate, decode time, and cold/warm latency separately. A basic transfer-time lower bound is bytes actually transferred divided by achieved bandwidth; queueing, request latency, and decoding add costs. A memory-mapped file can be served from the OS page cache, so mapping a file is not proof that an experiment measured physical disk reads.

IBM's 64-core mixed-signal chip is a primary example of analog in-memory computation: phase-change memory arrays perform computation alongside on-chip digital operations and communication. The work demonstrates ResNet and LSTM inference and explicitly treats digital activation processing as part of the system. It supports the distinction between stored physical states, arithmetic, and surrounding digital processing. It does not demonstrate this project's proposed ternary operator set or a transformer with three alternate logical NOT operations. [Le Gallo et al., Nature Electronics 2023, author publication page](https://research.ibm.com/publications/a-64-core-mixed-signal-in-memory-compute-chip-based-on-phase-change-memory-for-deep-neural-network-inference)

**Inference for this project:** fewer target weight levels may simplify an analog representation, but end-to-end benefits require accounting for sensing, input/output conversion, calibration, noise, and movement between blocks. A software result cannot establish those hardware benefits. No analog device or circuit model was evaluated here.

## Bounded experiments and stop conditions

Start with the first stage; later stages are proposals, not work already run.

| Stage | Bounded experiment | Pass / fail evidence |
|---|---|---|
| 1. Operator specification | Enumerate 27 unary maps and all 9 binary input pairs; check algebraic laws on all 27 triples | Exact tables, counts, laws, and intentional counterexamples in the companion verification contract |
| 2. Representation preservation | Small integer matrices; paired signs and permutations; counterexamples for zero-moving swaps and ReLU | Equality for claimed invariants; explicit failing examples for unsupported transforms |
| 3. Storage only | Synthetic ternary blocks with controlled repetition, sparsity, and entropy; compare 2-bit, five-trit-byte, and template encodings | Exact round-trip recovery; total bytes including dictionary/metadata; expand-on-random-data cases reported |
| 4. Numeric kernel feasibility | Small dot products using the same integer inputs; compare stored layouts with a wider exact reference | Exact outputs before rescaling; overflow cases caught; decode cost measured separately |
| 5. Model relevance | Later, select one documented compatible small model and a fixed evaluation set | Define acceptable quality loss before running; record baseline, checkpoint, precision, shapes, hardware, and total memory |
| 6. Disk / analog follow-up | Only after a justified access pattern or device model exists | Measure end-to-end benefit in the intended workload; reject claims based only on payload size or operation counts |

For the region/template hypothesis, the first decisive question is whether exact repeated structure survives realistic weights and metadata accounting. If exact matches are rare, approximation becomes a different experiment: measure output distortion and model quality rather than describing it as lossless compression. For claims of runtime improvement, state whether the goal is batch-one latency, batched throughput, memory capacity, or energy. An improvement in one metric is not evidence for the others.

The research remains open at three points: the intended meaning of a "region" in real model tensors; which transforms preserve the desired computation; and whether those transforms create enough repeatable structure to outweigh representation and execution costs. The mathematical reference makes those questions precise enough to begin testing without first installing or training a model.

## Evidence and limits

All external technical claims above are attached to primary papers, author publication pages, or the official implementation repository. Mathematical counts, formulas, and counterexamples are direct derivations from the stated finite domain. Statements explicitly marked as inference are proposed connections, not results reported by those sources.

Raw retrieved responses were captured under `work/technical/` before filtering. The Nature landing page for the separate 34-tile speech chip did not yield usable body text; that paper is not used as evidence here. The analog discussion instead uses the accessible author page for the distinct 64-core paper. This is an initial orientation, not an exhaustive literature review, a priority claim, or a verified efficiency result for the proposed system.
