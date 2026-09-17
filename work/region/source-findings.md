# Source Findings: Ternary Memory Research Primary Sources

All quotes are verbatim from the saved capture files. Numbers are quoted, not paraphrased.

---

## 1. mx-formats — Microscaling Data Formats for Deep Learning (Rouhani et al. 2023)

- URL: https://arxiv.org/abs/2310.10537 (full text: https://arxiv.org/html/2310.10537)
- Checked: 2026-09-12T14:10:48Z (abs), 2026-09-12T14:10:51Z (html)
- Method: raw HTTP capture (curl), both pages returned full static HTML (200)

Quotes (from `mx-formats-html-raw.txt`):

- Block definition and shared scale (line 285): "A basic unit of data in an MX format represents a vector of *k* numbers and consists of a single *shared scale* *X* and *k* scalar *elements* ... This unit of data is called an MX block and is defined by the combination of *block size* *k*, scale data format, and element data format. The two data formats are independent of one another, and all *k* elements share the same element data format."
- Shared scale does not vary per element (line 318): "The shared scale *X* does not encode Inf."
- Concrete formats and E8M0 (line 325): "Table 1 shows the parameters that define the concrete MX formats, which are named by prepending \"MX\" to the name of the element data format. All concrete MX formats use E8M0 (an 8-bit exponent) as the format for the shared scale."
- Table 1, block size and formats (lines 350-377):
  - "MXFP8 | 32 | E8M0 | 8 | FP8 (E4M3 / E5M2) | 8"
  - "MXFP6 | 32 | E8M0 | 8 | FP6 (E2M3 / E3M2) | 6"
  - "MXFP4 | 32 | E8M0 | 8 | FP4 (E2M1) | 4"
  - "MXINT8 | 32 | E8M0 | 8 | INT8 | 8"
  (columns: Format Name | Block Size | Scale Data Format | Scale Bits | Element Data Format | Element Bit-width)

Notes: block size for all four concrete MX formats is 32 elements per shared scale; the shared scale is always E8M0 (8-bit exponent, no mantissa).

---

## 2. t-mac — T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge (Wei et al. 2024)

- URL: https://arxiv.org/abs/2407.00088 (full text: https://arxiv.org/html/2407.00088)
- Checked: 2026-09-12T14:10:54Z (abs), 2026-09-12T14:10:56Z (html)
- Method: raw HTTP capture (curl), both pages returned full static HTML (200)

Quotes (from `t-mac-html-raw.txt`):

- What the LUT contains, and multiplication replaced by lookup (line 371): "Since one bit can only represent two values, e.g., 1/-1, the bit patterns of a one-bit vector are limited. For example, if a one-bit matrix is partitioned into groups of four-element vector, the number of possible bit patterns (e.g., [1,1,1,-1] and [1,1,-1,-1]) for each group is only 2^4. Given an activation, it can be first computed with all possible bit patterns and saved in tables. The mpGEMM of activation and one-bit matrix is then transformed to table lookup indexed by each bit pattern in weight, and addition to accumulate the looked-up results. The mpGEMM is reduced to table lookup+add operations and no multiplication."
- Transforms multiplication to table lookup (line 338, abstract): "Specifically, T-MAC transforms the traditional data-type-centric multiplication to bit-wise table lookup, and enables a unified and scalable mpGEMM solution."
- Group size example, g=4 (line 545): "Taking the group size g=4, the tile size of the index matrix W_i[K_tk,M_tm]=[4,32], and the bit width as b=4 ... the 32 uint4 indices are first unpacked into uint8 bytes (blue) to ensure compatibility with the hardware data type and instructions. Subsequently, the uint8 indices are utilized to look up the table."
- Table size grows exponentially with group size (line 556): "The table size grows exponentially with the group size g. For instance, when g=4, the LUT is four times larger than the original activation... the LUT method uses 144 8-bit registers and llama.cpp uses 104 8-bit registers."
- Speedup numbers with exact conditions (abstract, line 340): "Evaluated on low-bit Llama and BitNet models, T-MAC demonstrates up to 4× increase in throughput and 70% reduction in energy consumption compared to llama.cpp. For BitNet-b1.58-3B, T-MAC delivers a token generation throughput of 30 tokens/s with a single core and 71 tokens/s with eight cores on M2-Ultra, and 11 tokens/s on Raspberry Pi 5."
- Additional speedup numbers with hardware (line 381): "We evaluated T-MAC performance on typical edge devices, including Apple M2 Ultra, Jetson AGX Orin, Surface Book 3, and Raspberry Pi 5. The T-MAC kernel speedup can reach up to 6.6× and an average of 3.6× compared to the SOTA on the CPU by llama.cpp... The e2e LLM inference speedup achieves 2.8× speedup for Llama-2-7B-2bit model... The inference performance can reach 11.1 tokens/s even on a Raspberry Pi for BitNet-b1.58-3B model... T-MAC is also energy efficient, reducing 60-70% energy compared to llama.cpp."

---

## 3. bitnet-cpu-report — 1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs (Wang et al. 2024)

- URL: https://arxiv.org/abs/2410.16144 (full text: https://arxiv.org/html/2410.16144)
- Checked: 2026-09-12T14:10:59Z (abs), 2026-09-12T14:11:02Z (html)
- Method: raw HTTP capture (curl), both pages returned full static HTML (200)

Quotes (from `bitnet-cpu-report-html-raw.txt`):

- Kernel names (line 486): "bitnet.cpp offers a suite of optimized kernels, including I2_S, TL1 and TL2. The kernels are designed for fast and lossless inference of 1.58-bit models on both x86 and ARM architectures."
- I2_S packing, bits per weight (line 503, Table 1 caption): "I2_S Kernel transforms each full-precision weight into a 2-bit value to save memory and bandwidth. When performing computation, the 2-bit weights are unpacked to the original values."
- I2_S description (line 506): "I2_S Kernel adopts the vanilla multiply-then-addition manner to perform the matrix multiplication... it transforms each full-precision weight into a 2-bit representation offline. During computation, it transforms the weights back to their original values and performs the vanilla GEMV operations."
- TL1 indexing (line 635, Table 2 caption): "TL1 Kernel transforms every two full-precision weights into 4-bit index and performs LUT computation."
- TL1 description, table size (line 638): "TL1 Kernel preprocesses every two full-precision weights by packing them into 4-bit index..., and pre-computes their corresponding activations into 3^2=9 values. The index-value pairs are stored in a lookup table to perform LUT computation... GEMV processing is performed using an int16 LUT and accumulation through addition."
- TL2 indexing (line 806, Table 3 caption): "TL2 Kernel compresses every three full-precision weights into a 1-bit sign (0 or 1) and a 4-bit index."
- TL2 vs TL1 (line 809): "TL2 Kernel is similar to TL1. The major difference is that it compresses every three weights into a 5-bit index, while TL1 compresses every two weights into a 4-bit index. Therefore, TL2 achieves a higher compression ratio than TL1... it employs LUT and reduces model size by 1/6 compared to TL1 Kernel, thereby lowering bandwidth requirements."
- Speedup ranges with hardware, exactly as written (abstract, line 248): "Extensive experiments demonstrate that bitnet.cpp achieves significant speedups, ranging from 2.37x to 6.17x on x86 CPUs and from 1.37x to 5.07x on ARM CPUs, across various model sizes."
- Speedup ranges with hardware, body text (line 262-263): "bitnet.cpp achieves speedups ranging from 1.37x to 5.07x on ARM CPUs, with larger models experiencing greater performance gains. Additionally, it reduces energy consumption by 55.4% to 70.0%, further boosting overall efficiency. On x86 CPUs, speedups range from 2.37x to 6.17x with energy reductions between 71.9% and 82.2%."
- Test hardware and further speedup detail (line 819, 825): "For ARM, we used a Mac Studio with an Apple M2 Ultra processor and 64GB of memory... For x86, a Surface Laptop Studio 2 with an Intel Core i7-13700H processor (14 cores, 20 threads) and 64GB of memory was used." / "bitnet.cpp consistently outpaces llama.cpp, with speedups ranging from 1.37x to 6.46x, depending on the model and architecture. On the Apple M2, speedups peak at 5.07x in the unlimited thread scenario, while on the Intel i7-13700H, bitnet.cpp achieves up to 6.46x in thread-limited scenarios."

---

## 4. llamacpp-tq — llama.cpp PR #8151, ternary types TQ1_0 and TQ2_0

- URL: https://github.com/ggml-org/llama.cpp/pull/8151
- Checked: 2026-09-12T14:11:05Z (page), 2026-09-12T14:11:10Z (API)
- Method: raw HTTP capture (curl) of the GitHub PR page succeeded (200) but is a heavy client-hydration shell; the PR body was instead extracted from the GitHub REST API JSON (`llamacpp-tq-api-raw.txt`) via `python3 json.load`, and the `body` field saved verbatim to `llamacpp-tq-body.txt` (this is a direct field-extraction of the raw JSON capture, not a re-fetch).

Quotes (from `llamacpp-tq-body.txt`):

- Bits per weight (line 3): "This adds `1.6875 bpw` and `2.0625 bpw` quant types for TriLMs and BitNet b1.58 models. For now, these are named `TQ1_0` and `TQ2_0`, respectively."
- TQ1_0 packing, 5 trits per byte via base-3/243 (line 6): "The `1.6875 bpw` type mostly relies on the fact that `3^5 == 243 < 256 == 2^8` to pack 5 trits per byte."
- TQ1_0 structure and block size (lines 368-376): "## Structure of `TQ1_0` ... This type relies on the fact that `3^5 == 243 < 256 == 2^8`. In a block of 256 elements, there are 240 elements encoded in 5 elements per byte, while the last 16 elements are encoded in 4 elements per byte. ... But there is also one `float16` scale per block, so the size of a block is 54 bytes making it a `1.6875 bpw` type. Even though it's not ideal, this is still `1.6875 / (log(3) / log(2)) ≈ 94%` of the best ternary packing efficiency."
- TQ2_0 structure and packing (lines 394-400): "## Structure of `TQ2_0` ... `TQ2_0` started as an experiment to see how fast a 2-bit type can be compared to a 1.6-bit type on compute-bound hardware. This packs each ternary value in 2 bits, which means each byte contains 4 values." (Byte-range table extends to element 256, confirming the same 256-element block size, with the scale stored at bytes 64-65.)
- Which is faster and why (line 24): "If you want to try `TQ2_0`, which is faster (but bigger) than `TQ1_0` on compute-bound hardware, you can replace `tq1_0` with `tq2_0`..."
- Speed comparison to other quants (line 34, 46): "`TQ2_0` is twice as fast as `Q4_K` on my laptop. It's *the* fastest quant on compute-bound AVX2-capable computers." / "`TQ1_0` is usually slightly faster than `Q4_K`, and that `TQ2_0` is by far the fastest quant on AVX2."

---

## 5. nvidia-2to4 — NVIDIA Ampere 2:4 structured sparsity

- Blog: https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/
- Paper: https://arxiv.org/abs/2104.08378 (full text: https://arxiv.org/html/2104.08378), Mishra et al. 2021
- Checked: 2026-09-12T14:11:13Z (blog), 2026-09-12T14:11:17Z (paper abs), 2026-09-12T14:11:26Z (paper html)
- Method: raw HTTP capture (curl), all pages returned full static/server-rendered HTML (200)

Quotes (from `nvidia-2to4-blog-raw.txt`):

- 2:4 pattern definition (line 374): "The NVIDIA A100 GPU adds support for fine-grained structured sparsity to its Tensor Cores. Sparse Tensor Cores accelerate a 2:4 sparsity pattern. In each contiguous block of four values, two values must be zero. This naturally leads to a sparsity of 50%, which is fine-grained."
- 2x throughput claim (line 395): "Sparse Tensor Cores accelerate this format by operating only on the nonzero values in the compressed matrix. They use the metadata that is stored with the nonzeros to pull only the necessary values from the other, uncompressed operand. So, for a sparsity of 2x, they can complete the same effective calculation in half the time."

Quotes (from `nvidia-2to4-arxiv-html-raw.txt`, Mishra et al. 2021):

- 2x math throughput (line 294, abstract): "Sparse Tensor Cores double math throughput for matrix-multiply operations when the first argument is a compressed 2:4 sparse matrix."
- Compressed storage dimension (line 392, Figure 1 caption): "Structured-sparse matrix (W) storage format. The uncompressed matrix is of dimension R × C and the compressed matrix is of dimension R × C/2."
- Metadata cost per nonzero, 2-bit indices (line 400): "With this pattern, only the 2 nonzero values in each group of 4 values need to be stored. Metadata to decode compressed format is stored separately, using 2-bits to encode the position of each nonzero value within the group of 4 values."
- Compressed storage size relative to dense, exact percentages (line 418-419): "Due to its 4-value block size, the 2:4 sparse storage format ... requires only 2-bits metadata per value, limiting storage overhead to 12.5% and 25% for 16b and 8b values, respectively. For 16-bit operands, storing a sparse tensor in compressed format leads to ∼44% savings in storage capacity: 4 dense elements require 4*16 = 64-bits of storage while 2:4 sparsity leads to 2*16-bits + 2*2-bits = 36-bits to store the two non-zero elements. For 8-bit operands, storing in compressed format saves ∼38% in memory capacity and bandwidth compared to the dense tensor."

Note: neither source states the storage reduction as a flat "about half plus metadata" figure; the paper instead gives the precise bit-level arithmetic above (36 bits vs 64 bits for 16-bit operands, ~44%/~38% savings), which is quoted verbatim rather than the rough estimate.

---

## 6. cheri-tags — CHERI tagged memory

- URL: https://www.cl.cam.ac.uk/research/security/ctsrd/cheri/cheri-faq.html and https://www.cl.cam.ac.uk/research/security/ctsrd/cheri/
- Checked: 2026-09-12T14:11:29Z (FAQ), 2026-09-12T14:11:32Z (main page)
- Method: raw HTTP capture (curl), both pages returned full static HTML (200)

Quotes (from `cheri-tags-faq-raw.txt`):

- One tag bit per capability-sized location, purpose (line 60): "Capabilities hold a virtual address as well as metadata describing the memory resources referenced by the pointer (bounds, permissions, ...) and also a 1-bit tag that protects the pointer itself (integrity, valid provenance, ...)."
- Storage overhead (line 284): "Tags add one bit of memory for every 128 or 256 bits of data, with a <1% memory overhead; they are maintained with cache lines, and obey normal cache-coherency rules."
- Out-of-band storage, partitioned physical memory (line 287): "In our CHERI prototype, we partition physical memory, setting aside a portion to hold tags, rather than requiring a change to memory interfaces. Currently, that partition is hard-coded, but it would ideally be managed by the firmware or software supervisor."

---

## 7. asan-shadow — AddressSanitizer algorithm

- URL: https://github.com/google/sanitizers/wiki/AddressSanitizerAlgorithm
- Checked: 2026-09-12T14:11:35Z
- Method: raw HTTP capture (curl), page returned full static HTML (200)

Quotes (from `asan-shadow-raw.txt`):

- Shadow mapping ratio (line 789): "AddressSanitizer maps 8 bytes of the application memory into 1 byte of the shadow memory."
- Shadow byte value meanings (lines 791-799): "There are only 9 different values for any aligned 8 bytes of the application memory: All 8 bytes in qword are unpoisoned (i.e. addressable). The shadow value is 0. All 8 bytes in qword are poisoned (i.e. not addressable). The shadow value is negative. First k bytes are unpoisoned, the rest 8-k are poisoned. The shadow value is k. This is guaranteed by the fact that malloc returns 8-byte aligned chunks of memory."
- Formula, 64-bit (line 832): "Shadow = (Mem >> 3) + 0x7fff8000;"
- Formula, 32-bit (line 861): "Shadow = (Mem >> 3) + 0x20000000;"

Note: the formula is written with the operand named `Mem`, not `Addr`, in this source; quoted exactly as written.

---

## 8. setun — Setun ternary computer

- URL: https://en.wikipedia.org/wiki/Setun
- Checked: 2026-09-12T14:11:38Z
- Method: raw HTTP capture (curl), page returned full static Parsoid HTML (200). The raw file's paragraphs are single very long lines (dense inline markup); a tag-stripped derivative of the same capture was saved as `setun-plaintext.txt` (no new network fetch) to make quotes locatable — line numbers below are from that derivative file.

Quotes (from `setun-plaintext.txt`, stripped from `setun-raw.txt`):

- Year and institution, developer, balanced ternary (line 662): "Setun (Russian: Сетунь) was a computer developed in 1958 at Moscow State University. It was built under the leadership of Sergei Sobolev and Nikolay Brusentsov. It was the first modern ternary computer, using the balanced ternary numeral system and three-valued ternary logic instead of the two-valued binary logic prevalent in other computers."
- Trits per word (line 665): "The characteristic operating memory consisted of 81 words of memory, each word composed of 18 trits (ternary digits) with additional 1944 words on magnetic drum (total of about 7 KB)."
- Number of machines built (line 665): "Fifty computers were built from 1959 until 1965, when production was halted."
- Number of machines built, alternate figure (line 683): "Only 50 Setun computers have been manufactured, 30 of which were used in the higher education institutions inside the Soviet Union."
- Reasoning for choosing ternary (line 703): "Brusentsov found the ternary number system superior to the binary number system: it allowed him to create very simple and reliable elements, and he needed only one seventh as many elements as Gutenmakher's computers... He also found the natural number-coding system used in the ternary system superior over the direct, reciprocal and supplementary number coding used in the binary system."

Note: the infobox on the same page separately lists "Released: 1959" and "Units sold: 50" (these appear in the raw HTML infobox markup, e.g. `"release_date":{"wt":"{{Start date and age|1959}}"}` and `"units_sold":{"wt":"50"}`), which is a discrepancy from the body text's "developed in 1958." Both are reported here rather than silently resolved. The source does not use the specific words "rounding" or "negation" to justify balanced ternary; the closest stated rationale is the component-count/element-count argument quoted above.

---

## 9. succinct-rank-select — Succinct data structure (rank/select)

- URL: https://en.wikipedia.org/wiki/Succinct_data_structure
- Checked: 2026-09-12T14:11:41Z
- Method: raw HTTP capture (curl), page returned full static Parsoid HTML with MathML (200). A tag-stripped derivative was saved as `succinct-plaintext.txt` (no new network fetch) to make quotes locatable — line numbers below are from that derivative file.

Quotes (from `succinct-plaintext.txt`, stripped from `succinct-rank-select-raw.txt`):

- Definition of rank(x) (lines 895-936, reconstructed from adjacent MathML/text nodes): "rank_q(x) returns the number of elements equal to q up to position x" [for q ∈ {0,1}].
- Definition of select(k) (lines 895, 978): "select_q(x) = min{k ∈ [0…n): rank_q(k) = x}" and "select_q(x) returns the position of the x-th occurrence of q."
- O(1) time with o(n) extra bits (line 1317): "this data structure supports rank queries in O(1) time and n+o(n) bits of space."
- o(n) overhead idea via large/small blocks (line 1026-1053): "It uses an idea similar to that for range-minimum queries; there are a constant number of recursions before stopping at a subproblem of a limited size. The bit array B is partitioned into large blocks of size l=lg^2(n) bits and small blocks of size s=lg(n)/2 bits."

Note: this article does not contain an explicit sentence connecting a bitvector-with-rank to "indexing into a compacted array of nonzeros"; that specific framing was not found in this source (see Not Found).

---

## 10. twn-ttq — Ternary Weight Networks (Li & Liu 2016) and Trained Ternary Quantization (Zhu et al. 2017)

### Ternary Weight Networks

- URL: https://arxiv.org/abs/1605.04711 (full text: https://arxiv.org/html/1605.04711)
- Checked: 2026-09-12T14:11:44Z (abs), 2026-09-12T14:11:47Z (html)
- Method: raw HTTP capture (curl), both pages returned full static HTML (200)

Quotes (from `twn-html-raw.txt`):

- Weight set (line 266): "We address the limited storage and computational resources issues by introducing ternary weight networks (TWNs), which constrain the weights to be ternary-valued: +1, 0 and -1."
- Threshold-based ternary function, formal definition (lines 321-325): "W̃_i = f(W_i|Δ) = { +1 if W_i>Δ; 0 if |W_i|≤Δ; -1 if W_i<-Δ }"
- Threshold rule for zero — **exact value as written is 0.75, not 0.7** (lines 360-361): "the approximated Δ* is α/3, which equals to (2/3)E(|W|). When W_i is generated from normal distributions N(0,σ²), the approximated Δ* is 0.6σ which equals to 0.75E(|W|). Thus, we can use a rule of thumb that Δ* ≈ 0.75E(|W|) ≈ (0.75/n)Σ|W_i| for simplicity."
- Storage reduction, 16x (abstract, line 231): "TWNs achieve up to 16× model compression rate and need fewer multiplications compared with the float32 precision counterparts."

### Trained Ternary Quantization

- URL: https://arxiv.org/abs/1612.01064 (full text: https://arxiv.org/html/1612.01064)
- Checked: 2026-09-12T14:11:50Z (abs), 2026-09-12T14:11:52Z (html)
- Method: raw HTTP capture (curl), both pages returned full static HTML (200)

Quotes (from `ttq-html-raw.txt`):

- Weight set via thresholding (line 400): "we normalize the full-precision weights to the range [-1, +1] by dividing each weight by the maximum weight. Next, we quantize the intermediate full-resolution weights to {-1, 0, +1} by thresholding."
- Separate positive/negative scales W_p and W_n (line 415): "To learn the ternary value (codebook), we introduce two quantization factors W^p_l and W^n_l for positive and negative weights in each layer l."
- Formal ternary weight equation with W_p/W_n (lines 420-424): "w^t_l = { W^p_l : w̃_l>Δ_l ; 0 : |w̃_l|≤Δ_l ; -W^n_l : w̃_l<-Δ_l }"
- W_p and W_n are independently trained (line 430): "the scaling coefficients W^p_l and W^n_l are two independent parameters and are trained together with other parameters."
- Asymmetry benefit (line 460): "The asymmetry of W^p_l ≠ W^n_l enables neural networks to have more model capacity."
- Storage reduction, 16x with 2-bit weights (line 813): "As for spatial compression, by substituting 32-bit weights with 2-bit ternary weights, our model is approximately 16× smaller than original 32-bit AlexNet."

---

## Not Found

The following requested facts could not be located verbatim in the captured sources; they are reported as not found rather than guessed or paraphrased from memory:

- **cheri-tags**: An explicit sentence stating that the tag table is "not addressable by ordinary loads and stores" was not found in `cheri-tags-faq-raw.txt` or `cheri-tags-main-raw.txt`. The closest statement found is that CHERI "partition[s] physical memory, setting aside a portion to hold tags, rather than requiring a change to memory interfaces" (line 287 of the FAQ), which implies but does not state the ordinary-load/store restriction.
- **cheri-tags**: An explicit sentence stating that "the tag bit is cleared when data is written non-capability-wise" was not found in either captured CHERI page.
- **succinct-rank-select**: No sentence in the captured Wikipedia article explicitly describes "a bitvector-with-rank being used to index into a compacted array of nonzeros." The article defines rank/select and their O(1)/o(n) complexity, but does not connect this to sparse-array indexing as an application.
- **setun**: The source does not use the words "rounding" or "negation" to explain why balanced ternary was chosen; the rationale found instead is a component/element-count efficiency argument (see quote above), and there is a discrepancy between the article body ("developed in 1958") and its infobox ("Released: 1959") that is reported rather than resolved.
- **twn-ttq (TWN)**: The task's expected threshold value ("delta ≈ 0.7 × mean|W|") does not match the source; the paper's actual rule of thumb is **Δ\* ≈ 0.75 × E(|W|)** (quoted above). This is flagged as a correction rather than reported as the hypothesized 0.7 figure.
