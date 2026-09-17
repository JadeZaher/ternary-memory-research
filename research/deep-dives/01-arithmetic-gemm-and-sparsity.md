# Deep Dive 01: Arithmetic Upgrades, Multiplication-Free GEMM, and Physical Sparsity

**Date:** 2026-09-15  
**Stage:** Phase I-B / Gate 7  
**Status:** PASSED & VERIFIED ON NVIDIA RTX 4060  
**Primary Code:** [`experiments/bitlinear.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/bitlinear.py), [`experiments/test_arithmetic_upgrade.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/test_arithmetic_upgrade.py), [`experiments/bench_gemm_paths.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/bench_gemm_paths.py)  
**Verification Ledgers:** [`outputs/arithmetic-upgrade-test.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/arithmetic-upgrade-test.json), [`outputs/gemm-paths-benchmark.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/gemm-paths-benchmark.json), [`outputs/math-verification.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/math-verification.json)

---

## 1. The Why: Motivation, Theoretical Foundations & Novice Orientation

### The Computational Problem with Standard Transformers
In standard neural networks (e.g., standard FP16 or BF16 LLMs), computing the output of a linear layer $y = x W$ requires matrix multiplications:
$$y_j = \sum_{i=1}^{d_{\text{in}}} x_i \cdot W_{ij}$$
If $d_{\text{in}} = 768$ and $d_{\text{out}} = 2048$, calculating one token output requires $768 \times 2048 = 1,572,864$ floating-point multiply-accumulate (MAC) operations. Multipliers are silicon-hungry: an FP16 multiplier requires sign, exponent, and mantissa alignment circuits, consuming significant energy and chip area.

### What Changes in Ternary Representation?
In 1.58-bit ternary neural networks (inspired by BitNet b1.58), every weight entry is constrained to three discrete states:
$$\widetilde{W}_{ij} \in \{-1, 0, +1\}$$
Notice what happens to the scalar multiplication $x_i \cdot \widetilde{W}_{ij}$:
- If $\widetilde{W}_{ij} = +1$, $x_i \cdot (+1) = +x_i$ (simple addition).
- If $\widetilde{W}_{ij} = -1$, $x_i \cdot (-1) = -x_i$ (simple subtraction).
- If $\widetilde{W}_{ij} = 0$, $x_i \cdot 0 = 0$ (operation completely skipped).

`[Mathematical Derivation]` Multiplication is eliminated entirely. Matrix multiplication simplifies to **masked conditional addition and subtraction**:
$$y_j = \sum_{i: \widetilde{W}_{ij} = +1} x_i - \sum_{i: \widetilde{W}_{ij} = -1} x_i$$

### Novice Worked Example
Suppose an input vector is $x = [2.5, -1.0, 4.0, 0.5]$ and a ternary weight column is $W = [+1, 0, -1, +1]^T$.
Instead of 4 floating-point multiplications:
1. First element: $W_0 = +1 \implies \text{add } 2.5 \implies \text{accumulator} = 2.5$.
2. Second element: $W_1 = 0 \implies \text{skip } (-1.0) \implies \text{accumulator} = 2.5$.
3. Third element: $W_2 = -1 \implies \text{subtract } 4.0 \implies \text{accumulator} = 2.5 - 4.0 = -1.5$.
4. Fourth element: $W_3 = +1 \implies \text{add } 0.5 \implies \text{accumulator} = -1.5 + 0.5 = -1.0$.

Total operations: 2 additions, 1 subtraction, 0 multiplications.

### Key Theoretical Questions Addressed
1. **Numerical Parity:** Does implementing matrix multiplication via pure conditional masking match cuBLAS floating-point GEMM within floating-point tolerance?
2. **Physical Sparsity:** What percentage of weights naturally quantize to exact numerical zeros ($0$)? Does a numerical zero translate into skipped computation?
3. **Accumulator Bit-Width Bounds:** If we accumulate integer sums, how many bits are required in the hardware accumulator to guarantee that integer overflow *never* occurs?

---

## 2. The How: Mathematical Formalisms & Implementation Details

### A. Weight Ternarization via Straight-Through Estimator (STE)
During training, we store high-precision FP32 master weights $W$. In the forward pass, we ternarize $W$ using absmean scaling:
$$\gamma = \frac{1}{d_{\text{in}} d_{\text{out}}} \sum_{i,j} |W_{ij}|$$
$$\widetilde{W} = \text{clip}\left(\left\lfloor \frac{W}{\gamma + \epsilon} \right\rceil, -1, +1\right)$$
Because the rounding function $\lfloor \cdot \rceil$ has zero derivative almost everywhere, standard backpropagation would fail. We use the Straight-Through Estimator (STE):
$$\frac{\partial \mathcal{L}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial \widetilde{W}}$$
Gradient updates pass directly to the FP32 master weights, which drift continuously until their sign or magnitude flips the discrete quantized state.

### B. Masked Additive Kernel Implementation
In [`experiments/bitlinear.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/bitlinear.py), function `ternary_additive_gemm` implements this:
```python
def ternary_additive_gemm(x: torch.Tensor, W_tilde: torch.Tensor) -> torch.Tensor:
    # W_tilde in {-1, 0, +1}, shape (d_out, d_in)
    # x shape (batch, seq_len, d_in)
    pos_mask = (W_tilde == 1.0)   # (d_out, d_in)
    neg_mask = (W_tilde == -1.0)  # (d_out, d_in)
    
    # Broadcasted accumulation: sum(x * pos) - sum(x * neg)
    # Zero weights (W_tilde == 0) are excluded from both masks!
    out = torch.matmul(x, pos_mask.t().to(x.dtype)) - torch.matmul(x, neg_mask.t().to(x.dtype))
    return out
```

### C. Mathematical Proof: 20-Bit Overflow-Free Accumulator Bound
`[Mathematical Derivation]`  
Let activations be quantized to $b_{\text{act}} = 8$ bits signed integer: $x_i \in [-128, 127]$.  
Let ternary weights be $\widetilde{W}_{ij} \in \{-1, 0, +1\}$.  
For an inner product of dimension $K = d_{\text{in}}$:
$$y_j = \sum_{i=1}^K x_i \widetilde{W}_{ij}$$
The absolute theoretical maximum value occurs when all $x_i = -128$ and $\widetilde{W}_{ij} = -1$ (or $x_i = +127$ and $\widetilde{W}_{ij} = +1$):
$$|y_j|_{\max} \le \sum_{i=1}^K \max|x_i| \cdot |\widetilde{W}_{ij}| \le K \cdot 128 = 2^7 \cdot K$$
For standard transformer dimensions $K = 768$ (or even $K = 4096$):
- When $K = 768$: $|y_j|_{\max} \le 768 \times 128 = 98,304$.
  - Required bits: $\lceil \log_2(98,304 + 1) \rceil + 1 \text{ (sign bit)} = 17 + 1 = 18\text{ bits}$.
- When $K = 4096$: $|y_j|_{\max} \le 4096 \times 128 = 524,288 = 2^{19}$.
  - Required bits: $19 + 1 = 20\text{ bits}$.

Therefore, a **20-bit signed integer accumulator** provides an absolute, unbreakable mathematical guarantee against accumulator overflow for any hidden dimension up to $K = 4096$, completely eliminating the need for expensive 32-bit floating-point ALUs in dedicated hardware.

---

## 3. The Outcome: Empirical Findings, Baselines & Benchmarks

### A. Numerical Parity Audit
Audited via [`experiments/test_arithmetic_upgrade.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/test_arithmetic_upgrade.py) across 1,000 randomized matrices on RTX 4060:
- **Maximum Absolute Error ($\text{atol}$):** $1.192 \times 10^{-6}$ (exact floating-point epsilon parity).
- **Mean Squared Error:** $0.00000000$.
- **Assertion:** `torch.allclose(out_additive, out_dense, atol=1e-5)` **PASSED**.

### B. Physical Sparsity Audit
Across the projections of BitRoute-135M (Q, K, V, O, Gate, Up, Down projections):
- **Total ternary weight entries:** $134,133,540$.
- **Positive weights (+1):** $37.6\%$.
- **Negative weights (-1):** $37.5\%$.
- **Exact zero weights (0):** **$24.9\%$**.

`[Empirical Measured Result]` **One out of every four operations is completely skipped.** In a physical memory crossbar or sparse additive engine, 24.9% of memory reads and accumulator cycles draw zero current.

### C. Hardware Emulation Reality: Why Emulation Lags cuBLAS
When benchmarking `ternary_additive_gemm` on commodity GPU hardware ([`outputs/gemm-paths-benchmark.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/gemm-paths-benchmark.json)):
- Standard cuBLAS FP16 Tensor Core GEMM: $0.21\,\text{ms}$.
- Emulated PyTorch Masked Additive GEMM: $0.38\,\text{ms}$ ($0.545\times$ speed of cuBLAS).

**Why?** Commodity GPUs (like the RTX 4060) have physical silicon hardware dedicated specifically to dense $16\times 16$ tensor multiplication. Simulating addition by constructing boolean masks and dispatching two separate kernel passes suffers from memory bandwidth overhead. This establishes that **physical acceleration requires dedicated silicon (ASIC / FPGA / crossbar) rather than software emulation on commodity GPUs.**

---

## 4. Context Preservation & Next Steps
- Full test suite: `python experiments/test_arithmetic_upgrade.py`
- All tests pass on CUDA.
- The 20-bit accumulator proof was formally integrated into Paper 1 ([`research/paper-1-bitroute-systems.md`](file:///c:/Users/atooz/Programming/ternary-memory-research/research/paper-1-bitroute-systems.md)).
