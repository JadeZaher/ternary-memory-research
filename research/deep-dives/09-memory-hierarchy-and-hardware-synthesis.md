# Deep Dive 09: Memory Hierarchy, Bit-Packing & Neuromorphic Hardware Synthesis

**Date:** 2026-09-15  
**Stage:** Phase I & Phase III Hardware Specification  
**Status:** VALIDATED IN SIMULATION & BIT-PACKING ENGINES  
**Primary Code:** [`experiments/region_encoding.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/region_encoding.py), [`experiments/continuation_simulator.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/continuation_simulator.py), [`experiments/bench_inference_dtype.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/bench_inference_dtype.py)  
**Verification Ledgers:** [`outputs/region-encoding.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/region-encoding.json), [`outputs/continuation-simulation.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/continuation-simulation.json), [`outputs/inference-dtype-benchmark.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/inference-dtype-benchmark.json)

---

## 1. The Why: Motivation, The Memory Hierarchy & Novice Orientation

### The Physical Memory Hierarchy
In computer architecture, memory is organized in a pyramid:
1. **Registers & ALUs:** Fastest (< 1 cycle), but extremely tiny (kilobytes).
2. **On-Chip SRAM (L1/L2/L3 Cache):** Very fast (a few cycles, high bandwidth), typically 16MB to 64MB.
3. **Off-Chip DRAM (HBM/GDDR):** Slow (hundreds of cycles, restricted memory bus bandwidth), large (gigabytes).

In standard FP16 deep learning, large models (billions of parameters) cannot fit in on-chip SRAM. The hardware must continuously fetch weights across the memory bus from DRAM, which consumes over **80% of total inference energy**!

### The Two Critical Memory Quantities: $D$ and $P$
`[Theoretical Inference]`  
In our research framework, we formalize the memory footprint into two distinct quantities:
- **$D$ (Static Storage Payload):** The total physical size of all model parameters when stored at rest in off-chip DRAM.
- **$P$ (Active Working Set Footprint):** The volume of parameters and activations that must reside simultaneously in high-speed on-chip SRAM cache during active execution.

By reducing weights from 16-bit floats to 1.58-bit trits ($\pm 1, 0$), both $D$ and $P$ shrink by over **$10\times - 16\times$**.

---

## 2. The How: Bit-Packing Encodings & Hardware Crossbars

### A. Ternary Bit-Packing Encodings
In [`experiments/region_encoding.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/region_encoding.py), we compared two encoding schemes:

1. **2-Bit Direct Encoding:**
   Store each trit in 2 binary bits:
   - `00` $\to 0$
   - `01` $\to +1$
   - `11` $\to -1$
   - `10` $\to$ Unused / Sentinel
   *Efficiency:* Exactly 4 trits per 8-bit byte ($2.00$ bits/trit). Simple masking and bit-shifting hardware.

2. **Base-3 Radix Packing:**
   Notice that $3^5 = 243 \le 256 = 2^8$.
   We can pack **5 independent ternary trits into a single 8-bit byte**:
   $$\text{Byte} = t_0 + 3 \cdot t_1 + 9 \cdot t_2 + 27 \cdot t_3 + 81 \cdot t_4, \quad t_i \in \{0, 1, 2\}$$
   *Efficiency:* $\frac{8 \text{ bits}}{5 \text{ trits}} = \mathbf{1.60\text{ bits/trit}}$! This approaches the information-theoretic Shannon bound ($\log_2(3) \approx 1.585\text{ bits}$).

### B. Physical Neuromorphic Crossbar Mapping (Phase III)
`[Architectural Proposal]`  
In Phase III, ternary weights are mapped directly to physical memristive crossbars (RRAM or Phase Change Memory):

```
Voltages V_i ───►  [ RRAM ] ───►  [ RRAM ] ───►  [ RRAM ]
                     │              │              │
                   G = +1         G = 0          G = -1
                     │              │              │
                     ▼              ▼              ▼
           Accumulated Current: I_j = Σ V_i · G_ij (Kirchhoff's Law)
```

In a memristor crossbar:
- Each cell has a programmable conductance $G_{ij} \in \{+G_0, 0, -G_0\}$.
- Multiplications occur physically in analog via Ohm's Law: $I_{ij} = V_i \cdot G_{ij}$.
- Additions occur physically via Kirchhoff's Current Law: $I_j = \sum_i I_{ij}$.
- **Zero-Current Skipping:** When $W_{ij} = 0$, $G_{ij} = 0$. **No electrical current flows through the crossbar cell.** Zero-weight sparsity (24.9%) translates into literal zero energy dissipation at the physical device level!

---

## 3. The Outcome: Empirical Findings & Benchmarks

### A. Bit-Packing Compression Audit
Audited via [`experiments/region_encoding.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/region_encoding.py) ([`outputs/region-encoding.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/region-encoding.json)):

| Model | Uncompressed FP32 | Standard FP16 | 2-Bit Direct Packed | Base-3 Radix Packed | Compression vs FP32 |
|---|---|---|---|---|---|
| **FlowTrit-40M** | 160.00 MB | 80.00 MB | 10.00 MB | **8.05 MB** | **$19.87\times$** |
| **BitRoute-135M** | 536.53 MB | 268.27 MB | 33.53 MB | **27.02 MB** | **$19.86\times$** |

### B. Continuation Traffic Simulator
Audited via [`experiments/continuation_simulator.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/continuation_simulator.py) ([`outputs/continuation-simulation.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/continuation-simulation.json)):
- Simulating a 1,000-token generation workload across a constrained memory bus:
  - Standard FP16 Model: Transferred **268.2 GB** across the memory bus.
  - BitRoute-135M with 50% Layer Bypass: Transferred **13.5 GB** across the memory bus (**$19.8\times$ total DRAM traffic reduction**).

---

## 4. Context Preservation & Next Steps
- Implementation scripts: `experiments/region_encoding.py`, `experiments/continuation_simulator.py`
- Hardware synthesis roadmap formally documented in Paper 1 and Executive Synthesis ([`research/synthesis-executive-summary.md`](file:///c:/Users/atooz/Programming/ternary-memory-research/research/synthesis-executive-summary.md)).
