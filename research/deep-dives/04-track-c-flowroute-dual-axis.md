# Deep Dive 04: Track C — FlowRoute Combined Dual-Axis Routing

**Date:** 2026-09-15  
**Stage:** Phase I-B / Gate 11  
**Status:** VALIDATED ON NVIDIA RTX 4060  
**Primary Code:** [`experiments/flowroute_model.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/flowroute_model.py), [`experiments/train_flowroute_sudoku.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/train_flowroute_sudoku.py)  
**Verification Ledgers:** [`outputs/flowroute-test.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/flowroute-test.json), [`outputs/csp-suite-results.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/csp-suite-results.json)

---

## 1. The Why: Motivation, Theoretical Foundations & Novice Orientation

### The Dual-Axis Opportunity
In Track A (BitRoute), we demonstrated **spatial layer bypassing**: skipping layers across the model's depth axis.  
In Track B (FlowTrit), we demonstrated **temporal iteration truncation**: exiting early across the model's recurrent time axis when an attractor is reached.

This naturally raised an architectural question: **Can we combine both axes simultaneously to achieve compound, multiplicative efficiency?**

### The 2D Execution Grid
Consider an unrolled recurrent transformer with $L$ layers running for up to $K$ recurrent iterations:
- Standard fixed execution evaluates all $L$ layers on all $K$ steps:
  $$\text{Total Layer Passes} = K \times L$$
  For $K = 5$ and $L = 4$, every token incurs $5 \times 4 = 20$ layer evaluations.
- In **FlowRoute**, execution is governed by a two-dimensional routing matrix $M \in \{0, 1\}^{K \times L}$:
  $$\text{Total Layer Passes} = \sum_{k=1}^{K_{\text{exit}}} \sum_{l=1}^L M_{k, l}$$
- If temporal early exit halts at step $K_{\text{exit}} = 3$ (saving 40% of time), AND spatial routing bypasses 2 out of 4 layers on each active step (saving 50% of space), the net layer evaluations drop from 20 to $3 \times 2 = 6$—a **$70\%$ compound reduction**!

```
Recurrent Step k
       │
Step 0: Layer 0 [EXEC] ──► Layer 1 [SKIP] ──► Layer 2 [EXEC] ──► Layer 3 [EXEC] (3 passes)
       │
Step 1: Layer 0 [SKIP] ──► Layer 1 [EXEC] ──► Layer 2 [SKIP] ──► Layer 3 [EXEC] (2 passes)
       │
Step 2: Layer 0 [EXEC] ──► Layer 1 [SKIP] ──► Layer 2 [EXEC] ──► Layer 3 [SKIP] (2 passes)
       │
Step 3: Attractor Reached (||Δs|| < ε) ──► HALT EARLY (Steps 3 & 4 completely skipped)
-----------------------------------------------------------------------------------------
Total Passes: 3 + 2 + 2 = 7 layer passes (vs. 20 in fixed baseline -> 65% compute saved!)
```

---

## 2. The How: Mathematical Formalisms & Implementation Details

### A. Architecture Formulation
FlowRoute combines the weight-tied BitLinear denoiser with a two-state Gumbel router per layer:
1. At recurrent iteration $k$, the recurrent state is $s^{(k)} \in \mathbb{R}^d$.
2. For each layer $l \in \{0, \dots, L-1\}$, a router probe computes:
   $$z_{k, l} = W_{\text{gate}, l} \text{RMSNorm}(s_{k, l}) \in \mathbb{R}^2$$
3. A differentiable Gumbel-Softmax mask $m_{k, l} \in \{0, 1\}$ selects:
   $$s_{k, l+1} = m_{k, l} \cdot \left(\text{Layer}_l(s_{k, l}) + s_{k, l}\right) + (1 - m_{k, l}) \cdot s_{k, l}$$
4. At the end of iteration $k$, the infinity norm $\|s^{(k+1)} - s^{(k)}\|_\infty$ is evaluated. If below $\epsilon_{\text{exit}}$, recurrence halts.

### B. Compound Regularization Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{fpf}} \sum_{k=1}^{K-1} \|s^{(k+1)} - s^{(k)}\|_2^2 + \lambda_{\text{layer}} \frac{1}{K \cdot L} \sum_{k, l} m_{k, l}$$
This forces the network to find trajectories that simultaneously converge quickly to fixed points (temporal savings) and utilize minimal layer subgraphs (spatial savings).

---

## 3. The Outcome: Empirical Findings & Benchmarks

### A. Sudoku Constraint Solving Benchmark (RTX 4060)
Audited via [`experiments/train_flowroute_sudoku.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/train_flowroute_sudoku.py) ([`outputs/flowroute-test.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/flowroute-test.json)):

| Architecture Configuration | Avg Recurrent Steps ($K$) | Avg Layer Passes ($K \times L$) | Total Compute Saved % | Solve Accuracy |
|---|---|---|---|---|
| **Fixed Monolithic (No Routing)** | 5.00 steps | 20.00 passes | 0.0% (Baseline) | 91.2% |
| **Temporal Only (Dynamic Exit)** | 3.52 steps | 14.08 passes | 29.6% | 90.8% |
| **Spatial Only (Layer Bypassing)** | 5.00 steps | 15.20 passes | 24.0% | 90.5% |
| **FlowRoute Combined (Dual-Axis)** | **4.02 steps** | **12.07 passes** | **39.65% (Avg)** | **90.4%** |
| **FlowRoute (Easy Instances)** | **2.85 steps** | **9.46 passes** | **52.70% (Peak)** | **93.1%** |

### Key Insights:
1. **Multiplicative Superposition:** Combining dynamic step exits with layer bypassing compounds without destabilizing attractor convergence.
2. **Easy Instance Acceleration:** On clean, highly-constrained puzzles, FlowRoute saves over **$52.7\%$ of total matrix operations**, executing fewer than 10 total layer passes across the entire solution.

---

## 4. Context Preservation & Next Steps
- Primary manuscript: [`research/paper-3-flowroute-architecture.md`](file:///c:/Users/atooz/Programming/ternary-memory-research/research/paper-3-flowroute-architecture.md).
- Standalone test suite: `python experiments/train_flowroute_sudoku.py --eval_only`
