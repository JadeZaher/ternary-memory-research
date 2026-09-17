# Gate 15: Track D-4 — Frontier Scale & Multi-Corpus Pretraining (NaviTrit-135M)

**Prepared:** 2026-09-17  
**Stage:** Phase II / Gate 15 (Queued & Specified)  
**Hardware Target:** Local NVIDIA RTX 4060 (8GB VRAM) & Cloud Cluster Scaling Extension  
**Governing Discipline:** Rule from `AGENTS.md` (*"A future model experiment requires a specified model, hardware, workload, baseline, and resource cap."*)

---

## 1. Hardware Feasibility Analysis: Can This Train on an 8GB GPU?

To maintain absolute scientific and engineering discipline, we calculate the exact memory and compute limits of training on a single **NVIDIA RTX 4060 (8GB VRAM)**.

### A. The VRAM Memory Equation (Why 1.58-Bit Training Still Requires Memory)
`[Mathematical Derivation]`  
While ternary weights at *inference* occupy only 1.58 bits per parameter, *training* from scratch requires high-precision master copies and optimizer buffers:
$$\text{Total VRAM} = \mathcal{M}_{\text{weights}} + \mathcal{M}_{\text{grads}} + \mathcal{M}_{\text{opt}} + \mathcal{M}_{\text{acts}} + \mathcal{M}_{\text{cuda}}$$
For a model with $N$ parameters:
1. **FP32 Master Weights:** $4 \times N$ bytes (required for the Straight-Through Estimator to accumulate small fractional gradient steps).
2. **Gradients:** $2 \times N$ bytes (in BF16/FP16) or $4 \times N$ bytes (FP32).
3. **AdamW Optimizer States ($m_t, v_t$):** $8 \times N$ bytes in standard FP32 AdamW, or $2 \times N$ bytes with 8-bit Adam (`bitsandbytes`).
4. **Activations:** $\mathcal{O}(B \times S \times L \times d_{\text{model}})$. With gradient checkpointing, this drops to $\sim 1.0 - 1.5\,\text{GB}$.

#### Parameter Scale vs. Memory Matrix:
| Parameter Count ($N$) | Forward Inference Weight Footprint | Training VRAM (Standard AdamW) | Training VRAM (8-Bit Adam + Checkpointing) | Feasible on 8GB RTX 4060? |
|---|---|---|---|---|
| **45.6M (Current Gate 14)** | **9.4 MB** | ~1.8 GB | ~1.2 GB | **YES (Validated: 3.5 GB peak)** |
| **135.0M (NaviTrit-135M)** | **27.0 MB** | ~3.8 GB | **~2.4 GB** | **YES (100% Feasible locally)** |
| **350.0M (Medium)** | **70.0 MB** | ~7.2 GB | **~4.8 GB** | **YES (Tight fit locally)** |
| **1.0 Billion (1B)** | **200.0 MB** | ~18.0 GB | ~10.5 GB | **NO (Exceeds 8GB; requires 16GB+ or CPU offload)** |
| **2.0 Billion (2B)** | **400.0 MB** | ~34.0 GB | ~20.0 GB | **NO (Requires 24GB+ / A100)** |
| **7.0 Billion (7B)** | **1.4 GB** | ~115.0 GB | ~68.0 GB | **NO (Requires multi-GPU cluster)** |

**Conclusion on Model Scale:**
* **Local Pre-Training:** **NaviTrit-135M** is the optimal, compute-dense architecture that trains comfortably within the 8GB VRAM envelope (~2.4 – 3.8 GB).
* **Local Inference:** A pre-trained **1B or 2B ternary model** *can* run inference on your machine easily, because inference requires only the 200MB–400MB packed weights!

---

### B. Compute Throughput Math (Tokens vs. Wall Clock Time on RTX 4060)
On the RTX 4060, our measured training throughput on ternary graph models is:
$$\text{Throughput} \approx 6,500 - 7,000 \text{ tokens/second}$$

Let us calculate the wall clock time required for various dataset token sizes:

| Dataset Size | Wall Clock Time on Single RTX 4060 | Practical Feasibility |
|---|---|---|
| **5.12M Tokens (TinyStories Current)** | **12.3 minutes** (Measured) | Daily iteration |
| **50M Tokens (Curated Multi-Corpus Slice)** | **~2.1 hours** | Overnight / afternoon run |
| **100M Tokens (Full Multi-Corpus Pretraining)** | **~4.2 hours** | Standard single session |
| **1.0B Tokens (Extended Foundation Run)** | **~42.0 hours (1.75 days)** | Weekend run |
| **10.0B Tokens (Chinchilla-Optimal for 135M)** | **~420 hours (17.5 days)** | Requires background / multi-GPU |
| **4.0 Trillion Tokens (BitNet 2B4T Scale)** | **~18.4 YEARS** | Requires multi-node supercomputer |

---

## 2. Formal Specification of Track D-4 (Gate 15)

In compliance with `AGENTS.md`, we formally specify the 5 required parameters for the new track:

### 1. Specified Model: `NaviTrit-135M`
*   **Base Parameters:** 134,133,540 (~135M).
*   **Structure:** 12 Stationary Multi-Head Attention Tiles, 12 Stationary SwiGLU FFN Tiles.
*   **Hidden Dimension:** $d_{\text{model}} = 768$, $n_{\text{heads}} = 12$, $d_{\text{ff}} = 2048$.
*   **Non-Monotonic Enhancements:**
    *   Hop-Conditioned Tile Modulation (FiLM $\gamma_t, \beta_t = \text{MLP}(t)$).
    *   Dedicated Recurrent Reasoning Core ($v_{\text{reason}}$, Node $2L = 24$): Weight-tied BitLinear denoiser with internal contraction halting $\|\Delta s\|_\infty < \epsilon_{\text{reason}}$.
    *   Enhanced Flow Navigation Controller with Attention Diversity ($\ge 40\%$) and Layer Entropy penalties.

### 2. Specified Hardware
*   **Primary Execution:** Single local NVIDIA GeForce RTX 4060 GPU (8GB VRAM).
*   **Precision:** Master weights in FP32, activations in BF16/FP16, forward linear weights strictly in $\{-1, 0, +1\}$ via STE.
*   **Memory Management:** PyTorch gradient checkpointing enabled on recurrent steps; 8-bit AdamW optimizer.

### 3. Specified Workload & Multi-Corpus Dataset Mixture
To solve both narrative coherence and factual/relational reasoning, we define a **streaming balanced 4-way corpus mixture**:

```
                              MULTI-CORPUS MIXTURE (100M Tokens)
                                              │
         ┌───────────────────┬────────────────┴───────────────────┬───────────────────┐
         ▼                   ▼                                    ▼                   ▼
┌──────────────────┐┌──────────────────┐                ┌──────────────────┐┌──────────────────┐
│  FineWeb-EDU     ││  TinyStories     │                │  OpenMath / GSM  ││  The Stack Mini  │
│  (50% - 50M tok) ││  (20% - 20M tok) │                │  (15% - 15M tok) ││  (15% - 15M tok) │
│  - Factual logic ││  - Syntax & flow │                │  - Multi-hop     ││  - Algorithmic   │
│  - World knowl.  ││  - Entity track  │                │    reasoning     ││    syntax        │
└──────────────────┘└──────────────────┘                └──────────────────┘└──────────────────┘
```

1.  **HuggingFace FineWeb-EDU (50M tokens):** High-quality web text filtered for educational value, providing world knowledge and factual anchoring.
2.  **TinyStories / Stories (20M tokens):** Clean grammatical and narrative continuity.
3.  **GSM8K & Synthetic Math Problems (15M tokens):** Multi-step arithmetic and symbolic word problems to exercise the Recurrent Reasoning Core.
4.  **The Stack / Python Mini-Code (15M tokens):** Structured programmatic code requiring long-range indentation and bracket binding.

### 4. Specified Baseline
*   **Baseline 1 (Monotonic Feedforward BitRoute-135M):** Standard 12-layer sequential pipeline running through all 12 layers ($0 \to 1 \to \dots \to 11$).
*   **Baseline 2 (Unmodulated NaviTrit-135M):** Non-monotonic routing without hop modulation or reasoning core.

### 5. Specified Resource Cap
*   **Total Tokens:** Strict ceiling of **100,000,000 tokens (100M tokens)**.
*   **Wall Clock Cap:** **6.0 hours** maximum execution time on RTX 4060.
*   **VRAM Cap:** **6.0 GB** maximum allocated VRAM (leaving 2.0 GB headroom for OS/display).

---

## 3. Cloud / Cluster Extension Roadmap (For Multi-GPU / Future Compute)

If external compute or cloud credits become available in the future:
*   **Stage 1:** Train NaviTrit-135M on 2.7 Billion tokens (Chinchilla compute-optimal) on $4\times \text{A100}$ (approx 18 hours).
*   **Stage 2:** Scale to **NaviTrit-1B** (24 layers, $d_{\text{model}}=2048$, 200 MB ternary footprint) trained on 20B tokens on an $8\times \text{H100}$ node.

---

## 4. Conductor Status
- Registered as **Gate 15: Track D-4 — Frontier Scale Multi-Corpus Pretraining** in [`research/conductor-track.md`](file:///c:/Users/atooz/Programming/ternary-memory-research/research/conductor-track.md) and [`outputs/conductor-registry.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/conductor-registry.json).
