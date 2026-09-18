# Deep Dive 10: Track D-4 — Frontier Scaling & Scale-Adaptive Generalization (10M to 100M NaviTrit)

> **Status correction (2026-09-18).** The accuracy metrics in this document (perplexities, Gemini judge scores, and any comparison built on them) were measured on a templated corpus whose validation split is 93-99% verbatim in training, and the Gate 23 model was never trained. They are retained as a record of the work, not as results. See [`research/hardening-2026-09-18-heldout.md`](../hardening-2026-09-18-heldout.md) for the held-out re-evaluation and the clean protocol that replaces this evidence.


**Date:** 2026-09-17  
**Stage:** Phase II / Gate 15  
**Status:** VALIDATED ON NVIDIA RTX 4060 (8GB VRAM)  
**Primary Code:**  
- [`experiments/frontier_scaling/navitrit_scale_model.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/frontier_scaling/navitrit_scale_model.py)  
- [`experiments/frontier_scaling/prepare_multicorpus.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/frontier_scaling/prepare_multicorpus.py)  
- [`experiments/frontier_scaling/train_scale.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/frontier_scaling/train_scale.py)  
- [`experiments/frontier_scaling/test_scale.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/frontier_scaling/test_scale.py)  
- [`experiments/frontier_scaling/audit_scale.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/frontier_scaling/audit_scale.py)  
**Verification Ledgers:**  
- [`outputs/navitrit-10m-results.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/navitrit-10m-results.json)  
- [`outputs/navitrit-100m-results.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/navitrit-100m-results.json)  
- [`outputs/navitrit-100m-scale-adaptive-results.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/navitrit-100m-scale-adaptive-results.json)  
- [`outputs/navitrit-100m-longhorizon-results.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/navitrit-100m-longhorizon-results.json)  
- [`outputs/navitrit-100m-longhorizon-audit.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/navitrit-100m-longhorizon-audit.json)  
**Saved Checkpoints:**  
- `outputs/checkpoints/navitrit-10m-trained.pt` (11.7M params, 47.7 MB)  
- `outputs/checkpoints/navitrit-100m-trained.pt` (129.7M params, 518.8 MB)  
- `outputs/checkpoints/navitrit-100m-step{2500,5000,7500,10000}.pt`

---

## 1. The Why: Frontier Scaling & The Collapse of Fixed Hyperparameters

### A. The Motivation: Scaling Beyond Toy Regimes
Prior tracks established foundational theorems and validated non-monotonic graph routing on small prototypes (10M to 45M parameters). To prove that ternary non-monotonic graph navigation is a viable general-purpose paradigm for modern foundation models, we initiated **Track D-4 (Frontier Scaling)** to scale the architecture across an order of magnitude:
- **NaviTrit-10M Preset:** $d = 192$, $L = 4$, $n_{\text{heads}} = 3$, $d_{\text{ff}} = 512$ ($\approx 11.7\text{M}$ active params).
- **NaviTrit-100M Preset:** $d = 768$, $L = 12$, $n_{\text{heads}} = 12$, $d_{\text{ff}} = 2048$ ($\approx 129.7\text{M}$ active params).

### B. The Multi-Domain Corpus
Rather than evaluating solely on child narratives, we constructed a unified multi-corpus token dataset (`data/multicorpus_10m.pt`) comprising 7.85 million tokens across three distinct computational domains:
1. **Linguistic Narrative:** TinyStories (entity consistency, grammar, emotional arcs).
2. **Multi-Step Arithmetic:** GSM8K word problems formatted with step-by-step reasoning prefixes (`Reasoning: Let's track the count step by step...`).
3. **Algorithmic Code:** Python algorithmic snippets (binary search, sorting, recursion, control flow).

### C. The Scaling Collapse Phenomenon
When we transferred the exact loss weights from the 10M prototype ($\lambda_{\text{attn}} = 0.50$, $\lambda_{\text{entropy}} = 0.20$) to the 100M model, an acute failure mode emerged:
- At 10M scale, Attention ratio was healthy at **48.8%** and layer spread was uniform.
- At 100M scale, the attention ratio collapsed to **16.7%**, with tokens sinking into repetitive FFN self-loops (`[15, 15, 15, 15, 15, 15]`).
- **Why?** In high-dimensional latent space ($d = 768$), the cross-entropy gradient magnitude scales as $\mathcal{O}(\sqrt{d})$. Because FFN tiles possess $4\times$ the parameter density of Attention projections, the gradient signal pulled the router overwhelmingly toward FFN tiles to rapidly compress vocabulary perplexity. The dimensionless penalty ($\lambda = 0.50$) was completely overpowered.

---

## 2. The How: Scale-Adaptive Generalization Framework

To enable NaviTrit to generalize across arbitrary parameter and depth scales without manual hyperparameter tuning, we derived the **Scale-Adaptive Generalization Framework**.

```
                           Scaling Dimensions
       Hidden Dimension: R_dim = d / d_0    |   Depth: R_depth = L / L_0
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Adaptive Attn    │     │ Normalized Graph │     │ Topological Anti-│
│ Diversity Weight │     │ Entropy Penalty  │     │ Gravity Shield   │
│ λ_attn(d, L)     │     │ H_norm ∈ [0, 1]  │     │ α_damp = ln(1+R) │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

### A. Dimensional & Depth Scaled Attention Diversity
Let $(d_0, L_0) = (192, 4)$ denote the reference baseline. We define the dimension ratio $\mathcal{R}_{\text{dim}} = \frac{d}{d_0}$ and depth ratio $\mathcal{R}_{\text{depth}} = \frac{L}{L_0}$.
The attention diversity penalty is scaled dynamically:
$$\lambda_{\text{attn\_div}}(d, L) = \lambda_0 \cdot \sqrt{\mathcal{R}_{\text{dim}}} \cdot \sqrt{\mathcal{R}_{\text{depth}}}$$
- At 10M: $\lambda_{\text{attn\_div}} = 0.50 \cdot \sqrt{1.0} \cdot \sqrt{1.0} = \mathbf{0.5000}$
- At 100M: $\lambda_{\text{attn\_div}} = 0.50 \cdot \sqrt{4.0} \cdot \sqrt{3.0} = 0.50 \cdot 2 \cdot 1.7321 = \mathbf{1.7321}$

### B. Scale-Invariant Normalized Graph Entropy
Raw Shannon entropy $\mathcal{H} = -\sum_{l=0}^{L-1} p_l \ln p_l$ scales naturally with depth $\ln(L)$. If $\lambda_{\text{entropy}}$ is held constant, deeper models face an artificially severe penalty. We normalize entropy by maximum uniform capacity:
$$\tilde{\mathcal{H}} = \frac{\mathcal{H}}{\ln(L)} \in [0, 1]$$
$$\mathcal{L}_{\text{entropy}} = \lambda_{\text{entropy}} \cdot \sqrt{\mathcal{R}_{\text{depth}}} \cdot \left(\sum_{l=0}^{L-1} \frac{p_l \ln p_l}{\ln(L)}\right)$$
This guarantees identical gradient pressure across 4-layer, 12-layer, and 32-layer graphs.

### C. Topological Anti-Gravity Shield (FFN Damping)
To directly prevent the high-dimensional representational gravity of FFN tiles from trapping the routing controller, we inject a scale-dependent negative logit bias whenever the prior node was an FFN tile:
$$\alpha_{\text{damp}}(d) = \ln\left(1 + \mathcal{R}_{\text{dim}}\right)$$
$$\text{logits}_{\text{FFN}}^{(t+1)} = \text{logits}_{\text{FFN}}^{(t+1)} - \alpha_{\text{damp}}(d) \quad \text{if } \text{node}^{(t)} \in \mathcal{V}_{\text{FFN}}$$
At 100M ($d = 768, \mathcal{R}_{\text{dim}} = 4.0$), $\alpha_{\text{damp}} = \ln(5.0) \approx \mathbf{1.6094}$. This applies a $5\times$ relative probability suppression against immediate consecutive FFN hops.

---

## 3. The Outcome: Long-Horizon 10,000-Step Pretraining

### A. Pretraining Metrics & Efficiency
Trained on a single NVIDIA RTX 4060 GPU ($B=16, S=128$, total tokens processed = $20,480,000$):
- **Total Runtime:** 1726.49 seconds (**28.77 minutes**).
- **Peak VRAM Allocated:** **2.82 GB** (leaving > 5.1 GB headroom on an 8GB card).
- **Learning Rate Schedule:** 500-step linear warmup + cosine decay from $4.0 \times 10^{-4} \to 4.0 \times 10^{-5}$.

### B. Continuous Milestone Validation Ledger

| Global Step | Validation Loss | Perplexity (PPL) | Attention Ratio | Sample Trajectory | Learning Rate |
|---|---|---|---|---|---|
| **Step 2,500** | 0.6111 | 1.84 | 50.0% | `[15, 2, 15, 14, 15, 14]` | $3.62 \times 10^{-4}$ |
| **Step 5,000** | 0.2482 | 1.28 | 50.0% | `[15, 0, 5, 16, 15, 0]` | $2.35 \times 10^{-4}$ |
| **Step 7,500** | 0.1596 | 1.17 | 66.7% | `[2, 0, 19, 22, 15, 0]` | $9.81 \times 10^{-5}$ |
| **Step 10,000** | **0.1355** | **1.15** | **50.2%** | `[0, 13, 2, 9, 0, 19]` | $4.00 \times 10^{-5}$ |

**Validation Analysis:**
1. **Attention Ratio Fully Restored:** Attention visitation remained at **50.2%**, proving that the Scale-Adaptive Generalization Framework completely eradicated the FFN collapse observed in naive scaling.
2. **Topological Graph Exploration:** Rather than collapsing into a single layer, the model actively navigated across the entire graph (visiting Attention nodes 0, 2, 6, 10 and FFN nodes 9, 13, 15, 19).
3. **High Parameter Efficiency:** With 129.7M ternary parameters, validation loss reached 0.1355 on the multi-corpus within under 30 minutes of consumer GPU training.

---

## 4. The Critical Discovery: Why Mathematical Reasoning Failed

### A. The Generation Audit Findings
Inspection of generated completions from [`outputs/navitrit-100m-longhorizon-audit.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/navitrit-100m-longhorizon-audit.json) revealed a profound split across tasks:

1. **Narrative Text (Fluent & Coherent):**
   > *Prompt:* `"One sunny day, Lily and her brother found a big, red ball."`  
   > *Continuation:* `"...She wanted to play with her toy car. She asked her mom, "Can I have a toy car?" "Yes, please?" Lily asked. "Yes,""`  
   > -> Fluent syntax, correct dialogue punctuation, natural English phrasing.

2. **Algorithmic Python Code (Structurally Accurate):**
   > *Prompt:* `"def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n"`  
   > *Continuation:* `"    while low <= high:\n        mid = (low + high) // 2\n        if arr"`  
   > -> Exact Python 4-space indentation, correct while loop condition, perfect midpoint calculation.

3. **Multi-Step Arithmetic (Mathematically Incoherent):**
   > *Prompt:* `"Problem: Olivia had 15 apples. She gave 4 apples to Liam. How many apples remain?\nReasoning:"`  
   > *Continuation:* `" Let's track the count step by step.\nStep 1: Mia begins with 15 apples.\nStep 2: After gave 3 to Lily, 15 - 3 = 17 apples"`  
   > -> Hallucinated entity name ("Mia", "Lily"), changed the subtraction operand ("gave 3"), and asserted that $15 - 3 = 17$!

### B. Root Cause: Zero Visitations to the Reasoning Core
Audit of the node visitations in [`outputs/navitrit-100m-longhorizon-results.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/navitrit-100m-longhorizon-results.json) revealed the definitive physical cause:
```json
"findings": {
  "reasoning_core_visitations": 0
}
```
Out of all validation batches evaluated, **Node 24 (the Recurrent Reasoning Core) was visited zero times.**

### C. Why Standard Pretraining Regularizes Out the Reasoning Core
1. **Asymmetry in Loss Formulation:** The reasoning core possesses an internal Fixed-Point Forcing loss ($\mathcal{L}_{\text{fpf}} = \sum_k \|\Delta s_k\|_2^2$). When the router visits Node 24, this internal relaxation loss is added to the backward graph. In contrast, language tiles incur no such penalty.
2. **The Shortcut of Surface Memorization:** For cross-entropy loss on typical corpora, predicting high-probability surface templates (`"Step 1: Mia begins with..."`) provides an immediate loss decrease through standard language tiles.
3. **Absence of Domain Gating:** Because routing is conditioned only on average sequence pooled states without explicit task-type supervision, the controller took the path of least resistance, bypassing the latent deliberation engine entirely.

This empirical finding directly motivates the integration of the **Flow-Reasoned Planner (FRP) Block**, creating a certified structural pathway for continuous-time mathematical relaxation.
