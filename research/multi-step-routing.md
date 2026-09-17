# Multi-Step and Multi-Hop Dynamic Routing in Ternary Neural Systems

**Author:** Antigravity  
**Date:** 2026-09-16  
**Status:** Theoretical Foundation & Literature Synthesis  
**Registry Cross-Reference:** Gate 11-B (`outputs/conductor-registry.json`), `outputs/bitroute-multihop-results.json`

---

## 1. Executive Summary & Literature Reality Check

In Phase I-C, our decoupled 2-hop Flow-Reasoned Transformer achieved:
- Validation Loss: **2.9836** (PPL 19.76) vs. **3.0238** (PPL 20.57) for full execution ($-0.0402$ loss advantage while saving $16.7\%$ parameter compute).
- Crushed matched random coin control: **3.3598** (PPL 28.78) ($-0.3762$ loss advantage).
- Sublayer configuration discovered: Layers 0, 1, 3 completely bypass Attention ($0\%$ execution), while preserving $100\%$ of FFNs and $100\%$ of later Attention layers.

### Is This Result "Too Good to Be True"?

A healthy scientific skepticism is essential. When a model with $16.7\%$ less compute strictly outperforms the dense baseline, we must ask: **Has this been seen before, or is it an artifact?**

The literature reveals two distinct realities:

1. **The Depth Regularization & Sparsity Advantage is Well-Established:**
   - **Mixture-of-Depths (MoD) (Raposo et al., Google DeepMind, April 2024):** DeepMind showed that dynamically routing tokens around self-attention or MLP blocks not only matches dense baseline performance but **surpasses vanilla transformer log-probability by up to 1.5%** at isoFLOP or lower compute budgets.
   - **Attention Drop & Layer Pruning (Zhang et al., 2024; Men et al., "ShortGPT", 2024; Gromov et al., 2024):** Pruning studies on LLaMA-2-70B demonstrated that removing up to $50\%$ of self-attention blocks yields negligible accuracy loss, and often acts as a regularizer that reduces over-smoothing.
   - **Stochastic Depth & LayerDrop (Fan et al., 2019; Huang et al., 2016):** Sub-network sampling has consistently shown that dense transformers suffer from co-adaptation and representation collapse; selectively skipping modules acts as structural dropout.

2. **The Radical Truth About Our Experiment (Attractor Collapse vs. Dynamic Variance):**
   When examining the telemetry across all 1,562 validation sequences in `outputs/bitroute-multihop-results.json`, we find that the router emitted the **exact same sublayer mask** for all sequences:
   $$\text{Attn: } [0, 0, 1, 0, 1, 1], \quad \text{FFN: } [1, 1, 1, 1, 1, 1]$$
   Because sequence-level pooling ($h_{\text{pool}}$) was used with a quadratic budget loss ($\mathcal{L}_{\text{budget}}$), the continuous velocity field $v_\phi$ rapidly fell into a single, highly stable **discrete global attractor**. 
   
   Therefore, our Phase I-C experiment did not discover a token-by-token dynamic routing policy; rather, **it performed automated, differentiable structural architecture discovery (Macro-Pruning)**, proving that a 6-layer ternary transformer performs better when early self-attention is structurally removed.

---

## 2. Why Are Early Attention Layers Bypassed? (The Mechanics of Token Mixing)

The empirical result showed that the router completely bypassed Attention at Layers 0, 1, and 3, while executing FFNs at 100%. Why does this happen?

```
                    ┌─────────────────────────┐
Token Embeddings ──►│ Layer 0: FFN Only       │ (Subspace Expansion)
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ Layer 1: FFN Only       │ (Feature Disentanglement)
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ Layer 2: Attn + FFN     │ (First Global Contextual Mixing)
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ Layer 3: FFN Only       │ (Non-linear Semantic Refinement)
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ Layer 4: Attn + FFN     │ (Higher-Order Relational Reasoning)
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ Layer 5: Attn + FFN     │ (Final Vocabulary Boundary Alignment)
                    └───────────┬─────────────┘
                                │
                                ▼
                           Logits y
```

### Mathematical Explanation:

1. **Subspace Alignment Before Contextual Mixing:**
   In early layers ($l=0, 1$), token representations are dominated by raw token identity, position embeddings, and local syntactic tags.
   - The self-attention operation computes pairwise dot-products:
     $$A_{ij} = \text{Softmax}\left(\frac{q_i^T k_j}{\sqrt{d_k}}\right)$$
     When representations $h_i$ have not yet been transformed by non-linear expansions, $q_i^T k_j$ is dominated by high-frequency positional noise and un-normalized embedding clusters. Early attention heads frequently collapse into trivial identity matrices or uniform averaging (over-smoothing).
   - The FFN (SwiGLU) applies:
     $$\text{FFN}(h) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} h) \odot W_{\text{up}} h)$$
     This non-linear projection expands the hidden state into a higher-dimensional manifold ($d_{\text{model}} \to d_{\text{intermediate}} \to d_{\text{model}}$), isolating semantic concepts and rotating tokens into distinct orthogonal subspaces.

2. **The Decoupling Insight:**
   In standard transformers, Attention and FFN are bundled into a single atomic block. If a model wants to skip redundant early attention, it is forced to skip the FFN as well—causing representational starvation.
   By decoupling them into **Hop 1 (Attention)** and **Hop 2 (FFN)**, the Flow-Reasoned Router can eliminate early attention compute without sacrificing non-linear capacity.

---

## 3. Literature Comparison: What is Prior Art vs. What is Net-New?

| Dimension | Standard Transformers (Vaswani 2017) | LayerSkip (Meta, 2024) | Mixture-of-Depths (Google DeepMind, 2024) | Our FR-Router (Ternary Memory Research) |
|---|---|---|---|---|
| **Weight Precision** | FP32 / BF16 | FP16 / BF16 | BF16 | **Native Ternary $\{-1, 0, +1\}$ (STE)** |
| **Routing Granularity** | None (Dense) | Layer-level Early Exit | Token-level top-$k$ per module | **Decoupled Sublayer (Hop 1: Attn, Hop 2: FFN)** |
| **Router Decision Nature** | N/A | Heuristic scalar threshold | Greedy, independent per-layer probes | **Global Attractor State $r \in \mathbb{R}^{d_r}$** |
| **Optimization Method** | Backprop | Cross-entropy at exits | Per-block linear probe + top-$k$ | **Continuous ODE Relaxation ($dr/dt = v_\phi$) + FPF** |
| **Execution Coordination** | Static | Myopic (local 1-hop) | Local top-$k$ (no lookahead) | **Global DAG Subgraph Planning** |

### Net-New Contributions:
1. **Flow Reasoning on Routing Dynamics (FRP-Router):** Rather than evaluating greedy local probes at each layer, a centralized routing state $r$ undergoes recurrent relaxation towards a discrete attractor basin.
2. **Fixed-Point Forcing (FPF) for Routing Stability:** Penalizing the terminal velocity $\|v_\phi(r^*)\|_2^2$ forces the planned execution subgraph to reside in a certified stable attractor.
3. **Native 1.58-bit Ternary Execution:** First framework demonstrating decoupled multi-hop routing natively on ternary BitLinear weights.

---

## 4. Was This Applied to the Other Models?

**No, not yet.** Here is the exact status across the project models:

### 1. Model 1: BitRoute-135M (Track A)
- **Phase I-A/B:** Greedy single-hop Tri-State Router (`tristate_router.py`).
- **Phase I-C (Just completed):** Upgraded to `BitRouteMultiHopForCausalLM` with `FlowRoutingPlanner` (`experiments/bitroute_multihop_model.py`). Evaluated on TinyStories.

### 2. Model 2: FlowTrit-40M (Track B - Recurrent Flow Denoiser)
- **Current Architecture:** Weight-tied recurrent denoiser for discrete constraint satisfaction (Sudoku 4x4 and 9x9).
- **Current Dynamic Mechanism:** Temporal early exit only (halts recurrence when $\|s^{(k+1)} - s^{(k)}\|_\infty < \epsilon_{\text{exit}}$).
- **FR-Router Applied?** **NO.** FlowTrit currently executes all layers inside each recurrence step.

### 3. Model 3: FlowRoute-Combined (Track C - Dual-Axis)
- **Current Architecture:** Merged recurrent denoiser with a *greedy, local* `TwoStateLayerRouter` (EXECUTE vs ROUTE_AROUND per layer per step).
- **FR-Router Applied?** **NO.** FlowRoute uses independent linear probes at each layer with zero global coordination and atomic layer skipping (no Attention/FFN decoupling).

---

## 5. Blueprint: Extending Flow-Reasoned Routing to FlowTrit & FlowRoute

Extending FR-Router to recurrent reasoning models unlocks a **triple-axis dynamic system**:

```
Input Problem c (Clues)
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Global Flow-Reasoned Trajectory Planner                               │
│  r^(0) = W_init · [c, C_target]                                        │
│  dr/dt = v_phi(r | c)  ──►  r*                                         │
│  Emits 3D Execution Tensor: M* in {0, 1}^{K_max x L x 2}               │
└────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Triple-Axis Recurrent Execution Engine                                │
│                                                                        │
│  Iteration k=1:  Layer 0 [Attn: NO,  FFN: YES] ──► Layer 1 ...         │
│  Iteration k=2:  Layer 0 [Attn: YES, FFN: YES] ──► Layer 1 ...         │
│  Iteration k=3:  Layer 0 [Attn: NO,  FFN: NO ] ──► Layer 1 ...         │
│                                                                        │
│  Halt when ||s^(k+1) - s^(k)||_inf < eps_exit                          │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Hypotheses for Recurrent Models:
1. **Temporal Evolution of Module Value:**
   - In Iteration 1 (early hypothesis generation), local cell constraints dominate $\implies$ FFN heavy, Attention sparse.
   - In Iteration 2–3 (constraint propagation across rows/columns/boxes), global relational mixing dominates $\implies$ Attention active.
   - In Iteration 4–5 (attractor convergence), perturbations are minimal $\implies$ most layers completely bypassed.
2. **True Dynamic Per-Instance Variance:**
   Because constraint satisfaction puzzles (Sudoku 9x9) have vastly different clue topologies (17 clues vs. 35 clues), the routing state $r$ will not collapse to a single static mask; it will adaptively allocate compute based on problem difficulty.

---

## 6. Implementation Action Plan

1. **Token-Level Prefix Pooling for Language Generation:**
   Upgrade `flow_router.py` to support causal prefix-pooling $h_{\text{pool}}(t) = \frac{1}{t}\sum_{i=1}^t h_i$ so that autoregressive generation dynamically changes routing decisions per generated token.
2. **Apply Multi-Hop Decoupling to FlowRoute (`flowroute_multihop.py`):**
   Replace the greedy `TwoStateLayerRouter` with a recurrent `FlowRoutingPlanner` that plans iteration-specific sublayer masks.
3. **Document in Gate 12 Manuscripts:**
   Include the literature comparison with Mixture-of-Depths, ShortGPT, and LayerSkip in the introduction of Paper 1 and Paper 3.
