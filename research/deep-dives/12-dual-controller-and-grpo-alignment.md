
> **Status correction (2026-09-18).** The accuracy metrics in this document (perplexities, Gemini judge scores, and any comparison built on them) were measured on a templated corpus whose validation split is 93-99% verbatim in training, and the Gate 23 model was never trained. This was internal working output that was never published; it is kept unchanged as part of the research record, and the ideas in it are what the current experiments test. See [`research/hardening-2026-09-18-heldout.md`](../hardening-2026-09-18-heldout.md) for the held-out re-evaluation and the clean protocol that replaces this evidence.

﻿# Deep Dive 12: Hierarchical Dual-Controller Architecture & Backbone-Frozen GRPO Policy Alignment for NaviTrit-100M

**Date:** 2026-09-17  
**Stage:** Phase II / Gate 16-B  
**Status:** IN PROGRESS (TRAINING & VERIFICATION)  
**Primary Code:**  
- [`experiments/frontier_scaling/navitrit_dual_model.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/frontier_scaling/navitrit_dual_model.py)  
- [`experiments/frontier_scaling/train_scale_dual_grpo.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/frontier_scaling/train_scale_dual_grpo.py)  
- [`experiments/frontier_scaling/hybrid_verifier.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/frontier_scaling/hybrid_verifier.py)  
- [`experiments/frontier_scaling/test_dual_model.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/frontier_scaling/test_dual_model.py)  
- [`experiments/frontier_scaling/gemini_benchmark.py`](file:///c:/Users/atooz/Programming/ternary-memory-research/experiments/frontier_scaling/gemini_benchmark.py)  
**Verification Ledgers:**  
- [`outputs/navitrit-100m-dual-grpo-results.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/navitrit-100m-dual-grpo-results.json)  
- [`outputs/navitrit-100m-gemini-benchmark.json`](file:///c:/Users/atooz/Programming/ternary-memory-research/outputs/navitrit-100m-gemini-benchmark.json)  
**Evaluated Checkpoints:**  
- `outputs/checkpoints/navitrit-100m-trained.pt` (Pretrained 10k backbone)  
- `outputs/checkpoints/navitrit-100m-dual-grpo.pt` (Aligned Dual-Controller)  

---

## 1. Executive Summary & Architectural Motivation

Following the 10,000-step pretraining run of NaviTrit-100M (Gate 15), objective frontier LLM benchmarking via Gemini 2.5 Flash revealed a critical insight:
1. **Strong Syntactic & Algorithmic Emergence (8.5/10 Code):** On Python code completion (`binary_search`), the ternary ($\pm 1, 0$) weights synthesized structurally valid code (`while low <= high: mid = (low + high) // 2`).
2. **Catastrophic Arithmetic Failure (1.0/10 Math):** On arithmetic deduction ($15 - 4$), the model produced hallucinated entities and erroneous arithmetic ($15 - 3 = 17$).
3. **Reasoning Core Bypass:** The latent Recurrent Reasoning Core (Node 24) experienced exactly zero visitations across pretraining due to the internal Fixed-Point Forcing (FPF) loss penalty.

To eliminate this deficit without destabilizing the high-performing code capabilities, we designed and implemented **Gate 16-B: The Hierarchical Dual-Controller Architecture with Frozen-Backbone GRPO Policy Alignment**.

---

## 2. Mathematical Architecture: The 4-Level Decoupled System

The Hierarchical Dual-Controller decouples global intent planning from local graph step transitions, slot-variable memory, and numerical relaxation:

```
                          ┌────────────────────────────────────────────────────────┐
                          │         Level 1: Global Flow Planner                   │
                          │   h_pool -> g_domain in [0, 1], T_budget in [4, 6]     │
                          │   b_prior[24] = +2.0 * g_domain                        │
                          └───────────────────────────┬────────────────────────────┘
                                                      │
                                                      ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Level 3: Persistent Entity Registers (Slot-Variable Memory)                                            │
 │ R = SlotAttention(Q_slots, h_ctx) in R^{B x 4 x d}                                                     │
 │ h_injected = h_token + tanh(gamma_inject) * OutProj(CrossAttention(h_token, R))                        │
 └────────────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                              │
                                              ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Level 2: Local Flow Navigation Controller (Neural ODE Velocity Field)                                  │
 │ dr/dt = v_phi(r | h_injected + g_domain, node_prev) in R^{3 * d_route}                                 │
 │ logits = W_head * (r + dt * dr) + b_prior - alpha_damp * delta(prev_node, curr_node)                   │
 └────────────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       │ (Hop t: Node Selection)                     │
                       ▼                                             ▼
           ┌───────────────────────┐                     ┌───────────────────────┐
           │ Stationary Language   │                     │ Level 4: Recurrent    │
           │ Tiles (Nodes 0..23)   │                     │ Reasoning Core        │
           │ Attn_l / FFN_l        │                     │ (Node 24: Latent ODE) │
           └───────────────────────┘                     │ ds/dt = D_theta(s|c)  │
                                                         └───────────────────────┘
```

### A. Level 1: Global Flow Planner
Given sequence context summary $h_{\text{pool}} = \frac{1}{S} \sum_{i=1}^S h_i$:
1. Computes domain intent scalar $g_{\text{domain}} \in [0, 1]$ indicating mathematical/algorithmic density.
2. Predicts discrete computation hop budget $T_{\text{budget}} = \text{argmax}(W_{\text{budget}} h_{\text{pool}}) + 2 \in [4, 6]$.
3. Projects node prior bias: $\mathbf{b}_{\text{prior}}[24] = 2.0 \cdot g_{\text{domain}}$.

### B. Level 2: Local Dynamic Flow Controller
Traverses the stationary module graph using a continuous Neural ODE velocity field:
$$\frac{d\mathbf{r}}{dt} = v_\phi(\mathbf{r} \mid \mathbf{h}_{\text{ctx}} + \mathbf{g}_{\text{domain}}, \mathbf{e}_{\text{prev}})$$
$$\mathbf{r}_{t+1} = \text{LayerNorm}(\mathbf{r}_t + \Delta t \cdot \frac{d\mathbf{r}}{dt})$$
$$\mathbf{z}_t = W_{\text{head}} \mathbf{r}_{t+1} + \mathbf{b}_{\text{prior}} - \alpha_{\text{damp}} \mathbf{m}_{\text{self-loop}}$$
Crucially, the input dimension is strictly formatted as $3 \times d_{\text{route}} = 384$, preserving **100% weight-compatibility** with the 10,000-step pretrained controller checkpoint (`controller.v_net.0.weight`).

### C. Level 3: Persistent Entity Registers
To eliminate entity bleed (e.g. "Olivia" mutating into "Mia", "15 apples" mutating into "30 crayons"), $M=4$ persistent latent memory slots $R \in \mathbb{R}^{B \times 4 \times d}$ are bound via slot-attention over the context:
$$R = \text{Softmax}\left(\frac{Q_{\text{slots}} (K_{\text{ctx}})^T}{\sqrt{d_k}}\right) V_{\text{ctx}}$$
Slot memory is injected into hidden representations at every hop:
$$\mathbf{h}_{\text{injected}} = \mathbf{h} + \tanh(\gamma_{\text{inject}}) \cdot W_{\text{out}} \text{CrossAttn}(\mathbf{h}, R)$$
By initializing $\gamma_{\text{inject}} = 0$, the registers provide an **exact identity warm-start** at initialization.

### D. Level 4: Recurrent Reasoning Core (Node 24)
A weight-tied BitLinear SwiGLU recurrent denoiser:
$$\mathbf{s}_{k+1} = \mathbf{s}_k + 0.5 \cdot W_{\text{state}} \left( W_{\text{down}} (\text{SiLU}(W_{\text{gate}} \mathbf{x}) \odot W_{\text{up}} \mathbf{x}) \right)$$
$$\mathbf{h}_{\text{out}} = \mathbf{h}_{\text{in}} + \mathbf{s}^*$$
By zero-initializing $W_{\text{state}}$ and its bias, Node 24 acts as an identity pass-through at step 0 ($\mathbf{s} = 0$), preventing representation corruption prior to arithmetic specialization.

---

## 3. Mathematical Autopsy of Initial GRPO Collapse

Our empirical audit of the initial GRPO alignment run revealed three foundational failure mechanisms:

### A. The Learned Repulsive Vector in `node_head`
In `outputs/checkpoints/navitrit-100m-trained.pt`, inspecting the weights of the classification head revealed:
$$\|W_{\text{head}}[0..23]\|_2 \approx 0.15 - 0.30$$
$$\|W_{\text{head}}[24]\|_2 = \mathbf{1.4131}, \quad \mathbf{b}_{\text{head}}[24] = -0.1148$$
Because Node 24 was penalized during pretraining by $\mathcal{L}_{\text{fpf}}$, the optimizer drove $W_{\text{head}}[24]$ into the exact opposite hemisphere of all controller states $\mathbf{r}$:
$$\mathbf{r} \cdot W_{\text{head}}[24] \approx -9.0$$
Even with a $+2.0$ prior boost, the resulting logit was $-7.037$ (compared to $+0.77$ for language tiles), yielding an activation probability of $P(\text{Node 24}) = e^{-7.8} \approx 0.04\%$.

**Resolution:** Resetting $W_{\text{head}}[24] = 0$ and $\mathbf{b}_{\text{head}}[24] = 0$ at warm-start restored neutral routing, enabling Node 24 to be actively selected when $g_{\text{domain}} = 1$.

### B. Degenerate 2-Node Mode Collapse
Without reference policy regularization, standard GRPO on sparse rewards collapsed into a 2-node limit cycle:
$$\mathcal{T} = [5, 10, 5, 10, 5] \quad (\text{FFN}_2 \leftrightarrow \text{Attn}_5)$$
Because Node 10 was an attention tile and the original anti-gravity shield only damped consecutive FFN nodes, the router discovered that oscillating between 5 and 10 was a zero-penalty path.

**Resolution:** 
1. Generalizing anti-gravity damping to all immediate self-loops:
$$\mathbf{z}[\text{prev\_node}] = \mathbf{z}[\text{prev\_node}] - \alpha_{\text{damp}}$$
2. Adding KL divergence regularization against the frozen reference controller $\pi_{\text{ref}}$:
$$\mathcal{L}_{\text{KL}} = \beta_{\text{KL}} \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}), \quad \beta_{\text{KL}} = 0.05$$
3. Adding policy entropy bonus:
$$\mathcal{L}_{\text{ent}} = -\beta_{\text{ent}} \mathcal{H}(\pi_\theta), \quad \beta_{\text{ent}} = 0.02$$

### C. The Backbone Freezing Law
Updating ternary backbone weights ($W \in \{-1, 0, +1\}$) on small prompt templates causes catastrophic forgetting of general language representations within 600 steps. In our initial test, story prompts began generating math sentences ("Sam, 24 - 2 = 20 pencils remain").

**Resolution:**
The entire ternary language backbone (124.15M parameters: `embed_tokens`, `embed_positions`, `attn_tiles`, `attn_norms`, `ffn_tiles`, `ffn_norms`, `lm_head`, `final_norm`) is permanently frozen with `requires_grad = False`. Optimization is strictly partitioned:
1. **Router Policy Gradient:** Updates `global_planner`, `controller`, `entity_registers`.
2. **Reasoning Core Arithmetic Gradient:** Updates `reasoning_core` on target deduction tokens.

---

## 4. Multi-Task Hybrid Verifier & Reward Formulation

The hybrid verifier evaluates candidate trajectories without external API latency during training:

$$\mathcal{R}(y \mid x, \text{type}) = \begin{cases}
+1.5 & \text{if math exact target match } (y_{\text{pred}} == y_{\text{true}}) \cr
+0.5 & \text{if math contains valid equation } (a - b = c) \cr
-1.0 & \text{if math calculation error or hallucination} \cr
+1.0 & \text{if code syntax valid via AST and completes block} \cr
-0.5 & \text{if code syntax error} \cr
+1.0 & \text{if story retains prompt entity and diversity} \ge 0.80 \cr
-0.5 & \text{if story degenerate repetition}
\end{cases}$$

Group Relative Advantage:
$$A_k = \frac{\mathcal{R}_k - \text{mean}(\mathcal{R})}{\text{std}(\mathcal{R}) + 10^{-6}}$$
GRPO Objective:
$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{K} \sum_{k=1}^K A_k \ln \pi_\theta(\mathcal{T}_k) + 0.05 \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) - 0.02 \mathcal{H}(\pi_\theta)$$
