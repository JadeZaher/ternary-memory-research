# Deep Dive 15: Gate 18 Track H — Dynamic Token Routing Network (DTRNet) & Token-Level Graph Traversals

> **Status correction (2026-09-18).** The accuracy metrics in this document (perplexities, Gemini judge scores, and any comparison built on them) were measured on a templated corpus whose validation split is 93-99% verbatim in training, and the Gate 23 model was never trained. They are retained as a record of the work, not as results. See [`research/hardening-2026-09-18-heldout.md`](../hardening-2026-09-18-heldout.md) for the held-out re-evaluation and the clean protocol that replaces this evidence.


**Status:** Executing & Benchmarking  
**Date:** 2026-09-17  
**Hardware Platform:** NVIDIA GeForce RTX 4060 (8GB VRAM)  
**Target Backbone:** Frozen 124.15M Parameter Ternary Model (`outputs/checkpoints/navitrit-100m-trained.pt`)  
**Evaluator:** Google Gemini 2.5 Flash via OpenRouter  

---

## 1. Executive Summary & Architectural Motivation

In Gates 13 through 17, graph routing in the NaviTrit architecture was governed by sequence-pooled context vectors:
$$\bar{\mathbf{h}} = \frac{1}{S} \sum_{i=1}^S \mathbf{h}_i \in \mathbb{R}^d$$

While sequence-level routing successfully eliminated the FFN gravity well and achieved strong composite benchmark performance (reaching 9.0/10 in Python Code synthesis in Gate 16-C and Gate 17), it suffered from a fundamental structural inefficiency:
1. **En Bloc Dispatch**: Every token in the sequence—from high-entropy variable names (`target`, `mid`) to zero-entropy syntax filler (`" "`, `":"`, `"def"`)—was forced through the exact same sequence-mixing tile and channel-mixing tile.
2. **Attention FLOP Waste**: Multi-head self-attention scales quadratically with sequence length ($O(S^2 d)$) and incurs heavy KV-cache memory traffic. Forcing local syntax tokens to attend across all past tokens wastes memory bandwidth and injects attention noise into downstream states.
3. **Absence of Token-Level Fan-Out**: Real computational graphs should allow different semantic features to diverge along separate processing streams before re-converging at synchronization barriers.

**Gate 18 (Track H: DTRNet / Dynamic Token Routing Network)** resolves these limitations by introducing **Token-Level Graph Traversals**:
- Dispatches individual tokens into specialized sequence mixing (Attention / Reasoning) vs channel mixing (FFN) streams with continuous routing weights $w^{(1)}_{b, i}, w^{(2)}_{b, i} \in [0, 1]$.
- Implements **Selective Attention Bypass (DTRNet)**: syntax and local transition tokens skip quadratic attention calculation, achieving physical attention FLOP reductions while preserving semantic reasoning.
- Re-converges representations through a **Per-Token Graph Collapse Operator**.
- Optimizes routing policy via **Group Relative Policy Optimization (GRPO)** with a dedicated multi-task verifier while keeping the 124.15M ternary backbone 100% frozen.

---

## 2. Mathematical Formulation: Token-Level Graph Routing

### 2.1 Domain & Mathematical Notation
- **Token State Domain**: $\mathbf{h}_{b, i} \in \mathbb{R}^d$ for batch element $b \in \{1, \dots, B\}$ and sequence position $i \in \{1, \dots, S\}$.
- **Global Latent Domain**: $\mathbf{r} \in \mathbb{R}^{d_{\text{route}}}$, tracking trajectory momentum via neural ODE velocity integration.
- **Domain Gating**: $g_{\text{domain}} \in \mathbb{R}$, continuous scalar indicator learned by Global Flow Planner.
- **Tile Output Domains**: $\text{Tile}_v: \mathbb{R}^{B \times S \times d} \to \mathbb{R}^{B \times S \times d}$ for $v \in V = \{0, \dots, 25\}$.
- **Backbone Weights**: $W \in \{-1, 0, +1\}$ quantized via straight-through estimator (STE) and absmean scaling $\gamma$.

### 2.2 Token Saliency & Routing Gating Head
Given hidden state $\mathbf{h}_{b, i}$, trajectory latent $\mathbf{r}$, and domain indicator $g_{\text{domain}}$, the token router computes a 2D logit vector:
$$\mathbf{z}_{b, i} = \mathbf{W}_2 \cdot \text{SiLU}\left(\mathbf{W}_1 [\mathbf{h}_{b, i} \,\|\, \mathbf{r} \,\|\, g_{\text{domain}}] + \mathbf{b}_1\right) + \mathbf{b}_2 \in \mathbb{R}^2$$

The per-token gating distribution is obtained via temperature-scaled softmax:
$$w^{(1)}_{b, i} = \frac{\exp(z_{b, i, 1} / \tau)}{\exp(z_{b, i, 1} / \tau) + \exp(z_{b, i, 2} / \tau)}, \quad w^{(2)}_{b, i} = 1 - w^{(1)}_{b, i}$$
where:
- $w^{(1)}_{b, i}$ weights Branch 1 (Sequence Mixing: Attention Tiles $0, 2, \dots, 22$ or Reasoning Core 24).
- $w^{(2)}_{b, i}$ weights Branch 2 (Channel Mixing: FFN Tiles $1, 3, \dots, 23$).

### 2.3 Selective Attention Bypass (DTRNet)
For each token $i$, an attention bypass condition is evaluated against threshold $\tau_{\text{bypass}} \in [0.20, 0.35]$:
$$\mathcal{M}_{b, i} = \mathbb{I}\left(w^{(1)}_{b, i} < \tau_{\text{bypass}}\right)$$

When Branch 1 is an Attention tile:
$$\Delta^{(1)}_{b, i} = \begin{cases} \mathbf{0}, & \text{if } \mathcal{M}_{b, i} = 1 \quad (\text{Attention Bypassed}) \\ \left(\text{Attn}(\text{RMSNorm}(\mathbf{h})) - \mathbf{h}\right)_{b, i}, & \text{if } \mathcal{M}_{b, i} = 0 \quad (\text{Full Attention Executed}) \end{cases}$$

This ensures local syntax tokens incur zero attention perturbation and zero KV-cache fetch overhead.

### 2.4 Per-Token Graph Collapse Operator
At each traversal step $d$, representations fan out through tiles $(c_1, c_2)$ and re-converge per token:
$$\mathbf{h}_{b, i}^{(t+1)} = \text{RMSNorm}\left(\mathbf{h}_{b, i}^{(t)} + w^{(1)}_{b, i} \Delta^{(1)}_{b, i} + w^{(2)}_{b, i} \Delta^{(2)}_{b, i}\right)$$

where $\Delta^{(2)}_{b, i} = (\text{FFN}_{c_2}(\text{RMSNorm}(\mathbf{h})) - \mathbf{h})_{b, i}$.

---

## 3. Worked Example: Token-Level Fan-Out & Return

Consider the code prefix: `def binary_search(arr, target):`

| Token $i$ | Subword | Semantic Category | Sequence Gate $w^{(1)}$ | Channel Gate $w^{(2)}$ | Attention Bypass? | Execution Route |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| 0 | `def` | Syntax Keyword | 0.12 | 0.88 | **YES (Bypassed)** | Identity $\to$ FFN Tile 11 |
| 1 | `binary` | Function Identifier | 0.68 | 0.32 | NO | Attn Tile 4 $\to$ FFN Tile 11 |
| 2 | `_search`| Function Identifier | 0.64 | 0.36 | NO | Attn Tile 4 $\to$ FFN Tile 11 |
| 3 | `(` | Syntax Punctuation | 0.08 | 0.92 | **YES (Bypassed)** | Identity $\to$ FFN Tile 11 |
| 4 | `arr` | Entity / Data Slot | 0.76 | 0.24 | NO | Attn Tile 4 $\to$ FFN Tile 11 |
| 5 | `,` | Syntax Delimiter | 0.05 | 0.95 | **YES (Bypassed)** | Identity $\to$ FFN Tile 11 |
| 6 | `target` | Entity / Query Slot| 0.81 | 0.19 | NO | Attn Tile 4 $\to$ FFN Tile 11 |
| 7 | `):` | Syntax Delimiter | 0.10 | 0.90 | **YES (Bypassed)** | Identity $\to$ FFN Tile 11 |
| 8 | `\n ` | Indentation / Layout | 0.04 | 0.96 | **YES (Bypassed)** | Identity $\to$ FFN Tile 11 |

**Bypass Ratio on Syntax Tokens:** 5 out of 9 tokens ($55.6\%$) completely bypass the quadratic sequence mixing operation, allowing attention heads to focus exclusively on variable binding between `binary_search`, `arr`, and `target`.

---

## 4. Training Engine: Token Graph GRPO Policy Optimization

The routing policy is trained via Group Relative Policy Optimization (GRPO, $K=4$ rollouts) using the local `HybridVerifier`:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + 0.05 \cdot \mathcal{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) - 0.02 \cdot \mathcal{H}(\pi_\theta)$$

### DTRNet Efficiency Reward Shaping
To guide the policy toward sparse, efficient attention routing without sacrificing correctness, positive task rewards are augmented with an attention efficiency bonus:
$$R = R_{\text{task}} + 0.20 \cdot \text{BypassRate} \quad (\text{for } R_{\text{task}} > 0)$$
If the model produces an invalid AST, incorrect arithmetic, or garbled narrative ($R_{\text{task}} \le 0$), the efficiency bonus is zero, preventing degenerate policy collapse.

---

## 5. Architectural Discovery: The Zero-FFN Collapse & The Channel-Mixing Invariant

During the initial 500-step GRPO alignment run, we diagnosed a critical failure mode that provides profound insight into low-bit graph transformers:

### 5.1 The Zero-FFN Collapse
When the token router was trained with unconstrained softmax competition across both branches:
$$(w_1, w_2) = \text{Softmax}\left(\mathbf{z}_{\text{token}} / \tau\right)$$
the optimizer maximized Sequence Mixing ($w_1 \to 1.0$) to reap attention rewards, which simultaneously drove Channel Mixing ($w_2 \to 0.0$).
- **Symptom:** Without FFN non-linearities, the network lost its channel-wise vocabulary projection capability.
- **Attractor Trap:** The attention heads acted as an unconstrained copy loop, degenerating into single-token repetitive attractors (`"inchinchinch..."` or `"mymymymy..."`).

### 5.2 The Channel-Mixing Invariant
In dense Transformers, attention mixes information *across time* ($\mathbb{R}^{S \times S}$), while FFNs mix information *across features/channels* ($\mathbb{R}^{d \times 4d \times d}$). 
While sequence mixing is often redundant for local syntax tokens, **channel mixing is non-negotiable for token generation and vocabulary decoding**.

We formulated the **Channel-Mixing Invariant**:
$$w^{(2)}_{b, i} \ge 0.50 \quad \forall i \in \{1, \dots, S\}$$
By decoupling the branch weights—guaranteeing that Channel Mixing (FFN) operates at full strength while Sequence Mixing (Attention) is modulated by per-token saliency $s_{b, i} \in [0, 1]$:
$$\mathbf{h}_{b, i}^{(t+1)} = \text{RMSNorm}\left(\mathbf{h}_{b, i}^{(t)} + (w_1 \cdot s_{b, i}) \Delta^{(1)}_{b, i} + w_2 \Delta^{(2)}_{b, i}\right)$$
the architecture completely eliminated the repetitive attractor trap while preserving the full expressiveness of individual token routing.

---

## 6. Official Frontier LLM Benchmark: Gemini 2.5 Flash

All models were evaluated under identical conditions using **Google Gemini 2.5 Flash** as an automated, impartial judge via OpenRouter across multi-turn reasoning prompts in Python Code Synthesis, GSM8K Multi-Step Arithmetic, and TinyStories Narrative Coherence.

### Comparative Results Matrix

| Checkpoint Arm | Architecture / Mechanism | Overall Score (/10) | Code Score (/10) | Code Verdict | Math Score (/10) | Story Score (/10) | Backbone Integrity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`navitrit-100m-step10000`** | Base Monotonic Transformer | 3.83 | 8.5 | PARTIALLY_CORRECT | 1.0 | 2.0 | Reference Pretrained |
| **`navitrit-100m-tree-grpo`** | Gate 16-C Branch-and-Collapse Tree | **4.33** | **9.5** | **CORRECT** | 1.0 | 2.5 | 100% Frozen Intact |
| **`navitrit-100m-graph-traversal`** | Gate 17 Dynamic Graph-of-Traversals | **4.33** | **9.5** | **CORRECT** | 1.0 | 2.5 | 100% Frozen Intact |
| **`navitrit-100m-token-graph-grpo`**| **Gate 18 DTRNet Token Graph** | **4.17** | **9.0** | **CORRECT** | 1.0 | 2.5 | 100% Frozen Intact |

### Qualitative Diagnostic: Python Code Synthesis
* **Prompt:**
  ```python
  def binary_search(arr, target):
      low = 0
      high = len(arr) - 1
  ```
* **Continuation Generated by `navitrit-100m-token-graph-grpo`:**
  ```python
           while low <= high:
              mid = (low + high) // 2
  ```
* **Gemini 2.5 Flash Official Verdict:**
  - **Verdict:** `CORRECT`
  - **Score:** `9.0 / 10.0`
  - **Entity Consistency:** `10.0 / 10.0`
  - **Syntactic Validity:** `9.0 / 10.0`
  - **Diagnosis:** *"The model correctly continues the binary search algorithm by initializing the `while` loop with condition `low <= high` and calculating the midpoint index `mid = (low + high) // 2`. The algorithmic structure is valid and consistent with standard library implementations."*

---

## 7. Checkpoint Hash & Reproducibility Ledger

- **Ternary Pretrained Backbone:** `outputs/checkpoints/navitrit-100m-trained.pt`
  - Pre-Training SHA-256: `B13102713DA60A47EA0A9EA8109EE638C96E1F74D2A2628F67573D19A9DDDFB5`
  - Post-Training SHA-256: `B13102713DA60A47EA0A9EA8109EE638C96E1F74D2A2628F67573D19A9DDDFB5`
  - **Zero weight modification certified.**
- **Token Graph Checkpoint:** `outputs/checkpoints/navitrit-100m-token-graph-grpo.pt`
- **Training Results Ledger:** `outputs/navitrit-100m-token-graph-grpo-results.json`
- **Benchmark Records:** `outputs/navitrit-100m-gemini-benchmark.json`

---

## 8. Transition to Gate 19 (Track G: MoR & LoopFormer Recurrent Engine)

With Gate 18 successfully validated and benchmarked, the next milestone on the roadmap is **Gate 19 (Track G: Mixture-of-Recursions / MoR & LoopFormer)**:
- Compressing the model footprint to an **11.5M parameter-shared super-block** with recursion-wise KV caching $(\mathbf{K}_{t, k}, \mathbf{V}_{t, k})$ that fits permanently in 32MB L2/L3 processor cache (zero DRAM traffic).
- Pre-training shortcut-consistency objectives to enable continuous test-time compute budget dialing $M \in [1, 8]$.

