# Theory of Flow-Reasoned Dynamic Routing (FR-Router)

**Date:** 2026-09-16  
**Status:** Theoretical Foundation & Architecture Specification  
**Context:** Builds directly on empirical hardening results (`research/hardening-2026-09-16.md`, `outputs/bitroute-tinystories-lambda0.2.json`) and the limitations of local single-hop greedy dispatch.

---

## 1. The Mathematical Nature of Residual Representations

In modern decoder transformers with residual connections, token representations evolve along an additive integration path:

$$h_0 = \text{Embed}(x) + W_{\text{pos}}$$

$$h_{l+1} = h_l + \Delta h_l(h_l)$$

$$\Delta h_l(h_l) = f_l^{\text{attn}}(\text{LN}_1(h_l)) + f_l^{\text{ffn}}(\text{LN}_2(h_l + f_l^{\text{attn}}(\text{LN}_1(h_l))))$$

$$y = \text{Softmax}(W_{\text{head}} \cdot \text{LN}_{\text{final}}(h_L))$$

### Continuous Limit: Residual Streams as Neural ODEs
As layer depth $L \to \infty$ with appropriate normalization step $\Delta t = 1/L$:

$$\frac{dh(t)}{dt} = f(h(t), t; \theta)$$

The hidden representation $h(t) \in \mathbb{R}^{d_{\text{model}}}$ traces a continuous trajectory through semantic space. Each layer $l$ does not replace the state; it injects a **velocity vector** $\Delta h_l$ that rotates and shifts $h(t)$ toward the decision boundaries of the output projection $W_{\text{head}}$.

---

## 2. Why Greedy 1-Hop Routing Fails

In existing dynamic depth literature (LayerSkip, Early-Exit, Mixture-of-Depths) and our initial Phase I-A implementation, routing is formulated as **local, uncoordinated, greedy decisions**:

$$\text{At layer } l: \quad a_l = \text{Router}_l(h_l) \in \{\text{EXECUTE}, \text{ROUTE\_AROUND}, \text{EARLY\_EXIT}\}$$

This formulation suffers from three fatal mathematical defects:

### A. Non-Markovian Dependency & Subspace Starvation
Layer $l+k$ is parameterized assuming that preceding layers $0 \dots l$ have already rotated the state $h$ into its expected input distribution:

$$\mathbb{E}[h_{l+1}] = h_l + \mathbb{E}[\Delta h_l]$$

If router $R_l$ greedily sets $a_l = \text{ROUTE\_AROUND}$, then $\Delta h_l = 0$. While $h_l$ is passed forward untouched via the identity residual, the downstream transformation $f_{l+1}$ receives an input vector that is **off-manifold**. If $f_{l+1}$ depends on an attention feature extracted *only* by $f_l$, layer $l+1$ produces corrupted activations.

A local probe $R_l(h_l)$ is myopic: it measures only the current state $h_l$, with zero lookahead into whether layer $l+3$ strictly requires $\Delta h_l$.

### B. The Early-Exit Cliff & Vocabulary Alignment Tension
An early exit forces intermediate hidden state $h_l$ directly into the final linear classifier:

$$\hat{y}_l = \text{Softmax}(W_{\text{head}} \cdot \text{LN}(h_l))$$

In early layers ($l < L/3$), representation features are syntactic, lexical, and positional. Forcing $\hat{y}_l$ to predict the final token creates severe gradient distortion:
* It forces early layers to compress representations into vocabulary space prematurely, destroying their ability to serve as hierarchical feature extractors for later layers.
* Our empirical hardening sweep confirmed this: at $\lambda_{\text{sparse}} = 1.0$, the router collapsed into exiting at Layer 0 on 33.8% of tokens, exploding perplexity from $14.86 \to 142.56$.

---

## 3. The New Paradigm: Global Dynamic Routing State ($\mathcal{S}_{\text{route}}$)

Instead of distributed, uncoordinated per-layer gates, we decouple **trajectory planning** from **kernel execution**.

We introduce a dedicated **Dynamic Routing State** $r \in \mathbb{R}^{d_r}$ that evolves alongside the sequence tokens:

$$S_{\text{system}} = \langle H, r \rangle, \quad H \in \mathbb{R}^{B \times S \times d_{\text{model}}}, \quad r \in \mathbb{R}^{B \times d_r}$$

### Global Layer Value Field $\mathbf{V}(H, r)$
Instead of evaluating one layer at a time, the routing state projects an **all-layer value assessment** over the entire network execution graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

$$\mathbf{V}(H, r) = \begin{bmatrix} 
v_{\text{attn}, 0} & v_{\text{ffn}, 0} \\
v_{\text{attn}, 1} & v_{\text{ffn}, 1} \\
\vdots & \vdots \\
v_{\text{attn}, L-1} & v_{\text{ffn}, L-1}
\end{bmatrix} \in \mathbb{R}^{L \times 2}$$

Where $v_{m, l} \in [0, 1]$ represents the predicted **marginal utility** of executing module $m$ at depth $l$:

$$v_{m, l} \approx \Delta \mathcal{L}_{\text{task}}(\text{execute } m_l) - \lambda_{\text{cost}} \cdot \text{FLOPs}(m_l)$$

---

## 4. Flow Reasoning on the Routing Graph (FRP-Router)

To prevent myopic or noisy routing choices, we incorporate **Flow Reasoning (Continuous Flow Matching / Fixed-Point Forcing)** onto the routing state itself.

### A. The Routing Flow Attractor
Rather than emitting a single-shot probability distribution, the routing state $r$ undergoes a continuous relaxation process:

$$\frac{dr}{dt} = v_\phi(r(t) \,|\, H, C_{\text{budget}}), \quad t \in [0, 1]$$

Where $v_\phi$ is a lightweight velocity network parameterized by ternary weights.

1. At $t=0$, $r(0)$ is initialized from pooled sequence embeddings $H$ and the target compute budget $C_{\text{budget}}$.
2. Over flow steps $k = 1 \dots K_{\text{route}}$, $r$ flows toward a **stable discrete attractor**:
   $$r^{(k+1)} = r^{(k)} + \Delta t \cdot v_\phi(r^{(k)})$$
3. **Contraction Stopping:** When $\|r^{(k+1)} - r^{(k)}\|_\infty < \epsilon_{\text{route}}$, the trajectory plan has converged.

### B. Why Flow Reasoning on Routing Solves the Multi-Hop Problem
1. **Pondering Compute:** On easy tokens (e.g. punctuation, predictable grammar), $r$ reaches the attractor basin in $K_{\text{route}} = 1$ step, selecting a sparse module subgraph ($10\text{--}20\%$ execution).
2. **Deliberation on Hard Tokens:** On ambiguous or reasoning-intensive tokens (e.g. multi-step logic, rare entities), the flow trajectory circles through attractor space for $K_{\text{route}} = 3\dots 5$ steps, dynamically committing additional layers to the execution path.
3. **Holistic Subgraph Selection:** The planned path $\Pi^*$ is an end-to-end directed acyclic path through the network. A layer is only selected if the downstream modules in $\Pi^*$ are verified to leverage its output.

---

## 5. Architectural Blueprint for Implementation

```
Token Input x
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Flow-Reasoned Router Engine (FRP-Router)                   │
│                                                             │
│  1. Sequence Context Pool: h_pool = Pool(Embed(x))          │
│  2. Initial Routing State: r^(0) = W_r · h_pool             │
│  3. Recurrent Flow Refinement (k = 1 .. K_flow):            │
│       r^(k+1) = r^(k) + v_phi(r^(k) | h_pool, lambda_cost)  │
│  4. Fixed-Point Attractor Path Output:                      │
│       M* = Gumbel_Softmax(W_map · r*) in {0, 1}^{L x 2}     │
└─────────────────────────────────────────────────────────────┘
      │
      │  Execution Mask: M* = [ [1, 1], [1, 0], [0, 0], [1, 1] ... ]
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Sparse Residual Execution Engine (Ternary Kernels)         │
│                                                             │
│  Layer 0: Attention [EXEC] -> FFN [EXEC]                    │
│  Layer 1: Attention [EXEC] -> FFN [BYPASS] (2-Hop Split)    │
│  Layer 2: Attention [BYPASS] -> FFN [BYPASS] (Full Skip)    │
│  Layer 3: Attention [EXEC] -> FFN [EXEC]                    │
│                                                             │
│  Zero weights loaded into memory for bypassed modules.      │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
Final Logits y
```

### Key Equations for Loss and Training:
1. **Task Loss:** $\mathcal{L}_{\text{task}} = \text{CrossEntropy}(y, y_{\text{target}})$
2. **Trajectory Budget Loss:**
   $$\mathcal{L}_{\text{budget}} = \left( \frac{1}{2L} \sum_{l=0}^{L-1} (M_{l, \text{attn}} + M_{l, \text{ffn}}) - C_{\text{target}} \right)^2$$
3. **Fixed-Point Attractor Loss (FPF):**
   $$\mathcal{L}_{\text{fpf}} = \|v_\phi(r^*) - 0\|_2^2$$
   Forces converged routing states to be genuine fixed-point attractors.

---

## 6. Next Experimental Target

1. Implement `experiments/flow_router.py` containing the `GlobalRoutingState` and `FlowRoutingPlanner` modules.
2. Evaluate on the hardened TinyStories benchmark against:
   - Full Execution baseline
   - Greedy 1-hop TriStateRouter (the $\lambda=0.2$ baseline from `hardening-2026-09-16.md`)
   - Matched-rate random baseline
3. Verify whether FRP-Router eliminates the representation cliff and unlocks non-sequential layer skipping with zero perplexity degradation.
