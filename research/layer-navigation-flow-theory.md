# Non-Monotonic Token Navigation: Bidirectional Layer Hopping and Flow-Reasoned Graph Traversal in Ternary Neural Systems

**Author:** Antigravity  
**Date:** 2026-09-16  
**Status:** Architectural Specification & Mathematical Foundation  
**Track:** Track D (Phase II / Advanced Dynamical Architectures)  
**Model Codename:** NaviTrit (Flow-Navigated Ternary Architecture)

---

## 1. The Core Paradigm Shift: From Monotonic Pipelines to Layer Graphs

Modern deep neural networks, from original Transformers to Mamba and Mixture-of-Depths, enforce a strict topological constraint: **Monotonic Feedforward Execution**. A token enters at Layer $0$ and exits at Layer $L-1$, with layer index $l$ strictly increasing:

$$l \in \{0, 1, 2, \dots, L-1\}, \quad l_{t+1} = l_t + 1$$

Even existing dynamic depth techniques (LayerSkip, Early-Exit, MoD) merely allow skipping *forward* ($l_{t+1} > l_t$) or terminating early. None allow **passing backward** ($l_{t+1} < l_t$) or **re-activating the same layer** ($l_{t+1} = l_t$).

### The NaviTrit Paradigm: The Network as a Module Graph $\mathcal{G}$

We reformulate the $L$-layer network not as a sequential pipeline, but as a **fully connected graph of stationary computational nodes**:

$$\mathcal{G} = (\mathcal{V}, \mathcal{E})$$

Where each node $v \in \mathcal{V}$ is a decoupled ternary sublayer:
$$\mathcal{V} = \{ \text{Attn}_0, \text{FFN}_0, \text{Attn}_1, \text{FFN}_1, \dots, \text{Attn}_{L-1}, \text{FFN}_{L-1} \} \cup \{ \text{EXIT} \}$$

$|\mathcal{V}| = 2L + 1$ discrete destinations.

```
       ┌─────────────────────────────────────────────────────────────┐
       │                                                             │
       ▼                                                             │
┌──────────────┐     Forward Hop      ┌──────────────┐  Backward Hop │
│   Attn_l     │ ───────────────────► │    FFN_l     │ ──────────────┘
└──────┬───────┘                      └──────┬───────┘
       │                                     │
       │ Cross-Layer Loop                    │ Forward Skip
       ▼                                     ▼
┌──────────────┐                      ┌──────────────┐
│   Attn_k     │ ◄─────────────────── │   Attn_m     │
└──────┬───────┘                      └──────┬───────┘
       │                                     │
       └──────────────────┬──────────────────┘
                          │
                          ▼ [HALT / CONTRACTION]
                   ┌──────────────┐
                   │  EXIT HEAD   │ ──► Vocabulary Logits y
                   └──────────────┘
```

A token's trajectory is a **dynamic walk on $\mathcal{G}$**:
$$\Pi = \big( v^{(1)}, v^{(2)}, \dots, v^{(T)} \big), \quad T \le T_{\text{max}}$$

Where $v^{(t)} \in \mathcal{V}$. At each step $t$, the token can:
1. **Hop Forward:** Move deeper into the hierarchy ($v^{(t+1)}$ at higher layer index).
2. **Hop Backward:** Return to an earlier layer ($v^{(t+1)}$ at lower layer index) to re-contextualize or re-project.
3. **Self-Loop (Re-activate):** Re-execute the *same* layer ($v^{(t+1)} = v^{(t)}$) to iteratively refine representations through a single high-capacity module.
4. **Early Exit:** Terminate and project immediately to vocabulary space.

---

## 2. Mathematical Formalization of Non-Monotonic Token Trajectories

### A. The Residual State Accumulator
At navigation step $t \in \{0, \dots, T_{\text{max}}-1\}$, the hidden representation evolves additively:

$$h^{(t+1)} = h^{(t)} + f_{v^{(t)}}(\text{LN}(h^{(t)}))$$

Where $f_v$ is the ternary BitLinear transformation corresponding to node $v \in \mathcal{V}$:
- If $v = \text{Attn}_l$: $f_v(h) = \text{BitLinearAttention}_l(h)$
- If $v = \text{FFN}_l$: $f_v(h) = \text{BitLinearFFN}_l(h)$
- If $v = \text{EXIT}$: $h^{(t+1)} = h^{(t)}$ and computation halts.

In cumulative form:
$$h^{(T)} = h^{(0)} + \sum_{t=0}^{T-1} \Delta h_{v^{(t)}}$$

Notice that the final representation $h^{(T)}$ is the sum of velocity vectors sampled along an **arbitrary sequence of nodes**, not a fixed $0 \dots L-1$ progression!

### B. The Flow-Reasoned Navigation Controller ($S_{\text{nav}}$)
To choose node $v^{(t)}$, we deploy an internal **Navigation Routing State**:

$$r_t \in \mathbb{R}^{d_r}$$

The routing state evolves via a continuous velocity field conditioned on both the current token state $h^{(t)}$ and the previous hop $v^{(t-1)}$:

$$\frac{dr}{dt} = v_\phi(r_t \,|\, h^{(t)}, v^{(t-1)}, C_{\text{target}})$$

Discrete numerical integration:
$$r_{t+1} = r_t + \Delta \tau \cdot v_\phi(r_t)$$

From the relaxed state $r_t$, the next destination node $v^{(t)}$ is sampled across all $2L+1$ candidates:

$$\mathbf{p}^{(t)} = \text{Softmax}\left( W_{\text{dest}} \cdot r_t \right) \in \Delta^{2L}$$

During training, we use differentiable straight-through Gumbel-Softmax:
$$\mathbf{g}^{(t)} = \text{GumbelSoftmax}\left( W_{\text{dest}} \cdot r_t, \tau_{\text{temp}}, \text{hard}=\text{True} \right) \in \{0, 1\}^{2L+1}$$

$$h^{(t+1)} = \sum_{k=0}^{2L-1} g_k^{(t)} \cdot \Big( h^{(t)} + f_{v_k}(\text{LN}(h^{(t)})) \Big) + g_{\text{EXIT}}^{(t)} \cdot h^{(t)}$$

---

## 3. Why Backward Passing and Repeated Activations Solve Hard Problems

Standard feedforward transformers suffer from a fundamental architectural limitation: **Fixed Depth per Sub-Problem**.

Consider a complex reasoning sequence:
1. Identifying syntactic structure: Requires $1$ simple projection.
2. Resolving a 3-step transitive constraint ("$A > B, B > C, C > D$"): Requires multiple iterative attention passes to propagate relations between disjoint tokens.
3. Generating a predictable function word ("the", "is"): Requires minimal computation.

### A. Algorithmic Looping Without Parameter Duplication
In a standard transformer, to allow $3$ steps of relation propagation, one must build a $24$-layer model, forcing all $24$ layers to exist in memory even when $21$ of them are redundant for simple tokens.

In **NaviTrit**:
- The model contains only $L=6$ physical layers ($12$ sublayers).
- On the transitive constraint, the navigation state $r$ selects:
  $$\text{Attn}_2 \longrightarrow \text{Attn}_2 \longrightarrow \text{Attn}_2$$
- The token visits the **exact same physical ternary matrix $\widetilde{W}_{\text{attn}, 2}$ three times consecutively**, updating the residual stream iteratively!
- **Parameter Cost:** $1\times$ (resident in on-chip SRAM / L2 cache).
- **Compute Allocated:** $3\times$ (only where needed).

### B. Passing Backward to Resolve Semantic Ambiguities
Suppose a token arrives at Layer $4$ (high-level semantics) and discovers an ambiguity: two noun phrases could be the referent of "it".
- In a forward-only transformer: Layer $4$ cannot go back to Layer $1$ to re-extract low-level syntactic dependencies; it must make do with the corrupted residual representation.
- In **NaviTrit**: Layer $4$ emits a backward hop back to $\text{Attn}_1$. The model re-queries the syntactic attention heads with its updated high-level context vector, resolving the ambiguity before hopping forward to final classification.

---

## 4. Hardware Realization: "Ternary on Metal" Alignment

This non-monotonic navigation architecture is uniquely suited for **Phase III: Neuromorphic In-Memory Crossbars**:

1. **Stationary Weights, Dynamic Signals:**
   In physical analog/digital ternary crossbars (RRAM / PCM), weights $\widetilde{W} \in \{-1, 0, +1\}$ are permanently programmed into stationary conductive crossbar arrays.
   - Moving weights into compute units burns immense energy ($>80\%$ of modern GPU power is DRAM traffic).
   - In NaviTrit, **weights never move**.
2. **Bus-Directed Crossbar Activations:**
   The activation vector $h^{(t)}$ sits in a local register. The lightweight navigation router activates the row-drivers of Tile $l$. The current integrates across the crossbar via Kirchhoff's laws ($I = \sum G_{ij} V_j$) in $<1\,\text{ns}$, yielding $h^{(t+1)}$.
   - To re-activate Layer $l$, the controller simply pulses Tile $l$ again.
   - To bypass or hop, the controller directs the bus to Tile $k$.
   - **Zero weight payload streaming, zero DRAM reloading.**

---

## 5. Mathematical Convergence & Halting Guarantees

Allowing backward hops and loops introduces the theoretical risk of **infinite loops** or **chaotic divergence**. How does NaviTrit guarantee convergence?

### Theorem: Certified Attractor Halting via Fixed-Point Forcing

We define the dual stopping condition:
1. **Discrete Early Exit Node:** The router emits $v^{(t)} = \text{EXIT}$.
2. **Continuous Dynamical Contraction:** The change in the residual representation drops below an $\epsilon$-contraction threshold:
   $$\|h^{(t+1)} - h^{(t)}\|_\infty = \|f_{v^{(t)}}(\text{LN}(h^{(t)}))\|_\infty < \epsilon_{\text{contract}}$$

To guarantee that trajectories converge to a stable fixed point in $\le T_{\text{max}}$ hops, the training objective penalizes long walks and forces contraction:

$$\mathcal{L}_{\text{nav}} = \mathcal{L}_{\text{task}} + \lambda_{\text{hop}} \cdot \frac{T}{T_{\text{max}}} + \lambda_{\text{fpf}} \sum_{t=1}^T \|v_\phi(r_t)\|_2^2$$

- The hop penalty $\lambda_{\text{hop}}$ exerts an economic pressure on the router: every additional layer execution incurs a loss cost.
- The fixed-point forcing loss $\lambda_{\text{fpf}}$ forces the routing velocity to extinguish as the representation approaches the decision manifold.

---

## 6. Research Roadmap: Track D Gate Specifications

To preserve all validated code and findings from Phase I (Track A: BitRoute, Track B: FlowTrit, Track C: FlowRoute), this work is registered as a **dedicated new track**:

- **Track Name:** Track D (NaviTrit: Non-Monotonic Token Navigation)
- **Phase:** Phase II (Dynamical Architectures & Non-Sequential Scaling)
- **Gate 13 (Theoretical Alignment & Graph Simulator):**
  - Implement toy discrete layer graph simulator verifying backward hops and repeated activations on symbolic logic tasks.
- **Gate 14 (NaviTrit-60M Prototype):**
  - Full ternary causal model with $2L+1$ Gumbel-Softmax navigation controller and stationary BitLinear tiles.
- **Paper Deliverable (Paper 4):**
  - *Beyond the Pipeline: Non-Monotonic Token Navigation and Stationary Attractor Routing in Low-Bit Neural Architectures* (Target: ICLR / NeurIPS).
