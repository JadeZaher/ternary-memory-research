# Architecture Specification: Ternary Memory Regions, Dynamic Layer Routing, and Recurrent Flow Reasoning

Prepared 2026-09-15. Status: System architecture specification incorporating user decisions from the Conductor Track Grill Me session.
Complements `research/region-memory.md`, `research/flow-reasoning-and-ternary.md`, and `research/conductor-track.md`.

---

## 1. Architectural Decisions Summary

Based on the interactive Conductor alignment:
*   **Base Model Baseline:** Llama-3.2-1B / Qwen2.5-1.5B.
*   **Target Bottleneck:** Batch Size = 1 interactive generation on CPU / local DRAM (strictly memory-bandwidth bound).
*   **Layer Dispatch Mechanism (Tri-State Continuation Router):**
    Each block dynamically routes among three actions:
    1. `EXECUTE`: Stream ternary payload $P_l$ and compute full attention + FFN.
    2. `ROUTE_AROUND` (Bypass): Pass residual $h_{l+1} = h_l$ (or lightweight resident projection), loading **zero payload bytes** for layer $l$ while continuing to deeper layers.
    3. `EARLY_EXIT`: Halt execution at layer $l$ and emit candidate token.
*   **Reasoning Mode (FRM):** Full-network weight-tied recurrent flow denoiser with fixed-point contraction early exit ($\|s^{(k+1)} - s^{(k)}\|_\infty < \epsilon$).

---

## 2. The Tri-State Dynamic Continuation Router

```
                      Intermediate Hidden State h_l
                                    |
                                    v
                     +------------------------------+
                     | Resident Router Probe (D_r)  |
                     | 2-head softmax / gating      |
                     +------------------------------+
                                    |
             +----------------------+----------------------+
             |                      |                      |
             v                      v                      v
      [Action: EXIT]        [Action: BYPASS]       [Action: EXECUTE]
   Confidence >= tau_exit    Route Around Layer       Full Layer Path
             |                      |                      |
             v                      v                      v
      Emit Token y_hat        h_{l+1} = h_l          Stream P_l (TQ1_0)
      Halt generation         P_l NOT loaded         h_{l+1} = Layer_l(h_l)
      0 more layers read      0 bytes streamed       1.6875 bpw streamed
```

### 2.1 Formal Dispatch Logic
For layer $l$ and hidden state $h_l$:
1. The resident router descriptor $D_{\text{router}}[l]$ (FP16 linear projection $d_{\text{model}} \to 3$) computes dispatch logits:
   $$[z_{\text{exit}}, z_{\text{bypass}}, z_{\text{exec}}] = \text{softmax}(W_{\text{route}} h_l + b_{\text{route}})$$
2. **Priority Policy:**
   *   If $z_{\text{exit}} \ge \tau_{\text{exit}}$: Emit token immediately.
   *   Else if $z_{\text{bypass}} \ge \tau_{\text{bypass}}$:
       $$h_{l+1} = h_l$$
       Payload $P_l$ is **not fetched from DRAM**. The bus remains idle, saving bandwidth and latency.
   *   Else:
       $$h_{l+1} = h_l + \text{Layer}_l(h_l; P_l)$$
       Stream ternary payload $P_l$ ($1.6875$ bpw) into cache and compute.

### 2.2 Mathematical and Bandwidth Advantage
*   Traditional early exits only allow truncation: once you exit, deeper layers cannot be reached.
*   **Route-Around (Bypass)** allows *non-contiguous layer execution*: a token can skip lower-middle layers (e.g. syntactic processing layers that are redundant for that token) and still execute top-level semantic or reasoning layers.
*   In memory bandwidth accounting:
    $$\text{Bytes Loaded per Token} = \sum_{l \in \mathcal{L}_{\text{exec}}} \text{size}(P_l)$$
    where $\mathcal{L}_{\text{exec}} \subseteq \{1, \dots, L\}$.

---

## 3. Two-Tier Physical Memory Layout ($D$ vs $P$)

| Region | Memory Tier | Residency | Elements | Format | Bit Cost |
|---|---|---|---|---|---:|
| **$D_{\text{router}}$** | L1 / L2 Cache | Permanent | Dispatch probes per layer ($d \to 3$) | FP16 | $0.003$ bpw |
| **$D_{\text{scale}}$** | L1 / L2 Cache | Permanent | Block scales ($g=256$) | FP16 / E8M0 | $0.063$ bpw |
| **$D_{\text{rank}}$** | L1 / L2 Cache | Permanent | Two-level rank superblock directory | uint32 / uint16 | $0.063$ bpw |
| **$D_{\text{flow}}$** | Registers | Permanent | FPF convergence threshold $\epsilon$ | float32 | $< 64$ bits |
| **$P_{\text{stream}}$** | Host DRAM | Streamed on demand | Packed ternary weights $W \in \{-1, 0, +1\}$ | TQ1_0 format | $1.6875$ bpw |

**Total Resident Footprint ($D$):** $< 0.15$ bits per weight ($\approx 18\text{ MB}$ for a 1B parameter model). Easily fits entirely inside modern CPU L3 cache ($32\text{--}64\text{ MB}$).

---

## 4. Phase 4B: Weight-Tied Flow Reasoning Block

In the reasoning track, the entire network acts as a recurrent flow denoiser:

1. **Weight Loading:** The full-network ternary weights $P$ are streamed into cache **once**.
2. **Recurrent Iterations in Fast Cache:**
   For $k = 1, \dots, K_{\max}$:
   $$s_t^{(k+1)} = D_\theta^t(x_t \mid c, s_t^{(k)})$$
   Because the weights are weight-tied, zero additional DRAM reads occur during recurrence.
3. **Fixed-Point Early Exit:**
   At each recurrent step $k$:
   $$\Delta^{(k)} = \|s_t^{(k+1)} - s_t^{(k)}\|_\infty$$
   If $\Delta^{(k)} < \epsilon_{\text{converge}}$, exit recurrence and output $\hat{y} = \arg\max s_t^{(k+1)}$.

---

## 5. Verification Contract for Track 4

1. **Baseline Measure:** Load Llama-3.2-1B / Qwen2.5-1.5B; record baseline FP16 perplexity and latency per token on batch size 1 CPU inference.
2. **Quantization Pass:** Apply BitNet-style absmean or TWN thresholding ($0.75 \mathbb{E}[|W|]$) with block scaling $g=256$.
3. **Router Pass:** Train tri-state dispatch heads ($D_{\text{router}}$) on local traces; measure fraction of tokens routed through `EXECUTE`, `BYPASS`, and `EXIT`.
4. **Bandwidth Metric:** Confirm physical reduction in bytes streamed per token matches theoretical ledger.
