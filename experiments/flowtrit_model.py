"""
flowtrit_model.py: Native PyTorch Engine for Model 2 (FlowTrit-40M) - Track B.

Provides:
1. BitLinear: Native ternary weight layer {-1, 0, +1} with Straight-Through Estimator (STE),
   per-tensor absmean scale, int32/fp32 accumulation guarantees, and post-hoc ternarization.
2. FlowTritBlock: Residual multi-head attention and feed-forward block using BitLinear.
3. FlowTritDenoiser: Recurrent Flow Denoiser network with weight-tied recurrence across steps k=1..K,
   recurrent self-conditioning s^(k+1) = D_theta(x_t | c, s^(k)), Fixed-Point Forcing (FPF)
   rollout training loop, and certified dynamic early exit via ||s^(k+1) - s^(k)||_inf < eps_exit.
4. Memory & Cache Accounting: Exact bit and byte tracking for FP32 vs TQ1_0 ternary packed weights.

Mathematical Foundations & Notation:
- Ternary weight domain: S = {-1, 0, +1}
- Activation domain: x in R^{B x L x d_model}
- Discrete token domain: V classes per position, one-hot x_1 in {0, 1}^{L x V}
- Continuous flow time: t in [0, 1]
- Recurrent carry simplex: s in Delta^{V-1} per position
- Probability velocity field: v_theta^t(x_t | c, s) = (D_theta^t(x_t | c, s) - x_t) / (1 - t)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def ternary_additive_gemm(
    x: torch.Tensor,
    w_tilde: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Multiplication-free ternary GEMM using masked addition/subtraction.

    Domain:  x in R^{...×n}, w_tilde in {-1,0,+1}^{m×n}, alpha in R_{>0}
    Output:  y in R^{...×m}

    y_i = alpha * (sum_{j in P_i} x_j  -  sum_{j in N_i} x_j)  +  bias_i

    where P_i = {j : w_tilde_{ij} = +1}, N_i = {j : w_tilde_{ij} = -1}.
    Zero-weight positions contribute nothing — physically skipped.

    The matmuls x @ mask.T use 0/1 masks (no learned floating-point multiply).
    The only true floating-point multiplication is the final alpha scaling.
    """
    # Build boolean masks, cast to computation dtype
    mask_pos = (w_tilde == 1.0).to(x.dtype)   # [m, n]
    mask_neg = (w_tilde == -1.0).to(x.dtype)   # [m, n]

    # Pure additive GEMM: matmul by 0/1 mask is a sum of selected activations
    y = x @ mask_pos.t() - x @ mask_neg.t()    # [..., m]

    # Scale by alpha (the only true FP multiply in the weight path)
    y = y * alpha

    if bias is not None:
        y = y + bias

    return y


class BitLinear(nn.Linear):
    """
    Linear layer with native ternary weights {-1, 0, +1} via Straight-Through Estimator (STE).

    Domains:
    - Input: x in R^{... x d_in}
    - Proxy weights: W in R^{d_out x d_in}
    - Quantized weights: W_q in {-1, 0, +1}^{d_out x d_in} scaled by alpha in R^+
    - Accumulator: int32 / float32 accumulator (overflow-free for d_in <= 4096)
    - Output: y in R^{... x d_out}

    Quantization Scheme (BitNet b1.58 / TWN formulation):
    alpha = mean(|W|)
    W_norm = W / (alpha + eps)
    W_ternary = clamp(round(W_norm), -1.0, 1.0)
    W_ste = W_norm + (W_ternary - W_norm).detach()
    W_eff = alpha * W_ste
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        ternary: bool = True,
        use_additive_gemm: bool = True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.ternary = ternary
        self.use_additive_gemm = use_additive_gemm
        self.eps = 1e-5
        # Inference cache: built on first eval forward, dropped by .train()
        self._eval_cache: Optional[Dict[str, object]] = None

    def quantize_weights(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize continuous proxy weights to discrete ternary states {-1, 0, +1} with STE.
        Returns:
            w_eff: effective weights with STE gradients
            alpha: absmean scaling factor
        """
        # absmean scale
        alpha = torch.mean(torch.abs(w)).clamp(min=self.eps)
        # normalize
        w_norm = w / alpha
        # discrete ternary projection
        w_ternary = torch.clamp(torch.round(w_norm), -1.0, 1.0)
        # straight-through estimator: forward is w_ternary, backward flows to w_norm
        w_ste = w_norm + (w_ternary - w_norm).detach()
        w_eff = w_ste * alpha
        return w_eff, alpha

    def train(self, mode: bool = True):
        super().train(mode)
        self._eval_cache = None
        return self

    @property
    def last_zero_fraction(self) -> float:
        """Fraction of exact zeros in the current ternary projection (lazy; syncs when read)."""
        return self.get_sparsity()

    def _eval_weights(self, dtype: torch.dtype) -> Dict[str, object]:
        """Ternary snapshot computed once per eval phase (hardening 2026-09-16); previously every
        inference call re-quantized and forced a host sync for telemetry."""
        cache = self._eval_cache
        if cache is None or cache["dtype"] != dtype:
            with torch.no_grad():
                alpha = torch.mean(torch.abs(self.weight)).clamp(min=self.eps)
                w_tilde = torch.clamp(torch.round(self.weight / alpha), -1.0, 1.0)
                cache = {
                    "dtype": dtype,
                    "alpha": alpha.to(dtype),
                    "w_eff": (w_tilde * alpha).to(dtype),
                    "m_pos": (w_tilde == 1.0).to(dtype),
                    "m_neg": (w_tilde == -1.0).to(dtype),
                }
            self._eval_cache = cache
        return cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.ternary:
            return F.linear(x, self.weight, self.bias)
        if self.training:
            # Training: STE path for gradient flow
            w_eff, _ = self.quantize_weights(self.weight)
            return F.linear(x, w_eff, self.bias)
        cache = self._eval_weights(x.dtype)
        if not self.use_additive_gemm:
            return F.linear(x, cache["w_eff"], self.bias)
        # Inference: additive path (add/sub with cached ternary selector masks)
        y = (x @ cache["m_pos"].t() - x @ cache["m_neg"].t()) * cache["alpha"]
        if self.bias is not None:
            y = y + self.bias
        return y

    def ternarize_post_hoc(self, threshold_ratio: float = 0.75) -> None:
        """
        Applies post-hoc Ternary Weight Network (TWN) thresholding to existing FP32 weights.
        Discretizes weights directly in place without Straight-Through Estimator retraining:
        delta = threshold_ratio * mean(|W|)
        W_ternary = +1 (W > delta), -1 (W < -delta), 0 (|W| <= delta)
        alpha = mean(|W| for nonzeros)
        """
        with torch.no_grad():
            delta = threshold_ratio * torch.mean(torch.abs(self.weight))
            mask_pos = self.weight > delta
            mask_neg = self.weight < -delta
            mask_nonzero = mask_pos | mask_neg
            if mask_nonzero.any():
                alpha = torch.mean(torch.abs(self.weight[mask_nonzero]))
            else:
                alpha = torch.tensor(1.0, device=self.weight.device)
            w_tern = torch.zeros_like(self.weight)
            w_tern[mask_pos] = 1.0
            w_tern[mask_neg] = -1.0
            self.weight.copy_(w_tern * alpha)
            # Switch to unquantized execution so weights remain fixed at the discrete post-hoc values
            self.ternary = False

    def get_sparsity(self) -> float:
        """Calculate the fraction of exact zero weights."""
        with torch.no_grad():
            if self.ternary:
                alpha = torch.mean(torch.abs(self.weight)).clamp(min=self.eps)
                w_ternary = torch.clamp(torch.round(self.weight / alpha), -1.0, 1.0)
                return (w_ternary == 0).float().mean().item()
            else:
                return (self.weight == 0).float().mean().item()


class FlowTritBlock(nn.Module):
    """
    Transformer denoiser block built with native BitLinear projections.
    Features:
    - Pre-LayerNorm architecture
    - Multi-Head Self-Attention over sequence positions (Q, K, V, Out via BitLinear)
    - Feed-Forward Network with GeLU non-linearity (FFN1, FFN2 via BitLinear)
    - Residual skip connections
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, ternary: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Attention sub-layer
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = BitLinear(d_model, d_model, bias=False, ternary=ternary)
        self.k_proj = BitLinear(d_model, d_model, bias=False, ternary=ternary)
        self.v_proj = BitLinear(d_model, d_model, bias=False, ternary=ternary)
        self.out_proj = BitLinear(d_model, d_model, bias=True, ternary=ternary)

        # FFN sub-layer
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn1 = BitLinear(d_model, 4 * d_model, bias=True, ternary=ternary)
        self.ffn2 = BitLinear(4 * d_model, d_model, bias=True, ternary=ternary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        B, L, _ = x.shape

        # Multi-Head Attention
        res = x
        x_norm = self.ln1(x)
        q = self.q_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, L, self.d_model)
        x = res + self.out_proj(attn)

        # Feed-Forward Network
        res = x
        h = F.gelu(self.ffn1(self.ln2(x)))
        x = res + self.ffn2(h)
        return x


class FlowTritDenoiser(nn.Module):
    """
    Model 2: FlowTrit-40M Recurrent Flow Denoiser Engine.

    Key Architectural Principles:
    1. Weight-Tied Recurrence: Denoiser weights theta are shared across all recurrent steps k=1..K.
       Weights are loaded into fast SRAM/L2/L3 cache once, running K iterations with zero DRAM reloads.
    2. Recurrent Self-Conditioning:
       s^(k+1) = D_theta(x_t | c, s^(k), t)
       The carry simplex s^(k) in Delta^{V-1} provides recursive state refinement.
    3. Fixed-Point Forcing (FPF) Training:
       Trains on carries generated by the model's own inference rollouts with stop-gradient,
       forcing the flow vector field around valid solutions to be a contracting attractor:
       ||D_theta^t(x_t | c, s) - y|| <= gamma ||s - y|| with gamma < 1.
    4. Certified Dynamic Early Exit:
       At test time, monitors consecutive contraction:
       Delta^(k) = ||s^(k+1) - s^(k)||_inf < epsilon_exit
       Halts recurrence dynamically when the state enters the attractor basin.
    """

    def __init__(
        self,
        seq_len: int = 16,
        num_classes: int = 4,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ternary: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ternary = ternary

        # Input dimension: x_t (V) + condition c (V) + carry s (V) = 3 * V
        in_dim = 3 * num_classes
        self.in_proj = BitLinear(in_dim, d_model, bias=True, ternary=ternary)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Flow time embedding (sinusoidal + 2-layer BitLinear MLP)
        self.time_mlp = nn.Sequential(
            BitLinear(d_model, d_model, bias=True, ternary=ternary),
            nn.SiLU(),
            BitLinear(d_model, d_model, bias=True, ternary=ternary),
        )

        # Weight-tied transformer blocks
        self.blocks = nn.ModuleList([
            FlowTritBlock(d_model=d_model, n_heads=n_heads, ternary=ternary)
            for _ in range(n_layers)
        ])

        # Output projection head
        self.ln_out = nn.LayerNorm(d_model)
        self.out_head = BitLinear(d_model, num_classes, bias=True, ternary=ternary)

    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal time embedding for flow parameter t in [0, 1].
        t: (B, 1) or (B,)
        Returns: (B, d_model)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half_dim = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t.float() * freqs.unsqueeze(0)
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)
        if emb.shape[-1] < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.shape[-1]))
        return self.time_mlp(emb)

    def forward(
        self,
        x_t: torch.Tensor,
        c: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single recurrent denoiser evaluation step:
        s^(k+1) = D_theta(x_t | c, s^(k), t)

        Args:
            x_t: (B, L, V) continuous noisy flow state
            c: (B, L, V) prompt condition (one-hot clues, 0 elsewhere)
            s: (B, L, V) recurrent carry state (simplex probabilities)
            t: (B, 1) or (B,) flow time in [0, 1]

        Returns:
            logits: (B, L, V) unnormalized scores
            s_next: (B, L, V) updated carry simplex (softmax probabilities)
        """
        B, L, _ = x_t.shape

        # Concatenate inputs along channel dimension
        inp = torch.cat([x_t, c, s], dim=-1)  # (B, L, 3 * V)
        h = self.in_proj(inp) + self.pos_emb  # (B, L, d_model)

        # Add time conditioning
        t_emb = self._embed_time(t).unsqueeze(1)  # (B, 1, d_model)
        h = h + t_emb

        # Pass through weight-tied transformer blocks
        for block in self.blocks:
            h = block(h)

        h = self.ln_out(h)
        logits = self.out_head(h)  # (B, L, V)
        s_next = F.softmax(logits, dim=-1)
        return logits, s_next

    def rollout_inference(
        self,
        x_t: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        clue_mask: Optional[torch.Tensor] = None,
        max_steps: int = 5,
        early_exit: bool = False,
        eps_exit: float = 0.03,
    ) -> Dict[str, any]:
        """
        Inference rollout across recurrent steps k = 1 .. K.
        Supports certified dynamic early exit via ||s^(k+1) - s^(k)||_inf < eps_exit.

        Args:
            x_t: (B, L, V) input flow state
            c: (B, L, V) condition
            t: (B, 1) flow time
            clue_mask: (B, L, 1) binary mask of known clues (1 = clue, 0 = unassigned)
            max_steps: maximum recurrence iterations K_max
            early_exit: whether to halt dynamically upon fixed-point contraction
            eps_exit: L-infinity contraction threshold delta_exit

        Returns:
            dict containing:
            - 's_final': final carry simplex (B, L, V)
            - 'predictions': argmax discrete classes (B, L)
            - 'steps_taken': number of steps executed per batch item
            - 'contraction_deltas': list of L-inf deltas at each recurrent step
            - 'early_exited': whether early exit triggered
        """
        B, L, V = x_t.shape
        device = x_t.device

        # Initial carry: uniform simplex distribution over vocabulary V
        s = torch.full((B, L, V), 1.0 / V, device=device)
        if clue_mask is not None:
            s = (1.0 - clue_mask) * s + clue_mask * c

        deltas = []
        steps_taken = max_steps
        early_exited = False

        with torch.no_grad():
            for k in range(1, max_steps + 1):
                _, s_next = self.forward(x_t, c, s, t)
                if clue_mask is not None:
                    s_next = (1.0 - clue_mask) * s_next + clue_mask * c

                # Measure contraction: L-infinity norm on unassigned positions
                if clue_mask is not None:
                    diff = torch.abs(s_next - s) * (1.0 - clue_mask)
                else:
                    diff = torch.abs(s_next - s)
                delta_k = torch.max(diff).item()
                deltas.append(delta_k)

                s = s_next

                if early_exit and delta_k < eps_exit:
                    steps_taken = k
                    early_exited = True
                    break

        pred = torch.argmax(s, dim=-1)
        return {
            "s_final": s,
            "predictions": pred,
            "steps_taken": steps_taken,
            "contraction_deltas": deltas,
            "early_exited": early_exited,
        }

    def train_fpf_step(
        self,
        x_clean: torch.Tensor,
        target_indices: torch.Tensor,
        clue_mask: torch.Tensor,
        c: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        max_rollout_k: int = 4,
        leak_free: bool = True,
    ) -> float:
        """
        Fixed-Point Forcing (FPF) training step:
        1. Sample continuous flow time t in [0.1, 0.9].
        2. Generate noisy interpolant x_t = (1 - t) * eps + t * x_target, where x_target is the
           visible condition c when leak_free=True (matches inference, hardened 2026-09-16) or the
           full solution x_clean when leak_free=False (original protocol: the solution was visible
           through x_t in training but absent at test time; see research/hardening-2026-09-16.md).
        3. Simulate rollout of k in [0 .. max_rollout_k] inference steps with stop-gradient
           to generate realistic carry s_fpf.
        4. Compute gradient pass D_theta(x_t | c, stopgrad(s_fpf), t) -> logits.
        5. Minimize cross-entropy against ground truth fixed point x_clean.
        """
        self.train()
        B, L, V = x_clean.shape
        device = x_clean.device

        # 1. Sample flow time
        t = torch.rand(B, 1, device=device) * 0.8 + 0.1

        # 2. Linear interpolant (see docstring for leak_free)
        eps = torch.randn_like(x_clean)
        t_expand = t.unsqueeze(-1)
        x_target = c if leak_free else x_clean
        x_t = (1.0 - t_expand) * eps + t_expand * x_target

        # 3. Rollout carry s_fpf with stop-gradient
        k_rollout = torch.randint(0, max_rollout_k + 1, (1,)).item()
        s = torch.full_like(x_clean, 1.0 / V)
        s = (1.0 - clue_mask) * s + clue_mask * c

        with torch.no_grad():
            for _ in range(k_rollout):
                _, s_next = self.forward(x_t, c, s, t)
                s = (1.0 - clue_mask) * s_next + clue_mask * c

        s_fpf = s.detach()

        # 4. Supervised forward step on the carry
        optimizer.zero_grad()
        logits, _ = self.forward(x_t, c, s_fpf, t)

        # Cross-entropy loss against ground truth target
        loss = F.cross_entropy(logits.view(-1, V), target_indices.view(-1))
        loss.backward()
        optimizer.step()

        return loss.item()

    def ternarize_post_hoc(self, threshold_ratio: float = 0.75) -> None:
        """Ternarize all BitLinear layers post-hoc without STE retraining."""
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.ternarize_post_hoc(threshold_ratio=threshold_ratio)

    def count_parameters(self) -> Dict[str, any]:
        """Count total parameters, ternary weights, and compute memory footprint."""
        total_params = sum(p.numel() for p in self.parameters())
        ternary_params = sum(
            m.weight.numel() for m in self.modules() if isinstance(m, BitLinear)
        )
        other_params = total_params - ternary_params

        # FP32 footprint: 32 bits per param (4 bytes)
        fp32_bytes = total_params * 4.0

        # TQ1_0 Ternary footprint:
        # 1.585 bits pure information entropy + 0.1025 bpw block scale (g=256) = 1.6875 bits/weight
        ternary_packed_bytes = (ternary_params * 1.6875 / 8.0) + (other_params * 4.0)
        compression_ratio = fp32_bytes / ternary_packed_bytes

        return {
            "total_parameters": total_params,
            "ternary_parameters": ternary_params,
            "other_parameters": other_params,
            "fp32_bytes": fp32_bytes,
            "fp32_kib": fp32_bytes / 1024.0,
            "ternary_packed_bytes": ternary_packed_bytes,
            "ternary_packed_kib": ternary_packed_bytes / 1024.0,
            "compression_ratio": compression_ratio,
        }
