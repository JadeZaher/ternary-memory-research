"""
experiments/bitlinear.py: Native PyTorch BitLinear module with Straight-Through Estimator (STE).

Implements ternary weight quantization {-1, 0, +1} using absmean scaling:
    gamma = mean(|W|)  (computed either block-wise with g=256 or layer-wide)
    W_tilde = clip(round(W / gamma), -1, +1)
    W_eff = W_tilde * gamma

Backward pass passes gradients dL/dW_tilde straight through to master FP32 weights W.
Part of Track A (BitRoute-135M) in ternary-memory-research.

Hardening 2026-09-16 (see research/hardening-2026-09-16.md):
  - `ternary=False` gives a matched FP32 control with the same module interface.
  - In eval mode the ternary snapshot (W_tilde, gamma, additive masks) is computed once and
    cached until `.train()` is called again; previously every inference call re-quantized.
  - `last_zero_fraction` is a lazy property; the old eager `.item()` forced a GPU sync per layer per step.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def absmean_quantize_weights(
    weight: torch.Tensor,
    block_size: Optional[int] = 256,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ternarize weights to {-1.0, 0.0, +1.0} using absmean scaling with STE.

    Args:
        weight: FP32 master weight tensor of shape [out_features, in_features]
        block_size: Block size for block-scaled quantization (e.g. g=256).
                    If None or if weight.numel() is not divisible by block_size,
                    layer-wide scaling is applied.
        eps: Small epsilon to prevent division by zero.

    Returns:
        w_eff: Dequantized effective weight for forward pass with STE attached.
        w_tilde: Strict ternary weight tensor with elements in {-1.0, 0.0, +1.0}.
        gamma: Scale factor(s) applied.
    """
    orig_shape = weight.shape

    if block_size is not None and weight.numel() % block_size == 0:
        # Block-scaled absmean quantization
        w_reshaped = weight.view(-1, block_size)
        gamma = torch.mean(torch.abs(w_reshaped), dim=-1, keepdim=True).clamp(min=eps)
        w_scaled = w_reshaped / gamma
        w_tilde = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
        # Straight-Through Estimator (STE):
        # Forward returns w_tilde; backward passes dL/dW_tilde to w_scaled directly.
        w_quant = w_scaled + (w_tilde - w_scaled).detach()
        w_eff = (w_quant * gamma).view(orig_shape)
        w_tilde = w_tilde.view(orig_shape)
    else:
        # Layer-wide absmean quantization
        gamma = torch.mean(torch.abs(weight)).clamp(min=eps)
        w_scaled = weight / gamma
        w_tilde = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
        # Straight-Through Estimator (STE):
        w_quant = w_scaled + (w_tilde - w_scaled).detach()
        w_eff = w_quant * gamma

    return w_eff, w_tilde, gamma


def quantize_activations_8bit(
    x: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activations to 8-bit signed integers [-128, 127] using absmax scaling with STE.
    Used optionally for full BitNet b1.58 quantization.
    """
    scale = 127.0 / torch.max(torch.abs(x), dim=-1, keepdim=True)[0].clamp(min=eps)
    x_scaled = x * scale
    x_quant = torch.clamp(torch.round(x_scaled), -128.0, 127.0)
    # STE: forward gives x_quant, backward passes gradients through x_scaled
    x_eff = x_scaled + (x_quant - x_scaled).detach()
    return x_eff / scale, scale


def ternary_additive_gemm(
    x: torch.Tensor,
    w_tilde: torch.Tensor,
    gamma: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Multiplication-free ternary GEMM using masked addition/subtraction (reference form).

    Domain:  x in R^{...×n}, w_tilde in {-1,0,+1}^{m×n}, gamma in R_{>0}
    Output:  y in R^{...×m}

    y_i = gamma * (sum_{j in P_i} x_j  -  sum_{j in N_i} x_j)  +  bias_i

    where P_i = {j : w_tilde_{ij} = +1}, N_i = {j : w_tilde_{ij} = -1}.
    Zero-weight positions contribute nothing in the arithmetic model.

    Measured note: on a GPU that executes `x @ mask.T` through cuBLAS this is two dense GEMMs
    and is slower than one dense GEMM (outputs/arithmetic-upgrade-test.json). The savings it
    models are for hardware that can skip zeros; see experiments/bench_gemm_paths.py.
    """
    m_pos, m_neg, gamma_scalar = prepare_additive_masks(w_tilde, gamma, x.dtype)
    return ternary_additive_gemm_prepared(x, m_pos, m_neg, gamma_scalar, bias)


def prepare_additive_masks(
    w_tilde: torch.Tensor,
    gamma: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Build the +1 and -1 selector masks once. Block scales are folded into the masks;
    a layer-wide scalar gamma is returned separately so it stays a single multiply."""
    mask_pos = (w_tilde == 1.0).to(dtype)   # [m, n]
    mask_neg = (w_tilde == -1.0).to(dtype)  # [m, n]
    if gamma.dim() > 0 and gamma.numel() > 1:
        block_size = w_tilde.numel() // gamma.numel()
        gamma_matrix = gamma.expand(-1, block_size).reshape(w_tilde.shape).to(dtype)
        return mask_pos * gamma_matrix, mask_neg * gamma_matrix, None
    return mask_pos, mask_neg, gamma.to(dtype)


def ternary_additive_gemm_prepared(
    x: torch.Tensor,
    m_pos: torch.Tensor,
    m_neg: torch.Tensor,
    gamma_scalar: Optional[torch.Tensor],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Additive GEMM using pre-built masks (the cached inference path)."""
    y = x @ m_pos.t() - x @ m_neg.t()
    if gamma_scalar is not None:
        y = y * gamma_scalar
    if bias is not None:
        y = y + bias
    return y


class BitLinear(nn.Module):
    """
    BitLinear layer with ternary weights {-1, 0, +1} and Straight-Through Estimator.

    Stores master weights in FP32. During forward pass:
    1. Quantizes weights W to ternary W_tilde in {-1, 0, +1}.
    2. Scales by gamma (absmean per block or layer).
    3. Performs linear transformation.
    4. Propagates backward gradients to FP32 master weights without degradation.

    `ternary=False` turns the module into a plain FP32 linear layer (matched control).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        block_size: Optional[int] = 256,
        quantize_act: bool = False,
        use_additive_gemm: bool = True,
        ternary: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.quantize_act = quantize_act
        self.use_additive_gemm = use_additive_gemm
        self.ternary = ternary

        # Master weights stored in FP32
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        # Snapshot of the last ternary weights seen (telemetry / inspection)
        self.last_w_tilde: Optional[torch.Tensor] = None
        self.last_gamma: Optional[torch.Tensor] = None
        # Inference cache: built on first eval forward, dropped by .train()
        self._eval_cache: Optional[Dict[str, object]] = None

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming-style initialization adapted for ternary quantization
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self._eval_cache = None
        return self

    @property
    def last_zero_fraction(self) -> float:
        """Fraction of exact zeros in the last ternary snapshot (lazy; forces a sync when read)."""
        if self.last_w_tilde is None:
            return 0.0
        return float((self.last_w_tilde == 0).float().mean().item())

    def get_ternary_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate and return (W_tilde, gamma) without executing forward pass.
        W_tilde is strictly in {-1.0, 0.0, +1.0}.
        """
        with torch.no_grad():
            _, w_tilde, gamma = absmean_quantize_weights(
                self.weight, block_size=self.block_size
            )
        return w_tilde, gamma

    def _eval_weights(self, dtype: torch.dtype) -> Dict[str, object]:
        cache = self._eval_cache
        if cache is None or cache["dtype"] != dtype:
            with torch.no_grad():
                w_eff, w_tilde, gamma = absmean_quantize_weights(
                    self.weight, block_size=self.block_size
                )
                m_pos, m_neg, gamma_scalar = prepare_additive_masks(w_tilde, gamma, dtype)
                cache = {
                    "dtype": dtype,
                    "w_eff": w_eff.detach().to(dtype),
                    "w_tilde": w_tilde.detach(),
                    "gamma": gamma.detach(),
                    "m_pos": m_pos,
                    "m_neg": m_neg,
                    "gamma_scalar": gamma_scalar,
                }
            self._eval_cache = cache
        return cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.ternary:
            return F.linear(x, self.weight, self.bias)

        if self.quantize_act:
            x, _ = quantize_activations_8bit(x)

        if not self.training:
            cache = self._eval_weights(x.dtype)
            self.last_w_tilde = cache["w_tilde"]
            self.last_gamma = cache["gamma"]
            if self.use_additive_gemm:
                # Inference: additive path with cached masks (no gradient needed).
                return ternary_additive_gemm_prepared(
                    x, cache["m_pos"], cache["m_neg"], cache["gamma_scalar"], self.bias
                )
            return F.linear(x, cache["w_eff"], self.bias)

        # Training: STE path through w_eff for gradient flow
        w_eff, w_tilde, gamma = absmean_quantize_weights(
            self.weight, block_size=self.block_size
        )
        self.last_w_tilde = w_tilde.detach()
        self.last_gamma = gamma.detach()
        return F.linear(x, w_eff, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, block_size={self.block_size}, "
            f"quantize_act={self.quantize_act}, "
            f"use_additive_gemm={self.use_additive_gemm}, ternary={self.ternary}"
        )
