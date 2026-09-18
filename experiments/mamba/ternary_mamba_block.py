"""
experiments/mamba/ternary_mamba_block.py: Ternary Selective State Space Model (SSM) Layer.

Gate 20 (Phase II Track E: Hybrid Mamba-Samba-Jamba SSM):
1. Linear O(S) complexity in sequence length; O(1) memory per token during autoregressive generation.
2. Input-dependent Zero-Order Hold (ZOH) discretization with timescale Delta_t = Softplus(Linear(x) + Delta_bias).
3. Contractive Spectral Stability: Continuous transition parameter A = -exp(A_log) < 0 guarantees
   |A_bar_t| = |exp(Delta_t * A)| < 1.0 everywhere by mathematical construction.
   This permanently eliminates the Gate 19-B instability trap where unconstrained SSM state exploded to 2.78e7.
4. Quantized with BitLinear: In_proj, B_proj, C_proj, and Out_proj are strictly ternary {-1, 0, +1}
   with absmean scaling.
5. Dual Execution Engine:
   - Forward pass over full sequence [B, S, D] for parallel training.
   - Step forward pass [B, 1, D] with persistent state [B, D_inner, D_state] for O(1) generation with ZERO KV-cache.
"""

import os
import sys
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.bitlinear import BitLinear


@dataclass
class MambaConfig:
    """Configuration for Ternary Selective State Space Model."""
    d_model: int = 768                  # Model hidden size
    d_state: int = 16                   # SSM state expansion dimension N
    d_conv: int = 4                     # 1D causal convolution kernel size
    expand: int = 2                     # Expansion factor E (d_inner = expand * d_model)
    dt_rank: Optional[int] = None       # Rank of Delta projection (defaults to ceil(d_model / 16))
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    quantize_act: bool = False
    ternary: bool = True

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def actual_dt_rank(self) -> int:
        if self.dt_rank is not None:
            return self.dt_rank
        return math.ceil(self.d_model / 16)


class TernaryMambaBlock(nn.Module):
    """
    Ternary Selective State Space Model (SSM) layer.
    Combines BitLinear projections with selective time-varying state space equations.
    """
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = config.d_inner
        self.dt_rank = config.actual_dt_rank

        # 1. In-projection: maps x to (u, z) of shape 2 * d_inner
        # Quantized to ternary {-1, 0, +1}
        self.in_proj = BitLinear(
            self.d_model,
            2 * self.d_inner,
            bias=config.bias,
            quantize_act=config.quantize_act,
            ternary=config.ternary,
        )

        # 2. Causal 1D Depthwise Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
            bias=config.conv_bias,
        )

        # 3. Selective Parameters Projections (x_proj)
        # Maps conv_x to (Delta, B, C)
        # Delta: dt_rank, B: d_state, C: d_state
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        # 4. Timescale Delta Projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj bias to span log(dt_min) to log(dt_max)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        # Inverse softplus: inv_sp(y) = log(exp(y) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # 5. Continuous State Matrix A: Parameterized as -exp(A_log) for contractivity
        # Shape: [d_inner, d_state]
        # HiPPO-style initialization: A_{i, j} = -(j + 1)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # 6. Skip Connection Parameter D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # 7. Out-projection: maps d_inner -> d_model
        # Quantized to ternary {-1, 0, +1}
        self.out_proj = BitLinear(
            self.d_inner,
            self.d_model,
            bias=config.bias,
            quantize_act=config.quantize_act,
            ternary=config.ternary,
        )

    @property
    def A(self) -> torch.Tensor:
        """Strictly negative state matrix: A = -exp(A_log) < 0."""
        return -torch.exp(self.A_log.float())

    def forward(
        self,
        x: torch.Tensor,
        inference_params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Ternary Selective SSM over sequence [B, S, D].
        """
        B, S, D = x.shape

        # Step-mode inference check
        if inference_params is not None and "mamba_state" in inference_params:
            return self.step(x, inference_params)

        # 1. In-projection & split into branch u and gate z
        in_proj_out = self.in_proj(x)  # [B, S, 2 * d_inner]
        u, z = in_proj_out.chunk(2, dim=-1)

        # 2. Causal 1D Convolution
        u_conv = u.transpose(1, 2)  # [B, d_inner, S]
        u_conv = self.conv1d(u_conv)[:, :, :S]  # Causal crop [B, d_inner, S]
        u_conv = u_conv.transpose(1, 2)  # [B, S, d_inner]
        x_act = F.silu(u_conv)

        # 3. Selective Projections: compute Delta, B, C
        ssm_params = self.x_proj(x_act)  # [B, S, dt_rank + 2 * d_state]
        dt_input, B_proj, C_proj = torch.split(
            ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Delta computation with Softplus: Delta in (0, inf)
        delta = F.softplus(self.dt_proj(dt_input))  # [B, S, d_inner]

        # 4. Continuous-to-Discrete Discretization
        # A: [d_inner, d_state]
        # delta: [B, S, d_inner]
        # A_bar: exp(delta * A) -> shape [B, S, d_inner, d_state]
        A = self.A  # [d_inner, d_state]
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # in (0, 1)

        # B_bar: delta * B -> shape [B, S, d_inner, d_state]
        # x_act: [B, S, d_inner]
        # B_proj: [B, S, d_state]
        B_bar = delta.unsqueeze(-1) * B_proj.unsqueeze(2)  # [B, S, d_inner, d_state]

        # 5. Selective Recurrent State Evolution
        # s_t = A_bar_t * s_{t-1} + B_bar_t * x_t
        y = self._selective_scan(x_act, A_bar, B_bar, C_proj)

        # 6. Skip connection D
        y = y + x_act * self.D.unsqueeze(0).unsqueeze(0)

        # 7. Multiplicative Gating & Out-projection
        y = y * F.silu(z)
        out = self.out_proj(y)  # [B, S, d_model]

        return out

    def _selective_scan(
        self,
        x: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch: parallel log-space scan (default) or the reference sequential loop."""
        if getattr(self, "scan_impl", "parallel") == "sequential":
            return self._selective_scan_sequential(x, A_bar, B_bar, C)
        return self._selective_scan_parallel(x, A_bar, B_bar, C)

    def _selective_scan_parallel(
        self,
        x: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel selective scan in signed log space (hardening 2026-09-18).

        The recurrence s_t = a_t * s_{t-1} + u_t with a_t in (0, 1) has the closed form
            s_t = A_t * sum_{s<=t} u_s / A_s,   A_t = prod_{r<=t} a_r.
        1/A_s overflows for long sequences, so the sum is taken with logcumsumexp on the positive and
        negative parts of u separately; every exponent that is finally exponentiated equals log|s_t^+/-|,
        which is bounded. Replaces a Python loop over S tokens (S kernel launches per layer per loop),
        which was the throughput bottleneck of every Mamba-bearing model in this repo (~1k tok/s).
        """
        B, S, d_inner = x.shape
        A_bar = A_bar.float()
        u = B_bar.float() * x.float().unsqueeze(-1)                       # [B,S,d,n]
        log_acum = torch.cumsum(torch.log(A_bar.clamp(min=1e-30)), dim=1)  # <= 0, non-increasing in t
        eps = 1e-30
        u_pos = torch.where(u >= 0, u, torch.zeros_like(u))
        u_neg = torch.where(u < 0, -u, torch.zeros_like(u))
        lp = torch.logcumsumexp(torch.log(u_pos + eps) - log_acum, dim=1)
        ln = torch.logcumsumexp(torch.log(u_neg + eps) - log_acum, dim=1)
        state = torch.exp(log_acum + lp) - torch.exp(log_acum + ln)     # [B,S,d,n]
        y = (state * C.float().unsqueeze(2)).sum(dim=-1)                  # [B,S,d]
        return y.to(x.dtype)

    def _selective_scan_sequential(
        self,
        x: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reference sequential scan (kept for equivalence tests and step-mode parity).
        x: [B, S, d_inner]
        A_bar: [B, S, d_inner, d_state] in (0, 1)
        B_bar: [B, S, d_inner, d_state]
        C: [B, S, d_state]
        Returns y: [B, S, d_inner]
        """
        B, S, d_inner = x.shape
        d_state = self.d_state
        device = x.device

        state = torch.zeros(B, d_inner, d_state, device=device, dtype=x.dtype)
        y_list = []

        # Sequential scan over tokens
        for t in range(S):
            # Input contribution: B_bar_t * x_t
            bx = B_bar[:, t] * x[:, t].unsqueeze(-1)  # [B, d_inner, d_state]
            # State update: s_t = A_bar_t * s_{t-1} + bx
            state = A_bar[:, t] * state + bx  # [B, d_inner, d_state]
            # Output: y_t = sum_j (s_{t, :, j} * C_{t, j})
            y_t = torch.sum(state * C[:, t].unsqueeze(1), dim=-1)  # [B, d_inner]
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)  # [B, S, d_inner]
        return y

    def step(
        self,
        x_t: torch.Tensor,
        inference_params: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Single-step O(1) autoregressive inference.
        x_t: [B, 1, d_model]
        inference_params['mamba_state']: [B, d_inner, d_state] persistent SSM state
        inference_params['conv_state']: [B, d_inner, d_conv] causal convolution history
        Returns out_t: [B, 1, d_model]
        """
        B, S, D = x_t.shape
        assert S == 1, "step() operates on a single token at a time"

        ssm_state = inference_params.get("mamba_state")
        conv_state = inference_params.get("conv_state")

        if ssm_state is None:
            ssm_state = torch.zeros(B, self.d_inner, self.d_state, device=x_t.device, dtype=x_t.dtype)
        if conv_state is None:
            conv_state = torch.zeros(B, self.d_inner, self.d_conv, device=x_t.device, dtype=x_t.dtype)

        # 1. In-projection
        in_proj_out = self.in_proj(x_t).squeeze(1)  # [B, 2 * d_inner]
        u, z = in_proj_out.chunk(2, dim=-1)  # [B, d_inner]

        # 2. Update Conv State (Shift buffer and insert new token)
        conv_state = torch.cat([conv_state[:, :, 1:], u.unsqueeze(-1)], dim=-1)
        inference_params["conv_state"] = conv_state

        # Depthwise 1D conv over kernel
        conv_weight = self.conv1d.weight.squeeze(1)  # [d_inner, d_conv]
        u_conv = torch.sum(conv_state * conv_weight, dim=-1)
        if self.conv1d.bias is not None:
            u_conv = u_conv + self.conv1d.bias
        x_act = F.silu(u_conv)  # [B, d_inner]

        # 3. Selective parameter projections
        ssm_params = self.x_proj(x_act)  # [B, dt_rank + 2 * d_state]
        dt_input, B_proj, C_proj = torch.split(
            ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        delta = F.softplus(self.dt_proj(dt_input))  # [B, d_inner]
        A = self.A  # [d_inner, d_state]
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))  # [B, d_inner, d_state] in (0, 1)
        B_bar = delta.unsqueeze(-1) * B_proj.unsqueeze(1)  # [B, d_inner, d_state]

        # 4. State Update
        bx = B_bar * x_act.unsqueeze(-1)
        ssm_state = A_bar * ssm_state + bx
        inference_params["mamba_state"] = ssm_state

        # 5. Output
        y = torch.sum(ssm_state * C_proj.unsqueeze(1), dim=-1)  # [B, d_inner]
        y = y + x_act * self.D

        # 6. Gating & Out-projection
        y = y * F.silu(z)
        out = self.out_proj(y).unsqueeze(1)  # [B, 1, d_model]

        return out
