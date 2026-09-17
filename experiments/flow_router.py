"""
experiments/flow_router.py: Flow-Reasoned Dynamic Router (FR-Router).

Provides global trajectory planning over the complete transformer execution graph:
1. Global Routing State r in R^{d_route}: tracks sequence trajectory and downstream needs.
2. Flow-Reasoned Relaxation (FRP): runs continuous ODE velocity integration:
     dr/dt = v_phi(r | h_pool, C_target)
   relaxing r towards a discrete attractor basin over k = 1 .. K_flow steps.
3. 2-Hop Decoupled Module Mask:
     M* in {0, 1}^{B x L x 2}  (sublayer 0: Attention, sublayer 1: FFN)
   allowing FFN (67% of layer weights) to be skipped while Attention is retained.
4. Matched Random Control: supports fixed-rate random Bernoulli coin execution for
   unbiased baseline comparison.

Reference: research/dynamic-routing-flow-theory.md
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# Sublayer indices
SUBLAYER_ATTN = 0
SUBLAYER_FFN = 1
NUM_SUBLAYERS = 2

# Action indices for each sublayer
ACTION_BYPASS = 0
ACTION_EXECUTE = 1


@dataclass
class FlowRoutingDecision:
    """Output telemetry and execution masks from the Flow-Reasoned Router."""
    # Binary or straight-through mask of shape [B, L, 2] (1.0 = EXECUTE, 0.0 = BYPASS)
    execution_mask: torch.Tensor
    # Raw value logits of shape [B, L, 2, 2] (action 0: BYPASS, action 1: EXECUTE)
    logits: torch.Tensor
    # Softmax probabilities of shape [B, L, 2, 2]
    probs: torch.Tensor
    # Final relaxed routing state r* of shape [B, d_route]
    r_star: torch.Tensor
    # Number of flow relaxation steps taken per sample [B]
    flow_steps_taken: torch.Tensor
    # Fixed-point forcing residual ||v_phi(r*)||_2^2 for stability regularization
    fpf_loss: torch.Tensor
    # Mean active execution fractions across the batch
    mean_attn_exec: float
    mean_ffn_exec: float
    mean_overall_exec: float
    # Parameter-weighted compute cost: (Attn + 2*FFN) / (3*L)
    weighted_param_cost: float


class FlowRoutingVelocityNet(nn.Module):
    """
    Continuous velocity field network v_phi(r | c, C_target).
    Predicts state derivative dr/dt in R^{d_route}.
    """
    def __init__(self, d_route: int):
        super().__init__()
        self.d_route = d_route
        self.net = nn.Sequential(
            nn.LayerNorm(d_route),
            nn.Linear(d_route, d_route * 2),
            nn.GELU(),
            nn.Linear(d_route * 2, d_route),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, r: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        r: current routing state [B, d_route]
        condition: pooled context embedding + budget [B, d_route]
        """
        return self.net(r + condition)


class FlowRoutingPlanner(nn.Module):
    """
    Global Flow-Reasoned Routing Planner.
    
    Replaces myopic single-hop layer probes with a global routing trajectory.
    Relaxes an internal routing state r via continuous velocity integration
    towards a stable discrete attractor, then projects an all-layer multi-hop
    execution mask M* in {0, 1}^{B x L x 2}.
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_route: int = 128,
        max_flow_steps: int = 3,
        dt: float = 0.5,
        eps_contraction: float = 1e-3,
        gumbel_tau: float = 1.0,
        default_target_budget: float = 0.70,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_route = d_route
        self.max_flow_steps = max_flow_steps
        self.dt = dt
        self.eps_contraction = eps_contraction
        self.gumbel_tau = gumbel_tau
        self.default_target_budget = default_target_budget

        # Matched-rate random control policy: Optional float in [0, 1]
        # When set during eval, samples actions randomly to verify if routing beats chance
        self.random_policy: Optional[float] = None

        # 1. Sequence Context Encoder: h_pool in R^{d_model} -> condition in R^{d_route}
        # Inputs: pooled hidden state + 1-dim scalar target budget
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_route),
            nn.LayerNorm(d_route),
            nn.GELU(),
            nn.Linear(d_route, d_route),
        )

        # 2. Initial state projection W_init: condition -> r^{(0)}
        self.r_init_proj = nn.Linear(d_route, d_route)

        # 3. Flow velocity field v_phi(r | condition)
        self.velocity_net = FlowRoutingVelocityNet(d_route)

        # 4. Global Value Field Projector: r* -> [L, 2, 2] logits
        # (For each layer l in 0..L-1 and sublayer m in {Attn, FFN}: 2 logits [BYPASS, EXECUTE])
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_route),
            nn.Linear(d_route, num_layers * NUM_SUBLAYERS * 2),
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Biased slightly towards EXECUTE so untrained networks preserve flow
        with torch.no_grad():
            last_linear = self.value_head[1]
            nn.init.normal_(last_linear.weight, std=0.01)
            # Reshape bias to [num_layers, NUM_SUBLAYERS, 2]
            bias = torch.zeros(self.num_layers, NUM_SUBLAYERS, 2)
            bias[:, :, ACTION_EXECUTE] = 1.0
            bias[:, :, ACTION_BYPASS] = -0.5
            last_linear.bias.copy_(bias.flatten())

    def pool_sequence(
        self,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool sequence tokens h [B, S, d_model] into a single representation [B, d_model].
        """
        if h.dim() == 2:
            return h
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(h.dtype)  # [B, S, 1]
            sum_h = torch.sum(h * mask, dim=1)
            count = torch.clamp(mask.sum(dim=1), min=1.0)
            return sum_h / count
        return h.mean(dim=1)

    def forward(
        self,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_budget: Optional[float] = None,
        force_all_execute: bool = False,
    ) -> FlowRoutingDecision:
        """
        Plan the multi-hop execution subgraph for the entire transformer stack.

        Args:
            h: Hidden state tensor [B, S, d_model] or [B, d_model]
            attention_mask: Optional attention mask [B, S]
            target_budget: Optional target execution ratio in [0, 1]
            force_all_execute: If True, forces all modules to EXECUTE (baseline control)

        Returns:
            FlowRoutingDecision containing execution_mask [B, L, 2] and telemetry
        """
        B = h.shape[0]
        device = h.device
        dtype = h.dtype

        # 1. Pool sequence context
        h_pool = self.pool_sequence(h, attention_mask)  # [B, d_model]

        # 2. Target budget scalar
        budget_val = target_budget if target_budget is not None else self.default_target_budget
        budget_tensor = torch.full((B, 1), budget_val, device=device, dtype=dtype)
        ctx_input = torch.cat([h_pool, budget_tensor], dim=-1)  # [B, d_model + 1]

        # 3. Context condition & initial state r^{(0)}
        condition = self.context_encoder(ctx_input)  # [B, d_route]
        r = self.r_init_proj(condition)              # [B, d_route]

        # 4. Continuous Flow Relaxation Loop: dr/dt = v_phi(r | condition)
        converged = torch.zeros(B, dtype=torch.bool, device=device)
        steps_taken = torch.zeros(B, dtype=torch.long, device=device)

        for k in range(self.max_flow_steps):
            v = self.velocity_net(r, condition)
            delta = self.dt * v
            r_next = r + delta

            # Check contraction: ||delta||_inf < eps
            inf_norm = torch.norm(delta.detach(), p=float('inf'), dim=-1)
            is_converged = inf_norm < self.eps_contraction

            if self.training:
                # Keep differentiable during training
                r = r_next
                steps_taken += 1
            else:
                # Record step execution for non-converged samples
                steps_taken = torch.where(converged, steps_taken, steps_taken + 1)
                converged = converged | is_converged
                r = torch.where(converged.unsqueeze(-1), r, r_next)
                if converged.all():
                    break

        r_star = r

        # 5. Fixed-Point Forcing (FPF) loss: ||v_phi(r*)||_2^2
        v_terminal = self.velocity_net(r_star, condition)
        fpf_loss = torch.mean(torch.sum(v_terminal ** 2, dim=-1))

        # 6. Global Value Head: r* -> [B, L, 2, 2]
        raw_logits = self.value_head(r_star)
        logits = raw_logits.view(B, self.num_layers, NUM_SUBLAYERS, 2)
        probs = F.softmax(logits, dim=-1)  # [B, L, 2, 2]

        # 7. Action Selection & Mask Emission
        if force_all_execute:
            # Full baseline control
            execution_mask = torch.ones((B, self.num_layers, NUM_SUBLAYERS), device=device, dtype=dtype)
        elif self.random_policy is not None and not self.training:
            # Matched random coin control
            rand_uniform = torch.rand((B, self.num_layers, NUM_SUBLAYERS), device=device)
            execution_mask = (rand_uniform < self.random_policy).to(dtype)
        elif self.training:
            # Differentiable Gumbel-Softmax straight-through
            # Flatten over layers and sublayers to apply gumbel_softmax along action dim (last dim)
            gumbel_out = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True, dim=-1)
            # EXECUTE is action index 1
            execution_mask = gumbel_out[..., ACTION_EXECUTE]  # [B, L, 2]
        else:
            # Deterministic argmax at eval
            actions = torch.argmax(logits, dim=-1)  # [B, L, 2]
            execution_mask = (actions == ACTION_EXECUTE).to(dtype)

        # 8. Compute telemetry
        with torch.no_grad():
            attn_exec = execution_mask[..., SUBLAYER_ATTN].mean().item()
            ffn_exec = execution_mask[..., SUBLAYER_FFN].mean().item()
            overall_exec = execution_mask.mean().item()
            # Parameter-weighted cost: FFN has ~2x parameters of Attention
            # Weight = (1 * Attn + 2 * FFN) / 3
            weighted_cost = (attn_exec + 2.0 * ffn_exec) / 3.0

        return FlowRoutingDecision(
            execution_mask=execution_mask,
            logits=logits,
            probs=probs,
            r_star=r_star,
            flow_steps_taken=steps_taken,
            fpf_loss=fpf_loss,
            mean_attn_exec=attn_exec,
            mean_ffn_exec=ffn_exec,
            mean_overall_exec=overall_exec,
            weighted_param_cost=weighted_cost,
        )

    def compute_budget_loss(
        self,
        decision: FlowRoutingDecision,
        target_budget: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Computes the quadratic penalty between planned parameter-weighted cost and target budget.
        L_budget = ( (1/(3*L)) * sum(M_attn + 2*M_ffn) - C_target )^2
        """
        target = target_budget if target_budget is not None else self.default_target_budget
        mask = decision.execution_mask  # [B, L, 2]
        m_attn = mask[..., SUBLAYER_ATTN]  # [B, L]
        m_ffn = mask[..., SUBLAYER_FFN]    # [B, L]
        # Weighted module cost per sample
        weighted_per_sample = (m_attn.sum(dim=-1) + 2.0 * m_ffn.sum(dim=-1)) / (3.0 * self.num_layers)
        budget_loss = torch.mean((weighted_per_sample - target) ** 2)
        return budget_loss
