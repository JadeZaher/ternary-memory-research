"""
experiments/tristate_router.py: Tri-State Continuation Router for dynamic layer dispatch.

Each transformer layer contains a resident router probe D_router (FP16/FP32 linear projection
d_model -> 3) that computes logits for three discrete actions:
  0: EXECUTE      - Stream full ternary payload P_l and compute Attention + FFN.
  1: ROUTE_AROUND - Completely bypass layer l (h_{l+1} = h_l); 0 payload bytes loaded.
  2: EARLY_EXIT   - Immediately emit candidate token and halt further layer execution.

Reference: research/architecture-ternary-continuations.md Section 2.

Hardening 2026-09-16: decisions are now per-sample (`action_ids`, shape [B]). The scalar
`action`/`action_id` fields describe sample 0 and remain the physical dispatch signal for
batch-size-1 inference; batched callers must use `action_ids`. Previously the whole batch
followed sample 0.
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# Action constants
ACTION_EXECUTE = 0
ACTION_ROUTE_AROUND = 1
ACTION_EARLY_EXIT = 2

ACTION_NAMES = {
    ACTION_EXECUTE: "EXECUTE",
    ACTION_ROUTE_AROUND: "ROUTE_AROUND",
    ACTION_EARLY_EXIT: "EARLY_EXIT",
}

_ACTION_ALIASES = {
    "EXECUTE": ACTION_EXECUTE,
    "ROUTE_AROUND": ACTION_ROUTE_AROUND,
    "BYPASS": ACTION_ROUTE_AROUND,
    "EARLY_EXIT": ACTION_EARLY_EXIT,
    "EXIT": ACTION_EARLY_EXIT,
}


@dataclass
class RouterDecision:
    """Telemetry and dispatch decision from the TriStateRouter."""
    action: str            # sample-0 action: 'EXECUTE', 'ROUTE_AROUND', or 'EARLY_EXIT'
    action_id: int         # sample-0 action id: 0, 1, or 2
    logits: torch.Tensor   # Raw probe logits [B, 3]
    probs: torch.Tensor    # Softmax probabilities [B, 3]
    p_exec: float          # sample-0 probability of EXECUTE
    p_bypass: float        # sample-0 probability of ROUTE_AROUND
    p_exit: float          # sample-0 probability of EARLY_EXIT
    gumbel_weights: Optional[torch.Tensor] = None  # Differentiable one-hot weights [B, 3]
    action_ids: Optional[torch.Tensor] = None      # Per-sample action ids [B] (long)


class TriStateRouter(nn.Module):
    """
    Resident lightweight router probe D_router: projects h_l -> 3 dispatch logits.

    Legacy priority policy (legacy_dispatch=True):
      1. If p_exit >= tau_exit: EARLY_EXIT
      2. Else if p_bypass >= tau_bypass: ROUTE_AROUND
      3. Else: EXECUTE
    Trained policy (default): argmax at eval, hard Gumbel-Softmax sample in training.
    `random_policy` (eval only) replaces the probe with fixed action probabilities: the
    matched-rate random control used to test whether the trained probe beats chance.
    """

    def __init__(
        self,
        d_model: int,
        tau_exit: float = 0.8,
        tau_bypass: float = 0.5,
        force_action: Optional[str] = None,
        gumbel_tau: float = 1.0,
        legacy_dispatch: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.tau_exit = tau_exit
        self.tau_bypass = tau_bypass
        self.force_action = force_action
        self.gumbel_tau = gumbel_tau
        self.legacy_dispatch = legacy_dispatch
        self.random_policy: Optional[torch.Tensor] = None  # [3] action probabilities, eval only

        # Linear probe: d_model -> 3 logits [EXECUTE, ROUTE_AROUND, EARLY_EXIT]
        self.probe = nn.Linear(d_model, 3, bias=True)

        # Statistics / Call trackers (count every sample, not just sample 0)
        self.call_count = 0
        self.execute_count = 0
        self.bypass_count = 0
        self.exit_count = 0

        self.reset_parameters()

    def reset_parameters(self):
        # Default initialization with slight bias towards EXECUTE so untrained models run layers
        nn.init.normal_(self.probe.weight, std=0.02)
        with torch.no_grad():
            # [EXECUTE, ROUTE_AROUND, EARLY_EXIT]
            self.probe.bias.copy_(torch.tensor([1.0, -0.5, -1.0]))

    def reset_counters(self):
        """Reset profiling counters."""
        self.call_count = 0
        self.execute_count = 0
        self.bypass_count = 0
        self.exit_count = 0

    def forward(
        self,
        h: torch.Tensor,
        force_action: Optional[str] = None,
    ) -> RouterDecision:
        """
        Evaluate hidden state h and return routing decision.

        Args:
            h: Hidden state tensor of shape [batch_size, seq_len, d_model] or [batch_size, d_model]
            force_action: Optional action override ('EXECUTE', 'ROUTE_AROUND', 'EARLY_EXIT')
        """
        self.call_count += 1

        # Use active token representation for dispatch (last token position for autoregressive decoding)
        if h.dim() == 3:
            h_probe = h[:, -1, :]  # [batch_size, d_model]
        else:
            h_probe = h

        logits = self.probe(h_probe)  # [batch_size, 3]
        probs = F.softmax(logits, dim=-1)
        batch = logits.shape[0]

        override = force_action if force_action is not None else self.force_action
        gumbel_weights = None

        if override is not None:
            if override not in _ACTION_ALIASES:
                raise ValueError(f"Unknown forced action: {override}")
            action_id = _ACTION_ALIASES[override]
            action_ids = torch.full((batch,), action_id, dtype=torch.long, device=logits.device)
        elif self.random_policy is not None and not self.training:
            # Matched-rate random control: ignore the probe, sample actions from fixed probabilities.
            action_ids = torch.multinomial(
                self.random_policy.to(logits.device).expand(batch, -1), 1
            ).squeeze(-1)
            action_id = int(action_ids[0].item())
        elif self.legacy_dispatch:
            # Formal Priority Policy from research/architecture-ternary-continuations.md:
            exit_hit = probs[:, ACTION_EARLY_EXIT] >= self.tau_exit
            bypass_hit = probs[:, ACTION_ROUTE_AROUND] >= self.tau_bypass
            action_ids = torch.where(
                exit_hit,
                torch.full((batch,), ACTION_EARLY_EXIT, dtype=torch.long, device=logits.device),
                torch.where(
                    bypass_hit,
                    torch.full((batch,), ACTION_ROUTE_AROUND, dtype=torch.long, device=logits.device),
                    torch.full((batch,), ACTION_EXECUTE, dtype=torch.long, device=logits.device),
                ),
            )
            action_id = int(action_ids[0].item())
        else:
            # Differentiable Gumbel-Softmax dispatch
            if self.training:
                gumbel_weights = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True, dim=-1)
                action_ids = torch.argmax(gumbel_weights, dim=-1)
            else:
                action_ids = torch.argmax(logits, dim=-1)
                gumbel_weights = F.one_hot(action_ids, num_classes=3).to(logits.dtype)
            action_id = int(action_ids[0].item())

        action = ACTION_NAMES[action_id]

        # Sample-0 probabilities (one host sync)
        p0 = probs[0].detach().tolist()
        p_exec, p_bypass, p_exit = float(p0[0]), float(p0[1]), float(p0[2])

        # Telemetry over the whole batch
        counts = torch.bincount(action_ids, minlength=3).tolist()
        self.execute_count += int(counts[ACTION_EXECUTE])
        self.bypass_count += int(counts[ACTION_ROUTE_AROUND])
        self.exit_count += int(counts[ACTION_EARLY_EXIT])

        return RouterDecision(
            action=action,
            action_id=action_id,
            logits=logits,
            probs=probs,
            p_exec=p_exec,
            p_bypass=p_bypass,
            p_exit=p_exit,
            gumbel_weights=gumbel_weights,
            action_ids=action_ids,
        )
