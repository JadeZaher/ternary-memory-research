"""
experiments/flowroute_model.py: FlowRoute Combined Architecture (Track A x Track B).

Integrates:
1. Track A's TriState Continuation Router into each layer of Track B's Recurrent Flow Denoiser.
2. Two-level Dynamic Compute Scaling:
   - Temporal recurrence scaling: halts iteration k when ||s^(k+1) - s^(k)||_inf < eps_exit.
   - Spatial layer routing: inside each iteration k, layers dynamically choose EXECUTE vs ROUTE_AROUND.
3. Multiplicative Compute Savings:
   Total Layer Executions = (Iterations K_eff) x (Active Layers per Step L_eff).
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.flowtrit_model import BitLinear
from experiments.tristate_router import TriStateRouter, RouterDecision, ACTION_EXECUTE, ACTION_ROUTE_AROUND


class TwoStateLayerRouter(nn.Module):
    """
    Dedicated 2-action router for recurrent flow layers:
      0: EXECUTE (stream weights, compute Attention + FFN)
      1: ROUTE_AROUND (residual bypass: out = in, zero weight compute)
    """
    def __init__(self, d_model: int, tau: float = 1.0):
        super().__init__()
        self.probe = nn.Linear(d_model, 2, bias=True)
        self.gumbel_tau = tau
        # Eval-only matched-rate random control: bypass with this probability, ignoring the probe.
        self.random_bypass_prob: Optional[float] = None
        with torch.no_grad():
            # Initial bias towards EXECUTE so untrained layers execute fully
            self.probe.bias.copy_(torch.tensor([1.5, -1.0]))

    def forward(self, x: torch.Tensor, force_action: Optional[str] = None):
        """Returns (sample-0 action name, probs [B,2], one-hot weights [B,2], per-sample action ids [B])."""
        h_probe = x.mean(dim=1) if x.dim() == 3 else x
        logits = self.probe(h_probe)  # [B, 2]
        probs = F.softmax(logits, dim=-1)
        batch = logits.shape[0]

        if force_action == "ROUTE_AROUND":
            action_ids = torch.ones(batch, dtype=torch.long, device=x.device)
            gumbel_weights = F.one_hot(action_ids, num_classes=2).to(logits.dtype)
        elif force_action == "EXECUTE":
            action_ids = torch.zeros(batch, dtype=torch.long, device=x.device)
            gumbel_weights = F.one_hot(action_ids, num_classes=2).to(logits.dtype)
        elif self.random_bypass_prob is not None and not self.training:
            action_ids = (torch.rand(batch, device=x.device) < self.random_bypass_prob).long()
            gumbel_weights = F.one_hot(action_ids, num_classes=2).to(logits.dtype)
        elif self.training:
            gumbel_weights = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True, dim=-1)
            action_ids = torch.argmax(gumbel_weights, dim=-1)
        else:
            action_ids = torch.argmax(logits, dim=-1)
            gumbel_weights = F.one_hot(action_ids, num_classes=2).to(logits.dtype)

        action = "EXECUTE" if int(action_ids[0].item()) == 0 else "ROUTE_AROUND"
        return action, probs, gumbel_weights, action_ids


class FlowRouteBlock(nn.Module):
    """
    Transformer denoiser layer embedded with an internal resident TwoStateLayerRouter.
    Allows ROUTE_AROUND (residual bypass) within recurrent denoiser steps.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        ternary: bool = True,
        use_trained_router: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 2-action router: EXECUTE vs ROUTE_AROUND
        self.router = TwoStateLayerRouter(d_model=d_model)

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

    def _execute_sublayers(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        res = x
        x_norm = self.ln1(x)
        q = self.q_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, L, self.d_model)
        x = res + self.out_proj(attn)

        res = x
        h = F.gelu(self.ffn1(self.ln2(x)))
        x = res + self.ffn2(h)
        return x

    def forward(
        self,
        x: torch.Tensor,
        force_action: Optional[str] = None,
    ) -> Tuple[torch.Tensor, str, torch.Tensor]:
        action, probs, gumbel_weights, action_ids = self.router(x, force_action=force_action)
        batch = x.shape[0]
        exec_mask = action_ids == 0
        self.last_exec_mask = exec_mask  # [B] bool, read by rollout_inference for per-sample accounting

        # Training mode with differentiable Gumbel-Softmax blending
        if self.training:
            h_exec = self._execute_sublayers(x)
            g = gumbel_weights
            if x.dim() == 3:
                g = g.unsqueeze(1)  # [B, 1, 2]
            # Blend EXECUTE (0) and ROUTE_AROUND (1): g0 + g1 == 1.0 strictly
            out_x = g[..., 0:1] * h_exec + g[..., 1:2] * x
            return out_x, action, probs

        # Inference, batch size 1: discrete routing (the only path that physically skips compute)
        if batch == 1:
            if action == "ROUTE_AROUND":
                return x, action, probs
            return self._execute_sublayers(x), action, probs

        # Inference, batched: per-sample selection (correct statistics, no physical skip)
        if not bool(exec_mask.any()):
            return x, action, probs
        h_exec = self._execute_sublayers(x)
        m = exec_mask.view(batch, 1, 1).to(x.dtype)
        return m * h_exec + (1.0 - m) * x, action, probs

class FlowRouteDenoiser(nn.Module):
    """
    Combined Architecture: Recurrent Flow Denoiser with Dynamic Tri-State Layer Routing.
    """

    def __init__(
        self,
        seq_len: int = 16,
        num_classes: int = 4,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ternary: bool = True,
        use_trained_router: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ternary = ternary

        in_dim = 3 * num_classes
        self.in_proj = BitLinear(in_dim, d_model, bias=True, ternary=ternary)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Time MLP
        self.time_mlp = nn.Sequential(
            BitLinear(d_model, d_model, bias=True, ternary=ternary),
            nn.SiLU(),
            BitLinear(d_model, d_model, bias=True, ternary=ternary),
        )

        # Recurrent Transformer Layers with Routers
        self.blocks = nn.ModuleList([
            FlowRouteBlock(
                d_model=d_model,
                n_heads=n_heads,
                ternary=ternary,
                use_trained_router=use_trained_router,
            )
            for _ in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(d_model)
        self.out_head = BitLinear(d_model, num_classes, bias=True, ternary=ternary)
        self.last_probs: List[torch.Tensor] = []

    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        B = t.shape[0]
        half_dim = self.d_model // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        args = t.view(-1, 1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.time_mlp(emb).unsqueeze(1)

    def forward(
        self,
        x_t: torch.Tensor,
        c: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        force_layer_action: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        h_in = torch.cat([x_t, c, s], dim=-1)
        h = self.in_proj(h_in) + self.pos_emb + self._embed_time(t)

        actions = []
        probs_list = []
        for block in self.blocks:
            h, act, p = block(h, force_action=force_layer_action)
            actions.append(act)
            probs_list.append(p)

        self.last_probs = probs_list
        logits = self.out_head(self.ln_out(h))
        s_next = torch.softmax(logits, dim=-1)
        return logits, s_next, actions

    def compute_router_loss(self, lambda_sparse: float = 0.01) -> torch.Tensor:
        """Auxiliary loss encouraging ROUTE_AROUND in redundant layers."""
        if not hasattr(self, "last_probs") or not self.last_probs:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        p_execs = [p[:, 0].mean() for p in self.last_probs]
        return lambda_sparse * torch.stack(p_execs).mean()

    def rollout_inference(
        self,
        x_t: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        clue_mask: Optional[torch.Tensor] = None,
        max_steps: int = 5,
        early_exit: bool = False,
        eps_exit: float = 0.03,
        force_layer_action: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Inference rollout with both temporal early exit AND spatial layer bypass tracking.
        force_layer_action="EXECUTE" disables layer routing (step-exit-only control).
        """
        B, L, V = x_t.shape
        device = x_t.device
        s = torch.full((B, L, V), 1.0 / V, device=device)
        if clue_mask is not None:
            s = (1.0 - clue_mask) * s + clue_mask * c

        deltas = []
        total_layers_executed = 0
        total_layers_bypassed = 0
        steps_taken = max_steps
        early_exited = False

        with torch.no_grad():
            for k in range(1, max_steps + 1):
                _, s_next, acts = self.forward(x_t, c, s, t, force_layer_action=force_layer_action)
                if clue_mask is not None:
                    s_next = (1.0 - clue_mask) * s_next + clue_mask * c

                # Track layer bypass statistics per sample (not just sample 0)
                for block in self.blocks:
                    n_exec = int(block.last_exec_mask.sum().item())
                    total_layers_executed += n_exec
                    total_layers_bypassed += B - n_exec

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
        total_possible = steps_taken * self.n_layers * B
        layer_savings = total_layers_bypassed / max(total_possible, 1)

        return {
            "s_final": s,
            "predictions": pred,
            "steps_taken": steps_taken,
            "contraction_deltas": deltas,
            "early_exited": early_exited,
            "layers_executed": total_layers_executed,
            "layers_bypassed": total_layers_bypassed,
            "layer_saving_pct": round(layer_savings * 100, 1),
        }

    def train_fpf_step(
        self,
        x_clean: torch.Tensor,
        target_indices: torch.Tensor,
        clue_mask: torch.Tensor,
        c: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        max_rollout_k: int = 4,
        lambda_sparse: float = 0.01,
        leak_free: bool = True,
    ) -> float:
        """FPF step; leak_free builds x_t from the visible condition c (see FlowTritDenoiser.train_fpf_step)."""
        self.train()
        B, L, V = x_clean.shape
        device = x_clean.device

        t = torch.rand(B, 1, device=device) * 0.8 + 0.1
        eps = torch.randn_like(x_clean)
        t_expand = t.unsqueeze(-1)
        x_target = c if leak_free else x_clean
        x_t = (1.0 - t_expand) * eps + t_expand * x_target

        k_rollout = torch.randint(0, max_rollout_k + 1, (1,)).item()
        s = torch.full_like(x_clean, 1.0 / V)
        s = (1.0 - clue_mask) * s + clue_mask * c

        with torch.no_grad():
            for _ in range(k_rollout):
                _, s_next, _ = self.forward(x_t, c, s, t)
                s = (1.0 - clue_mask) * s_next + clue_mask * c

        s_fpf = s.detach()

        optimizer.zero_grad()
        logits, _, _ = self.forward(x_t, c, s_fpf, t)
        
        ce_loss = F.cross_entropy(logits.view(-1, V), target_indices.view(-1))
        router_loss = self.compute_router_loss(lambda_sparse=lambda_sparse)
        total_loss = ce_loss + router_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return total_loss.item()
