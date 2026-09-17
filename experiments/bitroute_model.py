"""
experiments/bitroute_model.py: Complete native PyTorch implementation of BitRoute-135M.

Architecture:
  - 12 Transformer Layers, d_model=768, 12 Attention Heads (d_head=64), d_ff=2048 (SwiGLU)
  - All projections (Q, K, V, O, Gate, Up, Down) use native ternary BitLinear {-1, 0, +1}
  - Each layer embeds a TriStateRouter probe (D_router) supporting:
      * EXECUTE: Full attention + FFN computation
      * ROUTE_AROUND: Pure residual bypass (h_{l+1} = h_l), zero BitLinear forward computation
      * EARLY_EXIT: Emits prediction from current hidden state and halts subsequent layers
  - Parameter count: ~134.1M to ~135M parameters.
  - Fully compatible with CUDA execution (e.g. RTX 4060).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.bitlinear import BitLinear
from experiments.tristate_router import (
    TriStateRouter,
    RouterDecision,
    ACTION_EXECUTE,
    ACTION_ROUTE_AROUND,
    ACTION_EARLY_EXIT,
)


@dataclass
class BitRouteConfig:
    """Configuration for BitRoute-135M."""
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    block_size: int = 256  # Block size for ternary absmean scaling (g=256)
    quantize_act: bool = False
    tie_word_embeddings: bool = False
    tau_exit: float = 0.8
    tau_bypass: float = 0.5
    use_trained_router: bool = False
    ternary: bool = True  # False = matched FP32 control (same architecture, plain linear layers)


class BitRouteRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class BitRouteAttention(nn.Module):
    """
    Multi-head causal self-attention using ternary BitLinear projections (Q, K, V, O).
    """

    def __init__(self, config: BitRouteConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        assert (
            self.head_dim * self.num_heads == self.hidden_size
        ), "hidden_size must be divisible by num_attention_heads"

        # Ternary BitLinear projections
        self.q_proj = BitLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            block_size=config.block_size,
            quantize_act=config.quantize_act,
            ternary=config.ternary,
        )
        self.k_proj = BitLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            block_size=config.block_size,
            quantize_act=config.quantize_act,
            ternary=config.ternary,
        )
        self.v_proj = BitLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            block_size=config.block_size,
            quantize_act=config.quantize_act,
            ternary=config.ternary,
        )
        self.o_proj = BitLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            block_size=config.block_size,
            quantize_act=config.quantize_act,
            ternary=config.ternary,
        )

        # Execution tracker
        self.forward_called = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.forward_called += 1
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=(attention_mask is None and seq_len > 1)
        )

        # Transpose back and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class BitRouteFFN(nn.Module):
    """
    SwiGLU feed-forward network using ternary BitLinear projections (Gate, Up, Down).
    """

    def __init__(self, config: BitRouteConfig):
        super().__init__()
        self.gate_proj = BitLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            block_size=config.block_size,
            quantize_act=config.quantize_act,
            ternary=config.ternary,
        )
        self.up_proj = BitLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            block_size=config.block_size,
            quantize_act=config.quantize_act,
            ternary=config.ternary,
        )
        self.down_proj = BitLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            block_size=config.block_size,
            quantize_act=config.quantize_act,
            ternary=config.ternary,
        )

        # Execution tracker
        self.forward_called = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_called += 1
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BitRouteTransformerBlock(nn.Module):
    """
    Single transformer layer integrating:
      1. TriStateRouter probe (D_router)
      2. Residual bypass mechanism (ROUTE_AROUND -> 0 payload bytes loaded)
      3. BitLinear Attention and FFN
    """

    def __init__(self, config: BitRouteConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.router = TriStateRouter(
            config.hidden_size,
            tau_exit=config.tau_exit,
            tau_bypass=config.tau_bypass,
            legacy_dispatch=not getattr(config, 'use_trained_router', True)
        )
        self.input_layernorm = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = BitRouteAttention(config)
        self.post_attention_layernorm = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = BitRouteFFN(config)

    def _execute_layer(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        norm_h = self.input_layernorm(hidden_states)
        attn_out = self.attn(norm_h, attention_mask=attention_mask)
        hidden_states = residual + attn_out

        residual = hidden_states
        norm_h2 = self.post_attention_layernorm(hidden_states)
        ffn_out = self.ffn(norm_h2)
        hidden_states = residual + ffn_out
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        force_action: Optional[str] = None,
        active_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, RouterDecision]:
        """
        active_mask: optional [B] bool, False for samples that already EARLY_EXITed upstream
        (batched evaluation only). Exited samples keep their hidden state unchanged.
        """
        # 1. Query the lightweight router probe
        decision = self.router(hidden_states, force_action=force_action)

        if self.training and decision.gumbel_weights is not None:
            h_exec = self._execute_layer(hidden_states, attention_mask)
            g = decision.gumbel_weights  # [batch, 3]
            if hidden_states.dim() == 3:
                g = g.unsqueeze(1)  # [batch, 1, 3]
            # EXECUTE blends in the computed layer. ROUTE_AROUND and EARLY_EXIT both keep the
            # residual: an exit means "no further layers", which is the identity at this layer.
            # (Before 2026-09-16 the EXIT weight was dropped, so a sampled exit zeroed h.)
            keep = g[..., 1:2] + g[..., 2:3]
            hidden_states = g[..., 0:1] * h_exec + keep * hidden_states
            return hidden_states, decision

        batch = hidden_states.shape[0]
        if batch == 1 or decision.action_ids is None:
            # Discrete dispatch: the only path that physically skips the layer payload.
            # ROUTE_AROUND and EARLY_EXIT both leave h_{l+1} = h_l; the model decides whether
            # an EXIT halts the remaining layers.
            if decision.action in ("EARLY_EXIT", "ROUTE_AROUND"):
                return hidden_states, decision
            return self._execute_layer(hidden_states, attention_mask), decision

        # Batched evaluation: per-sample selection. Correct routing statistics, no physical skip.
        exec_mask = decision.action_ids == ACTION_EXECUTE
        if active_mask is not None:
            exec_mask = exec_mask & active_mask
        if not bool(exec_mask.any()):
            return hidden_states, decision
        h_exec = self._execute_layer(hidden_states, attention_mask)
        m = exec_mask.view(batch, 1, 1).to(hidden_states.dtype)
        return m * h_exec + (1.0 - m) * hidden_states, decision


@dataclass
class BitRouteOutput:
    """Complete output and execution telemetry for BitRoute-135M."""
    logits: torch.Tensor
    executed_layers: List[int]
    bypassed_layers: List[int]
    exit_layer: Optional[int]
    routing_decisions: List[RouterDecision]
    payload_bytes_streamed: int
    full_payload_bytes: int
    bandwidth_saving_ratio: float
    # Fractions over all (sample, layer) pairs; they sum to 1. Correct for batched evaluation.
    layer_exec_fraction: float = 1.0
    layer_bypass_fraction: float = 0.0
    layer_exit_fraction: float = 0.0


class BitRouteForCausalLM(nn.Module):
    """
    BitRoute-135M: Native Ternary Transformer with Tri-State Dynamic Continuation Routing.
    """

    def __init__(self, config: Optional[BitRouteConfig] = None):
        super().__init__()
        self.config = config or BitRouteConfig()

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList([
            BitRouteTransformerBlock(self.config, layer_idx=i)
            for i in range(self.config.num_hidden_layers)
        ])
        self.norm = BitRouteRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Weight byte accounting (TQ1_0 format: 1.6875 bpw)
        self.weights_per_layer = (
            4 * (self.config.hidden_size * self.config.hidden_size) +
            3 * (self.config.hidden_size * self.config.intermediate_size)
        )
        self.bytes_per_layer_payload = int(self.weights_per_layer * 1.6875 / 8)
        self.total_layer_payload_bytes = self.bytes_per_layer_payload * self.config.num_hidden_layers

    def count_parameters(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        layer_params = sum(p.numel() for layer in self.layers for p in layer.parameters())
        embed_params = sum(p.numel() for p in self.embed_tokens.parameters())
        head_params = sum(p.numel() for p in self.lm_head.parameters())
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "layer_parameters": layer_params,
            "embed_parameters": embed_params,
            "head_parameters": head_params,
        }

    def compute_router_loss(self, lambda_sparse: float = 0.01) -> torch.Tensor:
        """
        Computes sparsity penalty on routing decisions to encourage ROUTE_AROUND/EARLY_EXIT.
        """
        if not hasattr(self, 'last_routing_decisions') or not self.last_routing_decisions:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            return torch.tensor(0.0, device=device)
            
        p_execs = []
        for dec in self.last_routing_decisions:
            if hasattr(dec, 'probs') and dec.probs is not None:
                # probs is [batch, 3], ACTION_EXECUTE is 0
                p_execs.append(dec.probs[..., ACTION_EXECUTE].mean())
                
        if not p_execs:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            return torch.tensor(0.0, device=device)
            
        mean_p_exec = torch.stack(p_execs).mean()
        return lambda_sparse * mean_p_exec

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        early_exit_enabled: bool = True,
        force_layer_actions: Optional[Dict[int, str]] = None,
    ) -> BitRouteOutput:
        """
        Forward pass with dynamic layer routing.

        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: optional attention mask
            early_exit_enabled: if True, halts forward pass when a router emits EARLY_EXIT
            force_layer_actions: optional dict mapping layer_idx -> forced action
        """
        hidden_states = self.embed_tokens(input_ids)
        batch = hidden_states.shape[0]
        num_layers = len(self.layers)

        executed_layers: List[int] = []
        bypassed_layers: List[int] = []
        exit_layer: Optional[int] = None
        decisions: List[RouterDecision] = []

        # Per-(sample, layer) status accumulators; exec + bypass + exit == batch * num_layers.
        active = torch.ones(batch, dtype=torch.bool, device=hidden_states.device)
        exec_total = 0.0
        bypass_total = 0.0
        exit_total = 0.0

        for idx, layer in enumerate(self.layers):
            force_act = force_layer_actions.get(idx, None) if force_layer_actions else None
            hidden_states, decision = layer(
                hidden_states,
                attention_mask=attention_mask,
                force_action=force_act,
                active_mask=active if (early_exit_enabled and batch > 1) else None,
            )
            decisions.append(decision)

            if batch == 1 or decision.action_ids is None:
                if decision.action == "EARLY_EXIT" and early_exit_enabled:
                    exit_layer = idx
                    exit_total += float(num_layers - idx)  # this layer and every deeper one
                    break
                elif decision.action in ("ROUTE_AROUND", "EARLY_EXIT"):
                    # An EXIT decision with early exit disabled acts as a bypass.
                    bypassed_layers.append(idx)
                    bypass_total += 1.0
                else:
                    executed_layers.append(idx)
                    exec_total += 1.0
                continue

            ids = decision.action_ids
            is_exec = (ids == ACTION_EXECUTE) & active
            is_exit = (ids == ACTION_EARLY_EXIT) & active
            if not early_exit_enabled:
                is_exit = torch.zeros_like(is_exit)
            is_bypass = active & ~is_exec & ~is_exit
            exec_total += float(is_exec.sum().item())
            bypass_total += float(is_bypass.sum().item())
            exit_total += float((~active).sum().item() + is_exit.sum().item())
            if early_exit_enabled:
                active = active & ~is_exit
            if bool(is_exec.any()):
                executed_layers.append(idx)
            else:
                bypassed_layers.append(idx)

        # Emit prediction using final norm + lm_head (whether full or early exit)
        norm_hidden = self.norm(hidden_states)
        logits = self.lm_head(norm_hidden)

        self.last_routing_decisions = decisions

        denom = float(batch * num_layers)
        layer_exec_fraction = exec_total / denom
        layer_bypass_fraction = bypass_total / denom
        layer_exit_fraction = exit_total / denom

        # Memory payload accounting (bytes streamed per sample, averaged over the batch):
        streamed_bytes = int(layer_exec_fraction * num_layers * self.bytes_per_layer_payload)
        full_bytes = self.total_layer_payload_bytes
        ratio = full_bytes / max(streamed_bytes, 1) if streamed_bytes > 0 else float("inf")

        return BitRouteOutput(
            logits=logits,
            executed_layers=executed_layers,
            bypassed_layers=bypassed_layers,
            exit_layer=exit_layer,
            routing_decisions=decisions,
            payload_bytes_streamed=streamed_bytes,
            full_payload_bytes=full_bytes,
            bandwidth_saving_ratio=round(ratio, 2),
            layer_exec_fraction=layer_exec_fraction,
            layer_bypass_fraction=layer_bypass_fraction,
            layer_exit_fraction=layer_exit_fraction,
        )
