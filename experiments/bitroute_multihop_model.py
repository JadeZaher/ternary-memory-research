"""
experiments/bitroute_multihop_model.py: Native Ternary Transformer with Flow-Reasoned Multi-Hop Routing.

Decouples each transformer layer into two independent execution hops:
  Hop 1: Attention (Q, K, V, O) - 33% of layer weights
  Hop 2: FFN (Gate, Up, Down)  - 67% of layer weights

Driven by global FlowRoutingPlanner:
  - Relaxed continuous attractor routing state r*
  - Emits execution mask M* in {0, 1}^{B x L x 2}
  - Batch-1 physical skip: when M_{l, ffn} = 0, FFN weights are never accessed,
    saving 67% of the DRAM memory streaming for that layer.

Reference: research/dynamic-routing-flow-theory.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.bitlinear import BitLinear
from experiments.flow_router import (
    FlowRoutingPlanner,
    FlowRoutingDecision,
    SUBLAYER_ATTN,
    SUBLAYER_FFN,
)
from experiments.bitroute_model import (
    BitRouteConfig,
    BitRouteRMSNorm,
    BitRouteAttention,
    BitRouteFFN,
)


@dataclass
class MultiHopBlockOutput:
    """Output and execution telemetry from a single multi-hop layer."""
    hidden_states: torch.Tensor
    attn_executed: bool
    ffn_executed: bool


class BitRouteMultiHopBlock(nn.Module):
    """
    Decoupled 2-Hop Transformer Layer:
      Hop 1: h' = h + M_attn * Attention(LN1(h))
      Hop 2: h'' = h' + M_ffn * FFN(LN2(h'))
    """
    def __init__(self, config: BitRouteConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.input_layernorm = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = BitRouteAttention(config)
        self.post_attention_layernorm = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = BitRouteFFN(config)

    def _execute_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_h = self.input_layernorm(hidden_states)
        attn_out = self.attn(norm_h, attention_mask=attention_mask)
        return hidden_states + attn_out

    def _execute_ffn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_h = self.post_attention_layernorm(hidden_states)
        ffn_out = self.ffn(norm_h)
        return hidden_states + ffn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask_attn: torch.Tensor,
        mask_ffn: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> MultiHopBlockOutput:
        """
        Execute 2-hop block driven by masks mask_attn [B] and mask_ffn [B].
        """
        B = hidden_states.shape[0]

        # -------------------------------------------------------------
        # HOP 1: Attention Sublayer (33% parameter payload)
        # -------------------------------------------------------------
        attn_executed = False
        if B == 1 and not self.training:
            # Physical skip: if mask is 0, never touch attention weights
            if mask_attn[0].item() > 0.5:
                hidden_states = self._execute_attention(hidden_states, attention_mask)
                attn_executed = True
        else:
            # Batched or training forward
            h_attn = self._execute_attention(hidden_states, attention_mask)
            attn_executed = True
            m_a = mask_attn.view(B, 1, 1).to(hidden_states.dtype)
            hidden_states = m_a * h_attn + (1.0 - m_a) * hidden_states

        # -------------------------------------------------------------
        # HOP 2: FFN Sublayer (67% parameter payload)
        # -------------------------------------------------------------
        ffn_executed = False
        if B == 1 and not self.training:
            # Physical skip: if mask is 0, never touch FFN weights
            if mask_ffn[0].item() > 0.5:
                hidden_states = self._execute_ffn(hidden_states)
                ffn_executed = True
        else:
            # Batched or training forward
            h_ffn = self._execute_ffn(hidden_states)
            ffn_executed = True
            m_f = mask_ffn.view(B, 1, 1).to(hidden_states.dtype)
            hidden_states = m_f * h_ffn + (1.0 - m_f) * hidden_states

        return MultiHopBlockOutput(
            hidden_states=hidden_states,
            attn_executed=attn_executed,
            ffn_executed=ffn_executed,
        )


@dataclass
class BitRouteMultiHopOutput:
    """Output and execution telemetry for BitRouteMultiHop LM."""
    logits: torch.Tensor
    routing_decision: FlowRoutingDecision
    payload_bytes_streamed: int
    full_payload_bytes: int
    bandwidth_saving_ratio: float
    total_budget_loss: torch.Tensor
    total_fpf_loss: torch.Tensor


class BitRouteMultiHopForCausalLM(nn.Module):
    """
    BitRoute Transformer with Flow-Reasoned Multi-Hop Routing.
    """
    def __init__(
        self,
        config: Optional[BitRouteConfig] = None,
        d_route: int = 128,
        max_flow_steps: int = 3,
        dt: float = 0.5,
        eps_contraction: float = 1e-3,
        default_target_budget: float = 0.70,
    ):
        super().__init__()
        self.config = config or BitRouteConfig()

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)

        # Global Flow-Reasoned Routing Planner
        self.flow_router = FlowRoutingPlanner(
            num_layers=self.config.num_hidden_layers,
            d_model=self.config.hidden_size,
            d_route=d_route,
            max_flow_steps=max_flow_steps,
            dt=dt,
            eps_contraction=eps_contraction,
            default_target_budget=default_target_budget,
        )

        # Decoupled 2-hop transformer layers
        self.layers = nn.ModuleList([
            BitRouteMultiHopBlock(self.config, layer_idx=i)
            for i in range(self.config.num_hidden_layers)
        ])

        self.norm = BitRouteRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Weight byte accounting (TQ1_0 format: 1.6875 bpw)
        # Attention: 4 * (d * d)
        self.attn_weights_per_layer = 4 * (self.config.hidden_size * self.config.hidden_size)
        self.attn_bytes_per_layer = int(self.attn_weights_per_layer * 1.6875 / 8)

        # FFN: 3 * (d * d_intermediate)
        self.ffn_weights_per_layer = 3 * (self.config.hidden_size * self.config.intermediate_size)
        self.ffn_bytes_per_layer = int(self.ffn_weights_per_layer * 1.6875 / 8)

        self.layer_payload_bytes = self.attn_bytes_per_layer + self.ffn_bytes_per_layer
        self.full_payload_bytes = self.layer_payload_bytes * self.config.num_hidden_layers

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_budget: Optional[float] = None,
        force_all_execute: bool = False,
    ) -> BitRouteMultiHopOutput:
        """
        Forward pass with global flow-reasoned multi-hop execution.
        """
        hidden_states = self.embed_tokens(input_ids)
        B = hidden_states.shape[0]

        # 1. Global Flow-Reasoned Trajectory Planning
        decision = self.flow_router(
            hidden_states,
            attention_mask=attention_mask,
            target_budget=target_budget,
            force_all_execute=force_all_execute,
        )
        mask = decision.execution_mask  # [B, L, 2]

        # 2. Sequential Decoupled Layer Execution
        bytes_streamed = 0
        for l_idx, layer in enumerate(self.layers):
            mask_attn = mask[:, l_idx, SUBLAYER_ATTN]
            mask_ffn = mask[:, l_idx, SUBLAYER_FFN]

            block_out = layer(
                hidden_states,
                mask_attn=mask_attn,
                mask_ffn=mask_ffn,
                attention_mask=attention_mask,
            )
            hidden_states = block_out.hidden_states

            # Accounting for batch-1 physical skips
            if B == 1 and not self.training:
                if block_out.attn_executed:
                    bytes_streamed += self.attn_bytes_per_layer
                if block_out.ffn_executed:
                    bytes_streamed += self.ffn_bytes_per_layer
            else:
                # Batched or training expectation
                attn_frac = mask_attn.mean().item()
                ffn_frac = mask_ffn.mean().item()
                bytes_streamed += int(attn_frac * self.attn_bytes_per_layer + ffn_frac * self.ffn_bytes_per_layer)

        # 3. Final Norm and Vocabulary Projection
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # 4. Routing Losses
        budget_loss = self.flow_router.compute_budget_loss(decision, target_budget)
        fpf_loss = decision.fpf_loss

        # Savings ratio
        savings_ratio = 1.0 - (bytes_streamed / max(1, self.full_payload_bytes))

        return BitRouteMultiHopOutput(
            logits=logits,
            routing_decision=decision,
            payload_bytes_streamed=bytes_streamed,
            full_payload_bytes=self.full_payload_bytes,
            bandwidth_saving_ratio=savings_ratio,
            total_budget_loss=budget_loss,
            total_fpf_loss=fpf_loss,
        )


def test_multihop_execution():
    """Quick verification of multi-hop transformer on local device."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Verifying BitRouteMultiHopForCausalLM on {device}...")

    config = BitRouteConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=128,
        ternary=True,
    )

    model = BitRouteMultiHopForCausalLM(config).to(device)

    # 1. Test batch-1 physical skip
    model.eval()
    input_ids = torch.randint(0, 1000, (1, 16), device=device)
    with torch.no_grad():
        out_eval = model(input_ids, target_budget=0.50)
    print(f"Batch-1 Eval Pass: Bytes streamed {out_eval.payload_bytes_streamed:,} / {out_eval.full_payload_bytes:,} ({out_eval.bandwidth_saving_ratio*100:.1f}% saved)")
    print(f"Attn Exec: {out_eval.routing_decision.mean_attn_exec:.2f}, FFN Exec: {out_eval.routing_decision.mean_ffn_exec:.2f}")

    # 2. Test training mode backward pass
    model.train()
    input_ids_train = torch.randint(0, 1000, (2, 16), device=device)
    out_train = model(input_ids_train, target_budget=0.60)
    loss = out_train.logits.sum() * 1e-4 + 0.1 * out_train.total_budget_loss + 0.01 * out_train.total_fpf_loss
    loss.backward()

    # Check gradients on flow router
    router_grad_norm = torch.norm(model.flow_router.value_head[1].weight.grad).item()
    print(f"Training Mode Pass: Loss = {loss.item():.4f}, Router Value Head Grad Norm = {router_grad_norm:.4e}")
    assert router_grad_norm > 0, "Router value head received zero grad!"
    print("Multi-hop model verification PASSED successfully!")


if __name__ == "__main__":
    test_multihop_execution()
