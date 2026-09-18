"""
experiments/frontier_scaling/navitrit_max_model.py: NaviTrit-Max (Scaled Multi-Tile Hybrid Architecture).

Scaled up for RTX 4060 GPU (8.58 GB VRAM, 24 MB L2 Cache):
- Hidden dimension: 1024 (up from 192 / 768)
- Intermediate dimension: 4096 (up from 512 / 2048)
- Attention heads: 16 (head dim 64)
- Max physical parameters: ~246.5 Million (6 layers with tied embeddings)
- Routing Block:
  1. Tile 0: Ternary Multi-Head Causal Attention (Flash SDPA, O(S^2) associative recall)
  2. Tile 1: Ternary Causal Depthwise Conv-Mixer (O(S) local sequence context)
  3. Tile 2: Wide Ternary SwiGLU FFN (General Linguistic & Syntactic Knowledge, 4096 dim)
  4. Tile 3: Wide Ternary SwiGLU FFN (Crop Science & Environmental General Knowledge, 4096 dim)
  5. Differentiable Softmax Dynamic Router with load-balancing loss
- Quantization: All projections strictly ternary {-1, 0, +1} via BitLinear with STE & absmean scaling
"""

import os
import sys
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.bitlinear import BitLinear
from experiments.bitroute_model import BitRouteRMSNorm


@dataclass
class NaviTritMaxConfig:
    """Configuration for NaviTrit-Max Scaled Hybrid Architecture."""
    vocab_size: int = 50257
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    num_layers: int = 6                # 6 layers x 32.5M + 51.5M tied embed = ~246.5M params
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-5
    block_size: int = 256              # Block size for ternary absmean quantization (g=256)
    ternary: bool = True               # Strict {-1, 0, +1} BitLinear quantization
    tie_word_embeddings: bool = True   # Ties LM head to input embedding to conserve 51.5M params
    num_experts: int = 2               # Expert 0: General, Expert 1: Agro-Environmental
    d_conv: int = 4                    # Causal 1D depthwise convolution kernel
    balance_loss_weight: float = 0.01  # Auxiliary MoE routing balance loss
    arch_mode: str = "deep_pipeline"   # "deep_pipeline" or "looped_dwp"
    max_loops: int = 6                 # Recursion depth if arch_mode == "looped_dwp"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class TernaryCausalAttention(nn.Module):
    """
    Ternary Multi-Head Causal Self-Attention using FlashAttention / PyTorch C++ SDPA.
    All projections (Q, K, V, O) are strictly ternary {-1, 0, +1}.
    """
    def __init__(self, config: NaviTritMaxConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        # Fused QKV projection
        self.qkv_proj = BitLinear(
            self.hidden_size,
            3 * self.hidden_size,
            bias=False,
            block_size=config.block_size,
            ternary=config.ternary,
        )
        # Out projection
        self.out_proj = BitLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            block_size=config.block_size,
            ternary=config.ternary,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x).view(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # PyTorch native C++ fused scaled dot-product attention (FlashAttention / memory efficient)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class TernaryCausalConvMixer(nn.Module):
    """
    Local Context & State Sequence Mixer using Causal 1D Depthwise Convolution and SiLU Gating.
    Provides O(S) linear token interaction with negligible GPU latency (~1 ms).
    """
    def __init__(self, config: NaviTritMaxConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.d_conv = config.d_conv

        # In-projection to branch u and gate z
        self.in_proj = BitLinear(
            self.hidden_size,
            2 * self.hidden_size,
            bias=False,
            block_size=config.block_size,
            ternary=config.ternary,
        )
        # Depthwise causal 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=config.d_conv,
            groups=self.hidden_size,
            padding=config.d_conv - 1,
            bias=True,
        )
        # Out-projection
        self.out_proj = BitLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            block_size=config.block_size,
            ternary=config.ternary,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        in_out = self.in_proj(x)
        u, z = in_out.chunk(2, dim=-1)

        # Causal convolution: [B, D, S] -> [B, D, S + kernel - 1] -> slice [:, :, :S]
        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :S].transpose(1, 2)
        # Gated non-linearity
        y = F.silu(u_conv) * torch.sigmoid(z)
        return self.out_proj(y)


class TernarySwiGLUExpert(nn.Module):
    """Wide Ternary SwiGLU Feedforward Expert."""
    def __init__(self, config: NaviTritMaxConfig):
        super().__init__()
        self.gate_up_proj = BitLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
            block_size=config.block_size,
            ternary=config.ternary,
        )
        self.down_proj = BitLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            block_size=config.block_size,
            ternary=config.ternary,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TernaryDualExpertFFN(nn.Module):
    """
    Routing Block containing 2 Wide Ternary SwiGLU Experts:
    - Expert 0: General Linguistic & Syntactic Knowledge
    - Expert 1: Agronomic, Crop Science & Environmental General Knowledge
    """
    def __init__(self, config: NaviTritMaxConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

        self.expert_0 = TernarySwiGLUExpert(config)  # General
        self.expert_1 = TernarySwiGLUExpert(config)  # Agro-Environmental

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Router logits: [B, S, 2]
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        # Expert outputs
        out_0 = self.expert_0(x)
        out_1 = self.expert_1(x)

        # Soft mixture
        p0 = router_probs[:, :, 0:1]
        p1 = router_probs[:, :, 1:2]
        y = p0 * out_0 + p1 * out_1

        # Load balancing auxiliary loss
        mean_p0 = p0.mean()
        mean_p1 = p1.mean()
        balance_loss = self.config.num_experts * (mean_p0 * mean_p0 + mean_p1 * mean_p1) - 1.0

        return y, router_probs, balance_loss


class NaviTritMaxLayer(nn.Module):
    """
    NaviTrit-Max Layer comprising:
    1. Pre-RMSNorm -> Ternary Multi-Head Attention -> Residual
    2. Pre-RMSNorm -> Ternary Causal Conv Mixer -> Residual
    3. Pre-RMSNorm -> Ternary Dual-Expert Routing FFN -> Residual
    """
    def __init__(self, config: NaviTritMaxConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm1 = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = TernaryCausalAttention(config)

        self.norm2 = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.conv_mixer = TernaryCausalConvMixer(config)

        self.norm3 = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.routing_ffn = TernaryDualExpertFFN(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Attention sub-layer
        x = x + self.attn(self.norm1(x))

        # 2. Local sequence convolution sub-layer
        x = x + self.conv_mixer(self.norm2(x))

        # 3. Dual-Expert Routing FFN sub-layer
        ffn_out, probs, balance_loss = self.routing_ffn(self.norm3(x))
        x = x + ffn_out

        return x, probs, balance_loss


class NaviTritMaxForCausalLM(nn.Module):
    """
    NaviTrit-Max Full Causal Language Model (~246.5M parameters).
    Optimized for multi-hour pretraining on NVIDIA RTX 4060 GPU.
    """
    def __init__(self, config: NaviTritMaxConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)

        # Positional embedding
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)

        # Layers
        if config.arch_mode == "looped_dwp":
            # 1 shared super-block looped dynamically
            self.layers = nn.ModuleList([NaviTritMaxLayer(config, layer_idx=0)])
        else:
            # Full deep sequential pipeline
            self.layers = nn.ModuleList([
                NaviTritMaxLayer(config, layer_idx=i) for i in range(config.num_layers)
            ])

        # Final RMSNorm
        self.final_norm = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output LM Head
        if config.tie_word_embeddings:
            self.lm_head = None  # Tied to tok_embeddings.weight
        else:
            self.lm_head = BitLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                block_size=config.block_size,
                ternary=config.ternary,
            )

    def count_parameters(self) -> Dict[str, Union[int, float]]:
        """Calculates exact total and trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Breakdown
        embed_params = sum(p.numel() for p in self.tok_embeddings.parameters()) + sum(p.numel() for p in self.pos_embeddings.parameters())
        layer_params = sum(p.numel() for p in self.layers.parameters())
        head_params = sum(p.numel() for p in self.lm_head.parameters()) if self.lm_head is not None else 0

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "embedding_parameters": embed_params,
            "layer_parameters": layer_params,
            "head_parameters": head_params,
            "total_millions": total_params / 1e6,
            "layers_count": len(self.layers),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(0, S, device=device).unsqueeze(0)
        h = self.tok_embeddings(input_ids) + self.pos_embeddings(positions)

        total_balance_loss = torch.tensor(0.0, device=device)
        all_routing_probs = []

        if self.config.arch_mode == "looped_dwp":
            super_block = self.layers[0]
            for loop_idx in range(self.config.max_loops):
                h, probs, b_loss = super_block(h)
                total_balance_loss = total_balance_loss + b_loss
                all_routing_probs.append(probs)
        else:
            for layer in self.layers:
                h, probs, b_loss = layer(h)
                total_balance_loss = total_balance_loss + b_loss
                all_routing_probs.append(probs)

        # Final normalization
        h = self.final_norm(h)

        # LM Head projection
        if self.lm_head is not None:
            logits = self.lm_head(h)
        else:
            # Tied embedding: logits = h @ W_emb^T
            logits = F.linear(h, self.tok_embeddings.weight)

        loss = None
        ce_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Composite loss with MoE load balance
            loss = ce_loss + self.config.balance_loss_weight * total_balance_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "logits": logits,
            "balance_loss": total_balance_loss,
            "routing_probs": all_routing_probs,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
    ) -> torch.Tensor:
        """Autoregressive generation for validation & qualitative evaluation."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_position_embeddings else input_ids[:, -self.config.max_position_embeddings:]
            out = self.forward(idx_cond)
            logits = out["logits"][:, -1, :]
            
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


if __name__ == "__main__":
    cfg = NaviTritMaxConfig()
    model = NaviTritMaxForCausalLM(cfg)
    info = model.count_parameters()
    print("=" * 60)
    print(f"NaviTrit-Max Model Configuration Audit:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60)
