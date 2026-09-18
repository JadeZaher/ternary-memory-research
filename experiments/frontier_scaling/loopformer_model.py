"""
experiments/frontier_scaling/loopformer_model.py: Looped Transformer & Mixture-of-Recursions (MoR) Engine.

Gate 19 (Track G):
1. Consolidates 12 physical transformer layers (85M params) into a single 7.09M parameter-shared
   recurrent super-block F_theta = (RMSNorm1, BitRouteAttention, RMSNorm2, BitRouteFFN).
2. Packed 1.58-bit footprint is 1.77 MB, residing 100% permanently in 32MB L2/L3 cache (0 DRAM traffic).
3. Recursion-Wise 2D KV Cache (K_{t,k}, V_{t,k}) for clean causal autoregressive generation.
4. Shortcut-Consistency Architecture: intermediate recursions m in {1,..,M-1} map directly to coherent
   vocabulary predictions via residual projection heads P_short^{(m)}, enabling continuous test-time
   compute budget scaling M in {1, 2, 4, 6, 8} without retraining.
5. Dynamic early halting head p_halt^{(k)} = sigmoid(w_halt^T h + b).
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
from experiments.bitroute_model import (
    BitRouteConfig,
    BitRouteRMSNorm,
    BitRouteAttention,
    BitRouteFFN,
)


@dataclass
class LoopFormerConfig:
    """Configuration for LoopFormer / MoR parameter-shared recurrent models."""
    vocab_size: int = 50257
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    block_size: int = 256
    
    # Recurrent compute budget parameters
    max_recursions: int = 8             # Maximum allowable recursion depth M_max
    default_recursions: int = 4         # Default operational recursion depth
    use_shortcut_consistency: bool = True
    consistency_lambda: float = 0.50     # Weight for intermediate exit loss
    consistency_mu: float = 0.10         # Weight for feature alignment MSE
    
    # Dynamic halting
    use_early_halting: bool = False
    halt_threshold: float = 0.70
    
    # Ternary quantization settings
    quantize_act: bool = False
    ternary: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class RecursionKVCache:
    """
    2D Key-Value Cache indexed across recursion depth k and token position t.
    Shape per recursion k: [batch_size, num_heads, seq_len, head_dim].
    """
    def __init__(self, max_recursions: int):
        self.max_recursions = max_recursions
        self.k_cache: Dict[int, torch.Tensor] = {}
        self.v_cache: Dict[int, torch.Tensor] = {}

    def update(
        self,
        recursion_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new key and value for given recursion level.
        key, value: [batch_size, num_heads, new_tokens, head_dim]
        Returns full cached keys and values for this recursion level.
        """
        if recursion_idx not in self.k_cache:
            self.k_cache[recursion_idx] = key
            self.v_cache[recursion_idx] = value
        else:
            self.k_cache[recursion_idx] = torch.cat([self.k_cache[recursion_idx], key], dim=2)
            self.v_cache[recursion_idx] = torch.cat([self.v_cache[recursion_idx], value], dim=2)
            
        return self.k_cache[recursion_idx], self.v_cache[recursion_idx]

    def reset(self):
        self.k_cache.clear()
        self.v_cache.clear()

    @property
    def current_seq_len(self) -> int:
        if 0 in self.k_cache:
            return self.k_cache[0].shape[2]
        return 0


class RecurrentSuperBlock(nn.Module):
    """
    Single parameter-shared ternary super-block F_theta.
    Consists of:
      - RMSNorm 1
      - BitRouteAttention (ternary Q, K, V, O projections)
      - RMSNorm 2
      - BitRouteFFN (ternary SwiGLU Gate, Up, Down projections)
      - Recursion Step Modulation Embedding e_step(k)
    """
    def __init__(self, config: LoopFormerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        # Sequence Mixing
        self.attn_norm = BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps)
        bitroute_cfg = BitRouteConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            block_size=config.block_size,
            quantize_act=config.quantize_act,
            ternary=config.ternary,
        )
        self.attn = BitRouteAttention(bitroute_cfg)

        # Channel Mixing
        self.ffn_norm = BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = BitRouteFFN(bitroute_cfg)

        # Recursion step condition embeddings: e_step(k) for k in 0..max_recursions-1
        self.step_embeddings = nn.Embedding(config.max_recursions, config.hidden_size)
        nn.init.normal_(self.step_embeddings.weight, mean=0.0, std=0.02)

    def forward_step(
        self,
        h: torch.Tensor,
        recursion_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[RecursionKVCache] = None,
    ) -> torch.Tensor:
        """
        Executes one full recursive pass through the parameter-shared super-block:
          h_tilde = h + e_step(k)
          h_mid = h + Attn(RMSNorm(h_tilde))
          h_out = h_mid + FFN(RMSNorm(h_mid))
        """
        batch_size, seq_len, _ = h.shape
        step_id = torch.tensor([recursion_idx], device=h.device)
        e_k = self.step_embeddings(step_id).view(1, 1, self.hidden_size)

        # Add step condition
        h_cond = h + e_k
        h_norm1 = self.attn_norm(h_cond)

        # Attention projection with recursion-wise KV handling
        if kv_cache is not None:
            # Autoregressive single-token / incremental generation with 2D cache
            q = self.attn.q_proj(h_norm1).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.attn.k_proj(h_norm1).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.attn.v_proj(h_norm1).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            k_all, v_all = kv_cache.update(recursion_idx, k, v)
            
            # If prefilling (seq_len > 1 and q length == k_all length), use causal masking.
            # If generating one token at a time (seq_len == 1), is_causal is False since all past keys are valid.
            is_causal_flag = (attention_mask is None and q.shape[2] > 1 and q.shape[2] == k_all.shape[2])
            attn_out = F.scaled_dot_product_attention(
                q, k_all, v_all,
                attn_mask=attention_mask,
                is_causal=is_causal_flag,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
            delta_attn = self.attn.o_proj(attn_out)
        else:
            # Full sequence forward pass (training or batch prompt evaluation)
            delta_attn = self.attn(h_norm1, attention_mask=attention_mask)

        h_mid = h + delta_attn
        h_norm2 = self.ffn_norm(h_mid)
        delta_ffn = self.ffn(h_norm2)
        h_out = h_mid + delta_ffn

        return h_out


class LoopFormerForCausalLM(nn.Module):
    """
    LoopFormer / Mixture-of-Recursions Causal Language Model.
    Replaces 12 distinct layers with recursive circulation through a single 7.09M super-block.
    """
    def __init__(self, config: LoopFormerConfig):
        super().__init__()
        self.config = config

        # Shared Token & Position Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Single Recurrent Super-Block
        self.super_block = RecurrentSuperBlock(config)

        # Final Readout Norm & Head
        self.ln_f = BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Shortcut Consistency Projectors P_short^{(m)} for intermediate exits
        # Initialized to near-zero so P_short(h) ~= h initially
        self.shortcut_projs = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(config.max_recursions)
        ])
        for proj in self.shortcut_projs:
            nn.init.zeros_(proj.weight)

        # Dynamic early halting head
        self.halt_head = nn.Linear(config.hidden_size, 1)
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, -2.0) # start with low halt probability

    def get_super_block_param_count(self) -> int:
        """Returns parameter count of the parameter-shared recurrent block."""
        return sum(p.numel() for p in self.super_block.parameters())

    def get_packed_footprint_mb(self) -> float:
        """
        Calculates physical footprint of the recurrent super-block packed in 1.58-bit:
        2 bits per weight / 8 bits per byte = 0.25 bytes per weight.
        """
        total_weights = self.get_super_block_param_count()
        return (total_weights * 2.0) / (8.0 * 1024 * 1024)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        recursion_budget: Optional[int] = None,
        return_all_recursions: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass circulating through super_block for `recursion_budget` iterations.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Positional encoding
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        h = self.embed_tokens(input_ids) + self.embed_positions(positions)

        M = recursion_budget if recursion_budget is not None else self.config.default_recursions
        M = min(max(1, M), self.config.max_recursions)

        intermediate_states: List[torch.Tensor] = []
        intermediate_logits: List[torch.Tensor] = []
        halt_probs: List[torch.Tensor] = []

        # Recursion Loop
        for k in range(M):
            h = self.super_block.forward_step(
                h,
                recursion_idx=k,
                attention_mask=attention_mask,
                kv_cache=None,
            )
            intermediate_states.append(h)

            if return_all_recursions or self.config.use_shortcut_consistency:
                # Project intermediate state through shortcut projector P_short^{(k)}
                # P_short(h) = h + W_short(h)
                shortcut_h = h + self.shortcut_projs[k](h)
                normed_short = self.ln_f(shortcut_h)
                logits_k = self.lm_head(normed_short)
                intermediate_logits.append(logits_k)

            if self.config.use_early_halting:
                halt_logit = self.halt_head(self.ln_f(h))
                halt_prob = torch.sigmoid(halt_logit)
                halt_probs.append(halt_prob)

        # Final readout
        final_h = self.ln_f(h)
        logits = self.lm_head(final_h)

        return {
            "logits": logits,
            "intermediate_states": intermediate_states,
            "intermediate_logits": intermediate_logits,
            "halt_probs": halt_probs,
            "recursion_budget": M,
        }

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        recursion_budget: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes Shortcut-Consistency Training Loss:
        L_total = L_LM(h^(M)) + sum_{m=1}^{M-1} lambda_m * [L_LM(P_short(h^(m))) + mu * ||h^(M) - P_short(h^(m))||^2]
        """
        out = self.forward(input_ids, recursion_budget=recursion_budget, return_all_recursions=True)
        M = out["recursion_budget"]
        final_logits = out["logits"]

        # 1. Final deep prediction loss
        loss_final = F.cross_entropy(final_logits.view(-1, self.config.vocab_size), targets.view(-1))

        # 2. Intermediate consistency losses
        loss_intermediate = torch.tensor(0.0, device=input_ids.device)
        loss_alignment = torch.tensor(0.0, device=input_ids.device)

        if M > 1 and len(out["intermediate_logits"]) >= M:
            final_rep = out["intermediate_states"][-1].detach() # Target representation to align with
            for m in range(M - 1):
                inter_logits_m = out["intermediate_logits"][m]
                inter_lm_loss = F.cross_entropy(inter_logits_m.view(-1, self.config.vocab_size), targets.view(-1))
                
                # Feature alignment penalty: ||final_h - P_short(h^(m))||^2
                inter_h_m = out["intermediate_states"][m] + self.shortcut_projs[m](out["intermediate_states"][m])
                align_loss = F.mse_loss(inter_h_m, final_rep)

                decay = (m + 1) / float(M) # earlier recursions receive lower weight
                loss_intermediate = loss_intermediate + decay * inter_lm_loss
                loss_alignment = loss_alignment + decay * align_loss

            loss_intermediate = loss_intermediate / float(M - 1)
            loss_alignment = loss_alignment / float(M - 1)

        total_loss = (
            loss_final
            + self.config.consistency_lambda * loss_intermediate
            + self.config.consistency_mu * loss_alignment
        )

        return {
            "total_loss": total_loss,
            "loss_final": loss_final,
            "loss_intermediate": loss_intermediate,
            "loss_alignment": loss_alignment,
            "recursion_budget": torch.tensor(float(M)),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
        recursion_budget: Optional[int] = None,
        use_kv_cache: bool = True,
        use_early_halting: bool = False,
    ) -> torch.Tensor:
        """
        Fast autoregressive generation with 2D Recursion-Wise KV Caching and dynamic compute dial M.
        """
        self.eval()
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        M = recursion_budget if recursion_budget is not None else self.config.default_recursions
        M = min(max(1, M), self.config.max_recursions)

        generated = input_ids.clone()
        kv_cache = RecursionKVCache(max_recursions=self.config.max_recursions) if use_kv_cache else None

        # Pre-fill prompt if using KV cache
        if use_kv_cache and seq_len > 1:
            # Process prompt tokens up to seq_len - 1 to populate KV cache
            prompt_prefix = generated[:, :-1]
            positions = torch.arange(seq_len - 1, device=device).unsqueeze(0).expand(batch_size, -1)
            h = self.embed_tokens(prompt_prefix) + self.embed_positions(positions)
            for k in range(M):
                h = self.super_block.forward_step(h, recursion_idx=k, kv_cache=kv_cache)

        for step in range(max_new_tokens):
            cur_seq_len = generated.shape[1]
            if cur_seq_len >= self.config.max_position_embeddings:
                break

            if use_kv_cache:
                # Only process the single latest token
                latest_token = generated[:, -1:]
                pos = torch.tensor([[cur_seq_len - 1]], device=device).expand(batch_size, 1)
                h = self.embed_tokens(latest_token) + self.embed_positions(pos)
                
                # Circulate latest token through all M recursions with 2D KV cache
                for k in range(M):
                    h = self.super_block.forward_step(h, recursion_idx=k, kv_cache=kv_cache)
                    if use_early_halting:
                        halt_p = torch.sigmoid(self.halt_head(self.ln_f(h))).item()
                        if halt_p >= self.config.halt_threshold and k >= 1:
                            break
                
                logits = self.lm_head(self.ln_f(h[:, -1, :]))
            else:
                # Full sequence forward pass (no cache fallback)
                out = self.forward(generated, recursion_budget=M)
                logits = out["logits"][:, -1, :]

            # Sampling
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def load_from_navitrit_checkpoint(self, checkpoint_path: str, source_layer: int = 0):
        """
        Initializes parameter-shared super-block from a pre-trained NaviTrit checkpoint.
        Extracts embeddings, lm_head, ln_f, and source_layer attention + FFN tiles.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        # 1. Load Embeddings & Readout Head
        if "embed_tokens.weight" in state:
            self.embed_tokens.weight.data.copy_(state["embed_tokens.weight"])
        if "embed_positions.weight" in state:
            self.embed_positions.weight.data.copy_(state["embed_positions.weight"])
        if "ln_f.weight" in state:
            self.ln_f.weight.data.copy_(state["ln_f.weight"])
        if "lm_head.weight" in state:
            self.lm_head.weight.data.copy_(state["lm_head.weight"])

        # 2. Extract representative Attention and FFN tile (source_layer)
        q_key = f"attn_tiles.{source_layer}.q_proj.weight"
        k_key = f"attn_tiles.{source_layer}.k_proj.weight"
        v_key = f"attn_tiles.{source_layer}.v_proj.weight"
        o_key = f"attn_tiles.{source_layer}.o_proj.weight"
        attn_norm_key = f"attn_norms.{source_layer}.weight"

        gate_key = f"ffn_tiles.{source_layer}.gate_proj.weight"
        up_key = f"ffn_tiles.{source_layer}.up_proj.weight"
        down_key = f"ffn_tiles.{source_layer}.down_proj.weight"
        ffn_norm_key = f"ffn_norms.{source_layer}.weight"

        if q_key in state:
            self.super_block.attn.q_proj.weight.data.copy_(state[q_key])
            self.super_block.attn.k_proj.weight.data.copy_(state[k_key])
            self.super_block.attn.v_proj.weight.data.copy_(state[v_key])
            self.super_block.attn.o_proj.weight.data.copy_(state[o_key])
            print(f"[LoopFormer] Successfully loaded Attention tile from {q_key}")

        if attn_norm_key in state:
            self.super_block.attn_norm.weight.data.copy_(state[attn_norm_key])

        if gate_key in state:
            self.super_block.ffn.gate_proj.weight.data.copy_(state[gate_key])
            self.super_block.ffn.up_proj.weight.data.copy_(state[up_key])
            self.super_block.ffn.down_proj.weight.data.copy_(state[down_key])
            print(f"[LoopFormer] Successfully loaded FFN tile from {gate_key}")

        if ffn_norm_key in state:
            self.super_block.ffn_norm.weight.data.copy_(state[ffn_norm_key])
