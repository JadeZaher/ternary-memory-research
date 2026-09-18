"""
experiments/frontier_scaling/navitrit_ifmor_model.py: Interleaved Functional Mixture-of-Recursions (IF-MoR).

Gate 19-B (Phase II Track G-Extended):
1. Widens the recurrent super-block to ~20M-25M parameters (~5-6 MB packed in 1.58-bit, 
   residing permanently in the 24MB on-chip L2 cache of the RTX 4060).
2. Interleaves 4 heterogeneous functional branches within each recursive loop:
   - Branch 1: Full Ternary Attention (O(S^2) AST parsing, identifier binding, precise grammar)
   - Branch 2: Kernelized Linear Attention (O(S) background context aggregation)
   - Branch 3: Ternary Mamba SSM with Auxiliary Numerical Supervision (recurrent state s_t in R^64
               acting as a working arithmetic scratchpad for carry/borrow operations)
   - Branch 4: Wide SwiGLU FFN (high-capacity channel mixing and vocabulary projection)
3. MoE Branch-Diversity Regularizer: L_balance = sum_{b=1}^4 (w_b_bar - 0.25)^2 to prevent
   branch collapse and foster a learned functional execution schedule across loops.
4. Dynamic test-time depth dial M in {1,..,6} with shortcut-consistency training.
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
from experiments.frontier_scaling.loopformer_model import RecursionKVCache


@dataclass
class IFMoRConfig:
    """Configuration for Interleaved Functional Mixture-of-Recursions (IF-MoR)."""
    vocab_size: int = 50257
    hidden_size: int = 768
    intermediate_size: int = 4096        # Wide FFN for vocabulary projection capacity
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    block_size: int = 256
    
    # Mamba State-Space Branch parameters
    d_state: int = 64                    # Recurrent scratchpad state dimension
    d_conv: int = 4                      # Local 1D causal convolution kernel size
    expand: int = 2                      # Mamba internal expansion factor
    
    # Recurrence & Depth dial
    max_loops: int = 6                  # Maximum allowable recursion depth
    default_loops: int = 4              # Operational default depth
    use_shortcut_consistency: bool = True
    consistency_lambda: float = 0.50
    consistency_mu: float = 0.10
    balance_lambda: float = 0.20        # Weight for MoE branch diversity loss
    arith_lambda: float = 1.00          # Weight for auxiliary arithmetic supervision
    
    # Quantization
    quantize_act: bool = False
    ternary: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class KernelizedLinearAttention(nn.Module):
    """
    O(S) Linear Attention using non-negative feature map phi(x) = ELU(x) + 1.
    Computes (Q @ (K^T @ V)) in linear time for global background context.
    """
    def __init__(self, config: IFMoRConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.q_proj = BitLinear(config.hidden_size, config.hidden_size, block_size=config.block_size, ternary=config.ternary)
        self.k_proj = BitLinear(config.hidden_size, config.hidden_size, block_size=config.block_size, ternary=config.ternary)
        self.v_proj = BitLinear(config.hidden_size, config.hidden_size, block_size=config.block_size, ternary=config.ternary)
        self.o_proj = BitLinear(config.hidden_size, config.hidden_size, block_size=config.block_size, ternary=config.ternary)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Non-negative kernel feature map
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Causal prefix cumsum of outer products: K^T @ V
        # k: [B, H, S, d_k], v: [B, H, S, d_v]
        kv = torch.einsum("bhsi,bhsj->bhsij", k, v)
        kv_cumsum = torch.cumsum(kv, dim=2)

        # Output projection: Q @ (cumsum(K^T V))
        out = torch.einsum("bhsi,bhsij->bhsj", q, kv_cumsum)

        # Normalization denominator: Q @ (cumsum(K))
        k_cumsum = torch.cumsum(k, dim=2)
        denom = torch.einsum("bhsi,bhsi->bhs", q, k_cumsum).unsqueeze(-1) + self.eps

        out = out / denom
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class TernaryMambaBranch(nn.Module):
    """
    Selective State-Space (Mamba) Branch with Auxiliary Arithmetic Supervision.
    Maintains a continuous recurrent state s_t in R^{d_state} acting as a numerical scratchpad.
    """
    def __init__(self, config: IFMoRConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.d_state = config.d_state
        self.d_inner = config.hidden_size * config.expand

        # Input projection into main channel and gate channel
        self.in_proj = BitLinear(config.hidden_size, 2 * self.d_inner, block_size=config.block_size, ternary=config.ternary)

        # Causal depthwise 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # Projections for Delta, B, C
        self.x_proj = nn.Linear(self.d_inner, 2 * self.d_state + self.hidden_size, bias=False)
        self.dt_proj = nn.Linear(self.hidden_size, self.d_inner, bias=True)

        # Continuous state matrix A: initialized to negative exponential scale
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = BitLinear(self.d_inner, config.hidden_size, block_size=config.block_size, ternary=config.ternary)

        # Auxiliary Arithmetic Supervision Head: decodes numeric values from s_t
        self.num_head = nn.Linear(self.d_state, 1, bias=True)
        nn.init.normal_(self.num_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.num_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes selective SSM forward pass.
        Returns:
          y: output hidden representation [B, S, D]
          final_s: final recurrent state [B, d_state]
          pred_num: predicted numeric value [B, S] for arithmetic loss
        """
        B, S, D = x.shape

        # 1. Project input to main and gate branches
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)

        # 2. Causal 1D Convolution
        x_conv = self.conv1d(x_main.transpose(1, 2))[:, :, :S].transpose(1, 2)
        x_act = F.silu(x_conv)

        # 3. Input-dependent parameters: Delta, B, C
        x_params = self.x_proj(x_act)
        B_mat = x_params[:, :, :self.d_state]
        C_mat = x_params[:, :, self.d_state : 2 * self.d_state]
        dt_raw = x_params[:, :, 2 * self.d_state :]
        dt = F.softplus(self.dt_proj(dt_raw)) # [B, S, d_inner]

        # 4. Discretization via ZOH
        A = -torch.exp(self.A_log.float()) # [d_inner, d_state]
        # dA = exp(dt * A) -> [B, S, d_inner, d_state]
        # For efficiency and numerical stability in PyTorch:
        dA = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, self.d_inner, self.d_state))
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2) # [B, S, 1, d_state]

        # 5. Recurrent State Scan
        # s_t = dA_t * s_{t-1} + dB_t * x_t
        # To record s_t across all tokens:
        states = []
        s_prev = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        ys = []
        for t in range(S):
            dA_t = dA[:, t] # [B, d_inner, d_state]
            dB_t = dB[:, t] # [B, d_inner, d_state]
            x_t = x_act[:, t].unsqueeze(-1) # [B, d_inner, 1]

            s_curr = dA_t * s_prev + dB_t * x_t
            # Output y_t = sum(s_curr * C_t, dim=-1) + D * x_t
            C_t = C_mat[:, t].unsqueeze(1) # [B, 1, d_state]
            y_t = torch.sum(s_curr * C_t, dim=-1) + self.D * x_act[:, t]
            ys.append(y_t)
            states.append(s_curr)
            s_prev = s_curr

        y_stack = torch.stack(ys, dim=1) # [B, S, d_inner]
        state_stack = torch.stack(states, dim=1) # [B, S, d_inner, d_state]

        # Gated modulation and projection
        y_gated = y_stack * F.silu(z)
        y_out = self.out_proj(y_gated)

        # 6. Extract mean state vector per token for arithmetic readout: [B, S, d_state]
        s_mean = torch.mean(state_stack, dim=2)
        pred_num = self.num_head(s_mean).squeeze(-1) # [B, S]
        final_s = s_mean[:, -1, :] # [B, d_state]

        return y_out, final_s, pred_num


class IFMoRSuperBlock(nn.Module):
    """
    Wide Interleaved Functional MoR Super-Block.
    Contains 4 heterogeneous functional branches:
      1. Full Ternary Attention (BitRouteAttention)
      2. Linear Attention (KernelizedLinearAttention)
      3. Ternary Mamba SSM (TernaryMambaBranch with arithmetic scratchpad)
      4. Wide SwiGLU FFN (BitRouteFFN with d_ff=4096)
    """
    def __init__(self, config: IFMoRConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Branch 1: Full Attention
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
        self.full_attn = BitRouteAttention(bitroute_cfg)

        # Branch 2: Linear Attention
        self.lin_attn = KernelizedLinearAttention(config)

        # Branch 3: Ternary Mamba SSM
        self.mamba = TernaryMambaBranch(config)

        # Branch 4: Wide SwiGLU FFN
        self.ffn = BitRouteFFN(bitroute_cfg)

        # Per-Loop Normalization Layers (one per loop iteration)
        self.loop_norms = nn.ModuleList([
            BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps)
            for _ in range(config.max_loops)
        ])

        # Learned Loop-Index Embeddings
        self.loop_embeddings = nn.Embedding(config.max_loops, config.hidden_size)
        nn.init.normal_(self.loop_embeddings.weight, mean=0.0, std=0.02)

        # Branch Router: [h || loop_embed] -> 4 branch weights
        self.branch_router = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
        )

        # Exit Router: [h] -> exit probability
        self.exit_router = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.constant_(self.exit_router[-1].bias, -2.0)

    def forward_step(
        self,
        h: torch.Tensor,
        loop_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[RecursionKVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes one recursive step through the 4-branch super-block.
        Returns:
          h_next: updated representation [B, S, D]
          branch_weights: router weights [B, S, 4]
          exit_prob: exit probability [B, S, 1]
          pred_num: predicted arithmetic value from Mamba state [B, S]
        """
        B, S, D = h.shape
        loop_id = torch.tensor([loop_idx], device=h.device)
        e_k = self.loop_embeddings(loop_id).view(1, 1, D).expand(B, S, D)

        # Normalize with loop-specific LayerNorm
        h_cond = h + e_k
        h_norm = self.loop_norms[loop_idx](h_cond)

        # Compute branch routing logits
        router_in = torch.cat([h_norm, e_k], dim=-1)
        branch_weights = F.softmax(self.branch_router(router_in), dim=-1) # [B, S, 4]

        # Execute 4 branches (deltas relative to input)
        delta_full = self.full_attn(h_norm, attention_mask=attention_mask)
        delta_lin = self.lin_attn(h_norm)
        delta_mamba, final_s, pred_num = self.mamba(h_norm)
        delta_ffn = self.ffn(h_norm)

        # Gated Combination across active branches
        # branch 0: Full Attention, branch 1: Linear Attention, branch 2: Mamba, branch 3: FFN
        w_attn = branch_weights[..., 0:1]
        w_lin = branch_weights[..., 1:2]
        w_ssm = branch_weights[..., 2:3]
        w_ffn = branch_weights[..., 3:4]

        combined_delta = (
            w_attn * delta_full
            + w_lin * delta_lin
            + w_ssm * delta_mamba
            + w_ffn * delta_ffn
        )

        h_next = h + combined_delta

        # Dynamic exit probability
        exit_prob = torch.sigmoid(self.exit_router(h_next))

        return h_next, branch_weights, exit_prob, pred_num


class NaviTritIFMoRForCausalLM(nn.Module):
    """
    NaviTrit-IFMoR: Interleaved Functional Mixture-of-Recursions Causal Language Model.
    Replaces the 7M single block with a wide ~20M-25M 4-branch super-block fitting in 24MB L2 cache.
    """
    def __init__(self, config: IFMoRConfig):
        super().__init__()
        self.config = config

        # Shared Token & Position Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Wide 4-Branch Super-Block
        self.super_block = IFMoRSuperBlock(config)

        # Final Readout Norm & Head
        self.ln_f = BitRouteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Shortcut Consistency Projectors P_short^{(m)}
        self.shortcut_projs = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(config.max_loops)
        ])
        for proj in self.shortcut_projs:
            nn.init.zeros_(proj.weight)

    def get_super_block_param_count(self) -> int:
        return sum(p.numel() for p in self.super_block.parameters())

    def get_packed_footprint_mb(self) -> float:
        """Footprint packed at 1.58-bit (2 bits per weight)."""
        return (self.get_super_block_param_count() * 2.0) / (8.0 * 1024 * 1024)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        recursion_budget: Optional[int] = None,
        return_all_recursions: bool = False,
    ) -> Dict[str, Any]:
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        h = self.embed_tokens(input_ids) + self.embed_positions(positions)

        M = recursion_budget if recursion_budget is not None else self.config.default_loops
        M = min(max(1, M), self.config.max_loops)

        intermediate_states = []
        intermediate_logits = []
        branch_weights_all = []
        pred_nums_all = []
        exit_probs_all = []

        for loop_idx in range(M):
            h, branch_w, exit_p, pred_num = self.super_block.forward_step(
                h, loop_idx=loop_idx, attention_mask=attention_mask
            )
            intermediate_states.append(h)
            branch_weights_all.append(branch_w)
            pred_nums_all.append(pred_num)
            exit_probs_all.append(exit_p)

            if return_all_recursions or self.config.use_shortcut_consistency:
                shortcut_h = h + self.shortcut_projs[loop_idx](h)
                logits_k = self.lm_head(self.ln_f(shortcut_h))
                intermediate_logits.append(logits_k)

        final_h = self.ln_f(h)
        logits = self.lm_head(final_h)

        return {
            "logits": logits,
            "intermediate_states": intermediate_states,
            "intermediate_logits": intermediate_logits,
            "branch_weights": branch_weights_all,
            "pred_nums": pred_nums_all,
            "exit_probs": exit_probs_all,
            "recursion_budget": M,
        }

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        numeric_targets: Optional[torch.Tensor] = None,
        recursion_budget: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes joint training loss:
        L = L_LM + lambda_sc * L_shortcut + lambda_bal * L_balance + lambda_arith * L_arith
        """
        out = self.forward(input_ids, recursion_budget=recursion_budget, return_all_recursions=True)
        M = out["recursion_budget"]
        final_logits = out["logits"]

        # 1. Primary LM Cross-Entropy Loss
        loss_lm = F.cross_entropy(final_logits.view(-1, self.config.vocab_size), targets.view(-1))

        # 2. Shortcut Consistency Loss
        loss_intermediate = torch.tensor(0.0, device=input_ids.device)
        loss_alignment = torch.tensor(0.0, device=input_ids.device)

        if M > 1 and len(out["intermediate_logits"]) >= M:
            final_rep = out["intermediate_states"][-1].detach()
            for m in range(M - 1):
                inter_logits = out["intermediate_logits"][m]
                inter_lm = F.cross_entropy(inter_logits.view(-1, self.config.vocab_size), targets.view(-1))
                inter_h = out["intermediate_states"][m] + self.shortcut_projs[m](out["intermediate_states"][m])
                align = F.mse_loss(inter_h, final_rep)
                decay = (m + 1) / float(M)
                loss_intermediate = loss_intermediate + decay * inter_lm
                loss_alignment = loss_alignment + decay * align

            loss_intermediate = loss_intermediate / float(M - 1)
            loss_alignment = loss_alignment / float(M - 1)

        loss_shortcut = (
            self.config.consistency_lambda * loss_intermediate
            + self.config.consistency_mu * loss_alignment
        )

        # 3. MoE Branch-Diversity Loss (forces balanced utilization of all 4 branches)
        # Average branch weight across all loops, tokens, and batch
        all_branch_w = torch.cat(out["branch_weights"], dim=1) # [B, M*S, 4]
        mean_branch_w = torch.mean(all_branch_w, dim=(0, 1))   # [4]
        loss_balance = torch.sum((mean_branch_w - 0.25) ** 2)

        # 4. Auxiliary Arithmetic Supervision Loss
        loss_arith = torch.tensor(0.0, device=input_ids.device)
        if numeric_targets is not None:
            # Predict from Mamba state at final loop step
            final_pred_num = out["pred_nums"][-1][:, -1] # [B]
            valid_mask = ~torch.isnan(numeric_targets)
            if valid_mask.any():
                loss_arith = F.mse_loss(final_pred_num[valid_mask], numeric_targets[valid_mask].float())

        total_loss = (
            loss_lm
            + loss_shortcut
            + self.config.balance_lambda * loss_balance
            + self.config.arith_lambda * loss_arith
        )

        return {
            "total_loss": total_loss,
            "loss_lm": loss_lm,
            "loss_shortcut": loss_shortcut,
            "loss_balance": loss_balance,
            "loss_arith": loss_arith,
            "mean_branch_usage": mean_branch_w.detach(),
            "recursion_budget": torch.tensor(float(M)),
        }

    def load_from_pretrained_checkpoint(self, checkpoint_path: str, source_layer: int = 0):
        """Warm-starts full attention and FFN tiles from pre-trained checkpoints."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        # Load Embeddings & Head
        for name in ["embed_tokens.weight", "embed_positions.weight", "ln_f.weight", "lm_head.weight"]:
            if name in state and hasattr(self, name.split(".")[0]):
                getattr(self, name.split(".")[0]).weight.data.copy_(state[name])

        # Load Full Attention from source layer
        q_key = f"attn_tiles.{source_layer}.q_proj.weight"
        k_key = f"attn_tiles.{source_layer}.k_proj.weight"
        v_key = f"attn_tiles.{source_layer}.v_proj.weight"
        o_key = f"attn_tiles.{source_layer}.o_proj.weight"
        if q_key in state:
            self.super_block.full_attn.q_proj.weight.data.copy_(state[q_key])
            self.super_block.full_attn.k_proj.weight.data.copy_(state[k_key])
            self.super_block.full_attn.v_proj.weight.data.copy_(state[v_key])
            self.super_block.full_attn.o_proj.weight.data.copy_(state[o_key])
            print(f"[IF-MoR] Warm-started Full Attention from {q_key}")

        # Also initialize Linear Attention projections with pre-trained attention weights
        if q_key in state:
            self.super_block.lin_attn.q_proj.weight.data.copy_(state[q_key])
            self.super_block.lin_attn.k_proj.weight.data.copy_(state[k_key])
            self.super_block.lin_attn.v_proj.weight.data.copy_(state[v_key])
            self.super_block.lin_attn.o_proj.weight.data.copy_(state[o_key])
            print(f"[IF-MoR] Warm-started Linear Attention from {q_key}")

        # Load FFN tiles
        gate_key = f"ffn_tiles.{source_layer}.gate_proj.weight"
        up_key = f"ffn_tiles.{source_layer}.up_proj.weight"
        down_key = f"ffn_tiles.{source_layer}.down_proj.weight"
        if gate_key in state and state[gate_key].shape == self.super_block.ffn.gate_proj.weight.shape:
            self.super_block.ffn.gate_proj.weight.data.copy_(state[gate_key])
            self.super_block.ffn.up_proj.weight.data.copy_(state[up_key])
            self.super_block.ffn.down_proj.weight.data.copy_(state[down_key])
            print(f"[IF-MoR] Warm-started SwiGLU FFN from {gate_key}")
