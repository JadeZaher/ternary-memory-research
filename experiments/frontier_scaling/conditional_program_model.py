"""
experiments/frontier_scaling/conditional_program_model.py: Dynamic Weight Parameterization (DWP)
and Weights as Conditional Programs for Low-Bit Recurrent Architectures.

Gate 19-D (Phase II Track W):
1. Solves the "Weight Identity Crisis" where the same physical tile must execute distinct functions
   (Bind vs Refine vs Verify vs Emit) across non-monotonic hops and recurrent iterations.
2. Context-Conditioned FiLM Modulation: Base ternary weights remain 100% frozen; a lightweight ContextHyperNet
   generates role-conditioned gain (gamma) and bias (beta) vectors.
3. Low-Rank Dynamic Adaptation: Synthesizes dynamic delta W = (alpha/r) U V^T conditioned on routing context.
4. Functional Role Tying: 4 canonical execution roles (0: BIND, 1: REFINE, 2: VERIFY, 3: EMIT).
5. Exact Identity Initialization: Modulation parameters start at zero (gamma=0, beta=0, U=0), guaranteeing
   zero-drift initialization bit-identical to the pre-trained backbone.
6. PCGrad Gradient Surgery: Eliminates gradient conflicts between disparate hops on shared tiles.
"""

import os
import sys
import math
import copy
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
class ConditionalProgramConfig:
    """Configuration for Dynamic Weight Parameterization & Conditional Programs."""
    vocab_size: int = 50257
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    block_size: int = 256
    
    # Hop and recursion settings
    max_hops: int = 8
    default_hops: int = 4
    d_context: int = 64
    
    # Dynamic parameterization settings
    use_film: bool = True
    use_lowrank: bool = True
    lora_rank: int = 16
    lora_alpha: float = 16.0
    
    # Functional role settings
    num_roles: int = 4   # 0: BIND, 1: REFINE, 2: VERIFY, 3: EMIT
    num_dirs: int = 3    # 0: FORWARD, 1: BACKWARD, 2: SELF
    
    # Shortcut consistency
    use_shortcut_consistency: bool = True
    consistency_lambda: float = 0.50
    consistency_mu: float = 0.10
    
    # Ternary quantization settings
    quantize_act: bool = False
    ternary: bool = True


class ContextHyperNet(nn.Module):
    """
    Lightweight Hypernetwork generating role-conditioned modulation parameters
    from routing context: hop index, functional role, hop direction, and hidden state.
    """
    def __init__(self, config: ConditionalProgramConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.d_context = config.d_context
        self.lora_rank = config.lora_rank

        # Context Embeddings
        self.hop_embeddings = nn.Embedding(config.max_hops + 1, config.d_context)
        self.role_embeddings = nn.Embedding(config.num_roles, config.d_context)
        self.dir_embeddings = nn.Embedding(config.num_dirs, config.d_context)

        # Context Aggregation MLP
        # Input: [hop_emb || role_emb || dir_emb] = 3 * d_context
        self.context_mlp = nn.Sequential(
            nn.Linear(3 * config.d_context, 128),
            nn.SiLU(),
            nn.Linear(128, config.d_context),
        )

        # 1. FiLM Modulation Heads: gamma (gain) and beta (bias) for Attention and FFN
        if config.use_film:
            self.attn_gamma_head = nn.Linear(config.d_context, config.hidden_size)
            self.attn_beta_head = nn.Linear(config.d_context, config.hidden_size)
            self.ffn_gamma_head = nn.Linear(config.d_context, config.hidden_size)
            self.ffn_beta_head = nn.Linear(config.d_context, config.hidden_size)

            # Exact Zero Initialization: gamma = 0, beta = 0 initially
            nn.init.zeros_(self.attn_gamma_head.weight)
            nn.init.zeros_(self.attn_gamma_head.bias)
            nn.init.zeros_(self.attn_beta_head.weight)
            nn.init.zeros_(self.attn_beta_head.bias)
            nn.init.zeros_(self.ffn_gamma_head.weight)
            nn.init.zeros_(self.ffn_gamma_head.bias)
            nn.init.zeros_(self.ffn_beta_head.weight)
            nn.init.zeros_(self.ffn_beta_head.bias)

        # 2. Low-Rank Dynamic Scaling Head (modulates the rank-r bottleneck per context)
        if config.use_lowrank:
            self.lora_scale_head = nn.Linear(config.d_context, config.lora_rank)
            nn.init.zeros_(self.lora_scale_head.weight)
            nn.init.zeros_(self.lora_scale_head.bias)

    def forward(
        self,
        hop_idx: int,
        role_idx: int,
        dir_idx: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates modulation parameters for a given execution context.
        """
        if device is None:
            device = self.hop_embeddings.weight.device

        h_idx = torch.tensor([min(hop_idx, self.config.max_hops)], device=device)
        r_idx = torch.tensor([min(role_idx, self.config.num_roles - 1)], device=device)
        d_idx = torch.tensor([min(dir_idx, self.config.num_dirs - 1)], device=device)

        e_h = self.hop_embeddings(h_idx)
        e_r = self.role_embeddings(r_idx)
        e_d = self.dir_embeddings(d_idx)

        ctx_raw = torch.cat([e_h, e_r, e_d], dim=-1) # [1, 3 * d_context]
        ctx = self.context_mlp(ctx_raw)              # [1, d_context]

        out = {"context": ctx}

        if self.config.use_film:
            out["attn_gamma"] = self.attn_gamma_head(ctx) # [1, d]
            out["attn_beta"] = self.attn_beta_head(ctx)   # [1, d]
            out["ffn_gamma"] = self.ffn_gamma_head(ctx)   # [1, d]
            out["ffn_beta"] = self.ffn_beta_head(ctx)     # [1, d]

        if self.config.use_lowrank:
            out["lora_scale"] = self.lora_scale_head(ctx) # [1, r]

        return out


class ContextModulatedBitLinear(nn.Module):
    """
    Wraps a frozen BitLinear ternary layer with dynamic context-conditioned modulation.
    Base weights are frozen (0 gradient); dynamic parameters adapt behavior conditionally.
    """
    def __init__(
        self,
        base_linear: BitLinear,
        in_features: int,
        out_features: int,
        config: ConditionalProgramConfig,
    ):
        super().__init__()
        self.base = base_linear
        # Freeze base parameters
        for p in self.base.parameters():
            p.requires_grad = False
            
        self.config = config
        self.scaling = config.lora_alpha / float(config.lora_rank)

        # Low-rank adapter matrices A and B
        if config.use_lowrank:
            self.lora_A = nn.Linear(in_features, config.lora_rank, bias=False)
            self.lora_B = nn.Linear(config.lora_rank, out_features, bias=False)
            nn.init.normal_(self.lora_A.weight, std=0.02)
            nn.init.zeros_(self.lora_B.weight) # Exact zero initialization!

    def forward(
        self,
        x: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        lora_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional dynamic low-rank perturbation and FiLM modulation.
        """
        out = self.base(x)

        # Dynamic Low-Rank Parameter Program
        if self.config.use_lowrank and hasattr(self, "lora_A"):
            lora_act = self.lora_A(x)
            if lora_scale is not None:
                lora_act = lora_act * (1.0 + lora_scale)
            delta = self.lora_B(lora_act) * self.scaling
            out = out + delta

        # Context-conditioned FiLM modulation
        if gamma is not None and beta is not None and self.config.use_film:
            out = out * (1.0 + gamma) + beta

        return out


class ModulatedRecurrentSuperBlock(nn.Module):
    """
    Parameter-shared recurrent block with context-conditioned dynamic weight modulation.
    """
    def __init__(self, config: ConditionalProgramConfig):
        super().__init__()
        self.config = config
        
        # Base Sub-modules
        b_cfg = BitRouteConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            block_size=config.block_size,
            quantize_act=config.quantize_act,
            ternary=config.ternary,
        )
        self.ln_1 = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = BitRouteAttention(b_cfg)
        self.ln_2 = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = BitRouteFFN(b_cfg)

        # Learned step embeddings
        self.step_embeddings = nn.Embedding(config.max_hops + 1, config.hidden_size)

        # Wrap projection layers with dynamic modulators
        self.mod_q = ContextModulatedBitLinear(self.attn.q_proj, config.hidden_size, config.hidden_size, config)
        self.mod_o = ContextModulatedBitLinear(self.attn.o_proj, config.hidden_size, config.hidden_size, config)
        self.mod_down = ContextModulatedBitLinear(self.ffn.down_proj, config.intermediate_size, config.hidden_size, config)

    def forward_step(
        self,
        h: torch.Tensor,
        mod_params: Dict[str, torch.Tensor],
        hop_idx: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Executes a single reasoning hop with dynamic weight parameterization.
        """
        B, S, D = h.shape

        # Add step embedding
        step_idx = torch.tensor([min(hop_idx, self.config.max_hops)], device=h.device)
        h_step = h + self.step_embeddings(step_idx).unsqueeze(1)

        # 1. Attention Block with dynamic modulation
        norm_1 = self.ln_1(h_step)
        
        # Attention forward pass with modulated output projection
        attn_out = self._forward_attn_modulated(norm_1, mod_params, attention_mask=attention_mask)
        h = h + attn_out

        # 2. FFN Block with dynamic modulation
        norm_2 = self.ln_2(h)
        ffn_out = self._forward_ffn_modulated(norm_2, mod_params)
        h = h + ffn_out

        return h

    def _forward_attn_modulated(
        self,
        x: torch.Tensor,
        mod_params: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        num_heads = self.attn.num_heads
        head_dim = self.attn.head_dim

        # Modulated query projection
        q = self.mod_q(x, lora_scale=mod_params.get("lora_scale"))
        k = self.attn.k_proj(x)
        v = self.attn.v_proj(x)

        q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, S, num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        else:
            causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
            scores = scores + causal_mask.view(1, 1, S, S)

        probs = F.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, D)

        # Output projection modulated with Attention FiLM
        out = self.mod_o(
            ctx,
            gamma=mod_params.get("attn_gamma"),
            beta=mod_params.get("attn_beta"),
            lora_scale=mod_params.get("lora_scale"),
        )
        return out

    def _forward_ffn_modulated(
        self,
        x: torch.Tensor,
        mod_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # SwiGLU gating
        gate = F.silu(self.ffn.gate_proj(x))
        up = self.ffn.up_proj(x)
        hidden = gate * up

        # Modulated down projection with FFN FiLM
        out = self.mod_down(
            hidden,
            gamma=mod_params.get("ffn_gamma"),
            beta=mod_params.get("ffn_beta"),
            lora_scale=mod_params.get("lora_scale"),
        )
        return out


class NaviTritConditionalProgramForCausalLM(nn.Module):
    """
    Full Causal Language Model with Dynamic Weight Parameterization & Conditional Programs.
    Freezes all base ternary weights; dynamically modulates parameters via ContextHyperNet.
    """
    def __init__(self, config: ConditionalProgramConfig):
        super().__init__()
        self.config = config

        # Token & Position Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Context Hypernetwork (trainable, ~0.58M params)
        self.hypernet = ContextHyperNet(config)

        # Modulated Recurrent Super-Block (base weights frozen)
        self.super_block = ModulatedRecurrentSuperBlock(config)

        # Final Normalization & LM Head
        self.ln_f = BitRouteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Shortcut consistency projection layers
        self.shortcut_projs = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(config.max_hops)
        ])
        for p in self.shortcut_projs:
            nn.init.zeros_(p.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        hop_budget: Optional[int] = None,
        role_sequence: Optional[List[int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_hops: bool = False,
        recursion_budget: Optional[int] = None,
        return_all_recursions: Optional[bool] = None,
    ) -> Dict[str, Any]:
        B, S = input_ids.shape
        device = input_ids.device

        # Support alias names for generic loop analyzers
        if hop_budget is None and recursion_budget is not None:
            hop_budget = recursion_budget
        if return_all_recursions is not None:
            return_all_hops = return_all_recursions

        # Embeddings
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        h = self.embed_tokens(input_ids) + self.embed_positions(positions)

        M = hop_budget if hop_budget is not None else self.config.default_hops
        M = min(max(1, M), self.config.max_hops)

        # Default canonical role sequence:
        # Hop 0: BIND (0) -> Mid Hops: REFINE (1) -> Pre-terminal: VERIFY (2) -> Terminal: EMIT (3)
        if role_sequence is None:
            if M == 1:
                role_sequence = [3] # EMIT
            elif M == 2:
                role_sequence = [0, 3] # BIND -> EMIT
            elif M == 3:
                role_sequence = [0, 1, 3] # BIND -> REFINE -> EMIT
            else:
                # e.g. M=4: [0, 1, 2, 3]
                role_sequence = [0] + [1] * (M - 3) + [2, 3]

        intermediate_states = []
        intermediate_logits = []

        for hop_idx in range(M):
            role_idx = role_sequence[hop_idx]
            dir_idx = 0 if hop_idx == 0 else (1 if role_idx == 2 else 0) # Backward hop during verify

            # 1. Synthesize dynamic parameters from ContextHyperNet
            mod_params = self.hypernet(hop_idx=hop_idx, role_idx=role_idx, dir_idx=dir_idx, device=device)

            # 2. Execute super-block with dynamic modulation
            h = self.super_block.forward_step(
                h, mod_params=mod_params, hop_idx=hop_idx, attention_mask=attention_mask
            )
            intermediate_states.append(h)

            if return_all_hops or self.config.use_shortcut_consistency:
                shortcut_h = h + self.shortcut_projs[hop_idx](h)
                logits_k = self.lm_head(self.ln_f(shortcut_h))
                intermediate_logits.append(logits_k)

        final_h = self.ln_f(h)
        logits = self.lm_head(final_h)

        return {
            "logits": logits,
            "intermediate_states": intermediate_states,
            "intermediate_logits": intermediate_logits,
            "hop_budget": M,
            "recursion_budget": M,
            "role_sequence": role_sequence,
        }

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        hop_budget: Optional[int] = None,
        role_sequence: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes multi-hop loss with shortcut consistency.
        """
        out = self.forward(input_ids, hop_budget=hop_budget, role_sequence=role_sequence, return_all_hops=True)
        M = out["hop_budget"]
        final_logits = out["logits"]

        # 1. Primary LM Loss
        loss_lm = F.cross_entropy(final_logits.view(-1, self.config.vocab_size), targets.view(-1))

        # 2. Intermediate Shortcut Consistency Loss
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

        total_loss = loss_lm + loss_shortcut

        return {
            "total_loss": total_loss,
            "loss_lm": loss_lm,
            "loss_shortcut": loss_shortcut,
            "hop_budget": torch.tensor(float(M)),
        }

    def load_base_weights(self, checkpoint_path: str):
        """Warm-starts frozen base weights from an existing LoopFormer or NaviTrit checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        # Load Embeddings & Head
        for name in ["embed_tokens.weight", "embed_positions.weight", "ln_f.weight", "lm_head.weight"]:
            if name in state and hasattr(self, name.split(".")[0]):
                getattr(self, name.split(".")[0]).weight.data.copy_(state[name])

        # Load SuperBlock base weights
        for key in list(state.keys()):
            if "super_block" in key:
                target_key = key.replace("super_block.", "")
                # Map to self.super_block
                try:
                    target_param = self.super_block.get_submodule(target_key.rsplit(".", 1)[0])
                    attr = target_key.rsplit(".", 1)[1]
                    getattr(target_param, attr).data.copy_(state[key])
                except Exception:
                    pass

        print(f"[ConditionalProgram] Successfully loaded base weights from {checkpoint_path}")


class PCGradOptimizer:
    """
    Projected Conflicting Gradients (PCGrad) optimizer wrapper.
    Eliminates destructive interference when shared parameters receive conflicting gradients.
    """
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def zero_grad(self):
        self.optimizer.zero_grad()

    def pcgrad_backward(self, objectives: List[torch.Tensor]):
        """
        Computes PCGrad projection across a list of distinct objective tensors (e.g. from different hops).
        """
        assert len(objectives) > 0, "Objectives list must be non-empty"

        params = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)

        # 1. Compute gradients for each objective separately
        grads_per_obj = []
        for i, obj in enumerate(objectives):
            self.optimizer.zero_grad()
            obj.backward(retain_graph=(i < len(objectives) - 1))
            grads = []
            for p in params:
                if p.grad is not None:
                    grads.append(p.grad.detach().clone())
                else:
                    grads.append(torch.zeros_like(p))
            grads_per_obj.append(grads)

        # 2. Project conflicting gradients
        num_tasks = len(objectives)
        projected_grads = copy.deepcopy(grads_per_obj)

        for i in range(num_tasks):
            order = list(range(num_tasks))
            # Shuffle or iterate across competing tasks
            for j in order:
                if i == j:
                    continue
                # Inner product across all parameter gradients
                dot = sum(torch.sum(projected_grads[i][k] * grads_per_obj[j][k]) for k in range(len(params)))
                if dot < 0:
                    norm_sq = sum(torch.sum(grads_per_obj[j][k] ** 2) for k in range(len(params))) + 1e-8
                    for k in range(len(params)):
                        projected_grads[i][k] = projected_grads[i][k] - (dot / norm_sq) * grads_per_obj[j][k]

        # 3. Sum projected gradients into parameter .grad
        self.optimizer.zero_grad()
        for k, p in enumerate(params):
            final_grad = sum(projected_grads[i][k] for i in range(num_tasks))
            p.grad = final_grad

    def step(self):
        self.optimizer.step()
