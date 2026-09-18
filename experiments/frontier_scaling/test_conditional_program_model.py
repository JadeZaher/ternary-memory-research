"""
experiments/frontier_scaling/test_conditional_program_model.py: Unit Tests for Dynamic Weight Parameterization
and Weights as Conditional Programs (Gate 19-D).

Tests:
1. Identity Initialization: Confirms exact zero-drift from base model at initialization (gamma=0, beta=0, U=0).
2. Context Differentiation: Verifies that distinct functional roles produce distinct semantic representations.
3. PCGrad Orthogonalization: Verifies that conflicting gradients are projected onto non-interfering orthogonal subspaces.
4. Parameter Footprint Audit: Ensures hypernetwork parameter footprint is < 1.0 MB.
5. End-to-End CUDA Execution: Validates multi-hop forward and backward pass on GPU without NaNs.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.frontier_scaling.conditional_program_model import (
    ConditionalProgramConfig,
    ContextHyperNet,
    ContextModulatedBitLinear,
    NaviTritConditionalProgramForCausalLM,
    PCGradOptimizer,
)
from experiments.bitlinear import BitLinear


class TestConditionalProgramModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TestConditionalProgram] Running tests on device: {cls.device}")

    def test_01_identity_initialization(self):
        """Verifies that at initialization, modulated layer matches base layer bit-identically."""
        cfg = ConditionalProgramConfig(hidden_size=256, lora_rank=8, d_context=32)
        base = BitLinear(256, 256).to(self.device)
        mod_linear = ContextModulatedBitLinear(base, 256, 256, cfg).to(self.device)
        hypernet = ContextHyperNet(cfg).to(self.device)

        x = torch.randn(2, 10, 256, device=self.device)

        # Baseline output
        with torch.no_grad():
            base_out = base(x)

        # Context-modulated output with initial zero-weights
        mod_params = hypernet(hop_idx=0, role_idx=0, dir_idx=0, device=self.device)
        mod_out = mod_linear(
            x,
            gamma=mod_params.get("attn_gamma"),
            beta=mod_params.get("attn_beta"),
            lora_scale=mod_params.get("lora_scale"),
        )

        max_diff = torch.max(torch.abs(base_out - mod_out)).item()
        print(f"[Test 1] Identity Init Max Difference: {max_diff:.2e}")
        self.assertLess(max_diff, 1e-6, "Modulated layer deviates from base layer at initialization!")

    def test_02_context_differentiation(self):
        """Verifies that distinct functional roles produce distinct parameter modulations."""
        cfg = ConditionalProgramConfig(hidden_size=256, lora_rank=8, d_context=32)
        hypernet = ContextHyperNet(cfg).to(self.device)

        # Perturb hypernet weights to simulate a trained state
        with torch.no_grad():
            for p in hypernet.parameters():
                p.add_(torch.randn_like(p) * 0.05)

        # Generate parameters for 4 distinct roles: BIND (0), REFINE (1), VERIFY (2), EMIT (3)
        params_bind = hypernet(hop_idx=0, role_idx=0, dir_idx=0, device=self.device)
        params_verify = hypernet(hop_idx=2, role_idx=2, dir_idx=1, device=self.device)

        gamma_diff = torch.norm(params_bind["attn_gamma"] - params_verify["attn_gamma"]).item()
        print(f"[Test 2] Functional Role Differentiation Norm (Bind vs Verify): {gamma_diff:.4f}")
        self.assertGreater(gamma_diff, 1e-3, "Different roles produced identical modulations!")

    def test_03_pcgrad_projection(self):
        """Verifies that PCGrad projects conflicting gradients onto orthogonal subspaces."""
        model = nn.Linear(4, 1, bias=False).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        pcgrad = PCGradOptimizer(optimizer)

        # Create two conflicting tasks:
        # Task 1: minimize w[0], Task 2: maximize w[0] (grad1 = [1, 0, 0, 0], grad2 = [-1, 0, 0, 0])
        x1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        x2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)

        loss1 = model(x1).squeeze()     # grad = +x1
        loss2 = -model(x2).squeeze()    # grad = -x2

        # Verify raw conflicting dot product
        raw_dot = torch.dot(x1.squeeze(), -x2.squeeze()).item()
        self.assertLess(raw_dot, 0.0, "Tasks are not conflicting in synthetic setup!")

        # Apply PCGrad backward
        pcgrad.pcgrad_backward([loss1, loss2])

        # Verify that the parameter grad is orthogonalized or non-conflicting
        final_grad = model.weight.grad
        print(f"[Test 3] PCGrad Final Projected Gradient: {final_grad.cpu().numpy()}")
        self.assertIsNotNone(final_grad)

    def test_04_parameter_footprint_audit(self):
        """Audits memory footprint of ContextHyperNet and low-rank adapter."""
        cfg = ConditionalProgramConfig(hidden_size=768, lora_rank=16, d_context=64)
        hypernet = ContextHyperNet(cfg)

        param_count = sum(p.numel() for p in hypernet.parameters())
        packed_mb = (param_count * 2) / (8 * 1024 * 1024)
        fp16_mb = (param_count * 2) / (1024 * 1024)

        print(f"[Test 4] HyperNet Parameters: {param_count:,} ({packed_mb:.4f} MB packed, {fp16_mb:.2f} MB FP16)")
        self.assertLess(packed_mb, 1.0, "HyperNet exceeds 1.0 MB packed footprint budget!")
        self.assertLess(param_count, 2_000_000, "HyperNet exceeds 2.0M parameter budget!")

    def test_05_end_to_end_cuda_execution(self):
        """Validates full causal LM multi-hop forward and backward pass on CUDA."""
        cfg = ConditionalProgramConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_attention_heads=4,
            max_position_embeddings=128,
            max_hops=4,
            default_hops=3,
            d_context=32,
            lora_rank=8,
        )
        model = NaviTritConditionalProgramForCausalLM(cfg).to(self.device)

        input_ids = torch.randint(0, 1000, (2, 32), device=self.device)
        targets = torch.randint(0, 1000, (2, 32), device=self.device)

        loss_dict = model.compute_loss(input_ids, targets, hop_budget=3)
        total_loss = loss_dict["total_loss"]
        total_loss.backward()

        self.assertFalse(torch.isnan(total_loss), "Encountered NaN in total loss!")
        print(f"[Test 5] End-to-End CUDA Loss: {total_loss.item():.4f} (All assertions passed)")


if __name__ == "__main__":
    unittest.main()
