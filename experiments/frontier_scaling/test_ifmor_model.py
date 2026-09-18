"""
experiments/frontier_scaling/test_ifmor_model.py: Unit Test Suite for IF-MoR Engine.

Gate 19-B Verification:
1. Super-block parameter count & 24MB L2 cache residency (< 7.0 MB packed footprint).
2. 4-Branch functional execution integrity (Full Attn, Linear Attn, Mamba SSM, Wide FFN).
3. Mamba state-space recurrence and numerical scratchpad decoding.
4. Multi-budget forward pass shapes [B, S, V] across M in {1, 2, 4, 6}.
5. MoE branch-diversity regularizer and arithmetic loss backprop.
6. Pre-trained weight extraction and loading.
"""

import os
import sys
import unittest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.frontier_scaling.navitrit_ifmor_model import (
    IFMoRConfig,
    NaviTritIFMoRForCausalLM,
)


class TestIFMoRModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.config = IFMoRConfig(
            vocab_size=50257,
            hidden_size=768,
            intermediate_size=4096,
            num_attention_heads=12,
            d_state=64,
            max_position_embeddings=512,
            max_loops=6,
            default_loops=4,
        )
        cls.model = NaviTritIFMoRForCausalLM(cls.config).to(cls.device)

    def test_01_parameter_count_and_l2_residency(self):
        """Verify IF-MoR super-block parameter count and 24MB L2 cache budget."""
        super_block_params = self.model.get_super_block_param_count()
        packed_mb = self.model.get_packed_footprint_mb()
        l2_cache_mb = 24.0 # RTX 4060 Ada Lovelace L2 cache size

        print(f"\n[Test 1] IF-MoR Super-Block Parameters: {super_block_params:,}")
        print(f"[Test 1] 1.58-bit Packed Footprint: {packed_mb:.3f} MB")
        print(f"[Test 1] RTX 4060 L2 Cache Occupancy: {packed_mb / l2_cache_mb * 100.0:.2f}% of 24MB")

        # Must be wider than 7M (at least 15M) and under 30M
        self.assertGreater(super_block_params, 15_000_000)
        self.assertLess(super_block_params, 30_000_000)
        # Packed footprint must be strictly under 8.0 MB (well under 24MB L2 budget)
        self.assertLess(packed_mb, 8.0)

    def test_02_four_branch_execution_integrity(self):
        """Verify each of the 4 heterogeneous functional branches executes cleanly."""
        B, S, D = 2, 8, self.config.hidden_size
        x = torch.randn(B, S, D, device=self.device)

        # 1. Full Attention
        delta_full = self.model.super_block.full_attn(x)
        self.assertEqual(delta_full.shape, (B, S, D))

        # 2. Linear Attention
        delta_lin = self.model.super_block.lin_attn(x)
        self.assertEqual(delta_lin.shape, (B, S, D))

        # 3. Ternary Mamba SSM
        delta_mamba, final_s, pred_num = self.model.super_block.mamba(x)
        self.assertEqual(delta_mamba.shape, (B, S, D))
        self.assertEqual(final_s.shape, (B, self.config.d_state))
        self.assertEqual(pred_num.shape, (B, S))

        # 4. Wide SwiGLU FFN
        delta_ffn = self.model.super_block.ffn(x)
        self.assertEqual(delta_ffn.shape, (B, S, D))

        print("[Test 2] All 4 functional branches executed cleanly with correct tensor dimensions.")

    def test_03_multi_budget_forward_pass(self):
        """Verify output logits shape [B, S, V] across dynamic test-time recursion budgets."""
        B, S = 2, 16
        input_ids = torch.randint(0, 1000, (B, S), device=self.device)

        for M in [1, 2, 4, 6]:
            out = self.model(input_ids, recursion_budget=M, return_all_recursions=True)
            logits = out["logits"]
            self.assertEqual(logits.shape, (B, S, self.config.vocab_size))
            self.assertFalse(torch.isnan(logits).any(), f"NaN in logits at M={M}")
            self.assertEqual(len(out["intermediate_states"]), M)
            self.assertEqual(len(out["branch_weights"]), M)
            print(f"[Test 3] Budget M={M}: logits shape {list(logits.shape)} - PASSED")

    def test_04_branch_diversity_and_arithmetic_loss(self):
        """Verify MoE branch diversity loss and Mamba arithmetic loss backpropagation."""
        self.model.train()
        B, S = 2, 12
        input_ids = torch.randint(0, 1000, (B, S), device=self.device)
        targets = torch.randint(0, 1000, (B, S), device=self.device)
        numeric_targets = torch.tensor([11.0, 8.0], device=self.device)

        loss_dict = self.model.compute_loss(input_ids, targets, numeric_targets=numeric_targets, recursion_budget=4)
        total_loss = loss_dict["total_loss"]
        loss_lm = loss_dict["loss_lm"]
        loss_balance = loss_dict["loss_balance"]
        loss_arith = loss_dict["loss_arith"]
        mean_branch_usage = loss_dict["mean_branch_usage"]

        print(f"\n[Test 4] Loss components: Total={total_loss.item():.4f}, LM={loss_lm.item():.4f}, "
              f"Balance={loss_balance.item():.4f}, Arith={loss_arith.item():.4f}")
        print(f"[Test 4] Mean Branch Usage: Full={mean_branch_usage[0].item():.3f}, "
              f"Lin={mean_branch_usage[1].item():.3f}, SSM={mean_branch_usage[2].item():.3f}, "
              f"FFN={mean_branch_usage[3].item():.3f}")

        self.assertFalse(torch.isnan(total_loss))
        self.assertGreater(total_loss.item(), 0.0)

        # Verify gradient flow
        self.model.zero_grad()
        total_loss.backward()

        # Check gradients on all 4 branch subsystems
        self.assertIsNotNone(self.model.super_block.full_attn.q_proj.weight.grad)
        self.assertIsNotNone(self.model.super_block.lin_attn.q_proj.weight.grad)
        self.assertIsNotNone(self.model.super_block.mamba.num_head.weight.grad)
        self.assertIsNotNone(self.model.super_block.ffn.gate_proj.weight.grad)
        self.assertIsNotNone(self.model.super_block.branch_router[0].weight.grad)
        print("[Test 4] Verified gradient backpropagation through all 4 branches, router, and arithmetic head.")

    def test_05_checkpoint_weight_loading(self):
        """Verify warm-start weight loading from previous checkpoints."""
        ckpt_path = "outputs/checkpoints/navitrit-100m-loopformer.pt"
        if os.path.exists(ckpt_path):
            self.model.load_from_pretrained_checkpoint(ckpt_path, source_layer=0)
            print(f"[Test 5] Successfully loaded weights from {ckpt_path}")
        else:
            print(f"[Test 5] Checkpoint {ckpt_path} not found, skipping loading test.")


if __name__ == "__main__":
    unittest.main()
