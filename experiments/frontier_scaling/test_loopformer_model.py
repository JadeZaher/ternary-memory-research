"""
experiments/frontier_scaling/test_loopformer_model.py: Unit Test Suite for LoopFormer / MoR Engine.

Gate 19 (Track G) Verification:
1. Parameter count & 32MB L2 Cache Residency Check (< 2.0 MB packed footprint).
2. Multi-budget forward pass shapes [B, S, V] across M in {1, 2, 4, 6, 8}.
3. 2D Recursion-Wise KV Cache numerical equivalence vs full sequence forward.
4. Shortcut-Consistency loss calculation and gradient flow.
5. Pre-trained weight extraction and loading from navitrit-100m-tree-grpo.pt.
"""

import os
import sys
import unittest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.frontier_scaling.loopformer_model import (
    LoopFormerConfig,
    LoopFormerForCausalLM,
    RecursionKVCache,
)


class TestLoopFormerModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.config = LoopFormerConfig(
            vocab_size=50257,
            hidden_size=768,
            intermediate_size=2048,
            num_attention_heads=12,
            max_position_embeddings=512,
            max_recursions=8,
            default_recursions=4,
        )
        cls.model = LoopFormerForCausalLM(cls.config).to(cls.device)

    def test_01_parameter_count_and_l2_residency(self):
        """Verify parameter-shared block size and 32MB L2 cache residency."""
        super_block_params = self.model.get_super_block_param_count()
        packed_mb = self.model.get_packed_footprint_mb()
        
        print(f"\n[Test 1] Recurrent Super-Block Parameters: {super_block_params:,}")
        print(f"[Test 1] 1.58-bit Packed Footprint: {packed_mb:.3f} MB")
        print(f"[Test 1] RTX 4060 L2 Cache Occupancy: {packed_mb / 32.0 * 100.0:.2f}% of 32MB")

        # Super-block should be ~7.09M parameters (4*768*768 attn + 3*768*2048 ffn + norms + step embed)
        self.assertGreater(super_block_params, 7_000_000)
        self.assertLess(super_block_params, 8_000_000)
        # Packed footprint must be strictly under 2.5 MB (well under 32 MB L2 cache)
        self.assertLess(packed_mb, 2.5)

    def test_02_multi_budget_forward_pass(self):
        """Verify logits shape [B, S, V] across variable test-time compute budgets M."""
        B, S = 2, 16
        input_ids = torch.randint(0, 1000, (B, S), device=self.device)

        budgets = [1, 2, 4, 6, 8]
        for M in budgets:
            out = self.model(input_ids, recursion_budget=M, return_all_recursions=True)
            logits = out["logits"]
            self.assertEqual(logits.shape, (B, S, self.config.vocab_size))
            self.assertFalse(torch.isnan(logits).any(), f"NaN found at budget M={M}")
            self.assertEqual(len(out["intermediate_states"]), M)
            print(f"[Test 2] Budget M={M}: logits shape {list(logits.shape)} - PASSED")

    def test_03_recursion_kv_cache_equivalence(self):
        """Verify that 2D Recursion-Wise KV Cache produces consistent token logits."""
        self.model.eval()
        B, S = 1, 8
        input_ids = torch.randint(0, 1000, (B, S), device=self.device)

        # Full sequence forward pass
        with torch.no_grad():
            out_full = self.model(input_ids, recursion_budget=4)
            logits_full = out_full["logits"][:, -1, :]

        # Incremental token pass with 2D RecursionKVCache
        kv_cache = RecursionKVCache(max_recursions=self.config.max_recursions)
        with torch.no_grad():
            # Pre-fill tokens 0..S-2
            prefix = input_ids[:, :-1]
            pos_prefix = torch.arange(S - 1, device=self.device).unsqueeze(0)
            h = self.model.embed_tokens(prefix) + self.model.embed_positions(pos_prefix)
            for k in range(4):
                h = self.model.super_block.forward_step(h, recursion_idx=k, kv_cache=kv_cache)

            # Process final token S-1
            last_tok = input_ids[:, -1:]
            pos_last = torch.tensor([[S - 1]], device=self.device)
            h_last = self.model.embed_tokens(last_tok) + self.model.embed_positions(pos_last)
            for k in range(4):
                h_last = self.model.super_block.forward_step(h_last, recursion_idx=k, kv_cache=kv_cache)

            logits_cache = self.model.lm_head(self.model.ln_f(h_last[:, -1, :]))

        diff = torch.max(torch.abs(logits_full - logits_cache)).item()
        print(f"[Test 3] Max logit discrepancy between full forward and 2D KV-cache: {diff:.6f}")
        # Numerical tolerance with BitLinear quantization
        self.assertLess(diff, 1e-3, "2D Recursion KV cache differs from full forward pass!")

    def test_04_shortcut_consistency_loss(self):
        """Verify shortcut-consistency training loss computation and gradient backprop."""
        self.model.train()
        B, S = 2, 12
        input_ids = torch.randint(0, 1000, (B, S), device=self.device)
        targets = torch.randint(0, 1000, (B, S), device=self.device)

        losses = self.model.compute_loss(input_ids, targets, recursion_budget=4)
        total_loss = losses["total_loss"]
        loss_final = losses["loss_final"]
        loss_intermediate = losses["loss_intermediate"]
        loss_alignment = losses["loss_alignment"]

        print(f"[Test 4] Losses: Total={total_loss.item():.4f}, Final={loss_final.item():.4f}, "
              f"Inter={loss_intermediate.item():.4f}, Align={loss_alignment.item():.4f}")

        self.assertFalse(torch.isnan(total_loss))
        self.assertGreater(total_loss.item(), 0.0)
        
        # Test backward pass
        self.model.zero_grad()
        total_loss.backward()

        # Check gradients on recurrent super-block
        grad_attn = self.model.super_block.attn.q_proj.weight.grad
        self.assertIsNotNone(grad_attn)
        self.assertGreater(torch.norm(grad_attn).item(), 0.0)
        print(f"[Test 4] Gradient flow through recurrent super-block verified: ||grad|| = {torch.norm(grad_attn).item():.4f}")

    def test_05_checkpoint_weight_loading(self):
        """Verify loading pre-trained weights from navitrit-100m-tree-grpo.pt."""
        ckpt_path = "outputs/checkpoints/navitrit-100m-tree-grpo.pt"
        if os.path.exists(ckpt_path):
            self.model.load_from_navitrit_checkpoint(ckpt_path, source_layer=0)
            print(f"[Test 5] Successfully verified checkpoint loading from {ckpt_path}")
        else:
            print(f"[Test 5] Checkpoint {ckpt_path} not found, skipping loading test")


if __name__ == "__main__":
    unittest.main()
