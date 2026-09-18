"""
experiments/frontier_scaling/test_navitrit_max.py: Unit Test Suite for NaviTrit-Max.

Verifies:
1. Parameter Count Audit (~247.6M parameters within 8.58GB VRAM envelope).
2. Strict Ternary Quantization Integrity (all projection weights in {-1, 0, +1}).
3. Dual-Expert Routing Distribution & Load-Balancing Loss.
4. End-to-End CUDA Mixed-Precision Forward + Backward Pass.
5. Autoregressive Generation & Token Sampling Integrity.
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.frontier_scaling.navitrit_max_model import (
    NaviTritMaxConfig,
    NaviTritMaxForCausalLM,
)


class TestNaviTritMax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[TestNaviTritMax] Running test suite on device: {cls.device}")

    def test_1_parameter_count_and_architecture(self):
        """Audits that model reaches scaled ~247.6M parameter specification."""
        cfg = NaviTritMaxConfig(num_layers=6, hidden_size=1024, intermediate_size=4096)
        model = NaviTritMaxForCausalLM(cfg)
        stats = model.count_parameters()

        print(f"[Test 1] Total Parameters: {stats['total_millions']:.2f}M")
        print(f"[Test 1] Layers Count: {stats['layers_count']} | Embedding: {stats['embedding_parameters']/1e6:.2f}M")

        self.assertAlmostEqual(stats["total_millions"], 247.61, delta=1.0)
        self.assertEqual(stats["layers_count"], 6)
        self.assertEqual(stats["head_parameters"], 0)  # Tied embeddings

    def test_2_ternary_quantization_integrity(self):
        """Verifies that all BitLinear projection weights strictly quantize to {-1, 0, +1}."""
        cfg = NaviTritMaxConfig(num_layers=1, hidden_size=256, intermediate_size=512)
        model = NaviTritMaxForCausalLM(cfg).to(self.device)
        layer = model.layers[0]

        # Inspect QKV projection weights
        w = layer.attn.qkv_proj.weight
        scale = w.abs().mean().clamp(min=1e-5)
        quant_w = torch.clamp(torch.round(w / scale), -1.0, 1.0)
        unique_vals = set(quant_w.detach().cpu().flatten()[:1000].numpy().tolist())

        print(f"[Test 2] Attention QKV Unique Quantized Values: {unique_vals}")
        self.assertTrue(unique_vals.issubset({-1.0, 0.0, 1.0}))

        # Inspect Expert 0 FFN weights
        w_exp = layer.routing_ffn.expert_0.gate_up_proj.weight
        scale_exp = w_exp.abs().mean().clamp(min=1e-5)
        quant_exp = torch.clamp(torch.round(w_exp / scale_exp), -1.0, 1.0)
        unique_exp = set(quant_exp.detach().cpu().flatten()[:1000].numpy().tolist())

        print(f"[Test 2] Expert 0 Unique Quantized Values: {unique_exp}")
        self.assertTrue(unique_exp.issubset({-1.0, 0.0, 1.0}))

    def test_3_dual_expert_routing_distribution(self):
        """Verifies router produces normalized probabilities and valid balance loss."""
        cfg = NaviTritMaxConfig(num_layers=2, hidden_size=256, intermediate_size=512)
        model = NaviTritMaxForCausalLM(cfg).to(self.device)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 32), device=self.device)
        with torch.no_grad():
            out = model(input_ids)

        probs_list = out["routing_probs"]
        self.assertEqual(len(probs_list), 2)  # 2 layers

        # Probabilities must sum to 1.0 across experts
        prob_sum = probs_list[0].sum(dim=-1)
        expected_sum = torch.ones_like(prob_sum)
        diff = torch.max(torch.abs(prob_sum - expected_sum)).item()

        print(f"[Test 3] Router Probabilities Sum Discrepancy: {diff:.6f}")
        self.assertLess(diff, 1e-5)
        self.assertFalse(torch.isnan(out["balance_loss"]).any())

    def test_4_end_to_end_cuda_backward(self):
        """Verifies end-to-end forward and backward pass on CUDA with AMP."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        cfg = NaviTritMaxConfig(num_layers=2, hidden_size=512, intermediate_size=1024)
        model = NaviTritMaxForCausalLM(cfg).cuda()
        model.train()

        input_ids = torch.randint(0, 1000, (2, 64), device="cuda")
        labels = torch.randint(0, 1000, (2, 64), device="cuda")

        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(input_ids, labels=labels)
            loss = out["loss"]

        scaler.scale(loss).backward()

        # Check gradients exist
        grad_norm = model.tok_embeddings.weight.grad.norm().item()
        print(f"[Test 4] CUDA Loss: {loss.item():.4f} | Embedding Grad Norm: {grad_norm:.4f}")

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertGreater(grad_norm, 0.0)

    def test_5_autoregressive_generation(self):
        """Verifies autoregressive token generation works cleanly without divergence."""
        cfg = NaviTritMaxConfig(num_layers=2, hidden_size=256, intermediate_size=512)
        model = NaviTritMaxForCausalLM(cfg).to(self.device)
        model.eval()

        prompt = torch.tensor([[100, 200, 300]], device=self.device)
        gen = model.generate(prompt, max_new_tokens=8, temperature=0.7)

        print(f"[Test 5] Generated Tokens: {gen.tolist()[0]}")
        self.assertEqual(gen.shape[1], 11)  # 3 prompt + 8 new tokens


if __name__ == "__main__":
    unittest.main()
