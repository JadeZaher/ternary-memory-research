"""
experiments/frontier_scaling/test_static_state_analyzer.py: Unit Tests for Static State Analyzer.

Gate 19-C Verification:
1. Spectral norm power iteration accuracy against torch.linalg.matrix_norm.
2. Lipschitz contraction bound calculation.
3. Loop-invariant channel detection across recursion iterations.
4. Empirical contraction rate and reachability bounding.
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.frontier_scaling.static_state_analyzer import (
    compute_spectral_norm,
    StaticStateAnalyzer,
)
from experiments.frontier_scaling.loopformer_model import (
    LoopFormerConfig,
    LoopFormerForCausalLM,
)


class TestStaticStateAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.config = LoopFormerConfig(
            vocab_size=50257,
            hidden_size=768,
            intermediate_size=2048,
            num_attention_heads=12,
            max_position_embeddings=512,
            max_recursions=6,
            default_recursions=4,
        )
        cls.model = LoopFormerForCausalLM(cls.config).to(cls.device)
        cls.analyzer = StaticStateAnalyzer(cls.model, device=cls.device)

    def test_01_spectral_norm_accuracy(self):
        """Verify power iteration spectral norm computation against PyTorch matrix norm."""
        A = torch.randn(128, 64, device=self.device)
        computed_sigma = compute_spectral_norm(A, num_iterations=40)
        expected_sigma = torch.linalg.matrix_norm(A, ord=2).item()

        rel_error = abs(computed_sigma - expected_sigma) / max(1e-6, expected_sigma)
        print(f"\n[Test 1] Spectral norm computed: {computed_sigma:.4f}, expected: {expected_sigma:.4f}, rel_error: {rel_error:.4f}")
        self.assertLess(rel_error, 0.05, "Power iteration spectral norm differs by > 5%!")

    def test_02_lipschitz_bounds_analysis(self):
        """Verify Lipschitz analysis computes valid bounds and contractivity checks."""
        report = self.analyzer.analyze_lipschitz_bounds()
        print(f"[Test 2] Max Spectral: {report['max_spectral_norm']:.4f}, Mean: {report['mean_spectral_norm']:.4f}")
        print(f"[Test 2] Estimated Lipschitz L: {report['estimated_lipschitz_constant']:.4f}, Contractive: {report['is_contractive_guaranteed']}")

        self.assertGreater(report["max_spectral_norm"], 0.0)
        self.assertGreater(report["mean_spectral_norm"], 0.0)
        self.assertLessEqual(report["estimated_lipschitz_constant"], 1.0)
        self.assertTrue(report["banach_fixed_point_decidable"])

    def test_03_loop_invariant_detection(self):
        """Verify loop-invariant channel identification across recursion passes."""
        input_ids = torch.randint(0, 1000, (1, 8), device=self.device)
        report = self.analyzer.detect_loop_invariants(input_ids, max_loops=4)

        print(f"[Test 3] Invariant Channels: {report['invariant_channels_count']}/{report['total_hidden_dims']} ({report['invariant_channel_ratio']*100:.2f}%)")
        print(f"[Test 3] Mean Variance Across Loops: {report['mean_variance_across_loops']:.6f}")

        self.assertIn("invariant_channels_count", report)
        self.assertGreaterEqual(report["invariant_channels_count"], 0)
        self.assertLessEqual(report["invariant_channels_count"], self.config.hidden_size)

    def test_04_state_reachability_and_contraction(self):
        """Verify empirical state contraction rate rho and displacement tracking."""
        input_ids = torch.randint(0, 1000, (1, 8), device=self.device)
        report = self.analyzer.audit_state_reachability(input_ids, max_loops=4)

        print(f"[Test 4] State Norms: {[round(n, 2) for n in report['state_norms_per_loop']]}")
        print(f"[Test 4] Contraction Rates rho: {[round(r, 4) for r in report['contraction_factors_rho']]}")

        self.assertEqual(len(report["state_norms_per_loop"]), 4)
        self.assertFalse(report["out_of_vocabulary_explosion_detected"])
        self.assertFalse(report["zero_representation_collapse_detected"])


if __name__ == "__main__":
    unittest.main()
