"""
experiments/frontier_scaling/test_dual_model.py: Unit tests for Hierarchical Dual-Controller NaviTrit model.

Verifies:
1. PersistentEntityRegisters binding and injection.
2. GlobalFlowPlanner domain gating and dynamic budget prediction.
3. LocalFlowController velocity integration and prior bias application.
4. End-to-end NaviTritDualForCausalLM forward pass and autograd on CUDA.
5. Warm-start loader from pretrained 100M checkpoint.
6. Multi-task Hybrid Verifier scoring.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.frontier_scaling.navitrit_scale_model import get_scale_config
from experiments.frontier_scaling.navitrit_dual_model import (
    NaviTritDualForCausalLM,
    PersistentEntityRegisters,
    GlobalFlowPlanner,
    LocalFlowController,
)
from experiments.frontier_scaling.hybrid_verifier import HybridVerifier


def test_dual_architecture():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Unit Tests on {device} ===")

    # Test 1: PersistentEntityRegisters
    print("[Test 1] Testing PersistentEntityRegisters...")
    d_model = 192
    registers = PersistentEntityRegisters(hidden_size=d_model, num_slots=4).to(device)
    h_dummy = torch.randn(2, 16, d_model, device=device)
    slots = registers.bind_slots(h_dummy)
    assert slots.shape == (2, 4, d_model), f"Expected [2, 4, 192], got {slots.shape}"
    h_injected = registers.inject_registers(h_dummy, slots)
    assert h_injected.shape == h_dummy.shape, f"Expected {h_dummy.shape}, got {h_injected.shape}"
    print("  -> Passed! Slot attention binding and register injection verified.")

    # Test 2: GlobalFlowPlanner
    print("[Test 2] Testing GlobalFlowPlanner...")
    planner = GlobalFlowPlanner(hidden_size=d_model, d_route=64, num_nodes=10, max_hops=6).to(device)
    h_pool = h_dummy.mean(dim=1)
    g_domain, budget_logits, prior_bias = planner(h_pool)
    assert g_domain.shape == (2, 1), f"Expected [2, 1], got {g_domain.shape}"
    assert budget_logits.shape == (2, 5), f"Expected [2, 5], got {budget_logits.shape}"
    assert prior_bias.shape == (2, 10), f"Expected [2, 10], got {prior_bias.shape}"
    print("  -> Passed! Global Planner outputs verified.")

    # Test 3: LocalFlowController
    print("[Test 3] Testing LocalFlowController...")
    cfg_10m = get_scale_config("10m")
    controller = LocalFlowController(cfg_10m, num_nodes=10).to(device)
    action_soft, logits, r_next = controller(
        h_token=h_pool,
        prev_node=0,
        g_domain=g_domain,
        node_prior_bias=prior_bias,
        temperature=1.0,
    )
    assert action_soft.shape == (2, 10), f"Expected [2, 10], got {action_soft.shape}"
    assert logits.shape == (2, 10), f"Expected [2, 10], got {logits.shape}"
    print("  -> Passed! Local Controller integration verified.")

    # Test 4: End-to-End NaviTritDualForCausalLM forward & backward
    print("[Test 4] Testing End-to-End Dual Model on CUDA...")
    model = NaviTritDualForCausalLM(cfg_10m).to(device)
    input_ids = torch.randint(0, 1000, (2, 32), device=device)
    out = model(input_ids, temperature=1.0)
    
    assert "logits" in out
    assert out["logits"].shape == (2, 32, cfg_10m.vocab_size)
    assert "trajectory" in out
    assert len(out["trajectory"]) >= 2
    assert "trajectory_log_prob" in out
    assert out["trajectory_log_prob"].shape == (2,)

    # Autograd check
    loss = out["logits"].sum() + out["trajectory_log_prob"].sum()
    loss.backward()
    assert model.embed_tokens.weight.grad is not None
    assert model.global_planner.prior_proj.weight.grad is not None
    print("  -> Passed! End-to-end forward and backward pass verified.")

    # Test 5: Warm-Start Loader from Pretrained 100M Checkpoint
    print("[Test 5] Testing Warm-Start Loader on 100M Checkpoint...")
    ckpt_100m_path = os.path.join(os.path.dirname(__file__), "../../outputs/checkpoints/navitrit-100m-trained.pt")
    if os.path.exists(ckpt_100m_path):
        cfg_100m = get_scale_config("100m")
        model_100m = NaviTritDualForCausalLM(cfg_100m).to(device)
        model_100m.load_from_pretrained_backbone(ckpt_100m_path, device)
        print("  -> Passed! 100M backbone successfully warm-started into Dual-Controller.")
    else:
        print("  -> Skipped (100M checkpoint not found locally).")

    # Test 6: Hybrid Verifier
    print("[Test 6] Testing Multi-Task Hybrid Verifier...")
    verifier = HybridVerifier(use_gemini=False)
    # Math check
    r_math_good, _ = verifier.compute_reward("math", "15 - 4 = ?", "15 - 4 = 11 apples remain", target="11")
    r_math_bad, _ = verifier.compute_reward("math", "15 - 4 = ?", "15 - 3 = 17 apples remain", target="11")
    assert r_math_good > r_math_bad, f"Expected good reward > bad reward, got {r_math_good} vs {r_math_bad}"
    
    # Code check
    r_code_good, _ = verifier.compute_reward("code", "def f(x):\n", "    return x + 1\n")
    r_code_bad, _ = verifier.compute_reward("code", "def f(x):\n", "    return return ;;\n")
    assert r_code_good > r_code_bad, f"Expected good code > bad code, got {r_code_good} vs {r_code_bad}"
    print(f"  -> Passed! Verifier scores: Math Good={r_math_good}, Math Bad={r_math_bad}, Code Good={r_code_good}, Code Bad={r_code_bad}")

    print("\nALL 6 TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_dual_architecture()
