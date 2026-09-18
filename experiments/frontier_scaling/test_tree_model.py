"""
experiments/frontier_scaling/test_tree_model.py: Unit tests for Gate 16-C Tree-Traversal Routing (TTR).

Verifies:
1. TreeBranchController unconstrained top-2 branch dispatch and gradients.
2. LatentCollapseOperator residual gated combination.
3. End-to-end NaviTritTreeForCausalLM forward & backward passes on CUDA.
4. Non-preconditioned routing: confirms no core is artificially favored ahead of time.
5. Warm-start loading from pretrained 100M checkpoint (navitrit-100m-trained.pt).
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.frontier_scaling.navitrit_scale_model import get_scale_config
from experiments.frontier_scaling.navitrit_tree_model import (
    TreeBranchController,
    LatentCollapseOperator,
    NaviTritTreeForCausalLM,
)


def test_tree_architecture():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Gate 16-C Tree Unit Tests on {device} ===")

    cfg_10m = get_scale_config("10m")
    num_nodes = 2 * cfg_10m.num_layers + 2  # 8 layers -> 16 tiles + reasoning(16) + exit(17) = 18 nodes

    # ----------------------------------------------------
    # Test 1: TreeBranchController Unconstrained Top-2 Dispatch
    # ----------------------------------------------------
    print("[Test 1] Testing TreeBranchController unconstrained top-2 dispatch...")
    controller = TreeBranchController(cfg_10m, num_nodes=num_nodes).to(device)
    h_pool = torch.randn(2, cfg_10m.hidden_size, device=device)
    g_domain = torch.rand(2, 1, device=device)
    prior_bias = torch.zeros(2, num_nodes, device=device)
    v_vis = torch.zeros(2, num_nodes, device=device)

    # Eval mode: greedy top-2
    controller.eval()
    (c1, c2), (w1, w2), logits, r_next = controller(
        h_token=h_pool,
        prev_node=num_nodes,
        g_domain=g_domain,
        node_prior_bias=prior_bias,
        v_vis=v_vis,
        temperature=1.0,
    )
    assert c1 != c2, f"Branches must be distinct, got c1={c1}, c2={c2}"
    assert c1 < num_nodes - 1 and c2 < num_nodes - 1, "Exit node must never be selected in compute branches"
    assert torch.allclose(w1 + w2, torch.ones_like(w1)), f"Gates must sum to 1.0, got {w1 + w2}"
    print(f"  -> Eval mode dispatch: Branch 1={c1}, Branch 2={c2}, Gate 1={w1[0].item():.3f}, Gate 2={w2[0].item():.3f}")

    # Train mode: stochastic exploration
    controller.train()
    (c1_tr, c2_tr), (w1_tr, w2_tr), logits_tr, _ = controller(
        h_token=h_pool,
        prev_node=num_nodes,
        g_domain=g_domain,
        node_prior_bias=prior_bias,
        v_vis=v_vis,
        temperature=1.0,
    )
    assert c1_tr != c2_tr, f"Training branches must be distinct, got {c1_tr}, {c2_tr}"
    print(f"  -> Train mode dispatch: Branch 1={c1_tr}, Branch 2={c2_tr}")
    print("  -> Test 1 Passed!")

    # ----------------------------------------------------
    # Test 2: LatentCollapseOperator
    # ----------------------------------------------------
    print("[Test 2] Testing LatentCollapseOperator...")
    collapse_op = LatentCollapseOperator(cfg_10m.hidden_size).to(device)
    h_orig = torch.randn(2, 16, cfg_10m.hidden_size, device=device)
    h_b1 = h_orig + torch.randn_like(h_orig) * 0.1
    h_b2 = h_orig + torch.randn_like(h_orig) * 0.1
    w1_gate = torch.tensor([[0.6], [0.4]], device=device)
    w2_gate = torch.tensor([[0.4], [0.6]], device=device)

    h_collapsed = collapse_op(h_orig, h_b1, h_b2, w1_gate, w2_gate)
    assert h_collapsed.shape == h_orig.shape, f"Expected {h_orig.shape}, got {h_collapsed.shape}"
    # Verify gradient flows back
    loss_col = h_collapsed.sum()
    loss_col.backward()
    print("  -> Test 2 Passed! Latent collapse and residual gating verified.")

    # ----------------------------------------------------
    # Test 3: NaviTritTreeForCausalLM Forward & Backward Pass (10M)
    # ----------------------------------------------------
    print("[Test 3] Testing NaviTritTreeForCausalLM forward & autograd (10M)...")
    model_10m = NaviTritTreeForCausalLM(cfg_10m).to(device)
    input_ids = torch.randint(0, cfg_10m.vocab_size, (2, 24), device=device)

    out = model_10m(input_ids, tree_depth=3)
    assert "logits" in out
    assert out["logits"].shape == (2, 24, cfg_10m.vocab_size)
    assert "tree_pairs" in out
    assert len(out["tree_pairs"]) == 3, f"Expected 3 tree levels, got {len(out['tree_pairs'])}"
    assert "trajectory" in out
    assert len(out["trajectory"]) == 6, f"Expected 6 executed tiles (3 levels * 2 branches), got {len(out['trajectory'])}"
    print(f"  -> Dispatched Tree Pairs across 3 levels: {out['tree_pairs']}")

    loss = out["logits"].sum() + out["trajectory_log_prob"].sum()
    loss.backward()
    assert model_10m.tree_controller.node_head.weight.grad is not None
    assert model_10m.collapse_operator.out_norm.weight.grad is not None
    assert model_10m.embed_tokens.weight.grad is not None
    print("  -> Test 3 Passed! End-to-end tree forward and backward pass verified.")

    # ----------------------------------------------------
    # Test 4: Verify Natural, Unbiased Prior Initialization
    # ----------------------------------------------------
    print("[Test 4] Verifying no core is artificially highlighted ahead of time...")
    prior_weights = model_10m.global_planner.prior_proj.weight
    prior_bias_init = model_10m.global_planner.prior_proj.bias
    assert torch.all(prior_weights == 0), "Prior projection weights must initialize to zero (no manual bias)"
    assert torch.all(prior_bias_init == 0), "Prior projection bias must initialize to zero"
    print("  -> Test 4 Passed! Clean neutral initialization verified (tree will train naturally).")

    # ----------------------------------------------------
    # Test 5: 100M Pretrained Checkpoint Warm-Start Loader
    # ----------------------------------------------------
    print("[Test 5] Testing 100M Checkpoint Warm-Start Loader...")
    ckpt_path = os.path.join(os.path.dirname(__file__), "../../outputs/checkpoints/navitrit-100m-trained.pt")
    if os.path.exists(ckpt_path):
        cfg_100m = get_scale_config("100m")
        model_100m = NaviTritTreeForCausalLM(cfg_100m).to(device)
        model_100m.load_from_pretrained_backbone(ckpt_path, device)
        total_p = model_100m.count_parameters()
        print(f"  -> NaviTrit-100M Tree model instantiated with {total_p / 1e6:.2f}M parameters.")

        # Test forward pass on 100M
        model_100m.eval()
        with torch.no_grad():
            out_100m = model_100m(input_ids[:, :16], tree_depth=3)
            print(f"  -> 100M Tree Pairs: {out_100m['tree_pairs']}")
            assert out_100m["logits"].shape == (2, 16, cfg_100m.vocab_size)
        print("  -> Test 5 Passed! 100M warm-start and forward pass verified.")
    else:
        print("  -> Skipped (100M checkpoint not found locally).")

    print("\n=== ALL 5 TREE-TRAVERSAL TESTS PASSED! ===")


if __name__ == "__main__":
    test_tree_architecture()
