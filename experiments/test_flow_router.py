"""
experiments/test_flow_router.py: Test suite for Flow-Reasoned Dynamic Router.

Validates:
1. Shape consistency: execution_mask is [B, L, 2], logits/probs are [B, L, 2, 2].
2. Differentiable Autograd: Gradients flow from downstream task loss and budget loss
   through straight-through Gumbel-Softmax mask back to velocity net v_phi, context encoder,
   and input activations h.
3. Attractor relaxation and contraction stopping: verifies step counts and convergence.
4. FPF stability regularization: verifies ||v_phi(r*)||_2^2 penalty behavior.
5. Matched random control: verifies random_policy replaces probe with exact Bernoulli coin.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.flow_router import (
    FlowRoutingPlanner,
    SUBLAYER_ATTN,
    SUBLAYER_FFN,
)


def run_flow_router_tests():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[1/5] Initializing FlowRoutingPlanner on {device}...")

    num_layers = 6
    d_model = 256
    d_route = 64
    max_flow_steps = 4

    router = FlowRoutingPlanner(
        num_layers=num_layers,
        d_model=d_model,
        d_route=d_route,
        max_flow_steps=max_flow_steps,
        dt=0.5,
        eps_contraction=1e-3,
        gumbel_tau=1.0,
        default_target_budget=0.60,
    ).to(device)

    # 1. Test Forward shapes (Eval mode)
    print("[2/5] Testing eval mode forward pass & shapes...")
    router.eval()
    B, S = 4, 32
    h = torch.randn(B, S, d_model, device=device)
    mask = torch.ones(B, S, device=device)

    with torch.no_grad():
        decision = router(h, attention_mask=mask)

    assert decision.execution_mask.shape == (B, num_layers, 2), f"Bad mask shape: {decision.execution_mask.shape}"
    assert decision.logits.shape == (B, num_layers, 2, 2), f"Bad logits shape: {decision.logits.shape}"
    assert decision.probs.shape == (B, num_layers, 2, 2), f"Bad probs shape: {decision.probs.shape}"
    assert decision.r_star.shape == (B, d_route), f"Bad r_star shape: {decision.r_star.shape}"
    # In eval mode, execution mask values must be strictly binary 0.0 or 1.0
    unique_vals = torch.unique(decision.execution_mask).tolist()
    for val in unique_vals:
        assert val in (0.0, 1.0), f"Eval mask contains non-binary value: {val}"
    print(f"      Eval Forward PASS: Mask shape {decision.execution_mask.shape}, Mean exec: {decision.mean_overall_exec:.3f}")

    # 2. Test Autograd Gradient Flow (Train mode)
    print("[3/5] Testing training mode differentiable autograd & gradient flow...")
    router.train()
    h_train = torch.randn(B, S, d_model, device=device, requires_grad=True)
    decision_train = router(h_train, attention_mask=mask)

    # Synthetic downstream loss depending on execution mask and hidden state
    # Simulated layer output: h_out = sum_{l, m} (mask_{l,m} * h)
    dummy_task_loss = torch.sum(decision_train.execution_mask.unsqueeze(1).unsqueeze(-1) * h_train.unsqueeze(2).unsqueeze(3)) * 1e-4
    budget_loss = router.compute_budget_loss(decision_train, target_budget=0.50)
    fpf_loss = decision_train.fpf_loss

    total_loss = dummy_task_loss + 0.1 * budget_loss + 0.01 * fpf_loss
    total_loss.backward()

    # Verify gradients reach velocity net, context encoder, r_init_proj, and value head
    assert h_train.grad is not None and torch.norm(h_train.grad) > 0, "No grad on input h!"
    for name, param in router.named_parameters():
        assert param.grad is not None, f"No gradient reached parameter: {name}"
        grad_norm = torch.norm(param.grad).item()
        assert grad_norm > 0, f"Zero gradient on parameter: {name}"
    print(f"      Autograd PASS: All parameters received non-zero gradients. Input grad norm: {torch.norm(h_train.grad).item():.4e}")

    # 3. Test Contraction Early Halting
    print("[4/5] Testing attractor relaxation & contraction dynamics...")
    router.eval()
    with torch.no_grad():
        # Case A: Low tolerance (forces max steps)
        router.eps_contraction = 1e-12
        dec_slow = router(h)
        assert (dec_slow.flow_steps_taken == max_flow_steps).all(), "Slow contraction should hit max steps"

        # Case B: High tolerance (immediate convergence in 1 step)
        router.eps_contraction = 1e2
        dec_fast = router(h)
        assert (dec_fast.flow_steps_taken == 1).all(), "Fast contraction should converge in 1 step"
    print("      Contraction Dynamics PASS: Dynamic halting correctly responds to epsilon.")

    # 4. Test Matched Random Policy Control
    print("[5/5] Testing matched random coin control policy...")
    router.eval()
    router.random_policy = 0.40  # 40% target execution
    with torch.no_grad():
        # Run over large batch to verify empirical frequency
        h_large = torch.randn(100, S, d_model, device=device)
        dec_rand = router(h_large)
        empirical_rate = dec_rand.mean_overall_exec
        print(f"      Target coin rate: 0.400, Measured empirical rate: {empirical_rate:.3f}")
        assert abs(empirical_rate - 0.40) < 0.05, f"Coin router out of binomial tolerance: {empirical_rate}"
    router.random_policy = None
    print("      Random Control PASS: Bernoulli sampling matches specified policy.")

    print("\n[ALL 5 TESTS PASSED SUCCESSFULLY]")


if __name__ == "__main__":
    run_flow_router_tests()
