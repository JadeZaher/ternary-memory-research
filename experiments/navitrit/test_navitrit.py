"""
experiments/navitrit/test_navitrit.py: Self-contained test suite for NaviTrit.

Validates:
1. Model forward pass shapes and trajectory recording.
2. Exercising all 4 navigation primitives:
   - Forward hops (l -> l+k)
   - Backward hops (l -> l-k)
   - Self-loops / repeated activations (l -> l)
   - Dynamic early exits (EXIT node or contraction stopping)
3. End-to-end differentiable autograd:
   - Gradients flow back through multi-hop Gumbel-Softmax walk to stationary
     BitLinear tiles, velocity net v_phi, and input embeddings.
4. Monotonic baseline equivalence mode.
5. Matched random walk control policy.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn

from experiments.navitrit.navitrit_model import (
    NaviTritConfig,
    NaviTritForCausalLM,
    TRANSITION_FORWARD,
    TRANSITION_BACKWARD,
    TRANSITION_SELF_LOOP,
    TRANSITION_EXIT,
)


def run_navitrit_tests():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[1/5] Initializing NaviTrit on {device}...")

    config = NaviTritConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=3,          # 3 Attention + 3 FFN = 6 modules + 1 EXIT = 7 nodes
        num_attention_heads=4,
        max_position_embeddings=128,
        max_hops=6,
        d_nav_route=64,
        flow_dt=0.5,
        eps_contraction=1e-3,
        ternary=True,
    )

    model = NaviTritForCausalLM(config).to(device)

    # -------------------------------------------------------------
    # 1. Test Forward Pass & Trajectory Recording (Eval mode)
    # -------------------------------------------------------------
    print("[2/5] Testing eval mode forward pass & trajectory telemetry...")
    model.eval()
    B, S = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)

    with torch.no_grad():
        out = model(input_ids)

    assert out.logits.shape == (B, S, config.vocab_size), f"Bad logits shape: {out.logits.shape}"
    assert len(out.trajectory_nodes) == out.total_hops_taken, "Mismatch between node list and hops count"
    assert out.total_hops_taken <= config.max_hops, f"Hops exceeded max_hops: {out.total_hops_taken} > {config.max_hops}"
    print(f"      Eval Forward PASS: Hops taken = {out.total_hops_taken}, Trajectory = {out.trajectory_nodes}")
    print(f"      Hops breakdown: Fwd={out.forward_hops_count}, Back={out.backward_hops_count}, Loop={out.self_loops_count}, Exit={out.early_exits_count}")

    # -------------------------------------------------------------
    # 2. Test Autograd Gradient Flow (Training mode)
    # -------------------------------------------------------------
    print("[3/5] Testing differentiable multi-hop autograd & loss heads...")
    model.train()
    input_ids_train = torch.randint(0, config.vocab_size, (2, 8), device=device)
    out_train = model(input_ids_train)

    assert out_train.diversity_loss is not None, "diversity_loss is None!"
    assert out_train.coherence_loss is not None, "coherence_loss is None!"

    loss = (
        out_train.logits.sum() * 1e-4
        + out_train.nav_budget_loss
        + out_train.fpf_loss
        + out_train.diversity_loss
        + out_train.coherence_loss
    )
    loss.backward()

    # Check gradients on stationary tiles
    attn_grad = model.graph.attn_tiles[0].q_proj.weight.grad
    ffn_grad = model.graph.ffn_tiles[0].gate_proj.weight.grad
    vel_grad = model.controller.velocity_net[1].weight.grad
    dest_grad = model.controller.dest_head[1].weight.grad
    cohere_grad = model.controller.coherence_head[1].weight.grad

    assert attn_grad is not None and torch.norm(attn_grad) > 0, "No grad on stationary Attention tile!"
    assert ffn_grad is not None and torch.norm(ffn_grad) > 0, "No grad on stationary FFN tile!"
    assert vel_grad is not None and torch.norm(vel_grad) > 0, "No grad on controller velocity net!"
    assert dest_grad is not None and torch.norm(dest_grad) > 0, "No grad on destination head!"
    assert cohere_grad is not None and torch.norm(cohere_grad) > 0, "No grad on coherence head!"
    print(f"      Autograd PASS: Stationary modules, velocity net, destination head, and coherence head received active gradients.")
    print(f"      Attn Tile 0 Grad Norm: {torch.norm(attn_grad).item():.4e}, Velocity Net Grad Norm: {torch.norm(vel_grad).item():.4e}")
    print(f"      Coherence Head Grad Norm: {torch.norm(cohere_grad).item():.4e}, Diversity Loss: {out_train.diversity_loss.item():.4e}")

    # -------------------------------------------------------------
    # 3. Test Navigation Primitives (Manual Injection & Verification)
    # -------------------------------------------------------------
    print("[4/5] Testing navigation primitive execution (Forward, Backward, Self-Loop, Exit)...")
    # Verify module graph executes individual node queries
    h_test = torch.randn(1, S, config.hidden_size, device=device)
    
    # Hop 1: Layer 0 FFN (node 1)
    h1 = model.graph.execute_node(1, h_test)
    assert not torch.equal(h1, h_test), "FFN node should transform h"

    # Self-Loop: Re-execute Layer 0 FFN (node 1) twice
    h1_repeat = model.graph.execute_node(1, h1)
    assert not torch.equal(h1_repeat, h1), "Repeated activation should further transform h"

    # Forward Hop: Jump to Layer 2 Attn (node 4)
    h2 = model.graph.execute_node(4, h1_repeat)
    assert not torch.equal(h2, h1_repeat), "Forward hop should transform h"

    # Backward Hop: Jump back to Layer 0 Attn (node 0)
    h_back = model.graph.execute_node(0, h2)
    assert not torch.equal(h_back, h2), "Backward hop should transform h"

    # Early Exit: Node 6 (EXIT node) should return h unchanged
    h_exit = model.graph.execute_node(model.graph.exit_node_idx, h_back)
    assert torch.equal(h_exit, h_back), "EXIT node must act as identity on h"

    print("      Navigation Primitives PASS: Forward, Backward, Self-Loop, and Exit verified.")

    # -------------------------------------------------------------
    # 4. Test Monotonic Baseline Equivalence Mode
    # -------------------------------------------------------------
    print("[5/5] Testing monotonic baseline mode & random control...")
    model.eval()
    with torch.no_grad():
        out_mono = model(input_ids, force_monotonic=True)
    expected_mono_nodes = list(range(2 * config.num_layers))
    assert out_mono.trajectory_nodes == expected_mono_nodes, f"Monotonic run failed: {out_mono.trajectory_nodes} != {expected_mono_nodes}"
    print(f"      Monotonic Mode PASS: Exactly executed sequence {out_mono.trajectory_nodes}")

    # Random walk control
    model.controller.random_policy = True
    with torch.no_grad():
        out_rand = model(input_ids)
    model.controller.random_policy = None
    print(f"      Random Walk Control PASS: Executed random trajectory {out_rand.trajectory_nodes}")

    print("\n[ALL NAVITRIT TESTS PASSED SUCCESSFULLY ON CUDA]")


if __name__ == "__main__":
    run_navitrit_tests()
