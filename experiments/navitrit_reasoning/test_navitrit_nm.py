"""
experiments/navitrit_reasoning/test_navitrit_nm.py: Unit Test Suite for NaviTrit-NM.

Verifies:
1. Weight Quantization: All stationary BitLinear weights in {-1, 0, +1}.
2. Hop-Conditioned Modulation: Different modulation at different hops.
3. Recurrent Reasoning Core: Trajectory enters v_reason, executes internal sub-steps.
4. Autograd & Gradient Flow: Backprop through STE, hop modulation, and router.
5. Deterministic Evaluation: Evaluation mode runs deterministically without Gumbel noise.
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.navitrit_reasoning.navitrit_nm_model import (
    NaviTritNMConfig,
    NaviTritNMForCausalLM,
    TRANSITION_REASONING,
)


def test_weight_quantization(device):
    print("\n[Test 1/5] Verifying Ternary Weight Quantization...")
    config = NaviTritNMConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_attention_heads=4,
        max_hops=4,
    )
    model = NaviTritNMForCausalLM(config).to(device)
    model.eval()
    
    # Check attention tiles
    for l, tile in enumerate(model.attn_tiles):
        for name, p in [("q_proj", tile.q_proj), ("k_proj", tile.k_proj), ("v_proj", tile.v_proj), ("o_proj", tile.o_proj)]:
            w_ternary, _ = p.get_ternary_weights()
            unique = torch.unique(w_ternary).tolist()
            for val in unique:
                assert val in [-1.0, 0.0, 1.0], f"Non-ternary value {val} in Layer {l} {name}"
    print("  [OK] Attention tile weights strictly in {-1, 0, +1}.")
    
    # Check reasoning core
    rc = model.reasoning_core
    for name, p in [("gate", rc.gate_proj), ("up", rc.up_proj), ("down", rc.down_proj), ("context", rc.context_proj)]:
        w_ternary, _ = p.get_ternary_weights()
        unique = torch.unique(w_ternary).tolist()
        for val in unique:
            assert val in [-1.0, 0.0, 1.0], f"Non-ternary value {val} in ReasoningCore {name}"
    print("  [OK] Reasoning Core weights strictly in {-1, 0, +1}.")


def test_hop_modulation(device):
    print("\n[Test 2/5] Verifying Hop-Conditioned Tile Modulation...")
    config = NaviTritNMConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_attention_heads=4,
        max_hops=4,
        use_hop_modulation=True,
    )
    model = NaviTritNMForCausalLM(config).to(device)
    model.eval()
    
    # Artificial pertubation of hop modulation MLP to test sensitivity
    with torch.no_grad():
        model.hop_modulation.mlp[-1].weight.fill_(0.1)
        model.hop_modulation.mlp[-1].bias.fill_(0.05)
        
    x = torch.randn(2, 8, config.hidden_size, device=device)
    
    # Execute Tile 0 at Hop 0 vs Hop 2
    out_hop0, _ = model._execute_tile(node_idx=0, h=x, hop_idx=0)
    out_hop2, _ = model._execute_tile(node_idx=0, h=x, hop_idx=2)
    
    diff = (out_hop0 - out_hop2).abs().max().item()
    assert diff > 1e-4, f"Hop modulation failed: outputs for Hop 0 and Hop 2 are identical (diff={diff})"
    print(f"  [OK] Hop modulation verified (activation divergence across hops: {diff:.4f}).")


def test_reasoning_core_execution(device):
    print("\n[Test 3/5] Verifying Dedicated Reasoning Core (Node 2L)...")
    config = NaviTritNMConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_attention_heads=4,
        max_hops=4,
        reasoning_sub_steps=3,
        use_reasoning_core=True,
    )
    model = NaviTritNMForCausalLM(config).to(device)
    model.eval()
    
    reasoning_node_idx = 2 * config.num_layers  # Node 4 for 2 layers
    input_ids = torch.randint(0, 1000, (2, 8), device=device)
    
    # Force trajectory to visit reasoning core at hop 1
    out = model(input_ids, force_nodes=[1, reasoning_node_idx, 0, 3])
    
    assert out.reasoning_hops_count >= 1, "Reasoning core was not executed!"
    reasoning_step = [s for s in out.trajectory if s.transition_type == TRANSITION_REASONING]
    assert len(reasoning_step) > 0, "Missing reasoning transition in trajectory!"
    assert reasoning_step[0].reasoning_steps_taken > 0, "Reasoning core took 0 sub-steps!"
    print(f"  [OK] Reasoning core executed successfully ({reasoning_step[0].reasoning_steps_taken} internal sub-steps).")


def test_autograd_flow(device):
    print("\n[Test 4/5] Verifying Autograd Gradient Flow...")
    config = NaviTritNMConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_attention_heads=4,
        max_hops=3,
        gumbel_hard=False,
    )
    model = NaviTritNMForCausalLM(config).to(device)
    model.train()
    
    input_ids = torch.randint(0, 1000, (2, 8), device=device)
    targets = torch.randint(0, 1000, (2, 8), device=device)
    
    out = model(input_ids)
    lm_loss = nn.CrossEntropyLoss()(out.logits.view(-1, config.vocab_size), targets.view(-1))
    total_loss = (
        lm_loss
        + out.loss_hop
        + out.loss_fpf
        + out.loss_state_fpf
        + out.loss_attn_div
        + out.loss_entropy
        + out.loss_cohere
    )
    
    total_loss.backward()
    
    # Assert gradients exist on master weights, hop modulation, and router
    attn_grad = model.attn_tiles[0].q_proj.weight.grad
    assert attn_grad is not None and torch.norm(attn_grad).item() > 0.0, "No gradient on attention weights!"
    
    hop_grad = model.hop_modulation.mlp[-1].weight.grad
    assert hop_grad is not None and torch.norm(hop_grad).item() > 0.0, "No gradient on hop modulation!"
    
    router_grad = model.controller.dest_head.weight.grad
    assert router_grad is not None and torch.norm(router_grad).item() > 0.0, "No gradient on router head!"
    
    rc_grad = model.reasoning_core.gate_proj.weight.grad
    assert rc_grad is not None and torch.norm(rc_grad).item() > 0.0, "No gradient on reasoning core!"
    
    print("  [OK] Autograd gradients verified across all model components.")


def test_deterministic_eval(device):
    print("\n[Test 5/5] Verifying Deterministic Evaluation Mode...")
    config = NaviTritNMConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_attention_heads=4,
        max_hops=4,
    )
    model = NaviTritNMForCausalLM(config).to(device)
    model.eval()
    
    input_ids = torch.randint(0, 1000, (1, 8), device=device)
    
    with torch.no_grad():
        out1 = model(input_ids)
        out2 = model(input_ids)
        
    diff = (out1.logits - out2.logits).abs().max().item()
    assert diff < 1e-5, f"Evaluation is not deterministic (max diff: {diff})"
    
    nodes1 = [s.selected_node for s in out1.trajectory]
    nodes2 = [s.selected_node for s in out2.trajectory]
    assert nodes1 == nodes2, f"Trajectories differed: {nodes1} vs {nodes2}"
    print(f"  [OK] Deterministic evaluation verified (trajectory: {nodes1}).")


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"=== Running NaviTrit-NM Unit Tests on {device} ===")
    
    test_weight_quantization(device)
    test_hop_modulation(device)
    test_reasoning_core_execution(device)
    test_autograd_flow(device)
    test_deterministic_eval(device)
    
    print("\n>>> ALL NAVITRIT-NM UNIT TESTS PASSED SUCCESSFULLY! <<<")
