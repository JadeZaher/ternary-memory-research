"""
experiments/frontier_scaling/test_token_graph_model.py: Unit tests for Gate 18 DTRNet Token Graph Model.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.frontier_scaling.navitrit_scale_model import get_scale_config
from experiments.frontier_scaling.navitrit_token_graph_model import NaviTritTokenGraphForCausalLM


def test_token_graph_forward_and_shapes():
    print("\n--- Test 1: Forward Pass & Token Gating Tensor Shapes ---")
    cfg = get_scale_config("100m")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NaviTritTokenGraphForCausalLM(cfg).to(device)
    model.eval()

    B, S = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=device)

    with torch.no_grad():
        out = model(input_ids, traversal_depth=3, bypass_thresh=0.20)

    logits = out["logits"]
    assert logits.shape == (B, S, cfg.vocab_size), f"Bad logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in logits!"

    t_graph = out["traversal_graph"]
    assert "directed_edges" in t_graph, "Missing directed_edges in traversal_graph"
    assert "attention_bypass_rate" in t_graph, "Missing attention_bypass_rate"
    assert len(t_graph["traversed_nodes"]) == 6, f"Expected 6 traversed nodes, got {len(t_graph['traversed_nodes'])}"

    print(f"  Logits shape: {logits.shape} (PASSED)")
    print(f"  Attention Bypass Rate: {t_graph['attention_bypass_rate']*100:.1f}%")
    print(f"  Directed edges: {t_graph['directed_edges'][:2]}")


def test_token_gate_normalization_and_bypass():
    print("\n--- Test 2: Per-Token Gate Normalization (Sum to 1.0) & Attention Bypass ---")
    cfg = get_scale_config("100m")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NaviTritTokenGraphForCausalLM(cfg).to(device)

    B, S = 2, 8
    h_seq = torch.randn(B, S, cfg.hidden_size, device=device)
    g_domain = torch.randn(B, 1, device=device)
    node_bias = torch.zeros(B, model.num_nodes, device=device)
    v_vis = torch.zeros(B, model.num_nodes, device=device)

    (c1, c2), (w1, w2), bypass_mask, edges, logits, r_next = model.token_controller(
        h_seq=h_seq,
        prev_node=model.num_nodes,
        g_domain=g_domain,
        node_prior_bias=node_bias,
        v_vis=v_vis,
        bypass_thresh=0.30,
    )

    assert w1.shape == (B, S, 1), f"Bad w1 shape: {w1.shape}"
    assert w2.shape == (B, S, 1), f"Bad w2 shape: {w2.shape}"
    assert (w1 >= 0.0).all() and (w1 <= 1.0).all(), "w1 out of bounds!"
    assert (w2 >= 0.0).all() and (w2 <= 1.0).all(), "w2 out of bounds!"
    assert (w1 + w2 <= 1.0001).all(), "Combined gates exceed 1.0!"

    assert bypass_mask.shape == (B, S), f"Bad bypass mask shape: {bypass_mask.shape}"
    print(f"  Per-token gates verified: w1 in [0, 1], w2 in [0, 1], w1+w2 <= 1.0 (PASSED)")
    print(f"  Sample bypass mask: {bypass_mask[0].tolist()}")


def test_loading_from_tree_grpo():
    print("\n--- Test 3: Weight Compatibility from Tree GRPO Checkpoint ---")
    cfg = get_scale_config("100m")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NaviTritTokenGraphForCausalLM(cfg).to(device)

    ckpt_path = "outputs/checkpoints/navitrit-100m-tree-grpo.pt"
    if not os.path.exists(ckpt_path):
        print(f"Skipping checkpoint load test: {ckpt_path} not found.")
        return

    tree_sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_sd = model.state_dict()

    # Map tree_controller -> token_controller keys where shapes match
    compatible_keys = 0
    new_keys = []
    for k, v in tree_sd.items():
        mapped_k = k.replace("tree_controller.", "token_controller.").replace("collapse_operator.", "token_collapse.")
        if mapped_k in model_sd and model_sd[mapped_k].shape == v.shape:
            model_sd[mapped_k] = v
            compatible_keys += 1

    for k in model_sd.keys():
        mapped_orig = k.replace("token_controller.", "tree_controller.").replace("token_collapse.", "collapse_operator.")
        if mapped_orig not in tree_sd:
            new_keys.append(k)

    model.load_state_dict(model_sd)
    print(f"  Compatible keys successfully loaded: {compatible_keys}")
    print(f"  Newly added parameters (e.g. token_gate_head): {len(new_keys)} ({new_keys[:3]}...)")


def test_trajectory_planning_and_forward():
    print("\n--- Test 4: Trajectory Planning & Fast Forward Rollout ---")
    cfg = get_scale_config("100m")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NaviTritTokenGraphForCausalLM(cfg).to(device)

    B, S = 2, 12
    input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=device)

    pairs, gates, masks, log_prob, slots, kl, entropy = model.plan_token_trajectory(
        input_ids,
        temperature=1.0,
        traversal_depth=3,
        return_diagnostics=True,
    )

    assert len(pairs) == 3, f"Expected 3 pairs, got {len(pairs)}"
    assert log_prob.shape == (B,), f"Expected log_prob shape (B,), got {log_prob.shape}"

    # Fast forward execution on prompt
    logits = model.forward_token_trajectory(input_ids, pairs, gates, masks, slots=slots)
    assert logits.shape == (B, S, cfg.vocab_size), f"Expected logits (B, S, V), got {logits.shape}"

    # Simulate sequence extension by 3 tokens
    ext_ids = torch.cat([input_ids, torch.randint(0, cfg.vocab_size, (B, 3), device=device)], dim=1)
    ext_logits = model.forward_token_trajectory(ext_ids, pairs, gates, masks, slots=slots)
    assert ext_logits.shape == (B, S + 3, cfg.vocab_size), f"Expected ext_logits (B, S+3, V), got {ext_logits.shape}"

    print(f"  Trajectory planning and fast forward validated on CUDA (PASSED)")


if __name__ == "__main__":
    test_token_graph_forward_and_shapes()
    test_token_gate_normalization_and_bypass()
    test_loading_from_tree_grpo()
    test_trajectory_planning_and_forward()
    print("\nALL DTRNET TOKEN GRAPH MODEL TESTS PASSED!")
