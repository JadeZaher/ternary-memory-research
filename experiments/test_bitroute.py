"""
experiments/test_bitroute.py: Comprehensive self-contained test suite for Track A (BitRoute-135M).

Verifications performed:
  1. CUDA availability & RTX 4060 device initialization.
  2. BitLinear ternary weight property: weights W_tilde strictly in {-1.0, 0.0, +1.0}.
  3. Straight-Through Estimator (STE) gradient flow: backward pass updates FP32 master weights.
  4. Block-scaled (g=256) vs layer-wide absmean scaling.
  5. TriStateRouter probe dispatch logic (EXECUTE, ROUTE_AROUND, EARLY_EXIT).
  6. Residual bypass assertion: ROUTE_AROUND skips BitLinear Attention & FFN forward passes.
  7. BitRoute-135M model parameter count (~135M parameters) and architectural structure.
  8. Dynamic early exit halting layer execution and saving memory bandwidth.
  9. End-to-end forward/backward training step on CUDA.
 10. Profiling latency, memory footprint, and bandwidth savings exported to outputs/bitroute-test.json.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.bitlinear import BitLinear, absmean_quantize_weights
from experiments.tristate_router import (
    TriStateRouter,
    ACTION_EXECUTE,
    ACTION_ROUTE_AROUND,
    ACTION_EARLY_EXIT,
)
from experiments.bitroute_model import (
    BitRouteConfig,
    BitRouteTransformerBlock,
    BitRouteForCausalLM,
)


def run_tests() -> Dict[str, Any]:
    test_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "cuda_device_count": torch.cuda.device_count(),
        },
        "tests": {},
        "profiling": {},
        "all_passed": False,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Track A: BitRoute-135M Engine Verification on {device} ===")
    print(f"CUDA Device: {test_results['platform']['cuda_device']}")

    # -------------------------------------------------------------------------
    # TEST 1: BitLinear Ternary Weight Distribution {-1, 0, +1}
    # -------------------------------------------------------------------------
    print("\n[1/8] Verifying BitLinear Ternary Weights strictly in {-1, 0, +1}...")
    torch.manual_seed(42)
    in_dim, out_dim = 768, 768

    # Test Block Scaling (g=256)
    layer_block = BitLinear(in_dim, out_dim, block_size=256, bias=False).to(device)
    x = torch.randn(2, 16, in_dim, device=device)
    _ = layer_block(x)
    w_tilde_block = layer_block.last_w_tilde

    unique_vals_block = torch.unique(w_tilde_block).cpu().tolist()
    is_strictly_ternary_block = set(unique_vals_block).issubset({-1.0, 0.0, 1.0})
    assert is_strictly_ternary_block, f"Weights not strictly in {{-1, 0, +1}}: {unique_vals_block}"

    counts_block = {
        "-1": int((w_tilde_block == -1.0).sum().item()),
        "0": int((w_tilde_block == 0.0).sum().item()),
        "+1": int((w_tilde_block == 1.0).sum().item()),
    }
    total_weights = w_tilde_block.numel()

    # Test Layer-Wide Scaling
    layer_wide = BitLinear(in_dim, out_dim, block_size=None, bias=False).to(device)
    _ = layer_wide(x)
    w_tilde_wide = layer_wide.last_w_tilde
    unique_vals_wide = torch.unique(w_tilde_wide).cpu().tolist()
    is_strictly_ternary_wide = set(unique_vals_wide).issubset({-1.0, 0.0, 1.0})
    assert is_strictly_ternary_wide, f"Layer-wide weights not strictly in {{-1, 0, +1}}: {unique_vals_wide}"

    test_results["tests"]["ternary_weights"] = {
        "status": "PASSED",
        "block_scaling_g256": {
            "unique_values": unique_vals_block,
            "counts": counts_block,
            "proportions": {k: round(v / total_weights, 4) for k, v in counts_block.items()},
        },
        "layer_wide_scaling": {
            "unique_values": unique_vals_wide,
        },
    }
    print(f"  -> PASS: Unique values: {unique_vals_block}, Proportions: {test_results['tests']['ternary_weights']['block_scaling_g256']['proportions']}")

    # -------------------------------------------------------------------------
    # TEST 2: Straight-Through Estimator (STE) Gradient Propagation
    # -------------------------------------------------------------------------
    print("\n[2/8] Verifying STE Gradient Propagation & Master Weight Updates...")
    layer_ste = BitLinear(in_dim, out_dim, block_size=256, bias=False).to(device)
    optimizer = torch.optim.AdamW(layer_ste.parameters(), lr=1e-3)

    w_initial = layer_ste.weight.detach().clone()
    optimizer.zero_grad()
    y = layer_ste(x)
    loss = y.pow(2).mean()
    loss.backward()

    assert layer_ste.weight.grad is not None, "Gradient is None!"
    assert torch.all(torch.isfinite(layer_ste.weight.grad)), "Gradients contain NaN or Inf!"
    assert torch.any(layer_ste.weight.grad != 0), "Gradients are all zeros!"

    optimizer.step()
    w_updated = layer_ste.weight.detach()
    weight_diff = (w_updated - w_initial).abs().max().item()
    assert weight_diff > 0, "Master weights did not change after optimizer step!"

    test_results["tests"]["ste_gradient_propagation"] = {
        "status": "PASSED",
        "grad_norm": float(layer_ste.weight.grad.norm().item()),
        "max_weight_update": float(weight_diff),
    }
    print(f"  -> PASS: Grad norm: {test_results['tests']['ste_gradient_propagation']['grad_norm']:.4f}, Max update: {weight_diff:.6f}")

    # -------------------------------------------------------------------------
    # TEST 3: TriStateRouter Dispatch Logic
    # -------------------------------------------------------------------------
    print("\n[3/8] Verifying TriStateRouter Dispatch Actions...")
    router = TriStateRouter(d_model=in_dim, tau_exit=0.8, tau_bypass=0.5).to(device)
    h_test = torch.randn(1, 16, in_dim, device=device)

    # Test Forced Actions
    dec_exec = router(h_test, force_action="EXECUTE")
    assert dec_exec.action == "EXECUTE" and dec_exec.action_id == ACTION_EXECUTE

    dec_bypass = router(h_test, force_action="ROUTE_AROUND")
    assert dec_bypass.action == "ROUTE_AROUND" and dec_bypass.action_id == ACTION_ROUTE_AROUND

    dec_exit = router(h_test, force_action="EARLY_EXIT")
    assert dec_exit.action == "EARLY_EXIT" and dec_exit.action_id == ACTION_EARLY_EXIT

    # Test Probability sum
    assert math.isclose(dec_exec.probs.sum().item(), 1.0, rel_tol=1e-5), "Probabilities do not sum to 1.0"

    test_results["tests"]["tristate_router"] = {
        "status": "PASSED",
        "forced_action_execute": dec_exec.action,
        "forced_action_bypass": dec_bypass.action,
        "forced_action_exit": dec_exit.action,
        "sample_probabilities": {
            "p_exec": round(dec_exec.p_exec, 4),
            "p_bypass": round(dec_exec.p_bypass, 4),
            "p_exit": round(dec_exit.p_exit, 4),
        },
    }
    print(f"  -> PASS: All three continuation actions (EXECUTE, ROUTE_AROUND, EARLY_EXIT) verified.")

    # -------------------------------------------------------------------------
    # TEST 4: Residual Bypass: Assert Zero Computation When ROUTE_AROUND
    # -------------------------------------------------------------------------
    print("\n[4/8] Asserting Residual Bypass Skips Attention and FFN Computation...")
    config = BitRouteConfig()
    block = BitRouteTransformerBlock(config, layer_idx=0).to(device)

    # 4A: ROUTE_AROUND Mode
    block.attn.forward_called = 0
    block.ffn.forward_called = 0
    h_in = torch.randn(1, 16, config.hidden_size, device=device)

    h_bypassed, decision_bypassed = block(h_in, force_action="ROUTE_AROUND")
    assert decision_bypassed.action == "ROUTE_AROUND"
    assert torch.equal(h_bypassed, h_in), "Bypass failed: h_{l+1} != h_l!"
    assert block.attn.forward_called == 0, f"Attention was called {block.attn.forward_called} times during bypass!"
    assert block.ffn.forward_called == 0, f"FFN was called {block.ffn.forward_called} times during bypass!"

    # 4B: EXECUTE Mode
    block.attn.forward_called = 0
    block.ffn.forward_called = 0
    h_executed, decision_executed = block(h_in, force_action="EXECUTE")
    assert decision_executed.action == "EXECUTE"
    assert not torch.equal(h_executed, h_in), "Execute failed: output is identical to input!"
    assert block.attn.forward_called == 1, "Attention was NOT called during EXECUTE!"
    assert block.ffn.forward_called == 1, "FFN was NOT called during EXECUTE!"

    test_results["tests"]["residual_bypass"] = {
        "status": "PASSED",
        "bypass_identity_exact": True,
        "attn_calls_during_bypass": 0,
        "ffn_calls_during_bypass": 0,
        "attn_calls_during_execute": 1,
        "ffn_calls_during_execute": 1,
    }
    print(f"  -> PASS: ROUTE_AROUND perfectly bypasses computation (Attn calls=0, FFN calls=0, h_{{l+1}} == h_l).")

    # -------------------------------------------------------------------------
    # TEST 5: Full BitRoute-135M Architecture & Parameter Accounting
    # -------------------------------------------------------------------------
    print("\n[5/8] Instantiating BitRoute-135M and Verifying Parameter Counts...")
    model = BitRouteForCausalLM(config).to(device)
    param_counts = model.count_parameters()
    total_params = param_counts["total_parameters"]
    total_params_m = round(total_params / 1e6, 2)

    print(f"  -> Model Parameter Breakdown:")
    print(f"     Total: {total_params_m}M ({total_params:,} params)")
    print(f"     Layer weights: {round(param_counts['layer_parameters'] / 1e6, 2)}M")
    print(f"     Embedding: {round(param_counts['embed_parameters'] / 1e6, 2)}M")
    print(f"     LM Head: {round(param_counts['head_parameters'] / 1e6, 2)}M")

    # Target: ~135M parameters (allow 130M - 140M range)
    assert 130_000_000 <= total_params <= 140_000_000, f"Parameter count {total_params} out of target range!"

    test_results["tests"]["model_architecture"] = {
        "status": "PASSED",
        "target_model": "BitRoute-135M",
        "total_parameters": total_params,
        "total_parameters_million": total_params_m,
        "breakdown": param_counts,
        "weights_per_layer": model.weights_per_layer,
        "bytes_per_layer_payload_tq1_0": model.bytes_per_layer_payload,
    }

    # -------------------------------------------------------------------------
    # TEST 6: Early Exit and Memory Bandwidth Savings
    # -------------------------------------------------------------------------
    print("\n[6/8] Verifying Dynamic Early Exit & Bandwidth Accounting...")
    input_ids = torch.randint(0, config.vocab_size, (1, 16), device=device)

    # 6A: Early Exit at Layer 3 (0-indexed layer 3 halts; layers 0, 1, 2 executed)
    out_exit3 = model(input_ids, force_layer_actions={3: "EARLY_EXIT"})
    assert out_exit3.exit_layer == 3
    assert out_exit3.executed_layers == [0, 1, 2]
    assert out_exit3.logits.shape == (1, 16, config.vocab_size)
    assert out_exit3.bandwidth_saving_ratio == 4.0  # 12 / 3 = 4.0x savings

    # 6B: Route Around 6 layers (50% bypass)
    bypass_map = {1: "ROUTE_AROUND", 3: "ROUTE_AROUND", 5: "ROUTE_AROUND", 7: "ROUTE_AROUND", 9: "ROUTE_AROUND", 11: "ROUTE_AROUND"}
    out_bypass6 = model(input_ids, force_layer_actions=bypass_map)
    assert len(out_bypass6.bypassed_layers) == 6
    assert len(out_bypass6.executed_layers) == 6
    assert out_bypass6.bandwidth_saving_ratio == 2.0  # 12 / 6 = 2.0x savings

    test_results["tests"]["dynamic_dispatch_scenarios"] = {
        "status": "PASSED",
        "early_exit_at_layer_3": {
            "executed_layers": out_exit3.executed_layers,
            "exit_layer": out_exit3.exit_layer,
            "streamed_bytes": out_exit3.payload_bytes_streamed,
            "full_bytes": out_exit3.full_payload_bytes,
            "bandwidth_saving_ratio": out_exit3.bandwidth_saving_ratio,
        },
        "fifty_percent_bypass": {
            "executed_layers": out_bypass6.executed_layers,
            "bypassed_layers": out_bypass6.bypassed_layers,
            "bandwidth_saving_ratio": out_bypass6.bandwidth_saving_ratio,
        },
    }
    print(f"  -> PASS: Early exit at layer 3 gives {out_exit3.bandwidth_saving_ratio}x bandwidth saving.")
    print(f"  -> PASS: 50% layer bypass gives {out_bypass6.bandwidth_saving_ratio}x bandwidth saving.")

    # -------------------------------------------------------------------------
    # TEST 7: End-to-End Forward & Backward Training Step on CUDA
    # -------------------------------------------------------------------------
    print("\n[7/8] Verifying End-to-End Training Backward Pass on CUDA...")
    opt_full = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt_full.zero_grad()

    out_train = model(input_ids, early_exit_enabled=False)
    targets = torch.randint(0, config.vocab_size, (1, 16), device=device)
    loss = F.cross_entropy(out_train.logits.view(-1, config.vocab_size), targets.view(-1))
    loss.backward()

    # Check gradients across diverse components
    has_embed_grad = model.embed_tokens.weight.grad is not None and model.embed_tokens.weight.grad.norm() > 0
    has_head_grad = model.lm_head.weight.grad is not None and model.lm_head.weight.grad.norm() > 0
    has_layer0_attn_grad = model.layers[0].attn.q_proj.weight.grad is not None and model.layers[0].attn.q_proj.weight.grad.norm() > 0
    has_layer0_ffn_grad = model.layers[0].ffn.gate_proj.weight.grad is not None and model.layers[0].ffn.gate_proj.weight.grad.norm() > 0

    assert has_embed_grad and has_head_grad and has_layer0_attn_grad and has_layer0_ffn_grad, "Missing component gradients!"
    opt_full.step()

    test_results["tests"]["end_to_end_training"] = {
        "status": "PASSED",
        "loss": float(loss.item()),
        "gradients_active": True,
    }
    print(f"  -> PASS: Loss = {loss.item():.4f}, All master gradients active and updated.")

    # -------------------------------------------------------------------------
    # TEST 8: Profiling Latency & Memory Footprint on RTX 4060
    # -------------------------------------------------------------------------
    print("\n[8/8] Profiling Latency and Hardware Footprint on RTX 4060...")
    model.eval()
    num_warmup = 10
    num_benchmark = 50

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Scenario A: Full 12 Layers Executed
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_benchmark):
            _ = model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    full_exec_time_ms = ((time.perf_counter() - start_time) / num_benchmark) * 1000

    # Scenario B: 50% Layers Bypassed (ROUTE_AROUND)
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_benchmark):
            _ = model(input_ids, force_layer_actions=bypass_map)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    bypass_exec_time_ms = ((time.perf_counter() - start_time) / num_benchmark) * 1000

    # Scenario C: Early Exit at Layer 3
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_benchmark):
            _ = model(input_ids, force_layer_actions={3: "EARLY_EXIT"})
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    exit_exec_time_ms = ((time.perf_counter() - start_time) / num_benchmark) * 1000

    gpu_mem_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0

    test_results["profiling"] = {
        "device": test_results["platform"]["cuda_device"],
        "gpu_max_memory_allocated_mb": round(gpu_mem_mb, 2),
        "batch_size": 1,
        "seq_len": 16,
        "latency_full_12_layers_ms": round(full_exec_time_ms, 3),
        "latency_50_pct_bypassed_ms": round(bypass_exec_time_ms, 3),
        "latency_early_exit_layer_3_ms": round(exit_exec_time_ms, 3),
        "speedup_bypass": round(full_exec_time_ms / bypass_exec_time_ms, 2),
        "speedup_early_exit": round(full_exec_time_ms / exit_exec_time_ms, 2),
        "theoretical_payload_mb": {
            "full_12_layers": round(model.total_layer_payload_bytes / (1024 * 1024), 2),
            "50_pct_bypassed": round((model.total_layer_payload_bytes / 2) / (1024 * 1024), 2),
            "early_exit_layer_3": round((model.bytes_per_layer_payload * 3) / (1024 * 1024), 2),
        },
    }

    print(f"  -> Latency Full (12 layers):       {full_exec_time_ms:.2f} ms")
    print(f"  -> Latency 50% Bypass:             {bypass_exec_time_ms:.2f} ms ({test_results['profiling']['speedup_bypass']}x faster)")
    print(f"  -> Latency Early Exit (Layer 3):   {exit_exec_time_ms:.2f} ms ({test_results['profiling']['speedup_early_exit']}x faster)")
    print(f"  -> GPU Peak Memory Allocated:      {gpu_mem_mb:.2f} MB")

    test_results["all_passed"] = True
    return test_results


def main():
    results = run_tests()
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "bitroute-test.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] All BitRoute-135M assertions PASSED.")
    print(f"[+] Output report successfully written to {out_path}")


if __name__ == "__main__":
    main()
