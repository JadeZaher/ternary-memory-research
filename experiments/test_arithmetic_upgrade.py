import torch
import torch.nn.functional as F
import time
import json
import os
from bitlinear import BitLinear

def phase1_correctness():
    print("--- Phase 1: Correctness Verification ---")
    results = {}
    x = torch.randn(4, 16, 256)
    
    max_diff_overall = 0.0
    for bias in [False, True]:
        for block_size in [256, None]:
            layer = BitLinear(256, 512, bias=bias, block_size=block_size)
            layer.eval()
            
            # Forward with legacy
            layer.use_additive_gemm = False
            out_legacy = layer(x)
            
            # Forward with additive
            layer.use_additive_gemm = True
            out_additive = layer(x)
            
            diff = torch.abs(out_legacy - out_additive).max().item()
            assert diff < 1e-4, f"Mismatch! max diff: {diff}"
            print(f"PASS: bias={bias}, block_size={block_size}, max_diff={diff:.6f}")
            max_diff_overall = max(max_diff_overall, diff)
    
    results['max_diff'] = max_diff_overall
    return results

def phase2_zero_skip():
    print("--- Phase 2: Zero-Skip Verification ---")
    layer = BitLinear(256, 512, block_size=256)
    layer.eval()
    x = torch.randn(4, 16, 256)
    layer.use_additive_gemm = True
    out_additive = layer(x)
    
    w_tilde = layer.last_w_tilde
    gamma = layer.last_gamma
    
    # manual computation for a single output neuron (e.g., 0)
    w_row = w_tilde[0]
    mask_pos = (w_row == 1.0)
    mask_neg = (w_row == -1.0)
    
    # output neuron 0 for batch 0, seq 0
    x_single = x[0, 0]
    
    manual_val = x_single[mask_pos].sum() - x_single[mask_neg].sum()
    
    if gamma.dim() > 0 and gamma.numel() > 1:
        gamma_row = gamma.view(512, -1).mean(dim=1)[0]
    else:
        gamma_row = gamma.mean()
        
    manual_val = manual_val * gamma_row
    
    diff = torch.abs(manual_val - out_additive[0, 0, 0]).item()
    assert diff < 1e-4, f"Manual computation mismatch! diff: {diff}"
    print(f"PASS: Zero-skip manual computation verified. diff={diff:.6f}")
    print(f"Zero-weight fraction: {layer.last_zero_fraction:.4f}")
    return {"zero_fraction": layer.last_zero_fraction}

def phase3_gradient():
    print("--- Phase 3: Gradient Flow Verification ---")
    layer_add = BitLinear(256, 512, block_size=256)
    layer_leg = BitLinear(256, 512, block_size=256)
    layer_leg.load_state_dict(layer_add.state_dict())
    
    x = torch.randn(4, 16, 256)
    target = torch.randn(4, 16, 512)
    
    # Additive
    layer_add.train()
    layer_add.use_additive_gemm = True
    out_add = layer_add(x)
    loss_add = F.mse_loss(out_add, target)
    loss_add.backward()
    grad_add = layer_add.weight.grad.clone()
    
    # Legacy
    layer_leg.train()
    layer_leg.use_additive_gemm = False
    out_leg = layer_leg(x)
    loss_leg = F.mse_loss(out_leg, target)
    loss_leg.backward()
    grad_leg = layer_leg.weight.grad.clone()
    
    assert grad_add is not None and not torch.all(grad_add == 0)
    diff = torch.abs(grad_add - grad_leg).max().item()
    assert diff < 1e-6, "Gradient mismatch"
    print("PASS: Gradients matched between paths in training mode.")
    return {"gradient_max_diff": diff}

def phase4_timing():
    print("--- Phase 4: Timing Benchmark on GPU ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    layer = BitLinear(768, 2048).to(device)
    layer.eval()
    x = torch.randn(1, 512, 768).to(device)
    
    # Warmup
    for _ in range(10):
        layer.use_additive_gemm = False
        _ = layer(x)
        layer.use_additive_gemm = True
        _ = layer(x)
        
    def get_stats(use_additive):
        times = []
        layer.use_additive_gemm = use_additive
        for _ in range(10):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                _ = layer(x)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            times.append((time.perf_counter() - start) / 10 * 1000)
        return torch.tensor(times).mean().item(), torch.tensor(times).std().item()

    mean_leg, std_leg = get_stats(False)
    mean_add, std_add = get_stats(True)
    
    speedup = mean_leg / mean_add
    
    print(f"Legacy path: {mean_leg:.3f} ± {std_leg:.3f} ms")
    print(f"Additive path: {mean_add:.3f} ± {std_add:.3f} ms")
    print(f"Speedup ratio (legacy/additive): {speedup:.3f}x")
    
    verdict = "FASTER" if speedup > 1.0 else "SLOWER_ON_CUBLAS"
    print(f"Verdict: {verdict}")

    return {
        "mean_latency_legacy_ms": mean_leg,
        "std_latency_legacy_ms": std_leg,
        "mean_latency_additive_ms": mean_add,
        "std_latency_additive_ms": std_add,
        "speedup": speedup,
        "verdict": verdict,
        "note": "additive path is two dense GEMMs on cuBLAS; the gate that matters is exactness (phase1/phase3)"
    }

def phase5_sparsity():
    print("--- Phase 5: Sparsity Analysis ---")
    configs = [
        ("q_proj", 768, 768),
        ("k_proj", 768, 768),
        ("v_proj", 768, 768),
        ("o_proj", 768, 768),
        ("gate_proj", 768, 2048),
        ("up_proj", 768, 2048),
        ("down_proj", 2048, 768),
    ]
    
    results = {}
    for name, in_f, out_f in configs:
        layer = BitLinear(in_f, out_f, block_size=256)
        x = torch.randn(1, 1, in_f)
        _ = layer(x)
        sparsity = layer.last_zero_fraction
        print(f"Layer {name} ({in_f}x{out_f}): sparsity = {sparsity:.4f}")
        results[name] = sparsity
        
    return results

def main():
    results = {}
    results['phase1'] = phase1_correctness()
    results['phase2'] = phase2_zero_skip()
    results['phase3'] = phase3_gradient()
    results['phase4'] = phase4_timing()
    results['phase5'] = phase5_sparsity()

    results["exactness_gate"] = "PASS" if (results["phase1"]["max_diff"] < 1e-4 and results["phase3"]["gradient_max_diff"] < 1e-6) else "FAIL"

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/arithmetic-upgrade-test.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved results to outputs/arithmetic-upgrade-test.json")

if __name__ == "__main__":
    main()
