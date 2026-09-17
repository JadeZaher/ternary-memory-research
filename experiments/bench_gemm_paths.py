"""
experiments/bench_gemm_paths.py: latency/accuracy benchmark across ternary GEMM execution paths.

Paths: dense_fp32(_layerwide) is the cuBLAS baseline on expanded ternary weights;
additive_reference rebuilds +1/-1 masks every call; additive_cached reuses pre-built
masks (models zero-skip hardware -- on cuBLAS both additive paths are literally two
dense GEMMs, x@m_pos.T - x@m_neg.T, so they are not expected to beat dense_fp32 here);
dense_bf16 is the same GEMM at half precision; int8_tensorcore runs a real int8
tensor-core matmul via torch._int_mm (requires a layer-wide scalar gamma).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from experiments.bitlinear import (
    absmean_quantize_weights,
    ternary_additive_gemm,
    prepare_additive_masks,
    ternary_additive_gemm_prepared,
)

SEED = 20260916
LAYER_SHAPES = [(768, 768), (768, 2048), (2048, 768)]
CUDA_INPUT_SHAPES = [(1, 1), (1, 512), (16, 128)]  # (batch, seq)
CPU_INPUT_SHAPES = [(1, 1), (1, 64)]
BLOCK_SIZE = 256


def shape_key(in_f, out_f):
    return f"{in_f}x{out_f}"


def input_key(batch, seq):
    return f"{batch}x{seq}"


def weight_storage_bytes(in_f, out_f):
    n = in_f * out_f
    return {
        "fp32_bytes": n * 4,
        "bf16_bytes": n * 2,
        "int8_bytes": n * 1,
        "packed_ternary_bytes": n * 1.6875 / 8,
        "_label": "derived, not measured",
    }


def expand_gamma(w_tilde, gamma):
    """Expand block gamma [numel/block, 1] to w_tilde's full shape; scalar gamma broadcasts as-is."""
    if gamma.dim() > 0 and gamma.numel() > 1:
        block = w_tilde.numel() // gamma.numel()
        return gamma.expand(-1, block).reshape(w_tilde.shape)
    return gamma


def build_weight(in_f, out_f, block_size, device):
    torch.manual_seed(SEED)
    weight = torch.empty(out_f, in_f, dtype=torch.float32, device=device)
    torch.nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
    _, w_tilde, gamma = absmean_quantize_weights(weight, block_size=block_size)
    w_tilde = w_tilde.detach()
    gamma = gamma.detach()
    w_eff = (w_tilde * expand_gamma(w_tilde, gamma)).detach()
    return w_eff, w_tilde, gamma


def time_cuda(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def time_cpu(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def error_stats(y, y_ref):
    y = y.detach().float()
    y_ref = y_ref.detach().float()
    max_abs_err = (y - y_ref).abs().max().item()
    denom = y_ref.abs().max().item()
    rel_err = max_abs_err / denom if denom > 0 else float("nan")
    return max_abs_err, rel_err


def run_int8(x, w_tilde_lw, gamma_lw):
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    M = x2d.shape[0]

    scale_x = 127.0 / x2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [M,1]
    x_int8 = torch.clamp(torch.round(x2d * scale_x), -128, 127).to(torch.int8)
    w_int8 = w_tilde_lw.to(torch.int8)  # [out,in], values in {-1,0,1}

    padded_m = False
    target_m = M
    if M < 32:
        target_m = 32
        padded_m = True
    elif M % 8 != 0:
        target_m = M + (8 - M % 8)
        padded_m = True

    if target_m != M:
        pad = torch.zeros(target_m - M, x_int8.shape[1], dtype=torch.int8, device=x_int8.device)
        x_int8_run = torch.cat([x_int8, pad], dim=0)
        scale_x_run = torch.cat(
            [scale_x, torch.ones(target_m - M, 1, dtype=scale_x.dtype, device=scale_x.device)], dim=0
        )
    else:
        x_int8_run = x_int8
        scale_x_run = scale_x

    y_int32 = torch._int_mm(x_int8_run, w_int8.t().contiguous())
    y_int32 = y_int32[:M]
    scale_x = scale_x_run[:M]

    y = y_int32.float() * (1.0 / scale_x) * gamma_lw
    return y.reshape(*orig_shape[:-1], -1), padded_m


def bench_cuda_shape(in_f, out_f, batch, seq, device, warmup, iters):
    torch.manual_seed(SEED)
    x = torch.randn(batch, seq, in_f, device=device)

    w_eff_bw, w_tilde_bw, gamma_bw = build_weight(in_f, out_f, BLOCK_SIZE, device)
    w_eff_lw, w_tilde_lw, gamma_lw = build_weight(in_f, out_f, None, device)
    m_pos, m_neg, gamma_scalar = prepare_additive_masks(w_tilde_bw, gamma_bw, x.dtype)

    results = {}

    def f_dense():
        return F.linear(x, w_eff_bw)

    mean, std = time_cuda(f_dense, warmup, iters)
    y_ref = f_dense()
    max_abs_err, rel_err = error_stats(y_ref, y_ref)
    results["dense_fp32"] = {
        "mean_ms": mean, "std_ms": std,
        "max_abs_err": max_abs_err, "rel_err": rel_err,
        "speedup_vs_dense_fp32": 1.0,
    }

    def f_add_ref():
        return ternary_additive_gemm(x, w_tilde_bw, gamma_bw)

    mean, std = time_cuda(f_add_ref, warmup, iters)
    y = f_add_ref()
    max_abs_err, rel_err = error_stats(y, y_ref)
    results["additive_reference"] = {
        "mean_ms": mean, "std_ms": std,
        "max_abs_err": max_abs_err, "rel_err": rel_err,
        "speedup_vs_dense_fp32": results["dense_fp32"]["mean_ms"] / mean,
    }

    def f_add_cached():
        return ternary_additive_gemm_prepared(x, m_pos, m_neg, gamma_scalar)

    mean, std = time_cuda(f_add_cached, warmup, iters)
    y = f_add_cached()
    max_abs_err, rel_err = error_stats(y, y_ref)
    results["additive_cached"] = {
        "mean_ms": mean, "std_ms": std,
        "max_abs_err": max_abs_err, "rel_err": rel_err,
        "speedup_vs_dense_fp32": results["dense_fp32"]["mean_ms"] / mean,
    }

    x_bf16 = x.to(torch.bfloat16)
    w_bf16 = w_eff_bw.to(torch.bfloat16)

    def f_bf16():
        return F.linear(x_bf16, w_bf16)

    mean, std = time_cuda(f_bf16, warmup, iters)
    y = f_bf16()
    max_abs_err, rel_err = error_stats(y, y_ref)
    results["dense_bf16"] = {
        "mean_ms": mean, "std_ms": std,
        "max_abs_err": max_abs_err, "rel_err": rel_err,
        "speedup_vs_dense_fp32": results["dense_fp32"]["mean_ms"] / mean,
    }

    def f_dense_lw():
        return F.linear(x, w_eff_lw)

    mean, std = time_cuda(f_dense_lw, warmup, iters)
    y_ref_lw = f_dense_lw()
    max_abs_err, rel_err = error_stats(y_ref_lw, y_ref_lw)
    results["dense_fp32_layerwide"] = {
        "mean_ms": mean, "std_ms": std,
        "max_abs_err": max_abs_err, "rel_err": rel_err,
        "speedup_vs_dense_fp32": results["dense_fp32"]["mean_ms"] / mean,
        "note": "reference for int8_tensorcore, uses layer-wide scalar gamma (block_size=None)",
    }

    try:
        def f_int8():
            y, _ = run_int8(x, w_tilde_lw, gamma_lw)
            return y

        mean, std = time_cuda(f_int8, warmup, iters)
        y, padded_m = run_int8(x, w_tilde_lw, gamma_lw)
        max_abs_err, rel_err = error_stats(y, y_ref_lw)
        results["int8_tensorcore"] = {
            "mean_ms": mean, "std_ms": std,
            "max_abs_err": max_abs_err, "rel_err": rel_err,
            "speedup_vs_dense_fp32": results["dense_fp32"]["mean_ms"] / mean,
            "padded_m": padded_m,
            "gamma_mode": "layer-wide scalar gamma (block_size=None); block gamma cannot be "
                          "factored out of a single int32 dequant multiply",
        }
    except Exception as e:
        results["int8_tensorcore"] = {"error": str(e)}

    return results


def bench_cpu_shape(in_f, out_f, batch, seq, warmup, iters):
    device = torch.device("cpu")
    torch.manual_seed(SEED)
    x = torch.randn(batch, seq, in_f, device=device)

    w_eff_bw, w_tilde_bw, gamma_bw = build_weight(in_f, out_f, BLOCK_SIZE, device)
    m_pos, m_neg, gamma_scalar = prepare_additive_masks(w_tilde_bw, gamma_bw, x.dtype)

    results = {}

    def f_dense():
        return F.linear(x, w_eff_bw)

    mean, std = time_cpu(f_dense, warmup, iters)
    y_ref = f_dense()
    results["dense_fp32"] = {"mean_ms": mean, "std_ms": std, "max_abs_err": 0.0, "rel_err": 0.0,
                              "speedup_vs_dense_fp32": 1.0}

    def f_add_ref():
        return ternary_additive_gemm(x, w_tilde_bw, gamma_bw)

    mean, std = time_cpu(f_add_ref, warmup, iters)
    y = f_add_ref()
    max_abs_err, rel_err = error_stats(y, y_ref)
    results["additive_reference"] = {"mean_ms": mean, "std_ms": std, "max_abs_err": max_abs_err,
                                      "rel_err": rel_err,
                                      "speedup_vs_dense_fp32": results["dense_fp32"]["mean_ms"] / mean}

    def f_add_cached():
        return ternary_additive_gemm_prepared(x, m_pos, m_neg, gamma_scalar)

    mean, std = time_cpu(f_add_cached, warmup, iters)
    y = f_add_cached()
    max_abs_err, rel_err = error_stats(y, y_ref)
    results["additive_cached"] = {"mean_ms": mean, "std_ms": std, "max_abs_err": max_abs_err,
                                   "rel_err": rel_err,
                                   "speedup_vs_dense_fp32": results["dense_fp32"]["mean_ms"] / mean}

    return results


def print_table(cuda_results):
    header = f"{'shape':<12}{'input':<10}{'path':<22}{'mean_ms':>10}{'std_ms':>10}{'speedup':>10}"
    print(header)
    print("-" * len(header))
    for shape_k, per_input in cuda_results.items():
        for input_k, paths in per_input.items():
            for path_name, stats in paths.items():
                if "error" in stats:
                    print(f"{shape_k:<12}{input_k:<10}{path_name:<22}{'ERROR':>10}{'':>10}{'':>10}")
                    continue
                print(f"{shape_k:<12}{input_k:<10}{path_name:<22}"
                      f"{stats['mean_ms']:>10.4f}{stats['std_ms']:>10.4f}{stats['speedup_vs_dense_fp32']:>10.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="5 warmup/5 timed iters, one shape only")
    parser.add_argument("--json", default="outputs/gemm-paths-benchmark.json")
    args = parser.parse_args()

    if args.smoke:
        layer_shapes = LAYER_SHAPES[:1]
        cuda_input_shapes = CUDA_INPUT_SHAPES[:1]
        cpu_input_shapes = CPU_INPUT_SHAPES[:1]
        warmup, iters = 5, 5
    else:
        layer_shapes = LAYER_SHAPES
        cuda_input_shapes = CUDA_INPUT_SHAPES
        cpu_input_shapes = CPU_INPUT_SHAPES
        warmup, iters = 10, 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)

    out = {
        "meta": {
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "cpu_threads": num_threads,
            "smoke": args.smoke,
        },
        "weight_storage_bytes": {},
        "cuda": {},
        "cpu": {},
    }

    for in_f, out_f in layer_shapes:
        sk = shape_key(in_f, out_f)
        out["weight_storage_bytes"][sk] = weight_storage_bytes(in_f, out_f)

        if device.type == "cuda":
            out["cuda"][sk] = {}
            for batch, seq in cuda_input_shapes:
                ik = input_key(batch, seq)
                out["cuda"][sk][ik] = bench_cuda_shape(in_f, out_f, batch, seq, device, warmup, iters)

        out["cpu"][sk] = {}
        for batch, seq in cpu_input_shapes:
            ik = input_key(batch, seq)
            out["cpu"][sk][ik] = bench_cpu_shape(in_f, out_f, batch, seq, warmup, iters)

    # Derived fields, not hard-coded verdicts.
    fastest_cuda_path_per_shape = {}
    additive_vs_dense_ratio = {}
    for sk, per_input in out["cuda"].items():
        for ik, paths in per_input.items():
            key = f"{sk}|{ik}"
            valid = {name: s for name, s in paths.items() if "error" not in s}
            if valid:
                fastest = min(valid.items(), key=lambda kv: kv[1]["mean_ms"])
                fastest_cuda_path_per_shape[key] = fastest[0]
            if "dense_fp32" in valid and "additive_cached" in valid:
                additive_vs_dense_ratio[key] = (
                    valid["dense_fp32"]["mean_ms"] / valid["additive_cached"]["mean_ms"]
                )

    out["fastest_cuda_path_per_shape"] = fastest_cuda_path_per_shape
    out["additive_vs_dense_ratio"] = additive_vs_dense_ratio

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)

    print_table(out["cuda"])
    print(f"\nSaved results to {args.json}")


if __name__ == "__main__":
    main()
