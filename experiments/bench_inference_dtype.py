"""
experiments/bench_inference_dtype.py: Full-model inference latency of BitRoute-135M under the
arithmetic paths that bench_gemm_paths.py found viable on this GPU.

Paths: fp32 additive (cached masks), fp32 dense, bf16 dense (autocast), and the pre-hardening
behaviour (re-quantize every call) reconstructed by clearing the eval cache before each forward.
Batch 1, all layers forced to EXECUTE, so the number is the pure arithmetic cost of the payload.
All numbers are measured.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from experiments.bitlinear import BitLinear
from experiments.bitroute_model import BitRouteForCausalLM, BitRouteConfig


def timed_forward(model, ids, force, warmup, iters, autocast_bf16=False, clear_cache=False):
    def one():
        if clear_cache:
            for m in model.modules():
                if isinstance(m, BitLinear):
                    m._eval_cache = None
        if autocast_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(ids, force_layer_actions=force)
        return model(ids, force_layer_actions=force)

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--json", default="outputs/inference-dtype-benchmark.json")
    args = ap.parse_args()

    torch.manual_seed(20260916)
    dev = torch.device("cuda")
    config = BitRouteConfig()  # the 135M reference shape: 12 layers, d 768, ff 2048, vocab 32000
    model = BitRouteForCausalLM(config).to(dev).eval()
    ids = torch.randint(0, config.vocab_size, (1, args.seq_len), device=dev)
    force = {l: "EXECUTE" for l in range(config.num_hidden_layers)}

    def set_path(additive):
        for m in model.modules():
            if isinstance(m, BitLinear):
                m.use_additive_gemm = additive
                m._eval_cache = None

    results = {}
    set_path(True)
    results["fp32_additive_requantize_each_call"] = timed_forward(model, ids, force, 5, args.iters, clear_cache=True)
    results["fp32_additive_cached"] = timed_forward(model, ids, force, 5, args.iters)
    set_path(False)
    results["fp32_dense_cached"] = timed_forward(model, ids, force, 5, args.iters)
    results["bf16_dense_cached_autocast"] = timed_forward(model, ids, force, 5, args.iters, autocast_bf16=True)
    set_path(True)
    results["bf16_additive_cached_autocast"] = timed_forward(model, ids, force, 5, args.iters, autocast_bf16=True)

    # Exactness of bf16 vs fp32 dense logits (max abs / max |ref|)
    set_path(False)
    with torch.no_grad():
        ref = model(ids, force_layer_actions=force).logits.float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, force_layer_actions=force).logits.float()
    rel_err = float((ref - out).abs().max() / ref.abs().max())

    base = results["fp32_additive_requantize_each_call"]
    out_json = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "model": "BitRoute-135M reference config, batch 1, all layers executed",
        "seq_len": args.seq_len,
        "iters": args.iters,
        "latency_ms": {k: round(v, 3) for k, v in results.items()},
        "speedup_vs_pre_hardening": {k: round(base / v, 2) for k, v in results.items()},
        "bf16_vs_fp32_logit_rel_err": rel_err,
        "label": "measured",
    }
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out_json, f, indent=2)
    for k, v in results.items():
        print(f"{k:40s} {v:8.3f} ms   x{base / v:.2f} vs pre-hardening")
    print(f"bf16 logit rel err vs fp32: {rel_err:.2e}")
    print(f"[+] written {args.json}")


if __name__ == "__main__":
    main()
