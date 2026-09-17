"""
experiments/benchmark_bitroute_inference.py: Greedy-generation inference benchmark for a trained
BitRoute checkpoint, logging per-token layer routing and measured throughput.

Hardening 2026-09-16: loads a real trained checkpoint (was: a freshly-initialized, untrained
model) so routing decisions and generated text reflect a router that has actually seen data.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer

from experiments.bitroute_model import BitRouteForCausalLM, BitRouteConfig

PROMPTS = [
    "Once upon a time, there was a little dog named Spot.",
    "The sun was bright and",
    "Lily wanted to",
]

DEFAULT_CHECKPOINT = "outputs/checkpoints/bitroute-tinystories-ternary_router.pt"
FALLBACK_CHECKPOINT = "outputs/checkpoints/bitroute-tinystories-ternary.pt"


def resolve_checkpoint(requested: str) -> str:
    if os.path.exists(requested):
        return requested
    if requested == DEFAULT_CHECKPOINT and os.path.exists(FALLBACK_CHECKPOINT):
        print(f"[!] {requested} not found; falling back to {FALLBACK_CHECKPOINT}")
        return FALLBACK_CHECKPOINT
    return ""


@torch.no_grad()
def generate_greedy(
    model: BitRouteForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> Dict:
    """Batch-1, full-recompute (no KV cache) greedy generation with per-token routing logs."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    curr_ids = input_ids.clone()
    prompt_len = curr_ids.shape[1]

    token_logs: List[Dict] = []
    t0 = time.time()
    for step in range(max_new_tokens):
        out = model(curr_ids)
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        curr_ids = torch.cat([curr_ids, next_token], dim=-1)
        token_logs.append({
            "step": step,
            "token": tokenizer.decode(next_token[0]),
            "token_id": int(next_token.item()),
            "executed_layers": out.executed_layers,
            "bypassed_layers": out.bypassed_layers,
            "exit_layer": out.exit_layer,
        })
    if device.type == "cuda":
        torch.cuda.synchronize()
    gen_time_s = time.time() - t0

    generated_text = tokenizer.decode(curr_ids[0])
    num_layers = len(model.layers)
    total_pairs = len(token_logs) * num_layers
    total_bypassed = sum(len(l["bypassed_layers"]) for l in token_logs)
    total_exited = sum(num_layers - l["exit_layer"] for l in token_logs if l["exit_layer"] is not None)

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "prompt_tokens": prompt_len,
        "new_tokens": max_new_tokens,
        "generation_time_s": round(gen_time_s, 4),  # measured
        "tokens_per_second": round(max_new_tokens / max(gen_time_s, 1e-6), 2),  # measured, batch 1, no KV cache
        "bypass_fraction": round(total_bypassed / max(total_pairs, 1), 4),  # measured
        "exit_fraction": round(total_exited / max(total_pairs, 1), 4),  # measured
        "token_logs": token_logs,
    }


def run(args: argparse.Namespace) -> Dict:
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    if not checkpoint_path:
        result = {"status": "NO_CHECKPOINT"}
        print(f"[!] No checkpoint found at {args.checkpoint} or fallback {FALLBACK_CHECKPOINT}. "
              f"Run experiments/train_bitroute_tinystories.py first.")
        print(json.dumps(result))
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/bitroute-inference-benchmark.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = BitRouteConfig(**ckpt["config"])
    model = BitRouteForCausalLM(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"================================================================================")
    print(f"BitRoute Inference Benchmark: checkpoint={checkpoint_path} arm={ckpt.get('arm')} ({device})")
    print(f"Measured, batch 1, no KV cache (full recompute per token).")
    print(f"================================================================================")

    prompt_results = []
    for prompt in PROMPTS:
        r = generate_greedy(model, tokenizer, prompt, args.max_new_tokens, device)
        prompt_results.append(r)
        print(f"\nPrompt: {prompt!r}")
        print(f"  Generated: {r['generated_text']!r}")
        print(f"  tokens/s: {r['tokens_per_second']} | bypass_fraction: {r['bypass_fraction']} | exit_fraction: {r['exit_fraction']}")

    overall_bypass = sum(r["bypass_fraction"] * r["new_tokens"] for r in prompt_results) / max(
        sum(r["new_tokens"] for r in prompt_results), 1
    )
    overall_exit = sum(r["exit_fraction"] * r["new_tokens"] for r in prompt_results) / max(
        sum(r["new_tokens"] for r in prompt_results), 1
    )
    overall_tok_per_s = sum(r["new_tokens"] for r in prompt_results) / max(
        sum(r["generation_time_s"] for r in prompt_results), 1e-6
    )

    results = {
        "status": "COMPLETED",
        "checkpoint": checkpoint_path,
        "arm": ckpt.get("arm"),
        "train_steps": ckpt.get("train_steps"),
        "seed": ckpt.get("seed"),
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
        "note": "measured, batch 1, no KV cache, full recompute per generated token",
        "overall_bypass_fraction": round(overall_bypass, 4),  # measured
        "overall_exit_fraction": round(overall_exit, 4),  # measured
        "overall_tokens_per_second": round(overall_tok_per_s, 2),  # measured
        "prompts": prompt_results,
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/bitroute-inference-benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results written to outputs/bitroute-inference-benchmark.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()
    run(args)
