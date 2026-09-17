"""
experiments/train_bitroute_multihop.py: Training and comparative benchmark of BitRoute
with Flow-Reasoned Multi-Hop Routing on TinyStories.

Compares:
1. Learned Flow-Reasoned Multi-Hop Routing (FR-Router ON)
2. Forced Full Execution Control (FR-Router FORCED)
3. Matched-Rate Random Control (Bernoulli coin at identical parameter-weighted budget)
4. Direct comparison to Phase I greedy 1-hop baseline (outputs/bitroute-tinystories-lambda0.2.json)

Reports:
- Sublayer execution breakdown (Attention vs FFN execution per layer)
- True parameter-weighted compute savings
- Validation loss and perplexity across all arms
- Measured inference latency and bandwidth reduction on RTX 4060
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.bitroute_model import BitRouteConfig
from experiments.bitroute_multihop_model import (
    BitRouteMultiHopForCausalLM,
    BitRouteMultiHopOutput,
)
from experiments.flow_router import (
    SUBLAYER_ATTN,
    SUBLAYER_FFN,
)
from experiments.train_bitroute_tinystories import (
    load_tinystories_tokens,
    build_loaders,
    build_optimizer,
    make_lr_lambda,
    set_seed,
)


def evaluate_multihop(
    model: BitRouteMultiHopForCausalLM,
    val_loader: DataLoader,
    device: torch.device,
    target_budget: float,
    force_all_execute: bool = False,
    random_policy: Optional[float] = None,
    eval_batches: int = 0,
) -> Dict:
    """Evaluates the multi-hop model over validation batches."""
    model.eval()
    model.flow_router.random_policy = random_policy

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0

    num_layers = model.config.num_hidden_layers
    attn_exec_counts = [0] * num_layers
    attn_bypass_counts = [0] * num_layers
    ffn_exec_counts = [0] * num_layers
    ffn_bypass_counts = [0] * num_layers
    flow_steps_hist = [0] * (model.flow_router.max_flow_steps + 1)

    with torch.no_grad():
        for b_idx, (x, y) in enumerate(val_loader):
            if eval_batches > 0 and b_idx >= eval_batches:
                break
            x, y = x.to(device), y.to(device)
            B, S = x.shape

            out = model(
                x,
                target_budget=target_budget,
                force_all_execute=force_all_execute,
            )

            # Compute CE loss
            logits = out.logits.view(-1, out.logits.size(-1))
            targets = y.view(-1)
            loss = loss_fn(logits, targets)

            total_loss += loss.item() * (B * S)
            total_tokens += B * S
            total_batches += 1

            # Accumulate telemetry
            mask = out.routing_decision.execution_mask  # [B, L, 2]
            for l in range(num_layers):
                a_exec = int((mask[:, l, SUBLAYER_ATTN] > 0.5).sum().item())
                f_exec = int((mask[:, l, SUBLAYER_FFN] > 0.5).sum().item())
                attn_exec_counts[l] += a_exec
                attn_bypass_counts[l] += (B - a_exec)
                ffn_exec_counts[l] += f_exec
                ffn_bypass_counts[l] += (B - f_exec)

            for step in out.routing_decision.flow_steps_taken.tolist():
                if step < len(flow_steps_hist):
                    flow_steps_hist[step] += 1

    model.flow_router.random_policy = None

    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, mean_loss))

    total_attn_slots = sum(attn_exec_counts) + sum(attn_bypass_counts)
    total_ffn_slots = sum(ffn_exec_counts) + sum(ffn_bypass_counts)

    attn_exec_rate = sum(attn_exec_counts) / max(1, total_attn_slots)
    ffn_exec_rate = sum(ffn_exec_counts) / max(1, total_ffn_slots)
    weighted_param_cost = (attn_exec_rate + 2.0 * ffn_exec_rate) / 3.0

    sublayer_histogram = []
    for l in range(num_layers):
        sublayer_histogram.append({
            "layer": l,
            "attn_exec": attn_exec_counts[l],
            "attn_bypass": attn_bypass_counts[l],
            "attn_exec_rate": round(attn_exec_counts[l] / max(1, attn_exec_counts[l] + attn_bypass_counts[l]), 3),
            "ffn_exec": ffn_exec_counts[l],
            "ffn_bypass": ffn_bypass_counts[l],
            "ffn_exec_rate": round(ffn_exec_counts[l] / max(1, ffn_exec_counts[l] + ffn_bypass_counts[l]), 3),
        })

    return {
        "val_loss": round(mean_loss, 4),
        "val_perplexity": round(ppl, 2),
        "attn_exec_fraction": round(attn_exec_rate, 4),
        "ffn_exec_fraction": round(ffn_exec_rate, 4),
        "weighted_param_cost": round(weighted_param_cost, 4),
        "bandwidth_saving_ratio": round(1.0 - weighted_param_cost, 4),
        "flow_steps_distribution": flow_steps_hist,
        "sublayer_histogram": sublayer_histogram,
        "evaluated_tokens": total_tokens,
        "evaluated_batches": total_batches,
    }


def benchmark_multihop_latency(
    model: BitRouteMultiHopForCausalLM,
    device: torch.device,
    seq_len: int = 128,
    runs: int = 50,
) -> float:
    """Measures batch-1 per-forward latency in milliseconds on CUDA."""
    model.eval()
    x = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)

    # Warmup
    for _ in range(10):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return round((t1 - t0) * 1000.0 / runs, 3)


def train_multihop_model(
    config: BitRouteConfig,
    args: argparse.Namespace,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    device: torch.device,
) -> Dict:
    """Trains the Flow-Reasoned Multi-Hop BitRoute model."""
    print(f"\n---> Initializing BitRouteMultiHop (Seed {args.seed})...")
    set_seed(args.seed)

    model = BitRouteMultiHopForCausalLM(
        config=config,
        d_route=args.d_route,
        max_flow_steps=args.max_flow_steps,
        dt=args.flow_dt,
        eps_contraction=args.eps_contraction,
        default_target_budget=args.target_budget,
    ).to(device)

    train_loader, val_loader = build_loaders(
        train_tokens, val_tokens, args.seq_len, args.batch_size, args.seed
    )
    optimizer = build_optimizer(model, args.lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(args.train_steps, warmup_iters=100)
    )
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    step = 0
    t_start = time.perf_counter()
    train_iter = iter(train_loader)

    print(f"---> Training Flow-Reasoned Multi-Hop Router for {args.train_steps} steps (Target Budget: {args.target_budget:.2f})...")

    while step < args.train_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Multi-hop forward
        out = model(x, target_budget=args.target_budget)

        # Task loss (CrossEntropy)
        logits = out.logits.view(-1, out.logits.size(-1))
        targets = y.view(-1)
        ce_loss = loss_fn(logits, targets)

        # Routing losses: Budget penalty and Fixed-Point Forcing (FPF)
        # Warmup routing penalty over router_warmup_frac
        progress = min(1.0, step / max(1, int(args.train_steps * args.router_warmup_frac)))
        effective_lambda_budget = args.lambda_budget * progress
        effective_lambda_fpf = args.lambda_fpf * progress

        total_loss = ce_loss + effective_lambda_budget * out.total_budget_loss + effective_lambda_fpf * out.total_fpf_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        step += 1
        if step % 200 == 0 or step == args.train_steps:
            dt = time.perf_counter() - t_start
            tok_per_sec = (step * args.batch_size * args.seq_len) / max(0.001, dt)
            print(
                f"step {step:4d}/{args.train_steps} | ce_loss: {ce_loss.item():.4f} | "
                f"budget_loss: {out.total_budget_loss.item():.4e} | fpf_loss: {out.total_fpf_loss.item():.4e} | "
                f"cost: {out.routing_decision.weighted_param_cost:.3f} | {tok_per_sec:.0f} tok/s"
            )

    train_time_s = round(time.perf_counter() - t_start, 2)
    print(f"---> Training completed in {train_time_s}s.")

    # -----------------------------------------------------------------
    # Three-Way Evaluation Protocol:
    # 1. Learned Routing ON
    # 2. Forced Full Execution Control
    # 3. Matched-Rate Random Control
    # -----------------------------------------------------------------
    print("\n---> Evaluating Arm 1: Learned Flow-Reasoned Routing ON...")
    router_on_eval = evaluate_multihop(
        model, val_loader, device, args.target_budget, eval_batches=args.eval_batches
    )
    print(f"     Val Loss: {router_on_eval['val_loss']:.4f}, PPL: {router_on_eval['val_perplexity']:.2f}")
    print(f"     Attn Exec: {router_on_eval['attn_exec_fraction']*100:.1f}%, FFN Exec: {router_on_eval['ffn_exec_fraction']*100:.1f}%, Weighted Cost: {router_on_eval['weighted_param_cost']*100:.1f}%")

    print("\n---> Evaluating Arm 2: Forced Full Execution Control...")
    forced_eval = evaluate_multihop(
        model, val_loader, device, args.target_budget, force_all_execute=True, eval_batches=args.eval_batches
    )
    print(f"     Val Loss: {forced_eval['val_loss']:.4f}, PPL: {forced_eval['val_perplexity']:.2f}")

    print("\n---> Evaluating Arm 3: Matched-Rate Random Control...")
    # Coin router matching the exact empirical weighted cost
    matched_rate = router_on_eval["weighted_param_cost"]
    random_eval = evaluate_multihop(
        model, val_loader, device, args.target_budget, random_policy=matched_rate, eval_batches=args.eval_batches
    )
    print(f"     Val Loss: {random_eval['val_loss']:.4f}, PPL: {random_eval['val_perplexity']:.2f} (Coin Rate: {matched_rate:.3f})")

    # Latency benchmark
    print("\n---> Benchmarking batch-1 latency on RTX 4060...")
    latency_ms = benchmark_multihop_latency(model, device, seq_len=args.seq_len)
    print(f"     Batch-1 Seq-{args.seq_len} Latency: {latency_ms:.3f} ms")

    # Save checkpoint
    checkpoint_path = None
    if args.save_checkpoints:
        os.makedirs("outputs/checkpoints", exist_ok=True)
        checkpoint_path = "outputs/checkpoints/bitroute-multihop-flow.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"     Checkpoint saved to {checkpoint_path}")

    return {
        "train_time_s": train_time_s,
        "checkpoint_path": checkpoint_path,
        "latency_ms_batch1_seq128": latency_ms,
        "router_on": router_on_eval,
        "forced_execute": forced_eval,
        "random_matched": random_eval,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--target-budget", type=float, default=0.70, help="Target parameter execution ratio (e.g. 0.70)")
    parser.add_argument("--lambda-budget", type=float, default=0.50, help="Quadratic penalty weight for target budget")
    parser.add_argument("--lambda-fpf", type=float, default=0.05, help="Fixed-point forcing velocity penalty")
    parser.add_argument("--d-route", type=int, default=128)
    parser.add_argument("--max-flow-steps", type=int, default=3)
    parser.add_argument("--flow-dt", type=float, default=0.5)
    parser.add_argument("--eps-contraction", type=float, default=1e-3)
    parser.add_argument("--router-warmup-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--eval-batches", type=int, default=0, help="0 = full validation set")
    parser.add_argument("--save-checkpoints", action="store_true", default=True)
    parser.add_argument("--smoke", action="store_true", help="Quick sanity run (steps=30, eval-batches=5)")
    parser.add_argument("--json", type=str, default="outputs/bitroute-multihop-results.json")
    parser.add_argument("--reference-json", type=str, default="outputs/bitroute-tinystories-lambda0.2.json")
    args = parser.parse_args()

    if args.smoke:
        args.train_steps = 30
        args.eval_batches = 5
        args.batch_size = min(args.batch_size, 4)
        args.seq_len = min(args.seq_len, 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"BitRoute Flow-Reasoned Multi-Hop Routing Benchmark on TinyStories ({device})")
    print("=" * 80)

    train_tokens, val_tokens = load_tinystories_tokens()
    print(f"[+] Loaded {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens")

    config = BitRouteConfig(
        vocab_size=50257,
        hidden_size=384,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=6,
        max_position_embeddings=512,
        block_size=256,
        ternary=True,
    )

    arm_results = train_multihop_model(config, args, train_tokens, val_tokens, device)

    # Reference comparison to Phase I greedy baseline
    ref_findings = {}
    if args.reference_json and os.path.exists(args.reference_json):
        with open(args.reference_json) as f:
            ref_data = json.load(f)
            ref_arms = ref_data.get("arms", {})
            if "fp32" in ref_arms:
                ref_findings["ref_fp32_val_loss"] = ref_arms["fp32"]["eval"]["val_loss"]
            if "ternary" in ref_arms:
                ref_findings["ref_ternary_val_loss"] = ref_arms["ternary"]["eval"]["val_loss"]
            if "ternary_router" in ref_arms:
                ref_findings["ref_greedy_val_loss"] = ref_arms["ternary_router"]["router_on"]["val_loss"]
                ref_findings["ref_greedy_ppl"] = ref_arms["ternary_router"]["router_on"]["val_perplexity"]
                ref_findings["ref_greedy_bypass"] = ref_arms["ternary_router"]["router_on"]["layer_bypass_fraction"]

    router_on = arm_results["router_on"]
    forced = arm_results["forced_execute"]
    random_matched = arm_results["random_matched"]

    findings = {
        "router_on_minus_forced_execute_val_loss": round(router_on["val_loss"] - forced["val_loss"], 4),
        "router_on_minus_random_matched_val_loss": round(router_on["val_loss"] - random_matched["val_loss"], 4),
        "router_beats_random_control": bool(router_on["val_loss"] < random_matched["val_loss"]),
        "router_beats_forced_execute": bool(router_on["val_loss"] < forced["val_loss"]),
        "attn_exec_fraction": router_on["attn_exec_fraction"],
        "ffn_exec_fraction": router_on["ffn_exec_fraction"],
        "weighted_param_cost": router_on["weighted_param_cost"],
        "bandwidth_saving_ratio": router_on["bandwidth_saving_ratio"],
    }
    findings.update(ref_findings)

    results = {
        "status": "COMPLETED",
        "model_architecture": "BitRouteMultiHop (Decoupled 2-Hop Attention/FFN + FlowRoutingPlanner)",
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": args.seed,
        "cli": vars(args),
        "arm": arm_results,
        "findings": findings,
    }

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results successfully written to {args.json}")


if __name__ == "__main__":
    main()
