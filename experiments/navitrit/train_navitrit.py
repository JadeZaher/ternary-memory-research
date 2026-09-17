"""
experiments/navitrit/train_navitrit.py: Training and 3-way comparative benchmark of NaviTrit on TinyStories.

Compares:
1. NaviTrit (Learned Non-Monotonic Graph Navigation with Flow Controller)
2. Monotonic Baseline (Fixed sequential 0 -> 1 -> ... -> 2L-1 pipeline)
3. Matched Random Walk Control (Random uniform node sampling on G)

Measures:
- Frequency of Forward hops, Backward hops, Self-loops, and Early exits
- Trajectory convergence and fixed-point forcing stability
- Validation loss, perplexity, parameter-weighted memory savings, and latency
- Output ledger: outputs/navitrit-results.json
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.navitrit.navitrit_model import (
    NaviTritConfig,
    NaviTritForCausalLM,
    NaviTritOutput,
    TRANSITION_FORWARD,
    TRANSITION_BACKWARD,
    TRANSITION_SELF_LOOP,
    TRANSITION_EXIT,
)
from experiments.train_bitroute_tinystories import (
    load_tinystories_tokens,
    build_loaders,
    build_optimizer,
    make_lr_lambda,
    set_seed,
)


def evaluate_navitrit(
    model: NaviTritForCausalLM,
    val_loader: DataLoader,
    device: torch.device,
    force_monotonic: bool = False,
    random_policy: Optional[bool] = None,
    eval_batches: int = 0,
) -> Dict:
    """Evaluates NaviTrit over validation batches and aggregates trajectory statistics."""
    model.eval()
    model.controller.random_policy = random_policy

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0

    total_hops_accum = 0
    fwd_hops_accum = 0
    back_hops_accum = 0
    loop_hops_accum = 0
    exit_hops_accum = 0
    node_visitation_counts = [0] * model.graph.total_nodes

    sample_trajectories = []

    with torch.no_grad():
        for b_idx, (x, y) in enumerate(val_loader):
            if eval_batches > 0 and b_idx >= eval_batches:
                break
            x, y = x.to(device), y.to(device)
            B, S = x.shape

            out = model(x, force_monotonic=force_monotonic)

            # Compute CE loss
            logits = out.logits.view(-1, out.logits.size(-1))
            targets = y.view(-1)
            loss = loss_fn(logits, targets)

            total_loss += loss.item() * (B * S)
            total_tokens += B * S
            total_batches += 1

            # Accumulate telemetry
            total_hops_accum += out.total_hops_taken * B
            fwd_hops_accum += out.forward_hops_count * B
            back_hops_accum += out.backward_hops_count * B
            loop_hops_accum += out.self_loops_count * B
            exit_hops_accum += out.early_exits_count * B

            for node in out.trajectory_nodes:
                if node < len(node_visitation_counts):
                    node_visitation_counts[node] += B

            if len(sample_trajectories) < 5:
                sample_trajectories.append({
                    "batch_idx": b_idx,
                    "hops": out.total_hops_taken,
                    "trajectory_nodes": out.trajectory_nodes,
                    "attn_ratio": out.attn_ratio,
                    "layer_entropy": out.layer_entropy,
                    "mean_coherence": out.mean_coherence,
                    "exit_reason": out.exit_reason,
                })

    model.controller.random_policy = None

    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, mean_loss))
    total_samples = total_batches * val_loader.batch_size if total_batches > 0 else 1

    avg_hops = total_hops_accum / max(1, total_samples)
    avg_fwd = fwd_hops_accum / max(1, total_samples)
    avg_back = back_hops_accum / max(1, total_samples)
    avg_loops = loop_hops_accum / max(1, total_samples)
    avg_exits = exit_hops_accum / max(1, total_samples)

    # Compute attention vs FFN ratio across all visited nodes
    attn_nodes = [2 * l for l in range(model.config.num_layers)]
    ffn_nodes = [2 * l + 1 for l in range(model.config.num_layers)]
    attn_visits = sum(node_visitation_counts[n] for n in attn_nodes if n < len(node_visitation_counts))
    ffn_visits = sum(node_visitation_counts[n] for n in ffn_nodes if n < len(node_visitation_counts))
    total_module_visits = attn_visits + ffn_visits
    eval_attn_ratio = round(attn_visits / max(1, total_module_visits), 4)

    return {
        "val_loss": round(mean_loss, 4),
        "val_perplexity": round(ppl, 2),
        "avg_hops_per_sequence": round(avg_hops, 2),
        "avg_forward_hops": round(avg_fwd, 2),
        "avg_backward_hops": round(avg_back, 2),
        "avg_self_loops": round(avg_loops, 2),
        "avg_early_exits": round(avg_exits, 2),
        "attention_module_ratio": eval_attn_ratio,
        "node_visitation_distribution": node_visitation_counts,
        "sample_trajectories": sample_trajectories,
        "evaluated_tokens": total_tokens,
        "evaluated_batches": total_batches,
    }



def benchmark_navitrit_latency(
    model: NaviTritForCausalLM,
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


def train_navitrit_model(
    config: NaviTritConfig,
    args: argparse.Namespace,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    device: torch.device,
) -> Dict:
    """Trains NaviTrit on TinyStories with non-monotonic graph navigation."""
    print(f"\n---> Initializing NaviTrit (Seed {args.seed})...")
    set_seed(args.seed)

    model = NaviTritForCausalLM(config).to(device)

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

    print(f"---> Training NaviTrit for {args.train_steps} steps (Max Hops: {config.max_hops})...")

    while step < args.train_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Non-monotonic forward pass
        out = model(x)

        # CrossEntropy loss
        logits = out.logits.view(-1, out.logits.size(-1))
        targets = y.view(-1)
        ce_loss = loss_fn(logits, targets)

        # Total navigation loss: CE + Hop cost + FPF velocity penalty + Diversity penalty + Coherence loss
        progress = min(1.0, step / max(1, int(args.train_steps * args.router_warmup_frac)))
        effective_lambda_hop = config.lambda_hop * progress
        effective_lambda_fpf = config.lambda_fpf * progress

        total_loss = (
            ce_loss
            + effective_lambda_hop * out.nav_budget_loss
            + effective_lambda_fpf * out.fpf_loss
            + out.diversity_loss
            + out.coherence_loss
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        step += 1
        if step % 200 == 0 or step == args.train_steps:
            dt = time.perf_counter() - t_start
            tok_per_sec = (step * args.batch_size * args.seq_len) / max(0.001, dt)
            print(
                f"step {step:4d}/{args.train_steps} | ce: {ce_loss.item():.4f} | "
                f"div: {out.diversity_loss.item():.4e} | coh: {out.coherence_loss.item():.4e} | "
                f"attn%: {out.attn_ratio * 100:.1f}% | ent: {out.layer_entropy:.2f} | "
                f"hops: {out.total_hops_taken} (Fwd:{out.forward_hops_count}, Back:{out.backward_hops_count}, Loop:{out.self_loops_count}) | "
                f"{tok_per_sec:.0f} tok/s"
            )

    train_time_s = round(time.perf_counter() - t_start, 2)
    print(f"---> Training completed in {train_time_s}s.")

    # -----------------------------------------------------------------
    # Three-Way Comparative Evaluation:
    # 1. NaviTrit (Learned Graph Navigation ON)
    # 2. Monotonic Sequential Baseline (Fixed 0 -> 1 -> ... -> 2L-1)
    # 3. Matched Random Walk Control
    # -----------------------------------------------------------------
    print("\n---> Evaluating Arm 1: NaviTrit (Learned Non-Monotonic Graph Navigation)...")
    navitrit_eval = evaluate_navitrit(model, val_loader, device, eval_batches=args.eval_batches)
    print(f"     Val Loss: {navitrit_eval['val_loss']:.4f}, PPL: {navitrit_eval['val_perplexity']:.2f}")
    print(f"     Avg Hops: {navitrit_eval['avg_hops_per_sequence']:.2f} (Fwd: {navitrit_eval['avg_forward_hops']:.2f}, Back: {navitrit_eval['avg_backward_hops']:.2f}, Loop: {navitrit_eval['avg_self_loops']:.2f})")
    print(f"     Attention Ratio: {navitrit_eval['attention_module_ratio']*100:.1f}%")
    print(f"     Sample Trajectory: {navitrit_eval['sample_trajectories'][0]['trajectory_nodes']}")

    print("\n---> Evaluating Arm 2: Monotonic Sequential Baseline (Fixed Feedforward Pipeline)...")
    monotonic_eval = evaluate_navitrit(model, val_loader, device, force_monotonic=True, eval_batches=args.eval_batches)
    print(f"     Val Loss: {monotonic_eval['val_loss']:.4f}, PPL: {monotonic_eval['val_perplexity']:.2f}")
    print(f"     Trajectory: {monotonic_eval['sample_trajectories'][0]['trajectory_nodes']}")

    print("\n---> Evaluating Arm 3: Matched Random Walk Control...")
    random_eval = evaluate_navitrit(model, val_loader, device, random_policy=True, eval_batches=args.eval_batches)
    print(f"     Val Loss: {random_eval['val_loss']:.4f}, PPL: {random_eval['val_perplexity']:.2f}")

    # Latency benchmark
    print("\n---> Benchmarking batch-1 latency on RTX 4060...")
    latency_ms = benchmark_navitrit_latency(model, device, seq_len=args.seq_len)
    print(f"     Batch-1 Seq-{args.seq_len} Latency: {latency_ms:.3f} ms")

    # Save checkpoint
    checkpoint_path = None
    if args.save_checkpoints:
        os.makedirs("outputs/checkpoints", exist_ok=True)
        checkpoint_path = "outputs/checkpoints/navitrit-flow-hardened.pt"
        torch.save(model.state_dict(), checkpoint_path)
        torch.save(model.state_dict(), "outputs/checkpoints/navitrit-flow.pt")
        print(f"     Checkpoints saved to {checkpoint_path} and outputs/checkpoints/navitrit-flow.pt")

    return {
        "train_time_s": train_time_s,
        "checkpoint_path": checkpoint_path,
        "latency_ms_batch1_seq128": latency_ms,
        "navitrit_learned": navitrit_eval,
        "monotonic_baseline": monotonic_eval,
        "random_walk_control": random_eval,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--num-layers", type=int, default=3, help="Number of physical layer pairs (3 Attn + 3 FFN = 6 modules)")
    parser.add_argument("--max-hops", type=int, default=6, help="Maximum trajectory length T_max")
    parser.add_argument("--lambda-hop", type=float, default=0.02, help="Cost penalty per hop")
    parser.add_argument("--lambda-fpf", type=float, default=0.01, help="Fixed-point forcing velocity penalty")
    parser.add_argument("--lambda-attn-div", type=float, default=0.5, help="Attention diversity penalty")
    parser.add_argument("--lambda-layer-entropy", type=float, default=0.2, help="Layer entropy penalty")
    parser.add_argument("--lambda-cohere", type=float, default=0.1, help="Semantic coherence loss weight")
    parser.add_argument("--min-attn-ratio", type=float, default=0.40, help="Minimum target attention fraction")
    parser.add_argument("--tau-cohere", type=float, default=0.85, help="Coherence threshold for certified early exit")
    parser.add_argument("--d-nav-route", type=int, default=128)
    parser.add_argument("--router-warmup-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--eval-batches", type=int, default=0, help="0 = full validation set")
    parser.add_argument("--save-checkpoints", action="store_true", default=True)
    parser.add_argument("--smoke", action="store_true", help="Quick sanity run (steps=30, eval-batches=5)")
    parser.add_argument("--json", type=str, default="outputs/navitrit-results.json")
    args = parser.parse_args()

    if args.smoke:
        args.train_steps = 30
        args.eval_batches = 5
        args.batch_size = min(args.batch_size, 4)
        args.seq_len = min(args.seq_len, 32)
        args.max_hops = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"NaviTrit: Non-Monotonic Token Navigation Benchmark on TinyStories ({device})")
    print("=" * 80)

    train_tokens, val_tokens = load_tinystories_tokens()
    print(f"[+] Loaded {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens")

    config = NaviTritConfig(
        vocab_size=50257,
        hidden_size=384,
        intermediate_size=1024,
        num_layers=args.num_layers,
        num_attention_heads=6,
        max_position_embeddings=512,
        block_size=256,
        max_hops=args.max_hops,
        d_nav_route=args.d_nav_route,
        lambda_hop=args.lambda_hop,
        lambda_fpf=args.lambda_fpf,
        lambda_attn_div=args.lambda_attn_div,
        lambda_layer_entropy=args.lambda_layer_entropy,
        lambda_cohere=args.lambda_cohere,
        min_attn_ratio=args.min_attn_ratio,
        tau_cohere=args.tau_cohere,
        ternary=True,
    )

    arm_results = train_navitrit_model(config, args, train_tokens, val_tokens, device)

    learned = arm_results["navitrit_learned"]
    mono = arm_results["monotonic_baseline"]
    rand = arm_results["random_walk_control"]

    findings = {
        "navitrit_minus_monotonic_val_loss": round(learned["val_loss"] - mono["val_loss"], 4),
        "navitrit_minus_random_val_loss": round(learned["val_loss"] - rand["val_loss"], 4),
        "navitrit_beats_random_control": bool(learned["val_loss"] < rand["val_loss"]),
        "navitrit_beats_monotonic_baseline": bool(learned["val_loss"] < mono["val_loss"]),
        "avg_hops_per_sequence": learned["avg_hops_per_sequence"],
        "attention_module_ratio": learned.get("attention_module_ratio", 0.0),
        "has_attention_diversity": bool(learned.get("attention_module_ratio", 0.0) >= 0.35),
        "has_backward_hops": bool(learned["avg_backward_hops"] > 0),
        "has_self_loops": bool(learned["avg_self_loops"] > 0),
        "has_early_exits": bool(learned["avg_early_exits"] > 0),
    }


    results = {
        "status": "COMPLETED",
        "model_architecture": "NaviTrit (Stationary Module Graph G + Flow-Reasoned Navigation Controller)",
        "track": "Track D (Paper 4: Non-Monotonic Token Navigation)",
        "gate": "Gate 13",
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": args.seed,
        "cli": vars(args),
        "arms": arm_results,
        "findings": findings,
    }

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results successfully written to {args.json}")


if __name__ == "__main__":
    main()
