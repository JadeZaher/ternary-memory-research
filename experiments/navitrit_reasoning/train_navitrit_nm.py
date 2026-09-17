"""
experiments/navitrit_reasoning/train_navitrit_nm.py: 2-Stage Adaptive Training for NaviTrit-NM.

Gate 14 (Track D-3): Non-Monotonic Training & Reasoning Track.
Trains NaviTrit-NM on TinyStories with:
1. Stage 1 (Backbone & Hop Modulation Adaptation):
   Pre-trains ternary backbone, reasoning core, and hop modulation across diverse graph walks
   to establish stable, non-divergent representations.
2. Stage 2 (Joint Navigation & Attractor Reasoning Alignment):
   End-to-end policy gradient with Hidden-State Contraction Regularization (L_state_fpf),
   Attention Diversity, and Coherence Certification.
3. Comparative Evaluations & Ablations:
   - NaviTrit-NM (Full Model)
   - Ablation: Monotonic Baseline
   - Ablation: Random Walk Control
4. Output ledger: outputs/navitrit-nm-results.json
5. Saved Checkpoint: outputs/checkpoints/navitrit-nm-trained.pt
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.navitrit_reasoning.navitrit_nm_model import (
    NaviTritNMConfig,
    NaviTritNMForCausalLM,
    NaviTritNMOutput,
    TRANSITION_FORWARD,
    TRANSITION_BACKWARD,
    TRANSITION_SELF_LOOP,
    TRANSITION_REASONING,
    TRANSITION_EXIT,
)
from experiments.train_bitroute_tinystories import (
    load_tinystories_tokens,
    build_loaders,
    build_optimizer,
    make_lr_lambda,
    set_seed,
)


def evaluate_navitrit_nm(
    model: NaviTritNMForCausalLM,
    val_loader: DataLoader,
    device: torch.device,
    force_nodes: Optional[List[int]] = None,
    eval_batches: int = 0,
) -> Dict:
    """Evaluates NaviTrit-NM over validation batches."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0

    total_hops_accum = 0
    fwd_hops_accum = 0
    back_hops_accum = 0
    loop_hops_accum = 0
    reason_hops_accum = 0
    early_exits_count = 0
    attn_module_hops = 0

    num_nodes = model.controller.num_nodes
    node_visitations = [0] * num_nodes
    sample_trajectories = []

    with torch.no_grad():
        for b_idx, (x, y) in enumerate(val_loader):
            if eval_batches > 0 and b_idx >= eval_batches:
                break
            x, y = x.to(device), y.to(device)

            out: NaviTritNMOutput = model(x, force_nodes=force_nodes)
            logits = out.logits
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            total_batches += 1

            total_hops_accum += out.total_hops_taken
            fwd_hops_accum += out.forward_hops_count
            back_hops_accum += out.backward_hops_count
            loop_hops_accum += out.self_loops_count
            reason_hops_accum += out.reasoning_hops_count
            if out.early_exit_taken:
                early_exits_count += 1

            for step in out.trajectory:
                node_visitations[step.selected_node] += 1
                if step.selected_node < 2 * model.num_layers and (step.selected_node % 2) == 0:
                    attn_module_hops += 1

            if b_idx < 5:
                nodes = [step.selected_node for step in out.trajectory]
                names = [step.node_name for step in out.trajectory]
                sample_trajectories.append({
                    "batch_idx": b_idx,
                    "hops": out.total_hops_taken,
                    "trajectory_nodes": nodes,
                    "trajectory_names": names,
                    "mean_coherence": round(out.terminal_coherence, 4),
                    "early_exit": out.early_exit_taken,
                })

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    n_seq = max(1, total_batches)

    total_valid_hops = max(1, sum(node_visitations))
    attn_ratio = float(attn_module_hops) / total_valid_hops

    return {
        "val_loss": round(avg_loss, 4),
        "val_perplexity": round(ppl, 2),
        "avg_hops_per_sequence": round(float(total_hops_accum) / n_seq, 2),
        "avg_forward_hops": round(float(fwd_hops_accum) / n_seq, 2),
        "avg_backward_hops": round(float(back_hops_accum) / n_seq, 2),
        "avg_self_loops": round(float(loop_hops_accum) / n_seq, 2),
        "avg_reasoning_hops": round(float(reason_hops_accum) / n_seq, 2),
        "early_exits_count": early_exits_count,
        "attention_module_ratio": round(attn_ratio, 4),
        "node_visitation_distribution": node_visitations,
        "sample_trajectories": sample_trajectories,
        "evaluated_tokens": total_tokens,
        "evaluated_batches": total_batches,
    }


def train_navitrit_nm(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load TinyStories tokens
    cache_path = os.path.join(os.path.dirname(__file__), "../../data/tinystories_tokens.pt")
    train_tokens, val_tokens = load_tinystories_tokens(cache_path)
    train_loader, val_loader = build_loaders(
        train_tokens, val_tokens, args.seq_len, args.batch_size, args.seed
    )

    # Instantiate NaviTrit-NM Config
    config = NaviTritNMConfig(
        vocab_size=50257,
        hidden_size=384,
        intermediate_size=1024,
        num_layers=args.num_layers,
        num_attention_heads=6,
        max_hops=args.max_hops,
        d_nav_route=args.d_nav_route,
        flow_dt=args.flow_dt,
        use_hop_modulation=True,
        use_reasoning_core=True,
        reasoning_sub_steps=args.reasoning_sub_steps,
        gumbel_tau=1.0,
        gumbel_hard=False,
        lambda_hop=args.lambda_hop,
        lambda_fpf=args.lambda_fpf,
        lambda_state_fpf=args.lambda_state_fpf,
        lambda_attn_div=args.lambda_attn_div,
        lambda_layer_entropy=args.lambda_layer_entropy,
        lambda_cohere=args.lambda_cohere,
        min_attn_ratio=args.min_attn_ratio,
        tau_cohere=args.tau_cohere,
    )

    model = NaviTritNMForCausalLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"NaviTrit-NM instantiated: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    optimizer = build_optimizer(model, lr=args.lr)
    lr_lambda = make_lr_lambda(args.train_steps, warmup_iters=100)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    stage1_steps = args.stage1_steps
    stage2_steps = args.train_steps - stage1_steps
    print(f"\nStarting 2-Stage Training:")
    print(f"  Stage 1 (Backbone & Hop Modulation Adaptation): {stage1_steps} steps")
    print(f"  Stage 2 (Joint Navigation & Attractor Reasoning): {stage2_steps} steps")

    train_iter = iter(train_loader)
    start_time = time.time()
    reasoning_node_idx = 2 * config.num_layers

    # Canonical paths for Stage 1 diverse exposure
    diverse_walks = [
        [0, 1, 2, 3, 4, 5],                               # Pure monotonic
        [1, 0, 3, 2, 5, 4],                               # Inverted intra-layer
        [1, 4, 4, 2, 5, reasoning_node_idx],              # Non-monotonic with reasoning visit
        [0, 2, reasoning_node_idx, 4, 5, 3],              # Early reasoning injection
        [1, reasoning_node_idx, reasoning_node_idx, 4, 2, 5], # Double reasoning deliberation
    ]

    model.train()
    for step in range(1, args.train_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if step <= stage1_steps:
            # Stage 1: Force diverse paths to expose stationary tiles to hop modulation
            walk = diverse_walks[step % len(diverse_walks)]
            out: NaviTritNMOutput = model(x, force_nodes=walk)
            lm_loss = loss_fn(out.logits.view(-1, config.vocab_size), y.view(-1))
            total_loss = lm_loss + out.loss_state_fpf
        else:
            # Stage 2: Learned autonomous navigation with all regularizers active
            out: NaviTritNMOutput = model(x)
            lm_loss = loss_fn(out.logits.view(-1, config.vocab_size), y.view(-1))
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 250 == 0 or step == args.train_steps:
            elapsed = time.time() - start_time
            stage_name = "Stage 1 (Adaptation)" if step <= stage1_steps else "Stage 2 (Joint Alignment)"
            print(
                f"[{stage_name} Step {step:4d}/{args.train_steps}] "
                f"Loss: {lm_loss.item():.4f} | "
                f"StateFPF: {out.loss_state_fpf.item():.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    train_duration = time.time() - start_time
    print(f"\nTraining completed in {train_duration:.2f}s ({train_duration/60:.1f} min)")

    # Save trained checkpoint
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "../../outputs/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "navitrit-nm-trained.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "args": vars(args),
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Comprehensive Comparative Benchmark
    print("\nRunning Evaluation Arms...")
    print("1. Evaluating NaviTrit-NM (Learned Autonomous Navigation)...")
    eval_learned = evaluate_navitrit_nm(model, val_loader, device, eval_batches=args.eval_batches)
    print(f"   NaviTrit-NM Val Loss: {eval_learned['val_loss']} (PPL {eval_learned['val_perplexity']})")
    print(f"   Reasoning Hops / Seq: {eval_learned['avg_reasoning_hops']}")
    print(f"   Attention Ratio: {eval_learned['attention_module_ratio']*100:.1f}%")

    print("2. Evaluating Monotonic Baseline Control (0 -> 1 -> 2 -> 3 -> 4 -> 5)...")
    eval_monotonic = evaluate_navitrit_nm(
        model, val_loader, device,
        force_nodes=[0, 1, 2, 3, 4, 5],
        eval_batches=args.eval_batches
    )
    print(f"   Monotonic Baseline Val Loss: {eval_monotonic['val_loss']} (PPL {eval_monotonic['val_perplexity']})")

    results_data = {
        "status": "COMPLETED",
        "model_architecture": "NaviTrit-NM (Non-Monotonic Graph + Hop Modulation + Recurrent Reasoning Core)",
        "track": "Gate 14 (Track D-3)",
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": args.seed,
        "cli": vars(args),
        "total_parameters": total_params,
        "train_time_s": round(train_duration, 2),
        "checkpoint_path": checkpoint_path,
        "arms": {
            "navitrit_nm_learned": eval_learned,
            "monotonic_baseline": eval_monotonic,
        },
        "findings": {
            "navitrit_nm_minus_monotonic_val_loss": round(eval_learned["val_loss"] - eval_monotonic["val_loss"], 4),
            "navitrit_nm_beats_monotonic": eval_learned["val_loss"] < eval_monotonic["val_loss"],
            "attention_ratio": eval_learned["attention_module_ratio"],
            "reasoning_hops_per_sequence": eval_learned["avg_reasoning_hops"],
            "has_backward_hops": eval_learned["avg_backward_hops"] > 0,
            "has_self_loops": eval_learned["avg_self_loops"] > 0,
            "has_reasoning_hops": eval_learned["avg_reasoning_hops"] > 0,
        }
    }

    out_json = os.path.join(os.path.dirname(__file__), "../../outputs/navitrit-nm-results.json")
    with open(out_json, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved results ledger to {out_json}")
    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NaviTrit-NM on TinyStories")
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--stage1_steps", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0006)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--max_hops", type=int, default=6)
    parser.add_argument("--reasoning_sub_steps", type=int, default=3)
    parser.add_argument("--d_nav_route", type=int, default=128)
    parser.add_argument("--flow_dt", type=float, default=0.5)
    parser.add_argument("--lambda_hop", type=float, default=0.02)
    parser.add_argument("--lambda_fpf", type=float, default=0.01)
    parser.add_argument("--lambda_state_fpf", type=float, default=0.05)
    parser.add_argument("--lambda_attn_div", type=float, default=0.5)
    parser.add_argument("--lambda_layer_entropy", type=float, default=0.2)
    parser.add_argument("--lambda_cohere", type=float, default=0.1)
    parser.add_argument("--min_attn_ratio", type=float, default=0.40)
    parser.add_argument("--tau_cohere", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--eval_batches", type=int, default=0)
    args = parser.parse_args()

    train_navitrit_nm(args)
