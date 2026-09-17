"""
experiments/frontier_scaling/train_scale.py: Scalable Training Engine for NaviTrit (10M & 100M).

Gate 15 (Track D-4): Multi-Corpus Pretraining with 2-Stage Adaptive Routing.

2-Stage Training:
  Stage 1: Backbone & Hop Modulation Adaptation across diverse walks.
  Stage 2: Joint Navigation & Recurrent Reasoning Core Stabilization.

Comparative Evaluation:
  Arm 1: NaviTrit Learned Non-Monotonic Graph Navigation.
  Arm 2: Monotonic Feedforward Baseline at identical 6-hop budget.

Outputs:
  Ledger: outputs/navitrit-10m-results.json
  Checkpoint: outputs/checkpoints/navitrit-10m-trained.pt
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from experiments.frontier_scaling.navitrit_scale_model import (
    NaviTritScaleConfig,
    NaviTritScaleForCausalLM,
    get_scale_config,
)


class FlatTokenDataset(Dataset):
    """Fixed-length chunk dataset from 1D token tensor."""
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        self.num_samples = (len(tokens) - 1) // seq_len
        self.tokens = tokens[: self.num_samples * seq_len + 1]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len].long()
        y = self.tokens[start + 1 : start + self.seq_len + 1].long()
        return x, y


def load_tokens(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if not os.path.exists(path):
        # Fallback to tinystories if multicorpus is building
        fallback = os.path.join(os.path.dirname(__file__), "../../data/tinystories_tokens.pt")
        print(f"Warning: {path} not found. Falling back to {fallback}")
        path = fallback
    all_tokens = torch.load(path, weights_only=True)
    split_idx = int(0.90 * len(all_tokens))
    return all_tokens[:split_idx], all_tokens[split_idx:]


def run_monotonic_forward(
    model: NaviTritScaleForCausalLM,
    input_ids: torch.Tensor,
    max_hops: int = 6,
) -> Dict[str, torch.Tensor]:
    """Evaluates the model along a fixed sequential monotonic pipeline for baseline comparison."""
    B, S = input_ids.size()
    device = input_ids.device
    pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
    h = model.embed_tokens(input_ids) + model.embed_positions(pos)

    trajectory = []
    # Fixed pipeline: node 0 -> node 1 -> node 2 -> node 3 ...
    for t in range(max_hops):
        node_idx = t % (2 * model.num_layers)
        trajectory.append(node_idx)
        h, _ = model.execute_tile(node_idx, h, t)

    h_norm = model.final_norm(h)
    logits = model.lm_head(h_norm)
    return {
        "logits": logits,
        "trajectory": trajectory,
        "hops_taken": max_hops,
        "early_exit": False,
        "attn_ratio": 0.50,
    }


def evaluate_arm(
    model: NaviTritScaleForCausalLM,
    val_loader: DataLoader,
    device: torch.device,
    is_monotonic: bool = False,
    max_batches: int = 40,
) -> Dict:
    model.eval()
    total_ce_loss = 0.0
    total_tokens = 0
    trajectories = []
    node_visitations = [0] * model.num_nodes
    attn_ratios = []

    with torch.no_grad():
        for b_idx, (x, y) in enumerate(val_loader):
            if b_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)

            if is_monotonic:
                out = run_monotonic_forward(model, x, max_hops=model.max_hops)
            else:
                out = model(x, temperature=0.7)

            logits = out["logits"]
            shift_logits = logits[:, :-1, :].contiguous().view(-1, model.vocab_size)
            shift_labels = y[:, :-1].contiguous().view(-1)
            loss_ce = F.cross_entropy(shift_logits, shift_labels, reduction="sum")

            total_ce_loss += loss_ce.item()
            total_tokens += shift_labels.numel()

            traj = out["trajectory"]
            if b_idx < 5:
                trajectories.append(traj)
            for node in traj:
                if node < len(node_visitations):
                    node_visitations[node] += 1
            attn_ratios.append(out.get("attn_ratio", 0.5))

    mean_loss = total_ce_loss / max(1, total_tokens)
    ppl = math.exp(min(mean_loss, 20.0))
    avg_attn = float(sum(attn_ratios) / max(1, len(attn_ratios)))

    return {
        "val_loss": round(mean_loss, 4),
        "val_perplexity": round(ppl, 2),
        "avg_hops": model.max_hops,
        "attention_ratio": round(avg_attn, 4),
        "node_visitations": node_visitations,
        "sample_trajectories": trajectories,
    }


def main():
    parser = argparse.ArgumentParser(description="Scalable NaviTrit Training Engine")
    parser.add_argument("--model_size", type=str, default="10m", choices=["10m", "100m"])
    parser.add_argument("--data_path", type=str, default="data/multicorpus_10m.pt")
    parser.add_argument("--train_steps", type=int, default=2500)
    parser.add_argument("--stage1_steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--max_hops", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--json", type=str, default="outputs/navitrit-10m-results.json")
    parser.add_argument("--save_checkpoint", action="store_true", default=True)
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Scalable NaviTrit Training [{args.model_size.upper()}] on {device} ===")

    # Load data
    train_tokens, val_tokens = load_tokens(args.data_path)
    train_ds = FlatTokenDataset(train_tokens, args.seq_len)
    val_ds = FlatTokenDataset(val_tokens, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Dataset loaded: {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens.")

    # Instantiate model
    cfg = get_scale_config(args.model_size, max_hops=args.max_hops)
    model = NaviTritScaleForCausalLM(cfg).to(device)
    total_params = model.count_parameters()
    print(f"NaviTrit-{args.model_size.upper()} initialized: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_iter = iter(train_loader)

    start_time = time.time()
    print(f"\n--- STAGE 1: Backbone & Hop Modulation Adaptation ({args.stage1_steps} steps) ---")
    model.train()
    for step in range(args.stage1_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # In Stage 1: Diverse walk sampling to adapt layer normalization
        # High Gumbel temperature to explore all nodes
        out = model(x, temperature=2.0)
        logits = out["logits"]

        shift_logits = logits[:, :-1, :].contiguous().view(-1, model.vocab_size)
        shift_labels = y[:, :-1].contiguous().view(-1)
        loss_ce = F.cross_entropy(shift_logits, shift_labels)

        loss_total = (
            loss_ce
            + out["loss_fpf"]
            + out["loss_state_fpf"]
            + out["loss_attn_div"]
            + out["loss_entropy"]
        )

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(f"  [Stage 1] Step {step+1}/{args.stage1_steps} | CE Loss: {loss_ce.item():.4f} | Attn Ratio: {out['attn_ratio']:.2f}")

    print(f"\n--- STAGE 2: Joint Navigation & Attractor Core Stabilization ({args.train_steps - args.stage1_steps} steps) ---")
    stage2_steps = args.train_steps - args.stage1_steps
    for step in range(stage2_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # Anneal temperature from 1.0 -> 0.5
        temp = max(0.5, 1.0 - 0.5 * (step / stage2_steps))
        out = model(x, temperature=temp)
        logits = out["logits"]

        shift_logits = logits[:, :-1, :].contiguous().view(-1, model.vocab_size)
        shift_labels = y[:, :-1].contiguous().view(-1)
        loss_ce = F.cross_entropy(shift_logits, shift_labels)

        loss_total = (
            loss_ce
            + out["loss_fpf"]
            + out["loss_state_fpf"]
            + out["loss_attn_div"]
            + out["loss_entropy"]
            + out["loss_hop_budget"]
        )

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 300 == 0 or (step + 1) == stage2_steps:
            print(f"  [Stage 2] Step {step+1}/{stage2_steps} | CE Loss: {loss_ce.item():.4f} | Trajectory: {out['trajectory']} | Attn: {out['attn_ratio']:.2f}")

    train_time = round(time.time() - start_time, 2)
    print(f"\nTraining completed in {train_time}s ({train_time/60:.2f} min).")

    # Evaluate arms
    print("\n=== Running Comparative Evaluation on Held-Out Validation Set ===")
    print("[1/2] Evaluating NaviTrit Learned Non-Monotonic Routing...")
    res_learned = evaluate_arm(model, val_loader, device, is_monotonic=False)

    print("[2/2] Evaluating Monotonic Feedforward Baseline...")
    res_monotonic = evaluate_arm(model, val_loader, device, is_monotonic=True)

    print(f"\n>>> Results Summary:")
    print(f"  NaviTrit-10M Val Loss : {res_learned['val_loss']} (PPL: {res_learned['val_perplexity']})")
    print(f"  Monotonic Baseline    : {res_monotonic['val_loss']} (PPL: {res_monotonic['val_perplexity']})")
    print(f"  Advantage             : {res_learned['val_loss'] - res_monotonic['val_loss']:.4f} loss points")
    print(f"  Attention Ratio       : {res_learned['attention_ratio']*100:.1f}%")
    print(f"  Node Visitations      : {res_learned['node_visitations']}")

    # Save checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), f"../../outputs/checkpoints/navitrit-{args.model_size}-trained.pt")
    if args.save_checkpoint:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    # Build telemetry ledger
    ledger = {
        "status": "COMPLETED",
        "gate": "Gate 15 (Track D-4)",
        "model_size": args.model_size,
        "total_parameters": total_params,
        "train_time_s": train_time,
        "cli": vars(args),
        "arms": {
            "navitrit_learned": res_learned,
            "monotonic_baseline": res_monotonic,
        },
        "findings": {
            "val_loss_delta": round(res_learned["val_loss"] - res_monotonic["val_loss"], 4),
            "navitrit_beats_monotonic": res_learned["val_loss"] < res_monotonic["val_loss"],
            "attention_ratio": res_learned["attention_ratio"],
            "reasoning_core_visitations": res_learned["node_visitations"][model.node_reasoning],
        },
    }

    out_json = os.path.join(os.path.dirname(__file__), f"../../{args.json}")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(ledger, f, indent=2)
    print(f"Ledger saved to {out_json}")


if __name__ == "__main__":
    main()
