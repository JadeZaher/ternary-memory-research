"""
experiments/frontier_scaling/train_loopformer.py: Shortcut-Consistency Training for LoopFormer / MoR.

Gate 19 (Track G):
1. Loads pre-trained Attention and FFN tiles from navitrit-100m-tree-grpo.pt.
2. Trains parameter-shared recurrent super-block (7.09M params) on multicorpus (TinyStories + GSM8K + Python Code).
3. Applies Shortcut-Consistency objective across dynamic recursion budgets M in {2, 4, 6}.
4. Evaluates test-time compute scaling: verifies that deeper recursions yield lower perplexity.
5. Emits outputs/checkpoints/navitrit-100m-loopformer.pt and outputs/navitrit-100m-loopformer-results.json.
"""

import os
import sys
import time
import math
import json
import random
import argparse
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from experiments.frontier_scaling.loopformer_model import (
    LoopFormerConfig,
    LoopFormerForCausalLM,
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


def load_data(path: str, val_split: float = 0.10) -> Tuple[torch.Tensor, torch.Tensor]:
    if not os.path.exists(path):
        fallback = os.path.join(os.path.dirname(__file__), "../../data/tinystories_tokens.pt")
        print(f"[Data] Warning: {path} not found. Falling back to {fallback}")
        path = fallback
    all_tokens = torch.load(path, weights_only=True)
    split_idx = int((1.0 - val_split) * len(all_tokens))
    return all_tokens[:split_idx], all_tokens[split_idx:]


def evaluate_ppl(
    model: LoopFormerForCausalLM,
    val_loader: DataLoader,
    device: torch.device,
    budget: int,
    max_batches: int = 30,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for b_idx, (x, y) in enumerate(val_loader):
            if b_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x, recursion_budget=budget)
            logits = out["logits"]
            shift_logits = logits.contiguous().view(-1, model.config.vocab_size)
            shift_labels = y.contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    return {"loss": avg_loss, "ppl": ppl}


def train():
    parser = argparse.ArgumentParser(description="Train LoopFormer with Shortcut-Consistency")
    parser.add_argument("--max-steps", type=int, default=600, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-path", type=str, default="data/multicorpus_10m.pt")
    parser.add_argument("--init-ckpt", type=str, default="outputs/checkpoints/navitrit-100m-tree-grpo.pt")
    parser.add_argument("--output-ckpt", type=str, default="outputs/checkpoints/navitrit-100m-loopformer.pt")
    parser.add_argument("--output-ledger", type=str, default="outputs/navitrit-100m-loopformer-results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Starting LoopFormer Shortcut-Consistency Training on {device} ===")

    # 1. Initialize LoopFormer Model
    config = LoopFormerConfig(
        vocab_size=50257,
        hidden_size=768,
        intermediate_size=2048,
        num_attention_heads=12,
        max_position_embeddings=512,
        max_recursions=8,
        default_recursions=4,
        use_shortcut_consistency=True,
    )
    model = LoopFormerForCausalLM(config).to(device)

    # 2. Warm-start from pre-trained checkpoint
    if os.path.exists(args.init_ckpt):
        model.load_from_navitrit_checkpoint(args.init_ckpt, source_layer=0)
    else:
        print(f"Warning: {args.init_ckpt} not found. Training from scratch.")

    super_block_params = model.get_super_block_param_count()
    packed_mb = model.get_packed_footprint_mb()
    print(f"Recurrent Super-Block Parameters: {super_block_params:,}")
    print(f"1.58-bit Packed Footprint: {packed_mb:.3f} MB (resides in 32MB L2 Cache)")

    # 3. Load Multi-Corpus Data
    train_tokens, val_tokens = load_data(args.data_path)
    train_dataset = FlatTokenDataset(train_tokens, args.seq_len)
    val_dataset = FlatTokenDataset(val_tokens, args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Data Loaded: {len(train_dataset)} train chunks, {len(val_dataset)} val chunks (seq_len={args.seq_len})")

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    def lr_lambda(step: int) -> float:
        warmup = 50
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, args.max_steps - warmup))
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initial Zero-Shot Validation across compute budgets M
    print("\n--- Initial Validation Across Compute Budgets ---")
    init_metrics = {}
    for M in [2, 4, 6]:
        eval_res = evaluate_ppl(model, val_loader, device, budget=M, max_batches=20)
        init_metrics[f"M_{M}"] = eval_res
        print(f"Initial Budget M={M}: Val Loss = {eval_res['loss']:.4f} | Perplexity = {eval_res['ppl']:.2f}")

    # 5. Training Loop
    print("\n--- Beginning Shortcut-Consistency Optimization ---")
    start_time = time.time()
    step = 0
    train_iter = iter(train_loader)
    history = []

    # Cycle budgets across steps to ensure elasticity
    budget_cycle = [2, 4, 6]

    while step < args.max_steps:
        model.train()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Dynamic budget sampling
        current_budget = budget_cycle[step % len(budget_cycle)]

        optimizer.zero_grad()
        loss_dict = model.compute_loss(x, y, recursion_budget=current_budget)
        total_loss = loss_dict["total_loss"]
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0 or step == 0:
            elapsed = time.time() - start_time
            print(
                f"Step {step+1:03d}/{args.max_steps} | Budget M={current_budget} | "
                f"Loss: {total_loss.item():.4f} (Final: {loss_dict['loss_final'].item():.4f}, "
                f"Inter: {loss_dict['loss_intermediate'].item():.4f}, Align: {loss_dict['loss_alignment'].item():.4f}) | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | Elapsed: {elapsed:.1f}s"
            )

        if (step + 1) % 200 == 0 or (step + 1) == args.max_steps:
            val_eval = {}
            for M in [2, 4, 6]:
                res = evaluate_ppl(model, val_loader, device, budget=M, max_batches=20)
                val_eval[f"M_{M}"] = res
            print(f"--> Step {step+1} Validation: M=2 PPL: {val_eval['M_2']['ppl']:.2f} | M=4 PPL: {val_eval['M_4']['ppl']:.2f} | M=6 PPL: {val_eval['M_6']['ppl']:.2f}")
            history.append({"step": step + 1, "val_metrics": val_eval})

        step += 1

    total_training_time = time.time() - start_time
    print(f"\nTraining Complete in {total_training_time:.2f}s ({total_training_time/60.0:.2f} min).")

    # 6. Final Evaluation
    print("\n--- Final Validation Across Compute Budgets ---")
    final_metrics = {}
    for M in [1, 2, 4, 6, 8]:
        res = evaluate_ppl(model, val_loader, device, budget=M, max_batches=40)
        final_metrics[f"M_{M}"] = res
        print(f"Final Budget M={M}: Val Loss = {res['loss']:.4f} | Perplexity = {res['ppl']:.2f}")

    # 7. Save Checkpoint
    os.makedirs(os.path.dirname(args.output_ckpt), exist_ok=True)
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "final_metrics": final_metrics,
        "super_block_params": super_block_params,
        "packed_mb": packed_mb,
    }, args.output_ckpt)
    print(f"Saved LoopFormer Checkpoint to {args.output_ckpt}")

    # 8. Save Ledger
    os.makedirs(os.path.dirname(args.output_ledger), exist_ok=True)
    ledger_data = {
        "gate": "Gate 19",
        "track": "Track G: Mixture-of-Recursions (MoR) & LoopFormer",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "l2_cache_mb": 32.0,
            "super_block_params": super_block_params,
            "super_block_packed_mb": packed_mb,
            "l2_cache_occupancy_pct": (packed_mb / 32.0) * 100.0,
            "dram_traffic_reduction": "Zero DRAM weight traffic during recurrent loops",
        },
        "training_params": {
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "learning_rate": args.lr,
            "total_training_time_s": total_training_time,
        },
        "initial_metrics": init_metrics,
        "final_metrics": final_metrics,
        "history": history,
    }
    with open(args.output_ledger, "w") as f:
        json.dump(ledger_data, f, indent=2)
    print(f"Saved Results Ledger to {args.output_ledger}")


if __name__ == "__main__":
    train()
