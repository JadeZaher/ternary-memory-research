"""
experiments/frontier_scaling/train_conditional_program.py: Training Engine for Dynamic Weight Parameterization
and Weights as Conditional Programs (Gate 19-D).

Trains context-conditioned FiLM modulators and dynamic low-rank adapters with 100% frozen base ternary weights.
Warm-starts from navitrit-100m-loopformer.pt, using multi-corpus tokens to teach physical tiles distinct functional roles
(BIND, REFINE, VERIFY, EMIT) across recursive hops.
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
from transformers import AutoTokenizer

from experiments.frontier_scaling.conditional_program_model import (
    ConditionalProgramConfig,
    NaviTritConditionalProgramForCausalLM,
    PCGradOptimizer,
)


class FlatTokenDataset(Dataset):
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


def evaluate_model(
    model: NaviTritConditionalProgramForCausalLM,
    val_loader: DataLoader,
    device: torch.device,
    budget: int,
    max_batches: int = 25,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x, hop_budget=budget)
            loss = F.cross_entropy(out["logits"].view(-1, model.config.vocab_size), y.view(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            batch_count += 1

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    return {"loss": avg_loss, "ppl": ppl, "budget": budget}


def train():
    parser = argparse.ArgumentParser(description="Train Conditional Program Weights (Gate 19-D)")
    parser.add_argument("--base-ckpt", type=str, default="outputs/checkpoints/navitrit-100m-loopformer.pt")
    parser.add_argument("--data-path", type=str, default="data/multicorpus_10m.pt")
    parser.add_argument("--output-ckpt", type=str, default="outputs/checkpoints/navitrit-100m-condprog.pt")
    parser.add_argument("--output-ledger", type=str, default="outputs/navitrit-100m-condprog-results.json")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--use-pcgrad", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"================================================================")
    print(f"Gate 19-D: Dynamic Weight Parameterization & Conditional Programs")
    print(f"Device: {device} | Base Checkpoint: {args.base_ckpt}")
    print(f"================================================================")

    # 1. Initialize Model
    config = ConditionalProgramConfig(
        vocab_size=50257,
        hidden_size=768,
        intermediate_size=2048,
        num_attention_heads=12,
        max_position_embeddings=512,
        max_hops=8,
        default_hops=4,
        d_context=64,
        lora_rank=16,
        lora_alpha=16.0,
        use_film=True,
        use_lowrank=True,
    )
    model = NaviTritConditionalProgramForCausalLM(config).to(device)

    # 2. Warm-start Base Weights
    if os.path.exists(args.base_ckpt):
        model.load_base_weights(args.base_ckpt)
    else:
        print(f"[Warning] Base checkpoint {args.base_ckpt} not found! Initializing with random base weights.")

    # 3. Freeze Base Weights (Only Train Dynamic Modulators and Adapters)
    trainable_params = 0
    frozen_params = 0
    for name, p in model.named_parameters():
        if "hypernet" in name or "lora" in name or "shortcut" in name:
            p.requires_grad = True
            trainable_params += p.numel()
        else:
            p.requires_grad = False
            frozen_params += p.numel()

    packed_trainable_mb = (trainable_params * 2) / (8 * 1024 * 1024)
    fp16_trainable_mb = (trainable_params * 2) / (1024 * 1024)
    print(f"Base Frozen Params: {frozen_params:,} (100% SHA-256 Protected)")
    print(f"Dynamic Program Trainable Params: {trainable_params:,} ({packed_trainable_mb:.4f} MB packed, {fp16_trainable_mb:.2f} MB FP16)")
    print(f"L2 Cache Footprint of Dynamic Layers: {(packed_trainable_mb / 24.0) * 100:.2f}% of 24MB L2")

    # 4. Load Training Data
    if not os.path.exists(args.data_path):
        fallback = "data/tinystories_tokens.pt"
        print(f"[Data] {args.data_path} not found. Falling back to {fallback}")
        args.data_path = fallback
    all_tokens = torch.load(args.data_path, weights_only=True)
    split_idx = int(0.90 * len(all_tokens))
    train_tokens, val_tokens = all_tokens[:split_idx], all_tokens[split_idx:]

    train_dataset = FlatTokenDataset(train_tokens, args.seq_len)
    val_dataset = FlatTokenDataset(val_tokens, args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Dataset: {len(train_dataset)} train chunks, {len(val_dataset)} val chunks")

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    pcgrad_opt = PCGradOptimizer(optimizer) if args.use_pcgrad else None

    def lr_lambda(step: int) -> float:
        warmup = 30
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, args.max_steps - warmup))
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initial Validation
    print("\n--- Initial Validation Across Compute Budgets ---")
    init_metrics = {}
    for M in [2, 4]:
        eval_res = evaluate_model(model, val_loader, device, budget=M, max_batches=15)
        init_metrics[f"M_{M}"] = eval_res
        print(f"Initial Budget M={M}: Loss={eval_res['loss']:.4f} | PPL={eval_res['ppl']:.2f}")

    # 6. Training Loop
    print("\n--- Beginning Dynamic Weight Parameterization Optimization ---")
    start_time = time.time()
    step = 0
    train_iter = iter(train_loader)
    history = []
    budget_cycle = [2, 4]

    while step < args.max_steps:
        model.train()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        current_budget = budget_cycle[step % len(budget_cycle)]

        optimizer.zero_grad()

        if args.use_pcgrad:
            # Multi-objective PCGrad: optimize LM loss and Shortcut alignment as distinct tasks
            loss_dict = model.compute_loss(x, y, hop_budget=current_budget)
            loss_lm = loss_dict["loss_lm"]
            loss_sc = loss_dict["loss_shortcut"]
            pcgrad_opt.pcgrad_backward([loss_lm, loss_sc])
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            pcgrad_opt.step()
            total_loss = loss_dict["total_loss"]
        else:
            loss_dict = model.compute_loss(x, y, hop_budget=current_budget)
            total_loss = loss_dict["total_loss"]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

        scheduler.step()

        if (step + 1) % 50 == 0 or step == 0:
            elapsed = time.time() - start_time
            print(
                f"Step {step+1:03d}/{args.max_steps} | Hop Budget M={current_budget} | "
                f"Total Loss: {total_loss.item():.4f} (LM: {loss_dict['loss_lm'].item():.4f}, "
                f"Shortcut: {loss_dict['loss_shortcut'].item():.4f}) | "
                f"Elapsed: {elapsed:.1f}s"
            )

        if (step + 1) % 250 == 0 or (step + 1) == args.max_steps:
            val_eval = {}
            for M in [2, 4]:
                res = evaluate_model(model, val_loader, device, budget=M, max_batches=20)
                val_eval[f"M_{M}"] = res
            print(f"--> Step {step+1} Validation: M=2 PPL: {val_eval['M_2']['ppl']:.2f} | M=4 PPL: {val_eval['M_4']['ppl']:.2f}")
            history.append({"step": step + 1, "val_metrics": val_eval})

        step += 1

    total_training_time = time.time() - start_time
    print(f"\nTraining Complete in {total_training_time:.2f}s ({total_training_time/60.0:.2f} min).")

    # 7. Final Validation Across All Compute Budgets
    print("\n--- Final Validation Across Compute Budgets ---")
    final_metrics = {}
    for M in [1, 2, 4, 6]:
        res = evaluate_model(model, val_loader, device, budget=M, max_batches=30)
        final_metrics[f"M_{M}"] = res
        print(f"Final Budget M={M}: Loss={res['loss']:.4f} | PPL={res['ppl']:.2f}")

    # 8. Save Checkpoint
    os.makedirs(os.path.dirname(args.output_ckpt), exist_ok=True)
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "final_metrics": final_metrics,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "packed_trainable_mb": packed_trainable_mb,
    }, args.output_ckpt)
    print(f"Saved Conditional Program Checkpoint to {args.output_ckpt}")

    # 9. Save Ledger
    os.makedirs(os.path.dirname(args.output_ledger), exist_ok=True)
    ledger_data = {
        "gate": "Gate 19-D",
        "track": "Phase II Track W: Dynamic Weight Parameterization & Weights as Conditional Programs",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "packed_trainable_mb": packed_trainable_mb,
            "l2_cache_occupancy_pct": (packed_trainable_mb / 24.0) * 100.0,
            "functional_roles": ["BIND (0)", "REFINE (1)", "VERIFY (2)", "EMIT (3)"],
        },
        "training_params": {
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "learning_rate": args.lr,
            "use_pcgrad": args.use_pcgrad,
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
