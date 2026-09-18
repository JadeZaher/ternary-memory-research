"""
experiments/frontier_scaling/train_ifmor.py: Joint Training Engine for IF-MoR with Mamba Arithmetic Supervision.

Gate 19-B Implementation:
1. Warm-starts from navitrit-100m-loopformer.pt.
2. Trains widened 20.6M 4-branch super-block (Full Attn, Linear Attn, Mamba SSM, SwiGLU FFN).
3. Interleaves Multi-Corpus tokens (Code, Stories, Math) with explicit numeric scratchpad supervision.
4. Optimizes joint loss: L_LM + L_shortcut + L_balance + L_arith.
5. Emits outputs/checkpoints/navitrit-100m-ifmor.pt and outputs/navitrit-100m-ifmor-results.json.
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

from experiments.frontier_scaling.navitrit_ifmor_model import (
    IFMoRConfig,
    NaviTritIFMoRForCausalLM,
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


def generate_arithmetic_sample(tokenizer: AutoTokenizer, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Generates an arithmetic prompt with explicit numerical target for the Mamba scratchpad."""
    names = ["Olivia", "Liam", "Lily", "Tom", "Sam", "Emma", "Jack", "Leo", "Mia", "Noah"]
    items = ["apples", "cookies", "books", "candies", "pencils", "marbles"]
    n1, n2 = random.sample(names, 2)
    item = random.choice(items)

    op_type = random.choice(["sub", "two_sub", "add"])
    if op_type == "sub":
        start = random.randint(15, 50)
        sub = random.randint(2, 12)
        ans = start - sub
        text = f"Problem: {n1} had {start} {item}. {n1} gave {sub} {item} to {n2}. How many {item} remain?\nReasoning: {start} - {sub} = {ans} {item} remain."
    elif op_type == "two_sub":
        start = random.randint(20, 50)
        s1 = random.randint(2, 8)
        s2 = random.randint(2, 8)
        ans = start - s1 - s2
        text = f"Problem: {n1} bakes {start} {item}. They sell {s1} in morning and {s2} in afternoon. How many {item} left?\nReasoning: {start} - {s1} - {s2} = {ans} {item} left."
    else:
        a = random.randint(5, 30)
        b = random.randint(5, 30)
        ans = a + b
        text = f"Problem: {n1} has {a} {item} and {n2} has {b} {item}. How many in total?\nReasoning: {a} + {b} = {ans} {item} in total."

    tokens = tokenizer.encode(text)
    if len(tokens) < seq_len + 1:
        tokens = tokens + [tokenizer.eos_token_id] * (seq_len + 1 - len(tokens))
    else:
        tokens = tokens[:seq_len + 1]

    t_tensor = torch.tensor(tokens, dtype=torch.long)
    x = t_tensor[:-1]
    y = t_tensor[1:]
    return x, y, float(ans)


def evaluate_ifmor(
    model: NaviTritIFMoRForCausalLM,
    val_loader: DataLoader,
    device: torch.device,
    budget: int,
    max_batches: int = 25,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_branch_usage = torch.zeros(4, device=device)
    batch_count = 0

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

            all_w = torch.cat(out["branch_weights"], dim=1)
            total_branch_usage += all_w.mean(dim=(0, 1))
            batch_count += 1

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    mean_w = (total_branch_usage / max(1, batch_count)).cpu().tolist()
    return {
        "loss": avg_loss,
        "ppl": ppl,
        "branch_usage": {
            "full_attn": mean_w[0],
            "lin_attn": mean_w[1],
            "mamba_ssm": mean_w[2],
            "swiglu_ffn": mean_w[3],
        }
    }


def train():
    parser = argparse.ArgumentParser(description="Train IF-MoR with Mamba Arithmetic Supervision")
    parser.add_argument("--max-steps", type=int, default=600, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-path", type=str, default="data/multicorpus_10m.pt")
    parser.add_argument("--init-ckpt", type=str, default="outputs/checkpoints/navitrit-100m-loopformer.pt")
    parser.add_argument("--output-ckpt", type=str, default="outputs/checkpoints/navitrit-100m-ifmor.pt")
    parser.add_argument("--output-ledger", type=str, default="outputs/navitrit-100m-ifmor-results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Starting IF-MoR Joint Training on {device} ===")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 1. Initialize IF-MoR Model
    config = IFMoRConfig(
        vocab_size=50257,
        hidden_size=768,
        intermediate_size=4096,
        num_attention_heads=12,
        d_state=64,
        max_position_embeddings=512,
        max_loops=6,
        default_loops=4,
        use_shortcut_consistency=True,
        balance_lambda=0.20,
        arith_lambda=0.10, # normalized scaling for numeric MSE
    )
    model = NaviTritIFMoRForCausalLM(config).to(device)

    # 2. Warm-start from previous checkpoint
    if os.path.exists(args.init_ckpt):
        model.load_from_pretrained_checkpoint(args.init_ckpt, source_layer=0)
    else:
        print(f"[IF-MoR] Warning: {args.init_ckpt} not found. Initializing fresh.")

    super_block_params = model.get_super_block_param_count()
    packed_mb = model.get_packed_footprint_mb()
    print(f"IF-MoR Super-Block Parameters: {super_block_params:,}")
    print(f"1.58-bit Packed Footprint: {packed_mb:.3f} MB (resides in 24MB L2 Cache, {packed_mb/24.0*100.0:.1f}%)")

    # 3. Load Multi-Corpus Tokens
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

    print(f"Multi-Corpus Chunks: {len(train_dataset)} train, {len(val_dataset)} val")

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        warmup = 50
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, args.max_steps - warmup))
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initial Validation
    print("\n--- Initial Validation Across Compute Budgets ---")
    init_metrics = {}
    for M in [2, 4]:
        eval_res = evaluate_ifmor(model, val_loader, device, budget=M, max_batches=15)
        init_metrics[f"M_{M}"] = eval_res
        print(f"Initial M={M}: Loss={eval_res['loss']:.4f} | PPL={eval_res['ppl']:.2f} | Branches: {eval_res['branch_usage']}")

    # 5. Training Loop
    print("\n--- Beginning IF-MoR Joint Optimization ---")
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

        # 25% of batches are explicit arithmetic scratchpad supervision samples
        numeric_targets = None
        if random.random() < 0.25:
            arith_xs, arith_ys, arith_nums = [], [], []
            for _ in range(args.batch_size):
                ax, ay, num = generate_arithmetic_sample(tokenizer, args.seq_len)
                arith_xs.append(ax)
                arith_ys.append(ay)
                arith_nums.append(num)
            x = torch.stack(arith_xs, dim=0).to(device)
            y = torch.stack(arith_ys, dim=0).to(device)
            numeric_targets = torch.tensor(arith_nums, dtype=torch.float32, device=device)

        current_budget = budget_cycle[step % len(budget_cycle)]

        optimizer.zero_grad()
        loss_dict = model.compute_loss(x, y, numeric_targets=numeric_targets, recursion_budget=current_budget)
        total_loss = loss_dict["total_loss"]
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0 or step == 0:
            elapsed = time.time() - start_time
            w = loss_dict["mean_branch_usage"].cpu().tolist()
            print(
                f"Step {step+1:03d}/{args.max_steps} | Budget M={current_budget} | "
                f"Loss: {total_loss.item():.4f} (LM: {loss_dict['loss_lm'].item():.4f}, "
                f"Bal: {loss_dict['loss_balance'].item():.4f}, Arith: {loss_dict['loss_arith'].item():.4f}) | "
                f"Branches: [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}] | "
                f"Elapsed: {elapsed:.1f}s"
            )

        if (step + 1) % 200 == 0 or (step + 1) == args.max_steps:
            val_eval = {}
            for M in [2, 4]:
                res = evaluate_ifmor(model, val_loader, device, budget=M, max_batches=20)
                val_eval[f"M_{M}"] = res
            print(f"--> Step {step+1} Validation: M=2 PPL: {val_eval['M_2']['ppl']:.2f} | M=4 PPL: {val_eval['M_4']['ppl']:.2f}")
            history.append({"step": step + 1, "val_metrics": val_eval})

        step += 1

    total_training_time = time.time() - start_time
    print(f"\nTraining Complete in {total_training_time:.2f}s ({total_training_time/60.0:.2f} min).")

    # 6. Final Evaluation
    print("\n--- Final Validation Across Compute Budgets ---")
    final_metrics = {}
    for M in [1, 2, 4, 6]:
        res = evaluate_ifmor(model, val_loader, device, budget=M, max_batches=30)
        final_metrics[f"M_{M}"] = res
        print(f"Final Budget M={M}: Loss={res['loss']:.4f} | PPL={res['ppl']:.2f} | Branches: {res['branch_usage']}")

    # 7. Save Checkpoint
    os.makedirs(os.path.dirname(args.output_ckpt), exist_ok=True)
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "final_metrics": final_metrics,
        "super_block_params": super_block_params,
        "packed_mb": packed_mb,
    }, args.output_ckpt)
    print(f"Saved IF-MoR Checkpoint to {args.output_ckpt}")

    # 8. Save Ledger
    os.makedirs(os.path.dirname(args.output_ledger), exist_ok=True)
    ledger_data = {
        "gate": "Gate 19-B",
        "track": "Track G-Extended: Interleaved Functional Mixture-of-Recursions (IF-MoR)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "l2_cache_mb": 24.0,
            "super_block_params": super_block_params,
            "super_block_packed_mb": packed_mb,
            "l2_cache_occupancy_pct": (packed_mb / 24.0) * 100.0,
            "branches": ["Full Attention", "Linear Attention", "Ternary Mamba SSM", "Wide SwiGLU FFN"],
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
