"""
experiments/frontier_scaling/train_navitrit_max.py: Production Multi-Hour Training Engine for NaviTrit-Max (247.6M).

Key Features:
1. Scaled Model: NaviTrit-Max (247.6M params, d_model=1024, d_intermediate=4096, 6 layers, 16 heads).
2. Data: 25M-Token Agro-Environmental Multi-Corpus (Crop Science, Climate/Env, Python Code, GSM8K, TinyStories).
3. Optimization: AdamW with warmup-cosine schedule, gradient clipping (1.0), and modern torch.amp mixed precision.
4. Fault-Tolerant Checkpointing: Resumable latest, best, and periodic checkpoints (every 500 steps).
5. Comprehensive Logging: Step latency, token throughput, peak VRAM, loss decomposition, and validation perplexity.
"""

import os
import sys
import time
import math
import json
import argparse
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.frontier_scaling.navitrit_max_model import (
    NaviTritMaxConfig,
    NaviTritMaxForCausalLM,
)


def get_cosine_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    """Calculates warmup + cosine decay learning rate."""
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    if step > total_steps:
        return min_lr
    decay_ratio = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts a random batch of sequences from the 1D token tensor."""
    max_idx = len(tokens) - seq_len - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([tokens[i : i + seq_len] for i in ix]).to(device=device, dtype=torch.long)
    y = torch.stack([tokens[i + 1 : i + seq_len + 1] for i in ix]).to(device=device, dtype=torch.long)
    return x, y


def evaluate(
    model: NaviTritMaxForCausalLM,
    val_tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_iters: int,
    device: torch.device,
) -> Dict[str, float]:
    """Computes validation cross-entropy loss and perplexity across eval_iters batches."""
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_balance = 0.0

    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch(val_tokens, batch_size, seq_len, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(x, labels=y)
                total_loss += out["loss"].item()
                total_ce += out["ce_loss"].item()
                total_balance += out["balance_loss"].item()

    avg_loss = total_loss / eval_iters
    avg_ce = total_ce / eval_iters
    avg_balance = total_balance / eval_iters
    ppl = math.exp(min(avg_ce, 20.0))

    model.train()
    return {
        "val_loss": avg_loss,
        "val_ce_loss": avg_ce,
        "val_balance_loss": avg_balance,
        "val_perplexity": ppl,
    }


def train(
    dataset_path: str = "data/multicorpus_agro_environmental_25m.pt",
    output_dir: str = "outputs/checkpoints",
    log_path: str = "outputs/navitrit-max-training-log.json",
    max_steps: int = 5000,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    seq_len: int = 512,
    base_lr: float = 1.5e-4,
    min_lr: float = 1.5e-5,
    warmup_steps: int = 200,
    eval_interval: int = 250,
    save_interval: int = 500,
    sample_interval: int = 500,
    device_str: str = "cuda",
    resume: bool = False,
    dry_run: bool = False,
):
    print("=" * 80)
    print(f"NaviTrit-Max (247.6M) Scaled Training Engine | Device: {device_str.upper()}")
    print(f"Max Steps: {max_steps:,} | Batch: {batch_size} (Grad Accum: {grad_accum_steps}) | Seq Len: {seq_len}")
    print("=" * 80)

    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # 1. Dataset Loading
    if not os.path.exists(dataset_path):
        fallback = "data/multicorpus_10m.pt"
        if os.path.exists(fallback):
            print(f"[Warning] {dataset_path} not found. Falling back to {fallback}!")
            dataset_path = fallback
        else:
            raise FileNotFoundError(f"Missing dataset {dataset_path}!")

    print(f"Loading pretraining token tensor from {dataset_path}...")
    tokens = torch.load(dataset_path, weights_only=True)
    print(f"Loaded {len(tokens):,} tokens ({len(tokens) * 4 / 1e6:.2f} MB).")

    # 95% Train / 5% Validation split
    n_train = int(0.95 * len(tokens))
    train_tokens = tokens[:n_train]
    val_tokens = tokens[n_train:]
    print(f"Train Tokens: {len(train_tokens):,} | Val Tokens: {len(val_tokens):,}")

    # 2. Model Initialization
    config = NaviTritMaxConfig(
        num_layers=6,
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        max_position_embeddings=1024,
        ternary=True,
        tie_word_embeddings=True,
    )
    model = NaviTritMaxForCausalLM(config).to(device)
    stats = model.count_parameters()
    print(f"Model Initialized: {stats['total_millions']:.2f}M parameters ({stats['trainable_parameters']:,} trainable).")

    # 3. Optimizer & Scaler
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=base_lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda")

    start_step = 0
    best_val_loss = float("inf")
    history = []

    # Checkpoint Resume
    latest_ckpt_path = os.path.join(output_dir, "navitrit-max-latest.pt")
    if resume and os.path.exists(latest_ckpt_path):
        print(f"Resuming from checkpoint: {latest_ckpt_path}...")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"Successfully resumed at step {start_step} (Best Val Loss: {best_val_loss:.4f}).")

    if dry_run:
        max_steps = min(max_steps, start_step + 50)
        print(f"[Dry Run Mode] Clamping execution to {max_steps} steps.")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    sample_prompt = "Photosynthesis in C4 plants differs from C3 plants because"

    model.train()
    tokens_per_step = batch_size * grad_accum_steps * seq_len
    start_time = time.time()
    step_t0 = time.time()

    print("\nStarting Training Loop...")
    for step in range(start_step, max_steps):
        # Update learning rate
        lr = get_cosine_lr(step, warmup_steps, max_steps, base_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_ce = 0.0
        accum_balance = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = get_batch(train_tokens, batch_size, seq_len, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(x, labels=y)
                loss = out["loss"] / grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += out["loss"].item()
            accum_ce += out["ce_loss"].item()
            accum_balance += out["balance_loss"].item()

        # Gradient clipping and optimizer step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        scaler.step(optimizer)
        scaler.update()

        step_t1 = time.time()
        step_duration = step_t1 - step_t0
        step_t0 = step_t1
        throughput = tokens_per_step / max(step_duration, 1e-5)

        avg_step_loss = accum_loss / grad_accum_steps
        avg_step_ce = accum_ce / grad_accum_steps
        avg_step_balance = accum_balance / grad_accum_steps

        # Logging every 10 steps
        if step % 10 == 0 or step == max_steps - 1:
            vram_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
            print(
                f"Step [{step:5d}/{max_steps:5d}] | "
                f"Loss: {avg_step_loss:6.4f} (CE: {avg_step_ce:6.4f}, Bal: {avg_step_balance:6.4f}) | "
                f"LR: {lr:.2e} | Grad: {grad_norm:5.2f} | "
                f"VRAM: {vram_gb:4.2f} GB | Speed: {throughput:5.0f} tok/s"
            )

        # Evaluation & Validation
        if (step > 0 and step % eval_interval == 0) or step == max_steps - 1:
            print(f"\n--- Running Validation Evaluation at Step {step} ---")
            val_metrics = evaluate(model, val_tokens, batch_size, seq_len, eval_iters=20, device=device)
            print(
                f"Validation Metrics -> Loss: {val_metrics['val_loss']:.4f} | "
                f"CE: {val_metrics['val_ce_loss']:.4f} | "
                f"Perplexity: {val_metrics['val_perplexity']:.2f}"
            )

            # Record history entry
            entry = {
                "step": step,
                "train_loss": avg_step_loss,
                "train_ce_loss": avg_step_ce,
                "val_loss": val_metrics["val_loss"],
                "val_ce_loss": val_metrics["val_ce_loss"],
                "val_perplexity": val_metrics["val_perplexity"],
                "lr": lr,
                "grad_norm": grad_norm,
                "throughput_tok_per_sec": throughput,
                "vram_gb": vram_gb,
                "elapsed_seconds": time.time() - start_time,
            }
            history.append(entry)

            with open(log_path, "w") as f:
                json.dump({"config": stats, "history": history}, f, indent=2)

            # Save best checkpoint
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_ckpt_path = os.path.join(output_dir, "navitrit-max-best.pt")
                torch.save(
                    {
                        "step": step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "config": config,
                    },
                    best_ckpt_path,
                )
                print(f"Saved new best checkpoint to {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")
            print("----------------------------------------------------\n")

        # Qualitative Sample Generation
        if (step > 0 and step % sample_interval == 0) or step == max_steps - 1:
            print("\n--- Qualitative Generation Sample ---")
            prompt_ids = tokenizer.encode(sample_prompt, return_tensors="pt").to(device)
            gen_ids = model.generate(prompt_ids, max_new_tokens=48, temperature=0.7)
            gen_text = tokenizer.decode(gen_ids[0].cpu().tolist())
            print(f"Generated text:\n{gen_text}")
            print("-------------------------------------\n")

        # Periodic Checkpoint Saving
        if (step > 0 and step % save_interval == 0) or step == max_steps - 1:
            # Latest checkpoint (resumable)
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "config": config,
                },
                latest_ckpt_path,
            )
            # Periodic step checkpoint
            step_ckpt_path = os.path.join(output_dir, f"navitrit-max-step-{step:05d}.pt")
            torch.save({"step": step, "model_state": model.state_dict(), "config": config}, step_ckpt_path)
            print(f"Saved checkpoint snapshots at step {step}.")

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time/3600:.2f} hours ({total_time:.1f} seconds)!")
    print(f"Final Model Checkpoint: {latest_ckpt_path}")
    print(f"Training Ledger: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NaviTrit-Max Scaled Pretraining Engine")
    parser.add_argument("--dataset", type=str, default="data/multicorpus_agro_environmental_25m.pt")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        seq_len=args.seq_len,
        base_lr=args.lr,
        resume=args.resume,
        dry_run=args.dry_run,
        device_str=args.device,
    )
