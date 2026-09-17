"""
experiments/train_bitroute_tinystories.py: Three-arm controlled comparison of BitRoute on TinyStories.

Hardening 2026-09-16: the previous ledger trained 40 steps at batch 4, gave the routed arm
double compute (init from an already-trained baseline, then trained again), evaluated routing
on batch 4 where the whole batch followed sample 0, and hard-coded status "PASSED". This
rewrite trains three arms from identical seeded init and data order for equal steps, evaluates
routing per-sample, and reports measured `status: "COMPLETED"` with numeric findings only.
"""

import argparse
import dataclasses
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
from torch.utils.data import Dataset, DataLoader

from experiments.bitroute_model import BitRouteForCausalLM, BitRouteConfig, BitRouteOutput
from experiments.tristate_router import ACTION_EXECUTE, ACTION_ROUTE_AROUND, ACTION_EARLY_EXIT, ACTION_NAMES

ARM_NAMES = ["fp32", "ternary", "ternary_router"]


def set_seed(seed: int) -> None:
    """Seed torch, python random, and CUDA RNG so arms share identical init/data order."""
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


class TokenChunkDataset(Dataset):
    """Fixed-length (input, target) chunks carved from a flat 1D token tensor."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        self.num_samples = (len(tokens) - 1) // seq_len
        self.tokens = tokens[: self.num_samples * seq_len + 1]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y


def load_tinystories_tokens(
    cache_path: str = "data/tinystories_tokens.pt",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loads the pre-tokenized 1D GPT-2 token tensor and splits it 90/10 train/val."""
    all_tokens = torch.load(cache_path, weights_only=True)
    split_idx = int(0.9 * len(all_tokens))
    return all_tokens[:split_idx], all_tokens[split_idx:]


def build_loaders(
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    seq_len: int,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    """Fresh DataLoaders with a seeded shuffle generator: identical data order per arm."""
    train_ds = TokenChunkDataset(train_tokens, seq_len)
    val_ds = TokenChunkDataset(val_tokens, seq_len)
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_model(config: BitRouteConfig, seed: int, device: torch.device) -> BitRouteForCausalLM:
    """Seed immediately before construction: identical init across arms with matching shapes
    (ternary flag only changes BitLinear.forward, not parameter shape or reset_parameters)."""
    set_seed(seed)
    model = BitRouteForCausalLM(config).to(device)
    return model


def build_optimizer(model: nn.Module, lr: float) -> torch.optim.AdamW:
    """AdamW, betas (0.9, 0.95), weight_decay 0.1 on >=2D (matrix) params only, 0.0 elsewhere."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def make_lr_lambda(total_steps: int, warmup_iters: int = 100):
    """Linear warmup for `warmup_iters` steps, then cosine decay to 10% of peak lr."""
    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return (step + 1) / warmup_iters
        progress = (step - warmup_iters) / max(1, total_steps - warmup_iters)
        progress = min(progress, 1.0)
        return 0.1 + 0.5 * (1.0 - 0.1) * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def layer_action_histograms(
    all_action_ids: List[List[torch.Tensor]],
    num_layers: int,
) -> List[Dict[str, int]]:
    """Bincount action_ids per layer over every eval batch, counting only samples still active
    (not yet EARLY_EXITed upstream). Exited samples never reach deeper layers, so including their
    decisions would inflate the deep-layer exit rate handed to the random control."""
    hists = []
    counts_per_layer = [torch.zeros(3, dtype=torch.long) for _ in range(num_layers)]
    num_batches = len(all_action_ids[0]) if num_layers > 0 else 0
    for b in range(num_batches):
        active = torch.ones_like(all_action_ids[0][b].cpu(), dtype=torch.bool)
        for layer_idx in range(num_layers):
            ids = all_action_ids[layer_idx][b].cpu()
            counts_per_layer[layer_idx] += torch.bincount(ids[active], minlength=3)
            active = active & (ids != ACTION_EARLY_EXIT)
    for layer_idx in range(num_layers):
        counts = counts_per_layer[layer_idx]
        hists.append({
            "EXECUTE": int(counts[ACTION_EXECUTE]),
            "ROUTE_AROUND": int(counts[ACTION_ROUTE_AROUND]),
            "EARLY_EXIT": int(counts[ACTION_EARLY_EXIT]),
        })
    return hists


@torch.no_grad()
def evaluate(
    model: BitRouteForCausalLM,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int,
    early_exit_enabled: bool = True,
    force_layer_actions: Optional[Dict[int, str]] = None,
    collect_action_ids: bool = False,
) -> Dict:
    """Measured val loss (micro-averaged CE per token), perplexity, and weighted routing
    fractions over eval batches. Optionally collects per-layer action_ids for histogramming."""
    model.eval()
    num_layers = len(model.layers)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    total_loss, total_tokens = 0.0, 0
    weighted_exec, weighted_bypass, weighted_exit, weight_sum = 0.0, 0.0, 0.0, 0.0
    action_ids_by_layer: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]

    num_batches = 0
    for x, y in val_loader:
        if max_batches > 0 and num_batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out: BitRouteOutput = model(
            x, early_exit_enabled=early_exit_enabled, force_layer_actions=force_layer_actions
        )
        loss = loss_fn(out.logits.view(-1, out.logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        total_tokens += y.numel()

        batch_actual = x.shape[0]
        weighted_exec += out.layer_exec_fraction * batch_actual
        weighted_bypass += out.layer_bypass_fraction * batch_actual
        weighted_exit += out.layer_exit_fraction * batch_actual
        weight_sum += batch_actual

        if collect_action_ids:
            for layer_idx, decision in enumerate(out.routing_decisions):
                if decision.action_ids is not None:
                    action_ids_by_layer[layer_idx].append(decision.action_ids.detach())

        num_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    result = {
        "val_loss": round(avg_loss, 4),
        "val_perplexity": round(ppl, 2),
        "layer_exec_fraction": round(weighted_exec / max(weight_sum, 1.0), 4),
        "layer_bypass_fraction": round(weighted_bypass / max(weight_sum, 1.0), 4),
        "layer_exit_fraction": round(weighted_exit / max(weight_sum, 1.0), 4),
        "evaluated_tokens": total_tokens,
        "evaluated_batches": num_batches,
    }
    if collect_action_ids:
        result["_action_ids_by_layer"] = action_ids_by_layer  # stripped before JSON dump
    return result


def train_arm(
    arm: str,
    config: BitRouteConfig,
    args: argparse.Namespace,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    device: torch.device,
) -> Dict:
    """Trains one arm (fp32 / ternary / ternary_router) from seeded scratch init."""
    print(f"\n--- Arm: {arm} ---")
    model = build_model(config, args.seed, device)
    train_loader, val_loader = build_loaders(train_tokens, val_tokens, args.seq_len, args.batch_size, args.seed)
    optimizer = build_optimizer(model, args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(args.train_steps))
    loss_fn = nn.CrossEntropyLoss()

    is_router_arm = arm == "ternary_router"
    warmup_steps = int(args.router_warmup_frac * args.train_steps) if is_router_arm else args.train_steps
    force_all_execute = {l: "EXECUTE" for l in range(config.num_hidden_layers)}

    model.train()
    train_iter = iter(train_loader)
    t0 = time.time()
    for step in range(1, args.train_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        if is_router_arm and step > warmup_steps:
            out = model(x)  # no forcing: differentiable Gumbel-Softmax routing
            ce_loss = loss_fn(out.logits.view(-1, config.vocab_size), y.view(-1))
            router_loss = model.compute_router_loss(lambda_sparse=args.lambda_sparse)
            loss = ce_loss + router_loss
        else:
            out = model(x, force_layer_actions=force_all_execute)
            ce_loss = loss_fn(out.logits.view(-1, config.vocab_size), y.view(-1))
            loss = ce_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 or step == args.train_steps:
            elapsed = time.time() - t0
            print(f"  [{arm}] step {step:5d}/{args.train_steps} loss={ce_loss.item():.4f} elapsed={elapsed:.1f}s")

    train_time_s = time.time() - t0
    print(f"  [{arm}] measured train time: {train_time_s:.2f}s")

    eval_batches = args.eval_batches
    results: Dict = {"train_time_s": round(train_time_s, 2)}

    if is_router_arm:
        router_on = evaluate(model, val_loader, device, eval_batches, early_exit_enabled=True, collect_action_ids=True)
        action_ids_by_layer = router_on.pop("_action_ids_by_layer")
        hists = layer_action_histograms(action_ids_by_layer, config.num_hidden_layers)
        results["router_on"] = router_on
        results["router_on"]["layer_action_histogram"] = hists

        forced = evaluate(model, val_loader, device, eval_batches, early_exit_enabled=True, force_layer_actions=force_all_execute)
        results["router_forced_execute"] = forced

        no_exit = evaluate(model, val_loader, device, eval_batches, early_exit_enabled=False)
        results["router_no_exit"] = no_exit

        # Matched-rate random control: per-layer action probabilities measured from router_on histogram.
        for layer_idx, hist in enumerate(hists):
            total = sum(hist.values())
            probs = torch.tensor([
                hist["EXECUTE"] / max(total, 1),
                hist["ROUTE_AROUND"] / max(total, 1),
                hist["EARLY_EXIT"] / max(total, 1),
            ])
            model.layers[layer_idx].router.random_policy = probs
        random_matched = evaluate(model, val_loader, device, eval_batches, early_exit_enabled=True)
        for layer in model.layers:
            layer.router.random_policy = None
        results["router_random_matched"] = random_matched
    else:
        results["eval"] = evaluate(model, val_loader, device, eval_batches, early_exit_enabled=True, force_layer_actions=force_all_execute)

    # Latency: batch 1, seq 128, random ids, 10 warmup + 50 timed forwards, arms (b)/(c) only.
    if arm in ("ternary", "ternary_router"):
        model.eval()
        sample_x = torch.randint(0, config.vocab_size, (1, 128), device=device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(50):
                _ = model(sample_x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat_ms = (time.time() - t0) * 1000 / 50
        results["latency_ms_batch1_seq128"] = round(lat_ms, 3)

    if args.save_checkpoints:
        ckpt_dir = "outputs/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"bitroute-tinystories-{arm}.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "config": dataclasses.asdict(config),
            "arm": arm,
            "train_steps": args.train_steps,
            "seed": args.seed,
        }, ckpt_path)
        print(f"  [{arm}] checkpoint saved: {ckpt_path}")

    return results


def run(args: argparse.Namespace) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"================================================================================")
    print(f"BitRoute TinyStories: fp32 / ternary / ternary_router controlled comparison ({device})")
    print(f"================================================================================")

    train_tokens, val_tokens = load_tinystories_tokens()
    print(f"[+] measured: {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens")

    base_kwargs = dict(
        vocab_size=50257,
        hidden_size=384,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=6,
        max_position_embeddings=512,
        block_size=256,
        use_trained_router=True,
    )

    # --arms trains a subset (e.g. a lambda sweep of the router arm); missing arms are read from
    # --reference-json so the cross-arm findings stay computable.
    arm_results: Dict[str, Dict] = {}
    selected = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.reference_json and os.path.exists(args.reference_json):
        with open(args.reference_json) as f:
            arm_results.update(json.load(f).get("arms", {}))
    for arm in ARM_NAMES:
        if arm not in selected:
            continue
        ternary = arm != "fp32"
        config = BitRouteConfig(ternary=ternary, **base_kwargs)
        arm_results[arm] = train_arm(arm, config, args, train_tokens, val_tokens, device)

    # findings: derived comparisons across arms
    fp32_val_loss = arm_results["fp32"]["eval"]["val_loss"]
    ternary_val_loss = arm_results["ternary"]["eval"]["val_loss"]
    router_on = arm_results["ternary_router"]["router_on"]
    forced = arm_results["ternary_router"]["router_forced_execute"]
    random_matched = arm_results["ternary_router"]["router_random_matched"]

    findings = {
        "ternary_minus_fp32_val_loss": round(ternary_val_loss - fp32_val_loss, 4),  # derived
        "router_on_minus_forced_execute_val_loss": round(router_on["val_loss"] - forced["val_loss"], 4),  # derived
        "router_on_minus_random_matched_val_loss": round(router_on["val_loss"] - random_matched["val_loss"], 4),  # derived
        "router_beats_random_control": bool(router_on["val_loss"] < random_matched["val_loss"]),  # derived
        "router_beats_forced_execute": bool(router_on["val_loss"] < forced["val_loss"]),  # derived
        "bypass_fraction_router_on": router_on["layer_bypass_fraction"],  # measured
    }

    latency_ternary = arm_results["ternary"].get("latency_ms_batch1_seq128")
    latency_router = arm_results["ternary_router"].get("latency_ms_batch1_seq128")
    if latency_ternary and latency_router:
        findings["router_latency_ratio_vs_ternary"] = round(latency_router / latency_ternary, 4)  # derived

    results = {
        "status": "COMPLETED",
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": args.seed,
        "config": base_kwargs,
        "cli": {
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "lambda_sparse": args.lambda_sparse,
            "router_warmup_frac": args.router_warmup_frac,
            "eval_batches": args.eval_batches,
        },
        "optimizer": "AdamW betas=(0.9, 0.95), weight_decay=0.1 on >=2D params only, 0.0 elsewhere; "
                     "linear warmup 100 steps then cosine to 10% of lr; grad clip 1.0; FP32, no autocast.",
        "arms": arm_results,
        "findings": findings,
    }

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results written to {args.json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--lambda-sparse", type=float, default=0.02)
    parser.add_argument("--router-warmup-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--eval-batches", type=int, default=0, help="0 = whole val set")
    parser.add_argument("--save-checkpoints", dest="save_checkpoints", action="store_true", default=True)
    parser.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false")
    parser.add_argument("--smoke", action="store_true", help="tiny fast run: steps=20, eval-batches=5, small batch/seq")
    parser.add_argument("--json", type=str, default="outputs/bitroute-tinystories-results.json")
    parser.add_argument("--arms", type=str, default=",".join(ARM_NAMES), help="comma-separated subset of arms to train")
    parser.add_argument("--reference-json", type=str, default="", help="results file supplying arms not trained in this run")
    args = parser.parse_args()

    if args.smoke:
        args.train_steps = 20
        args.eval_batches = 5
        args.batch_size = min(args.batch_size, 4)
        args.seq_len = min(args.seq_len, 32)

    run(args)
