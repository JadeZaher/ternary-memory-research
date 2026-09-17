"""
experiments/bench_algorithmic_suite.py: BitRoute three-arm comparison on synthetic algorithmic
sequence tasks (reverse / sort / mod_add / parity), chosen because a small transformer clearly
either learns or does not learn each one, and because the tasks differ in how much per-token
"work" they demand, giving the router genuinely different depth-usage incentives per task.

Each task produces (input_ids, target_ids, loss_mask): a random prefix, a SEP token, then the
answer; loss_mask marks only the answer positions (next-token prediction, teacher forcing).
Vocab layout: 0=PAD, 1=SEP, 2..11=digit/bit values 0-9 (vocab_size itself is 64, per config).
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.bitroute_model import BitRouteForCausalLM, BitRouteConfig, BitRouteOutput
from experiments.tristate_router import ACTION_EXECUTE, ACTION_ROUTE_AROUND, ACTION_EARLY_EXIT
from experiments.train_bitroute_tinystories import (
    set_seed,
    build_optimizer,
    make_lr_lambda,
    layer_action_histograms,
)

PAD_ID, SEP_ID, DIGIT_OFFSET = 0, 1, 2
ARM_NAMES = ["fp32", "ternary", "ternary_router"]
TASK_NAMES = ["reverse", "sort", "mod_add", "parity"]


def make_batch(task: str, batch_size: int, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds (x, y, loss_mask) for one synthetic task: prefix + SEP + answer, next-token targets,
    loss_mask=1 only on positions predicting the answer."""
    sep_col = torch.full((batch_size, 1), SEP_ID, dtype=torch.long)

    if task == "reverse":
        prefix = torch.randint(0, 10, (batch_size, 16), generator=generator)
        answer = prefix.flip(dims=[1])
    elif task == "sort":
        prefix = torch.randint(0, 10, (batch_size, 16), generator=generator)
        answer, _ = torch.sort(prefix, dim=1)
    elif task == "mod_add":
        prefix = torch.randint(0, 10, (batch_size, 8), generator=generator)
        answer = torch.cumsum(prefix, dim=1) % 10
    elif task == "parity":
        prefix = torch.randint(0, 2, (batch_size, 24), generator=generator)
        answer = torch.cumsum(prefix, dim=1) % 2
    else:
        raise ValueError(f"Unknown task: {task}")

    answer_len = answer.shape[1]
    seq = torch.cat([prefix + DIGIT_OFFSET, sep_col, answer + DIGIT_OFFSET], dim=1)
    x = seq[:, :-1]
    y = seq[:, 1:]
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[:, -answer_len:] = True
    return x, y, mask


def masked_cross_entropy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """CE over answer positions only; other positions get ignore_index=-100."""
    y_masked = y.clone()
    y_masked[~mask] = -100
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_masked.reshape(-1), ignore_index=-100)


def build_model(config: BitRouteConfig, seed: int, device: torch.device) -> BitRouteForCausalLM:
    set_seed(seed)
    return BitRouteForCausalLM(config).to(device)


@torch.no_grad()
def evaluate_algo(
    model: BitRouteForCausalLM,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    early_exit_enabled: bool = True,
    force_layer_actions: Optional[Dict[int, str]] = None,
    collect_action_ids: bool = False,
) -> Dict:
    """Token and exact-sequence accuracy on the answer positions, plus routing fractions."""
    model.eval()
    out: BitRouteOutput = model(x, early_exit_enabled=early_exit_enabled, force_layer_actions=force_layer_actions)
    preds = out.logits.argmax(dim=-1)

    correct = (preds == y) & mask
    token_acc = correct.sum().float() / mask.sum().float().clamp(min=1)

    # A sequence is exact iff every one of its answer positions is correct.
    per_row_ok = (correct | ~mask).all(dim=1)
    exact_acc = per_row_ok.float().mean()

    result = {
        "token_accuracy": round(float(token_acc), 4),
        "exact_seq_accuracy": round(float(exact_acc), 4),
        "layer_exec_fraction": round(out.layer_exec_fraction, 4),
        "layer_bypass_fraction": round(out.layer_bypass_fraction, 4),
        "layer_exit_fraction": round(out.layer_exit_fraction, 4),
    }
    if collect_action_ids:
        result["_action_ids_by_layer"] = [
            [d.action_ids.detach()] if d.action_ids is not None else [] for d in out.routing_decisions
        ]
    return result


def train_arm_on_task(
    task: str,
    arm: str,
    config: BitRouteConfig,
    args: argparse.Namespace,
    held_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Dict:
    print(f"\n--- Task: {task} | Arm: {arm} ---")
    model = build_model(config, args.seed, device)
    optimizer = build_optimizer(model, args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(args.train_steps))

    gen = torch.Generator()
    gen.manual_seed(args.seed)  # identical data order across arms of the same task

    is_router_arm = arm == "ternary_router"
    warmup_steps = int(args.router_warmup_frac * args.train_steps) if is_router_arm else args.train_steps
    force_all_execute = {l: "EXECUTE" for l in range(config.num_hidden_layers)}
    print_every = 100 if args.train_steps >= 100 else max(1, args.train_steps // 3)

    model.train()
    t0 = time.time()
    for step in range(1, args.train_steps + 1):
        x, y, mask = make_batch(task, args.batch_size, gen)
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        optimizer.zero_grad()
        if is_router_arm and step > warmup_steps:
            out = model(x)
            ce_loss = masked_cross_entropy(out.logits, y, mask)
            router_loss = model.compute_router_loss(lambda_sparse=args.lambda_sparse)
            loss = ce_loss + router_loss
        else:
            out = model(x, force_layer_actions=force_all_execute)
            ce_loss = masked_cross_entropy(out.logits, y, mask)
            loss = ce_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % print_every == 0 or step == args.train_steps:
            print(f"  [{task}/{arm}] step {step:5d}/{args.train_steps} loss={ce_loss.item():.4f}")

    train_time_s = time.time() - t0
    print(f"  [{task}/{arm}] measured train time: {train_time_s:.2f}s")

    hx, hy, hmask = held_out
    results: Dict = {"train_time_s": round(train_time_s, 2)}

    if is_router_arm:
        router_on = evaluate_algo(model, hx, hy, hmask, early_exit_enabled=True, collect_action_ids=True)
        action_ids_by_layer = router_on.pop("_action_ids_by_layer")
        hists = layer_action_histograms(action_ids_by_layer, config.num_hidden_layers)
        router_on["layer_action_histogram"] = hists
        results["router_on"] = router_on

        forced = evaluate_algo(model, hx, hy, hmask, early_exit_enabled=True, force_layer_actions=force_all_execute)
        results["router_forced_execute"] = forced

        for layer_idx, hist in enumerate(hists):
            total = sum(hist.values())
            probs = torch.tensor([
                hist["EXECUTE"] / max(total, 1),
                hist["ROUTE_AROUND"] / max(total, 1),
                hist["EARLY_EXIT"] / max(total, 1),
            ])
            model.layers[layer_idx].router.random_policy = probs
        random_matched = evaluate_algo(model, hx, hy, hmask, early_exit_enabled=True)
        for layer in model.layers:
            layer.router.random_policy = None
        results["router_random_matched"] = random_matched
    else:
        results["eval"] = evaluate_algo(model, hx, hy, hmask, early_exit_enabled=True, force_layer_actions=force_all_execute)

    return results


def run(args: argparse.Namespace) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"================================================================================")
    print(f"BitRoute Algorithmic Suite: fp32 / ternary / ternary_router ({device})")
    print(f"Tasks: {args.tasks}")
    print(f"================================================================================")

    base_kwargs = dict(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=128,
        block_size=128,
        use_trained_router=True,
    )

    task_results: Dict[str, Dict] = {}
    for task in args.tasks:
        held_out_gen = torch.Generator()
        held_out_gen.manual_seed(args.seed + 10_000 + TASK_NAMES.index(task))  # independent of training stream
        hx, hy, hmask = make_batch(task, args.eval_size, held_out_gen)
        hx, hy, hmask = hx.to(device), hy.to(device), hmask.to(device)

        arm_results: Dict[str, Dict] = {}
        for arm in ARM_NAMES:
            ternary = arm != "fp32"
            config = BitRouteConfig(ternary=ternary, **base_kwargs)
            arm_results[arm] = train_arm_on_task(task, arm, config, args, (hx, hy, hmask), device)

        fp32_exact = arm_results["fp32"]["eval"]["exact_seq_accuracy"]
        ternary_exact = arm_results["ternary"]["eval"]["exact_seq_accuracy"]
        router_on = arm_results["ternary_router"]["router_on"]
        random_matched = arm_results["ternary_router"]["router_random_matched"]

        findings = {
            "ternary_minus_fp32_exact_acc": round(ternary_exact - fp32_exact, 4),  # derived
            "router_minus_random_exact_acc": round(
                router_on["exact_seq_accuracy"] - random_matched["exact_seq_accuracy"], 4
            ),  # derived
            "bypass_fraction": router_on["layer_bypass_fraction"],  # measured
        }

        task_results[task] = {"arms": arm_results, "findings": findings}

    results = {
        "status": "COMPLETED",
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": args.seed,
        "config": base_kwargs,
        "cli": {
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lambda_sparse": args.lambda_sparse,
            "router_warmup_frac": args.router_warmup_frac,
            "eval_size": args.eval_size,
            "tasks": args.tasks,
        },
        "tasks": task_results,
    }

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results written to {args.json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-sparse", type=float, default=0.02)
    parser.add_argument("--router-warmup-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--tasks", type=str, default=",".join(TASK_NAMES), help="comma-separated subset of: " + ",".join(TASK_NAMES))
    parser.add_argument("--smoke", action="store_true", help="30 steps of one task, small batch/eval")
    parser.add_argument("--json", type=str, default="outputs/algorithmic-suite-results.json")
    args = parser.parse_args()
    args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.smoke:
        args.tasks = args.tasks[:1] if args.tasks else [TASK_NAMES[0]]
        args.train_steps = 30
        args.batch_size = min(args.batch_size, 16)
        args.eval_size = min(args.eval_size, 64)

    run(args)
