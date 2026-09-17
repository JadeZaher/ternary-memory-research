"""
experiments/train_flowtrit_sudoku9.py: Train and evaluate FlowTrit on 9x9 Sudoku constraint satisfaction.

Protocol bug fixed here: training previously built x_t = (1-t)*noise + t*x_clean, leaking the full
solution through x_t while inference only ever supplies x_t = 0.2*noise + 0.8*c (clues only). A probe
found the shipped 4x4 model at 75.6% solve rate as evaluated, 100% with training-style x_t, 90% with
x_t=0. leak_free=True (default) builds x_t from c during training so train and test match.

Rewrite of the original 60-solution / 15-epoch / batch-1 script, which evaluated at t=1 with
x_1 = puzzle (never matching the leak_free training interpolant) and hard-coded PASSED while all
four arms sat at chance (11% cell accuracy). This version trains thousands of solutions with batched
FPF steps and evaluates with the same t=0.8, x_t = 0.2*noise + 0.8*c convention used in training.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from experiments.flowtrit_model import FlowTritDenoiser
from experiments.sudoku9x9_dataset import (
    create_puzzle_9x9,
    encode_board_to_tensor,
    generate_valid_solution_9x9,
)

D_MODEL, N_LAYERS, N_HEADS = 192, 4, 4
SEQ_LEN, NUM_CLASSES = 81, 9
TIERS_ORDER = ["easy", "medium", "hard"]
CHANCE_CELL_ACCURACY_PCT = 11.11  # derived: 1/9 uniform-guess baseline over 9 classes


def measured(value) -> Dict[str, object]:
    return {"value": value, "label": "measured"}


def derived(value) -> Dict[str, object]:
    return {"value": value, "label": "derived"}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_board_solved_9x9(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> bool:
    """Both grids shape [81, 9] (one-hot); compares argmax values (1..9)."""
    pred_vals = pred_grid.argmax(dim=-1).view(9, 9) + 1
    target_vals = target_grid.argmax(dim=-1).view(9, 9) + 1
    return bool(torch.equal(pred_vals, target_vals))


# ==============================================================================
# Dataset construction
# ==============================================================================

def build_dataset(
    train_solutions_n: int,
    test_per_tier: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, List[Dict[str, object]]], float]:
    """Generates train_solutions_n complete boards (stacked on device) and test_per_tier puzzles/tier."""
    print(f"[+] Generating {train_solutions_n} training 9x9 solutions...")
    t0 = time.time()
    train_solutions_raw = [generate_valid_solution_9x9() for _ in range(train_solutions_n)]
    gen_seconds = time.time() - t0
    print(f"    done in {gen_seconds:.2f}s (measured)")

    train_tensor = torch.stack([encode_board_to_tensor(s) for s in train_solutions_raw]).to(device)  # [N, 81, 9]

    test_puzzles: Dict[str, List[Dict[str, object]]] = {"easy": [], "medium": [], "hard": []}
    for tier in TIERS_ORDER:
        for _ in range(test_per_tier):
            sol = generate_valid_solution_9x9()
            _, clues = create_puzzle_9x9(sol, tier=tier)
            test_puzzles[tier].append({
                "solution": encode_board_to_tensor(sol).to(device),
                "clue_indices": [r * 9 + c for (r, c) in clues],
            })

    return train_tensor, test_puzzles, gen_seconds


# ==============================================================================
# Training
# ==============================================================================

def train_arm(
    train_tensor: torch.Tensor,
    ternary: bool,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_rollout_k: int,
    device: torch.device,
    seed: int,
    arm_name: str,
) -> Tuple[FlowTritDenoiser, float]:
    """Batched FPF training; per-cell clue mask sampled with fraction ~ Uniform[0.25, 0.5] per sample."""
    set_all_seeds(seed)
    model = FlowTritDenoiser(
        seq_len=SEQ_LEN, num_classes=NUM_CLASSES, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, ternary=ternary
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = train_tensor.shape[0]

    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_train, (batch_size,), device=device)
        x_clean = train_tensor[idx]  # [B, 81, 9]
        target_indices = x_clean.argmax(dim=-1)

        clue_frac = torch.empty(batch_size, 1, 1, device=device).uniform_(0.25, 0.5)
        clue_mask = (torch.rand(batch_size, SEQ_LEN, 1, device=device) < clue_frac).float()
        c = x_clean * clue_mask

        loss = model.train_fpf_step(
            x_clean=x_clean,
            target_indices=target_indices,
            clue_mask=clue_mask,
            c=c,
            optimizer=optimizer,
            max_rollout_k=max_rollout_k,
            leak_free=True,
        )
        if step % 250 == 0 or step == steps:
            print(f"  [{arm_name} step {step:4d}/{steps}] loss={loss:.4f} elapsed={time.time() - t0:.1f}s")

    return model, time.time() - t0


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_sudoku9x9(
    model: FlowTritDenoiser,
    test_puzzles: Dict[str, List[Dict[str, object]]],
    device: torch.device,
    max_steps: int = 5,
    early_exit: bool = False,
    eps_exit: float = 0.03,
) -> Dict[str, object]:
    """Inference convention: t=0.8, x_t = 0.2*noise + 0.8*c (matches leak_free training interpolant)."""
    model.eval()
    tier_results = {}
    all_solved = 0
    all_total = 0
    all_steps: List[int] = []
    all_cell_acc: List[float] = []

    with torch.no_grad():
        for tier, puzzles in test_puzzles.items():
            solved = 0
            total = len(puzzles)
            steps_list = []
            cell_accs = []

            for p in puzzles:
                sol = p["solution"].unsqueeze(0)  # [1, 81, 9], already on device
                clue_indices = p["clue_indices"]

                clue_mask = torch.zeros(1, 81, 1, device=device)
                clue_mask[0, clue_indices, 0] = 1.0
                c = sol * clue_mask

                t = torch.tensor([[0.8]], device=device)
                noise = torch.randn_like(sol)
                x_t = 0.2 * noise + 0.8 * c

                res = model.rollout_inference(
                    x_t, c, t, clue_mask=clue_mask, max_steps=max_steps, early_exit=early_exit, eps_exit=eps_exit
                )
                steps_taken = res["steps_taken"]

                pred = res["s_final"][0].clone()
                pred[clue_indices] = sol[0, clue_indices]

                if check_board_solved_9x9(pred, sol[0]):
                    solved += 1
                steps_list.append(steps_taken)

                pred_vals = pred.argmax(dim=-1)
                target_vals = sol[0].argmax(dim=-1)
                non_clues = [i for i in range(81) if i not in clue_indices]
                if non_clues:
                    acc = (pred_vals[non_clues] == target_vals[non_clues]).float().mean().item() * 100
                    cell_accs.append(acc)

            avg_cell_acc = sum(cell_accs) / max(len(cell_accs), 1)
            tier_results[tier] = {
                "solved": solved,
                "total": total,
                "solve_rate_pct": round(solved / max(total, 1) * 100, 2),
                "cell_accuracy_pct": round(avg_cell_acc, 2),
                "avg_steps": round(sum(steps_list) / max(len(steps_list), 1), 3),
            }
            all_solved += solved
            all_total += total
            all_steps.extend(steps_list)
            all_cell_acc.extend(cell_accs)

    overall_cell_acc = sum(all_cell_acc) / max(len(all_cell_acc), 1)
    return {
        "overall_solve_rate_pct": round(all_solved / max(all_total, 1) * 100, 2),
        "overall_cell_accuracy_pct": round(overall_cell_acc, 2),
        "overall_avg_steps": round(sum(all_steps) / max(len(all_steps), 1), 3),
        "tier_results": tier_results,
    }


# ==============================================================================
# Main experiment
# ==============================================================================

def run_9x9_experiment(
    train_solutions: int = 3000,
    test_per_tier: int = 50,
    steps: int = 3000,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_rollout_k: int = 3,
    seed: int = 20260916,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_json: str = "outputs/flowtrit-sudoku9-results.json",
) -> Dict[str, object]:
    device = torch.device(device_name)
    print("=" * 80)
    print(f"Track B: FlowTrit on 9x9 Sudoku Benchmark ({device})")
    print("=" * 80)

    set_all_seeds(seed)
    train_tensor, test_puzzles, gen_seconds = build_dataset(train_solutions, test_per_tier, device)

    print("\n--- Training Model 1: FP32 Baseline ---")
    fp32_model, fp32_train_s = train_arm(
        train_tensor, ternary=False, steps=steps, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        max_rollout_k=max_rollout_k, device=device, seed=seed, arm_name="fp32",
    )
    eval_fp32 = evaluate_sudoku9x9(fp32_model, test_puzzles, device, max_steps=5)
    print(f"-> FP32 Solve Rate: {eval_fp32['overall_solve_rate_pct']}% | Cell Acc: {eval_fp32['overall_cell_accuracy_pct']}%")

    print("\n--- Evaluating Model 2: Post-Hoc Ternarized ---")
    posthoc_model = copy.deepcopy(fp32_model)
    posthoc_model.ternarize_post_hoc(threshold_ratio=0.75)
    eval_posthoc = evaluate_sudoku9x9(posthoc_model, test_puzzles, device, max_steps=5)
    print(f"-> Post-Hoc Solve Rate: {eval_posthoc['overall_solve_rate_pct']}% | Cell Acc: {eval_posthoc['overall_cell_accuracy_pct']}%")

    print("\n--- Training Model 3: Native FlowTrit (STE + FPF) ---")
    native_model, native_train_s = train_arm(
        train_tensor, ternary=True, steps=steps, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        max_rollout_k=max_rollout_k, device=device, seed=seed, arm_name="native",
    )
    eval_native = evaluate_sudoku9x9(native_model, test_puzzles, device, max_steps=5)
    print(f"-> Native Solve Rate: {eval_native['overall_solve_rate_pct']}% | Cell Acc: {eval_native['overall_cell_accuracy_pct']}%")

    print("\n--- Evaluating Model 4: Certified Dynamic Early Exit on Native ---")
    eval_dynamic = evaluate_sudoku9x9(native_model, test_puzzles, device, max_steps=6, early_exit=True, eps_exit=0.03)
    print(f"-> Dynamic Exit Solve Rate: {eval_dynamic['overall_solve_rate_pct']}% | Avg Steps: {eval_dynamic['overall_avg_steps']}")

    findings = {
        "fp32_learned_above_chance": eval_fp32["overall_cell_accuracy_pct"] > 25.0,
        "native_learned_above_chance": eval_native["overall_cell_accuracy_pct"] > 25.0,
        "native_minus_posthoc_cell_acc": round(
            eval_native["overall_cell_accuracy_pct"] - eval_posthoc["overall_cell_accuracy_pct"], 2),
        "native_minus_fp32_cell_acc": round(
            eval_native["overall_cell_accuracy_pct"] - eval_fp32["overall_cell_accuracy_pct"], 2),
    }

    results = {
        "status": "COMPLETED",
        "task": "Track B 9x9 Sudoku Generalization",
        "device": device_name,
        "seed": seed,
        "grid_size": "9x9 (81 positions, 9 classes)",
        "dataset": {
            "train_solutions": train_solutions,
            "test_per_tier": test_per_tier,
            "solution_generation_seconds": measured(round(gen_seconds, 2)),
        },
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "max_rollout_k": max_rollout_k,
            "leak_free": True,
            "fp32_train_seconds": measured(round(fp32_train_s, 2)),
            "native_train_seconds": measured(round(native_train_s, 2)),
        },
        "chance_cell_accuracy_pct": derived(CHANCE_CELL_ACCURACY_PCT),
        "models": {
            "fp32_baseline": eval_fp32,
            "post_hoc_ternarized": eval_posthoc,
            "native_flowtrit": eval_native,
            "native_dynamic_exit": eval_dynamic,
        },
        "findings": findings,
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results written to {out_path.resolve()}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-solutions", type=int, default=3000)
    parser.add_argument("--test-per-tier", type=int, default=50)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--json", type=str, default="outputs/flowtrit-sudoku9-results.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    train_solutions = args.train_solutions
    test_per_tier = args.test_per_tier
    steps = args.steps
    if args.smoke:
        train_solutions = 100
        test_per_tier = 3
        steps = 30

    run_9x9_experiment(
        train_solutions=train_solutions,
        test_per_tier=test_per_tier,
        steps=steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        output_json=args.json,
    )
