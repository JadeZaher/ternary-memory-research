"""
experiments/train_flowroute_sudoku.py: Train and evaluate FlowRoute (Track C Combined Architecture).

Protocol bug fixed here: training previously built x_t = (1-t)*noise + t*x_clean, leaking the full
solution through x_t while inference only ever supplies x_t = 0.2*noise + 0.8*c (clues only). A probe
found the shipped 4x4 model at 75.6% solve rate as evaluated, 100% with training-style x_t, 90% with
x_t=0. leak_free=True (default) builds x_t from c during training so train and test match.

Fixes three bugs in the previous version: (1) "step_exit_only" issued the identical rollout call as
"flowroute_combined" (both router and early-exit enabled in both); (2) there was no router-free
reference model to isolate the router's contribution from ordinary early-exit savings; (3) the
500-step model solved 0-1 puzzles, so its reported "compute savings" measured noise, not signal, and
status was hard-coded PASSED regardless of outcome. Also fixes: router-warmup freezes the router
probe's own parameters (requires_grad_(False)) for the first --router-warmup-frac of steps, relying on
the [1.5, -1.0] EXECUTE-biased init (TwoStateLayerRouter.reset in flowroute_model.py) to keep layers
executing while the backbone learns; the probe is unfrozen afterward.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from experiments.flowtrit_model import FlowTritDenoiser
from experiments.flowroute_model import FlowRouteDenoiser
from experiments.bench_csp_suite import TASK_SPECS, build_dataset, build_eval_puzzles, check_clue_adherence, set_all_seeds
from experiments.test_flowtrit import MiniSudoku4x4

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEQ_LEN, NUM_CLASSES = TASK_SPECS["sudoku4"]["seq_len"], TASK_SPECS["sudoku4"]["num_classes"]
TIERS = TASK_SPECS["sudoku4"]["tiers"]
TIERS_ORDER = ["easy", "medium", "hard"]
D_MODEL, N_LAYERS, N_HEADS = 128, 4, 4
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4
MAX_ROLLOUT_K = 4
FIXED_MAX_STEPS = 5
DYNAMIC_MAX_STEPS = 6
EPS_EXIT = 0.03


def measured(value) -> Dict[str, object]:
    return {"value": value, "label": "measured"}


def set_router_probes_trainable(model: FlowRouteDenoiser, trainable: bool) -> None:
    """Freezes/unfreezes only the router probe params, leaving the backbone always trainable."""
    for block in model.blocks:
        for p in block.router.probe.parameters():
            p.requires_grad_(trainable)


# ==============================================================================
# Training
# ==============================================================================

def train_flowroute(
    train_boards: List[List[int]],
    steps: int,
    lambda_sparse: float,
    router_warmup_frac: float,
    lr: float,
    seed: int,
) -> Tuple[FlowRouteDenoiser, float]:
    """Router-warmup: probe params frozen for the first router_warmup_frac of steps."""
    set_all_seeds(seed)
    model = FlowRouteDenoiser(
        seq_len=SEQ_LEN, num_classes=NUM_CLASSES, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        ternary=True, use_trained_router=True,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    warmup_steps = int(round(steps * router_warmup_frac))

    set_router_probes_trainable(model, False)
    t0 = time.time()
    for step in range(1, steps + 1):
        if step == warmup_steps + 1:
            set_router_probes_trainable(model, True)

        batch_samples = random.choices(train_boards, k=BATCH_SIZE)
        target_indices = torch.tensor(batch_samples, dtype=torch.long, device=DEVICE)
        x_clean = F.one_hot(target_indices, num_classes=NUM_CLASSES).float()

        clue_mask = torch.zeros(BATCH_SIZE, SEQ_LEN, 1, device=DEVICE)
        for i in range(BATCH_SIZE):
            n_clues = random.randint(TIERS["hard"], TIERS["easy"])
            idxs = random.sample(range(SEQ_LEN), n_clues)
            clue_mask[i, idxs, 0] = 1.0
        c = x_clean * clue_mask

        loss = model.train_fpf_step(
            x_clean=x_clean, target_indices=target_indices, clue_mask=clue_mask, c=c,
            optimizer=optimizer, max_rollout_k=MAX_ROLLOUT_K, lambda_sparse=lambda_sparse, leak_free=True,
        )
        if step % 250 == 0 or step == steps:
            print(f"  [flowroute step {step:4d}/{steps}] loss={loss:.4f} elapsed={time.time() - t0:.1f}s")

    return model, time.time() - t0


def train_flowtrit_reference(train_boards: List[List[int]], steps: int, lr: float, seed: int) -> Tuple[FlowTritDenoiser, float]:
    """Router-free reference: same architecture, data, steps, and seed as flowroute (no router)."""
    set_all_seeds(seed)
    model = FlowTritDenoiser(
        seq_len=SEQ_LEN, num_classes=NUM_CLASSES, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, ternary=True
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    t0 = time.time()
    for step in range(1, steps + 1):
        batch_samples = random.choices(train_boards, k=BATCH_SIZE)
        target_indices = torch.tensor(batch_samples, dtype=torch.long, device=DEVICE)
        x_clean = F.one_hot(target_indices, num_classes=NUM_CLASSES).float()

        clue_mask = torch.zeros(BATCH_SIZE, SEQ_LEN, 1, device=DEVICE)
        for i in range(BATCH_SIZE):
            n_clues = random.randint(TIERS["hard"], TIERS["easy"])
            idxs = random.sample(range(SEQ_LEN), n_clues)
            clue_mask[i, idxs, 0] = 1.0
        c = x_clean * clue_mask

        loss = model.train_fpf_step(
            x_clean=x_clean, target_indices=target_indices, clue_mask=clue_mask, c=c,
            optimizer=optimizer, max_rollout_k=MAX_ROLLOUT_K, leak_free=True,
        )
        if step % 250 == 0 or step == steps:
            print(f"  [reference step {step:4d}/{steps}] loss={loss:.4f} elapsed={time.time() - t0:.1f}s")

    return model, time.time() - t0


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_with_layers(
    model,
    eval_puzzles: List[Dict[str, object]],
    max_steps: int,
    early_exit: bool,
    eps_exit: float,
    force_layer_action: Optional[str] = None,
) -> Dict[str, object]:
    """
    Evaluates solve rate, steps, and layer-execution counts. Works for both FlowRouteDenoiser
    (real per-sample bypass accounting from rollout_inference) and FlowTritDenoiser (no routing:
    every layer executes every step, so layer_passes = steps_taken * n_layers).
    """
    model.eval()
    is_routed = isinstance(model, FlowRouteDenoiser)
    tier_stats = {t: {"total": 0, "solved": 0, "steps": 0, "layers": 0} for t in TIERS_ORDER}
    total_layers_executed = 0
    total_layers_bypassed = 0

    with torch.no_grad():
        for puzzle in eval_puzzles:
            board, clues, tier = puzzle["board"], puzzle["clues"], puzzle["tier"]
            target = torch.tensor(board, dtype=torch.long, device=DEVICE).unsqueeze(0)
            x_clean = F.one_hot(target, num_classes=NUM_CLASSES).float()
            clue_mask = torch.zeros(1, SEQ_LEN, 1, device=DEVICE)
            idxs = list(clues.keys())
            clue_mask[0, idxs, 0] = 1.0
            c = x_clean * clue_mask

            t_tensor = torch.tensor([[0.8]], device=DEVICE)
            noise = torch.randn_like(x_clean)
            x_t = 0.2 * noise + 0.8 * c

            if is_routed:
                result = model.rollout_inference(
                    x_t=x_t, c=c, t=t_tensor, clue_mask=clue_mask, max_steps=max_steps,
                    early_exit=early_exit, eps_exit=eps_exit, force_layer_action=force_layer_action,
                )
                layer_passes = result["layers_executed"]
                total_layers_executed += result["layers_executed"]
                total_layers_bypassed += result["layers_bypassed"]
            else:
                result = model.rollout_inference(
                    x_t=x_t, c=c, t=t_tensor, clue_mask=clue_mask,
                    max_steps=max_steps, early_exit=early_exit, eps_exit=eps_exit,
                )
                layer_passes = result["steps_taken"] * N_LAYERS
                total_layers_executed += layer_passes

            pred = result["predictions"].squeeze(0).tolist()
            solved = MiniSudoku4x4.is_valid_solution(pred) and check_clue_adherence(pred, clues)

            tier_stats[tier]["total"] += 1
            if solved:
                tier_stats[tier]["solved"] += 1
            tier_stats[tier]["steps"] += result["steps_taken"]
            tier_stats[tier]["layers"] += layer_passes

    tier_summary = {}
    total_solved = total_count = total_steps = total_layers = 0
    for tier, d in tier_stats.items():
        cnt = d["total"]
        tier_summary[tier] = {
            "solve_rate_pct": round(d["solved"] / cnt * 100, 2) if cnt else 0.0,
            "mean_steps": round(d["steps"] / cnt, 3) if cnt else 0.0,
            "mean_layer_passes": round(d["layers"] / cnt, 3) if cnt else 0.0,
        }
        total_solved += d["solved"]
        total_count += cnt
        total_steps += d["steps"]
        total_layers += d["layers"]

    overall_mean_layers = round(total_layers / total_count, 3) if total_count else 0.0
    compute_savings_pct = round((1.0 - overall_mean_layers / (FIXED_MAX_STEPS * N_LAYERS)) * 100.0, 2)

    return {
        "overall_solve_rate_pct": round(total_solved / total_count * 100, 2) if total_count else 0.0,
        "overall_mean_steps": round(total_steps / total_count, 3) if total_count else 0.0,
        "overall_mean_layer_passes": overall_mean_layers,
        "compute_savings_pct": compute_savings_pct,
        "tier_summary": tier_summary,
        "total_layers_executed": total_layers_executed,
        "total_layers_bypassed": total_layers_bypassed,
    }


def _bypass_fraction(mode_result: Dict[str, object]) -> float:
    exec_n = mode_result["total_layers_executed"]
    byp_n = mode_result["total_layers_bypassed"]
    return byp_n / max(exec_n + byp_n, 1)


# ==============================================================================
# Main experiment
# ==============================================================================

def run_flowroute_experiment(
    steps: int = 1500,
    seed: int = 1,
    lambda_sparse: float = 0.005,
    router_warmup_frac: float = 0.3,
    lr: float = 2e-3,
    puzzles_per_tier: int = 100,
    output_json: str = "outputs/flowroute-test.json",
) -> Dict[str, object]:
    print("=" * 80)
    print(f"Track C: FlowRoute Combined Engine (Spatial + Temporal Routing) on {DEVICE}")
    print("=" * 80)

    train_boards, test_boards, _ = build_dataset("sudoku4")
    puzzle_rng = random.Random(seed * 1_000_003 + 11)
    eval_puzzles = build_eval_puzzles(test_boards, SEQ_LEN, TIERS, puzzles_per_tier, puzzle_rng)
    print(f"[+] Loaded {len(train_boards)} train boards, {len(eval_puzzles)} eval puzzles "
          f"({puzzles_per_tier}/tier).")

    print("\n--- Training FlowRoute (FPF + Gumbel-Softmax router, warmup-frozen probe) ---")
    flowroute_model, t_flowroute = train_flowroute(train_boards, steps, lambda_sparse, router_warmup_frac, lr, seed)
    print(f"[+] FlowRoute trained in {t_flowroute:.2f}s (measured)")

    print("\n--- Training FlowTrit reference (no router, same seed/data/steps) ---")
    reference_model, t_reference = train_flowtrit_reference(train_boards, steps, lr, seed)
    print(f"[+] Reference trained in {t_reference:.2f}s (measured)")

    print("\n--- Evaluating FlowRoute Mode 1: fixed_all_layers (force EXECUTE, k=5) ---")
    eval_fixed_all = evaluate_with_layers(flowroute_model, eval_puzzles, max_steps=FIXED_MAX_STEPS,
                                           early_exit=False, eps_exit=EPS_EXIT, force_layer_action="EXECUTE")
    print(f"  -> solve_rate={eval_fixed_all['overall_solve_rate_pct']}% layer_passes={eval_fixed_all['overall_mean_layer_passes']}")

    print("\n--- Evaluating FlowRoute Mode 2: step_exit_only (force EXECUTE, dynamic k) ---")
    eval_step_exit = evaluate_with_layers(flowroute_model, eval_puzzles, max_steps=DYNAMIC_MAX_STEPS,
                                           early_exit=True, eps_exit=EPS_EXIT, force_layer_action="EXECUTE")
    print(f"  -> solve_rate={eval_step_exit['overall_solve_rate_pct']}% steps={eval_step_exit['overall_mean_steps']} "
          f"layer_passes={eval_step_exit['overall_mean_layer_passes']}")

    print("\n--- Evaluating FlowRoute Mode 3: layer_routing_only (router on, k=5) ---")
    eval_layer_only = evaluate_with_layers(flowroute_model, eval_puzzles, max_steps=FIXED_MAX_STEPS,
                                            early_exit=False, eps_exit=EPS_EXIT, force_layer_action=None)
    layer_routing_only_bypass_fraction = round(_bypass_fraction(eval_layer_only), 4)
    print(f"  -> solve_rate={eval_layer_only['overall_solve_rate_pct']}% "
          f"layer_passes={eval_layer_only['overall_mean_layer_passes']} bypass_fraction={layer_routing_only_bypass_fraction}")

    print("\n--- Evaluating FlowRoute Mode 4: combined (router on, dynamic k) ---")
    eval_combined = evaluate_with_layers(flowroute_model, eval_puzzles, max_steps=DYNAMIC_MAX_STEPS,
                                          early_exit=True, eps_exit=EPS_EXIT, force_layer_action=None)
    combined_bypass_fraction = _bypass_fraction(eval_combined)
    print(f"  -> solve_rate={eval_combined['overall_solve_rate_pct']}% steps={eval_combined['overall_mean_steps']} "
          f"layer_passes={eval_combined['overall_mean_layer_passes']} savings={eval_combined['compute_savings_pct']}%")

    print("\n--- Evaluating FlowRoute Mode 5: random_bypass_matched "
          f"(bypass_prob={combined_bypass_fraction:.4f}, matched to combined) ---")
    for block in flowroute_model.blocks:
        block.router.random_bypass_prob = combined_bypass_fraction
    eval_random_matched = evaluate_with_layers(flowroute_model, eval_puzzles, max_steps=DYNAMIC_MAX_STEPS,
                                                early_exit=True, eps_exit=EPS_EXIT, force_layer_action=None)
    for block in flowroute_model.blocks:
        block.router.random_bypass_prob = None
    print(f"  -> solve_rate={eval_random_matched['overall_solve_rate_pct']}% "
          f"layer_passes={eval_random_matched['overall_mean_layer_passes']}")

    print("\n--- Evaluating FlowTrit reference: fixed (k=5) ---")
    eval_ref_fixed = evaluate_with_layers(reference_model, eval_puzzles, max_steps=FIXED_MAX_STEPS, early_exit=False, eps_exit=EPS_EXIT)
    print(f"  -> solve_rate={eval_ref_fixed['overall_solve_rate_pct']}%")

    print("\n--- Evaluating FlowTrit reference: dynamic_exit ---")
    eval_ref_dynamic = evaluate_with_layers(reference_model, eval_puzzles, max_steps=DYNAMIC_MAX_STEPS, early_exit=True, eps_exit=EPS_EXIT)
    print(f"  -> solve_rate={eval_ref_dynamic['overall_solve_rate_pct']}% steps={eval_ref_dynamic['overall_mean_steps']}")

    findings = {
        "combined_minus_reference_fixed_solve_rate": round(
            eval_combined["overall_solve_rate_pct"] - eval_ref_fixed["overall_solve_rate_pct"], 2),
        "combined_minus_random_matched_solve_rate": round(
            eval_combined["overall_solve_rate_pct"] - eval_random_matched["overall_solve_rate_pct"], 2),
        "combined_savings_pct": eval_combined["compute_savings_pct"],
        "layer_routing_only_bypass_fraction": layer_routing_only_bypass_fraction,
        "router_beats_random_control": bool(
            eval_combined["overall_solve_rate_pct"] > eval_random_matched["overall_solve_rate_pct"]),
    }

    results = {
        "status": "COMPLETED",
        "task": "Track C Combined FlowRoute Architecture",
        "device": str(DEVICE),
        "seed": seed,
        "config": {
            "steps": steps,
            "lambda_sparse": lambda_sparse,
            "router_warmup_frac": router_warmup_frac,
            "lr": lr,
            "puzzles_per_tier": puzzles_per_tier,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
        },
        "train_seconds": {
            "flowroute": measured(round(t_flowroute, 2)),
            "flowtrit_reference": measured(round(t_reference, 2)),
        },
        "flowroute_modes": {
            "fixed_all_layers": eval_fixed_all,
            "step_exit_only": eval_step_exit,
            "layer_routing_only": eval_layer_only,
            "combined": eval_combined,
            "random_bypass_matched": eval_random_matched,
        },
        "flowtrit_reference_modes": {
            "fixed": eval_ref_fixed,
            "dynamic_exit": eval_ref_dynamic,
        },
        "findings": findings,
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved to {out_path.resolve()}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lambda-sparse", type=float, default=0.005)
    parser.add_argument("--router-warmup-frac", type=float, default=0.3)
    parser.add_argument("--puzzles-per-tier", type=int, default=100)
    parser.add_argument("--json", type=str, default="outputs/flowroute-test.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    steps = args.steps
    puzzles_per_tier = args.puzzles_per_tier
    if args.smoke:
        steps = 30
        puzzles_per_tier = 5

    run_flowroute_experiment(
        steps=steps,
        seed=args.seed,
        lambda_sparse=args.lambda_sparse,
        router_warmup_frac=args.router_warmup_frac,
        puzzles_per_tier=puzzles_per_tier,
        output_json=args.json,
    )
