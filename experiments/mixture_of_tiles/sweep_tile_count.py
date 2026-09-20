"""
experiments/mixture_of_tiles/sweep_tile_count.py: plan the fixed-bytes tile-count sweep.

Question the sweep answers: at a FIXED stored-byte budget, does quality survive splitting the block into
many small tiles? A larger tile count is what makes paging pay -- more, smaller tiles means a smaller hot
set and fewer bytes per token -- but only if the quality holds. This script solves for the
`intermediate_size` that keeps `tiles_ternary + tiles_dense` at the reference budget for each tile count,
and emits the exact training command. It runs NO training.

    python experiments/mixture_of_tiles/sweep_tile_count.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard rule: this folder never touches the GPU

import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

torch.set_num_threads(4)
CPU_DEVICE = torch.device("cpu")

from experiments.general_model.general_model import GeneralConfig, GeneralRoutedLM
from experiments.general_model.train_general import PRESETS
from experiments.mixture_of_tiles.tile_store import (
    MEGABYTE,
    packed_byte_length,
    ternary_bitlinear_layers,
)

OUTPUT_PATH = os.path.join("outputs", "mixture-of-tiles-sweep-plan.json")
INTERMEDIATE_STEP = 8            # the grid the planner searches; see `shape_constraints`
MIN_INTERMEDIATE = 8
MAX_INTERMEDIATE = 16384
BLOCK_SIZE = 256                 # BitLinear's absmean block

# The night-1 command every plan is a variation of; only --tile-kinds and the preset change.
BASE_TRAIN_FLAGS = [
    "--data-dir", "data/general",
    "--routing", "free",
    "--exit-mode", "value",
    "--exit-lambda", "0.02",
    "--explore", "0.3",
    "--min-hops", "2",
    "--max-hops", "8",
    "--exit-warmup", "500",
    "--router-cond", "state",
    "--adapter-cond", "hop",
    "--seq-len", "1024",
    "--batch-size", "4",
    "--grad-accum", "4",
    "--grad-checkpoint",
    "--time-budget-h", "8",
    "--readout-reserve-min", "30",
    "--eval-interval", "250",
    "--save-interval", "250",
    "--target-bpb", "1.3909",
]


def tile_kinds_for(count: int) -> List[str]:
    """Keep the Samba/Jamba 3:1 mamba:attn ratio by repeating the reference quartet."""
    if count % 4 != 0:
        raise ValueError(f"tile count {count} cannot hold a 3:1 mamba:attn ratio")
    return ["mamba", "mamba", "mamba", "attn"] * (count // 4)


def shape_constraints(hidden: int, intermediate: int, heads: int) -> Dict[str, Any]:
    """What `GeneralConfig` / `BitLinear` actually require of a candidate shape."""
    return {
        "intermediate_is_multiple_of_8": intermediate % INTERMEDIATE_STEP == 0,
        "swiglu_block_scaled": (intermediate * hidden) % BLOCK_SIZE == 0,     # else BitLinear falls back to
                                                                             # one layer-wide gamma
        "hidden_divisible_by_heads": hidden % heads == 0,
        "tile_count_multiple_of_4": True,
    }


def make_config(base: Dict[str, Any], count: int, intermediate: int) -> GeneralConfig:
    fields = dict(base)
    fields["tile_kinds"] = tile_kinds_for(count)
    fields["intermediate_size"] = int(intermediate)
    return GeneralConfig(**fields)


_MEASURE_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def measure(base: Dict[str, Any], count: int, intermediate: int) -> Dict[str, Any]:
    """Instantiate on CPU and read the real accounting off `count_parameters`, plus the packed store size."""
    key = (tuple(sorted((str(name), str(value)) for name, value in base.items())), count, intermediate)
    if key in _MEASURE_CACHE:
        return dict(_MEASURE_CACHE[key])
    config = make_config(base, count, intermediate)
    model = GeneralRoutedLM(config)
    footprint = model.count_parameters()
    packed = packed_store_bytes(model)
    del model
    result = {
        "tile_count": count,
        "intermediate_size": int(intermediate),
        "tile_params": int(footprint["params"]["tiles_ternary"] + footprint["params"]["tiles_dense"]),
        "tiles_ternary_params": int(footprint["params"]["tiles_ternary"]),
        "tiles_dense_params": int(footprint["params"]["tiles_dense"]),
        "tile_bytes": float(footprint["stored_bytes"]["tiles_ternary"]
                            + footprint["stored_bytes"]["tiles_dense"]),
        "tile_mb": float(footprint["tiles_packed_mb"]),
        "packed_store_bytes": int(packed["total"]),
        "packed_store_mb": packed["total"] / MEGABYTE,
        "packed_ternary_bytes": int(packed["ternary"]),
        "packed_gamma_bytes": int(packed["gamma"]),
        "packed_dense_bytes": int(packed["dense"]),
        "total_millions": float(footprint["total_millions"]),
        "stored_mb_total": float(footprint["stored_mb_total"]),
    }
    _MEASURE_CACHE[key] = result
    return dict(result)


def packed_store_bytes(model) -> Dict[str, int]:
    """Exact `tiles.bin` size for this model: base-3 ternary + fp16 gamma + fp16 dense."""
    ternary_bytes = 0
    gamma_bytes = 0
    ternary_names = set()
    for name, layer in ternary_bitlinear_layers(model.tiles):
        numel = layer.weight.numel()
        ternary_bytes += packed_byte_length(numel)
        blocks = numel // BLOCK_SIZE if numel % BLOCK_SIZE == 0 else 1
        gamma_bytes += blocks * 2
        ternary_names.add(f"{name}.weight")
    dense_bytes = sum(tensor.numel() * 2 for name, tensor in model.tiles.state_dict().items()
                      if name not in ternary_names)
    return {"ternary": ternary_bytes, "gamma": gamma_bytes, "dense": dense_bytes,
            "total": ternary_bytes + gamma_bytes + dense_bytes}


def solve_intermediate(base: Dict[str, Any], count: int, target_bytes: float,
                       tolerance: float) -> Dict[str, Any]:
    """Largest grid `intermediate_size` whose tile bytes stay at the reference budget (bytes rise strictly
    with intermediate, so a bisection on the grid is exact)."""
    floor = measure(base, count, MIN_INTERMEDIATE)
    if floor["tile_bytes"] > target_bytes * (1.0 + tolerance):
        floor["feasible"] = False
        floor["deviation"] = floor["tile_bytes"] / target_bytes - 1.0
        floor["reason"] = (
            "the per-tile floor already exceeds the budget: the sequence mixer (Mamba in_proj/out_proj or "
            "attention q/k/v/o) and the dense Mamba block (conv1d, x_proj, dt_proj, A_log, D) do not shrink "
            "with intermediate_size, so they scale linearly with tile count at fixed hidden_size")
        return floor

    low = MIN_INTERMEDIATE                                   # always known to fit
    high = min(MIN_INTERMEDIATE * 2, MAX_INTERMEDIATE)       # grown until it does not fit
    while high < MAX_INTERMEDIATE and measure(base, count, high)["tile_bytes"] <= target_bytes:
        low = high
        high = min(high * 2, MAX_INTERMEDIATE)
    while high - low > INTERMEDIATE_STEP:
        mid = ((low + high) // 2 // INTERMEDIATE_STEP) * INTERMEDIATE_STEP
        mid = max(low + INTERMEDIATE_STEP, min(mid, high - INTERMEDIATE_STEP))
        if measure(base, count, mid)["tile_bytes"] <= target_bytes:
            low = mid
        else:
            high = mid
    best = None
    for candidate in (low, min(low + INTERMEDIATE_STEP, MAX_INTERMEDIATE)):
        result = measure(base, count, candidate)
        result["deviation"] = result["tile_bytes"] / target_bytes - 1.0
        if best is None or abs(result["deviation"]) < abs(best["deviation"]):
            best = result
    best["feasible"] = abs(best["deviation"]) <= tolerance
    best["reason"] = "" if best["feasible"] else "no grid point lands inside the tolerance"
    return best


def preset_name(preset: str, intermediate: int, reference_intermediate: int) -> str:
    return preset if intermediate == reference_intermediate else f"{preset}i{intermediate}"


def train_command(base: Dict[str, Any], preset: str, count: int, intermediate: int,
                  reference_intermediate: int, tag: str) -> str:
    kinds = ",".join(tile_kinds_for(count))
    parts = ["python", "experiments/general_model/train_general.py",
             "--preset", preset_name(preset, intermediate, reference_intermediate),
             "--tile-kinds", kinds] + BASE_TRAIN_FLAGS + ["--tag", tag]
    return " ".join(parts)


def plan_sweep(preset: str = "p512", reference_count: int = 4, tile_counts: Optional[List[int]] = None,
               tolerance: float = 0.03, base_overrides: Optional[Dict[str, Any]] = None,
               verbose: bool = True) -> Dict[str, Any]:
    """Solve every tile count against the reference tile-byte budget and emit the plan."""
    tile_counts = tile_counts or [4, 8, 16, 32, 64]
    base = dict(PRESETS[preset]) if preset in PRESETS else {}
    base.update(base_overrides or {})
    base.setdefault("num_experts", 2)
    reference_intermediate = int(base["intermediate_size"])
    hidden = int(base["hidden_size"])
    heads = int(base["num_attention_heads"])

    reference = measure(base, reference_count, reference_intermediate)
    target_bytes = reference["tile_bytes"]
    if verbose:
        print(f"[reference] preset={preset} tiles={reference_count} intermediate={reference_intermediate} "
              f"-> tile params {reference['tile_params']:,}, tile bytes {target_bytes:,.0f} "
              f"({target_bytes / MEGABYTE:.3f} MB, packed {reference['packed_store_mb']:.3f} MB), "
              f"model {reference['total_millions']}M")

    plans: List[Dict[str, Any]] = []
    for count in tile_counts:
        solved = solve_intermediate(base, count, target_bytes, tolerance)
        intermediate = int(solved["intermediate_size"])
        tag = f"MoT-T{count}"
        entry = {
            **solved,
            "tile_kinds": tile_kinds_for(count),
            "mamba_tiles": 3 * count // 4,
            "attn_tiles": count // 4,
            "hidden_size": hidden,
            "num_experts": int(base["num_experts"]),
            "deviation_pct": 100.0 * solved["deviation"],
            "param_deviation_pct": 100.0 * (solved["tile_params"] / reference["tile_params"] - 1.0),
            "reference_tile_bytes": target_bytes,
            "shape_constraints": shape_constraints(hidden, intermediate, heads),
            "preset_name": preset_name(preset, intermediate, reference_intermediate),
            "preset_patch": None if intermediate == reference_intermediate else {
                preset_name(preset, intermediate, reference_intermediate):
                    {**{key: base[key] for key in ("hidden_size", "num_attention_heads", "dt_rank")
                        if key in base}, "intermediate_size": intermediate}},
            "tag": tag,
            "command": train_command(base, preset, count, intermediate, reference_intermediate, tag),
        }
        plans.append(entry)
        if verbose:
            flag = "ok " if entry["feasible"] else "INFEASIBLE"
            print(f"[plan] tiles={count:>3} intermediate={intermediate:>5} "
                  f"tile MB {entry['tile_mb']:>7.3f} packed MB {entry['packed_store_mb']:>7.3f} "
                  f"dev {entry['deviation_pct']:+6.2f}% params {entry['param_deviation_pct']:+7.2f}%  {flag}")
            if entry["reason"]:
                print(f"         {entry['reason']}")

    payload = {
        "preset": preset,
        "reference": {**reference, "tile_kinds": tile_kinds_for(reference_count)},
        "tolerance": tolerance,
        "objective": "count_parameters() stored_bytes[tiles_ternary] + stored_bytes[tiles_dense] "
                     "held at the reference value (the track's stored-byte objective)",
        "grid": {"intermediate_step": INTERMEDIATE_STEP, "min_intermediate": MIN_INTERMEDIATE},
        "known_constraints": [
            "train_general.py has no --intermediate-size flag: intermediate_size comes from PRESETS, so a "
            "non-reference plan needs the one-line PRESETS entry carried in `preset_patch`.",
            "intermediate_size is searched on a multiple-of-8 grid; BitLinear needs "
            "(intermediate x hidden) % 256 == 0 for block-scaled gamma, which multiples of 8 satisfy at "
            "hidden 512 and at hidden 64.",
            "tile counts must be multiples of 4 to hold the 3:1 mamba:attn ratio exactly.",
        ],
        "plans": plans,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default="p512")
    parser.add_argument("--counts", default="4,8,16,32,64")
    parser.add_argument("--reference-count", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    counts = [int(value) for value in args.counts.split(",") if value.strip()]
    payload = plan_sweep(preset=args.preset, reference_count=args.reference_count,
                         tile_counts=counts, tolerance=args.tolerance)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[sweep] wrote {args.output}")


if __name__ == "__main__":
    main()
