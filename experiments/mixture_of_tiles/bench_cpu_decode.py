"""
experiments/mixture_of_tiles/bench_cpu_decode.py: measure the flash-resident decode loop on CPU.

One row per (cache capacity, injected miss latency): tokens/s, bytes per generated token, tiles per token,
cache hit rate, per-sequence working set, mean miss latency. Writes `outputs/mixture-of-tiles-bench.json`.

    python experiments/mixture_of_tiles/bench_cpu_decode.py --capacity 1,2,4 --prompt-tokens 64 --new-tokens 32
    python experiments/mixture_of_tiles/bench_cpu_decode.py --checkpoint outputs/checkpoints/general-G1-full-best.pt
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard rule: this folder never touches the GPU

import sys
import json
import shutil
import argparse
import tempfile
from dataclasses import asdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

torch.set_num_threads(4)
CPU_DEVICE = torch.device("cpu")

from experiments.general_model.general_model import GeneralConfig, GeneralRoutedLM
from experiments.general_model.train_general import PRESETS
from experiments.mixture_of_tiles.cpu_decoder import StreamingDecoder
from experiments.mixture_of_tiles.tile_store import (
    MEGABYTE,
    TileCache,
    TileStore,
    ternary_bitlinear_layers,
)

OUTPUT_PATH = os.path.join("outputs", "mixture-of-tiles-bench.json")


def build_model(args) -> Dict[str, Any]:
    """Load a trained checkpoint if one was given and exists; otherwise a random-init model, and say so."""
    if args.checkpoint and os.path.exists(args.checkpoint):
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = GeneralConfig(**payload["config"])
        model = GeneralRoutedLM(config)
        model.load_state_dict(payload["model_state_dict"])
        source = f"checkpoint {args.checkpoint} (step {payload.get('step', '?')})"
        trained = True
    else:
        if args.checkpoint:
            print(f"[model] checkpoint {args.checkpoint} not found; falling back to random init")
        preset = dict(PRESETS[args.preset])
        config = GeneralConfig(
            tile_kinds=[kind.strip() for kind in args.tile_kinds.split(",") if kind.strip()],
            max_hops=args.max_hops, min_hops=args.min_hops, **preset)
        model = GeneralRoutedLM(config)
        source = f"RANDOM INIT preset={args.preset} (NOT a trained model)"
        trained = False
    model.eval()
    model.allow_exit = bool(args.allow_exit)
    for _, layer in ternary_bitlinear_layers(model):
        # CPU inference uses the single dense GEMM through w_eff; BitLinear's additive path is two dense
        # GEMMs plus two mask matrices per layer, which doubles both RAM and time on this hardware.
        layer.use_additive_gemm = bool(args.additive_gemm)
    return {"model": model, "config": config, "source": source, "trained": trained}


def run_one(model, store: TileStore, prompt: torch.Tensor, args, capacity: int,
            injected: float) -> Dict[str, Any]:
    cache = TileCache(store, capacity_tiles=capacity, prefetch=args.prefetch,
                      injected_miss_latency_ms_per_mb=injected)
    decoder = StreamingDecoder(model, cache, build_masks=bool(args.additive_gemm))
    try:
        result = decoder.generate(prompt, args.new_tokens)
    finally:
        decoder.detach()
    result["capacity_tiles"] = capacity
    result["injected_miss_ms_per_mb"] = injected
    return result


def format_row(result: Dict[str, Any]) -> str:
    return (f"  {result['capacity_tiles']:>8}  {result['injected_miss_ms_per_mb']:>10.2f}  "
            f"{result['tokens_per_second']:>9.3f}  {result['mean_mb_per_token']:>11.3f}  "
            f"{result['mean_tiles_per_token']:>11.2f}  {result['hit_rate'] * 100:>8.1f}%  "
            f"{result['working_set_size']:>11}  {result['mean_miss_latency_ms']:>13.2f}  "
            f"{result['tile_usage_entropy_bits']:>8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="a general-<tag>-best.pt; random init is used when absent or missing")
    parser.add_argument("--preset", choices=list(PRESETS), default="p512")
    parser.add_argument("--tile-kinds", default="mamba,mamba,mamba,attn")
    parser.add_argument("--capacity", default="1,2,4", help="comma-separated tile-cache capacities")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--injected-miss-ms-per-mb", type=float, default=0.33,
                        help="modelled cold-NVMe cost; 0.33 ms/MB is about 3 GB/s")
    parser.add_argument("--no-cold", action="store_true", help="skip the injected-latency rows")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--allow-exit", action="store_true",
                        help="let the value head exit early; meaningless on a random-init model")
    parser.add_argument("--additive-gemm", action="store_true",
                        help="keep BitLinear's mask-based additive path instead of the single dense GEMM")
    parser.add_argument("--max-hops", type=int, default=8)
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--store-dir", default=None, help="where tiles.bin lives; a temp dir by default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    built = build_model(args)
    model, config = built["model"], built["config"]
    print(f"[model] {built['source']}")
    if not built["trained"]:
        print("[model] a random-init tile router routes close to uniformly: the byte and latency numbers "
              "below are real, the ROUTING PATTERN they come from is not a learned one.")
    footprint = model.count_parameters()
    print(f"[model] {footprint['total_millions']}M params, tiles={config.tile_kinds}, "
          f"hops={config.max_hops}, allow_exit={model.allow_exit}, "
          f"additive_gemm={bool(args.additive_gemm)}")

    store_dir = args.store_dir or tempfile.mkdtemp(prefix="mixture-of-tiles-")
    table = TileStore.build(model, store_dir, canonicalize=True, verbose=True)

    capacities = [int(value) for value in args.capacity.split(",") if value.strip()]
    injections = [0.0] if args.no_cold else sorted({0.0, float(args.injected_miss_ms_per_mb)})
    prompt = torch.randint(0, config.vocab_size, (1, args.prompt_tokens), dtype=torch.long)

    results: List[Dict[str, Any]] = []
    with TileStore.open(store_dir) as store:
        print(f"\n  {'capacity':>8}  {'inj ms/MB':>10}  {'tok/s':>9}  {'MB/token':>11}  "
              f"{'tiles/token':>11}  {'hit rate':>9}  {'workingset':>11}  {'miss ms':>13}  {'H bits':>8}")
        for capacity in capacities:
            for injected in injections:
                result = run_one(model, store, prompt, args, capacity, injected)
                results.append(result)
                print(format_row(result))

    payload = {
        "model_source": built["source"],
        "trained": built["trained"],
        "config": asdict(config),
        "footprint": footprint,
        "byte_table": table,
        "protocol": "greedy batch-1, full recompute per token, no KV cache, CPU only, torch threads 4",
        "injected_latency_is": "accounted, not slept (Windows page cache makes a real cold read unmeasurable "
                               "after the first pass; see AGENTS.md)",
        "args": vars(args),
        "rows": [{key: value for key, value in result.items() if key != "per_token"} for result in results],
        "per_token": {f"cap{result['capacity_tiles']}-inj{result['injected_miss_ms_per_mb']}":
                      result["per_token"] for result in results},
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\n[bench] wrote {args.output}")

    if args.store_dir is None:
        shutil.rmtree(store_dir, ignore_errors=True)
        print(f"[bench] removed scratch store {store_dir}")


if __name__ == "__main__":
    main()
