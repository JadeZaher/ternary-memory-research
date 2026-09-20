"""
experiments/mixture_of_tiles/bench_cpu_decode.py: measure the flash-resident decode loop on CPU.

One row per (mode, prompt length, cache capacity, injected miss latency): tokens/s, bytes per generated
token, tiles per token, cache hit rate, per-sequence working set, mean miss latency; cached rows also carry
the prefill time and decode-only tokens/s. Writes a new `outputs/mixture-of-tiles-state-cache.json` ledger by default.

    python experiments/mixture_of_tiles/bench_cpu_decode.py --mode both --prompt-tokens 32,64,128 --new-tokens 16 --no-cold
    python experiments/mixture_of_tiles/bench_cpu_decode.py --checkpoint outputs/checkpoints/general-G1-full-best.pt

Machine rules (research/disk-inference-tier.md section 4): refuses to start while GPU memory in use is above
`--gpu-guard-mib` or free RAM is below `--min-free-ram-gb`, stops when its own resident set passes
`--max-rss-gb`, and prints every token as it is generated.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard rule: this folder never touches the GPU

import sys
import json
import math
import argparse
import tempfile
import subprocess
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

torch.set_num_threads(4)
CPU_DEVICE = torch.device("cpu")

from experiments.general_model.general_model import GeneralConfig, GeneralRoutedLM
from experiments.general_model.train_general import PRESETS
from experiments.mixture_of_tiles.cached_decoder import CachedStreamingDecoder
from experiments.mixture_of_tiles.cpu_decoder import StreamingDecoder
from experiments.mixture_of_tiles.tile_store import (
    MEGABYTE,
    TileCache,
    TileStore,
    ternary_bitlinear_layers,
)

OUTPUT_PATH = os.path.join("outputs", "mixture-of-tiles-state-cache.json")
GIGABYTE = 1024 ** 3


# ------------------------------------------------------------------ machine guards --------------
def gpu_memory_used_mib() -> Optional[int]:
    """GPU memory in use according to nvidia-smi (None when there is no nvidia-smi); sums all GPUs."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if not lines or any(not line.isdigit() for line in lines):
        return None
    return sum(int(line) for line in lines)


def free_ram_gb() -> Optional[float]:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.virtual_memory().available / GIGABYTE


def resident_set_gb() -> Optional[float]:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / GIGABYTE


class MachineGuardError(RuntimeError):
    """Raised when machine telemetry is unavailable or exceeds a safety threshold."""


def check_machine_guards(args, *, gpu_usage=None, verbose=True) -> Dict[str, Any]:
    """Check GPU usage, available memory and process RSS before benchmark work."""
    status = {"gpu_memory_used_mib": gpu_memory_used_mib() if gpu_usage is None else gpu_usage,
              "free_ram_gb": free_ram_gb(), "rss_gb": resident_set_gb(),
              "gpu_guard_mib": args.gpu_guard_mib, "min_free_ram_gb": args.min_free_ram_gb,
              "max_rss_gb": args.max_rss_gb, "guards_ignored": False}
    problems = []
    for key in ("gpu_memory_used_mib", "free_ram_gb", "rss_gb"):
        if status[key] is None:
            problems.append(f"required telemetry unavailable: {key}")
    if status["gpu_memory_used_mib"] is not None and status["gpu_memory_used_mib"] > args.gpu_guard_mib:
        problems.append(f"GPU memory in use is {status['gpu_memory_used_mib']} MiB > guard {args.gpu_guard_mib} MiB "
                        "(a spilling trainer saturates RAM bandwidth; see research/disk-inference-tier.md section 4)")
    if status["free_ram_gb"] is not None and status["free_ram_gb"] < args.min_free_ram_gb:
        problems.append(f"free RAM is {status['free_ram_gb']:.1f} GB < {args.min_free_ram_gb} GB")
    if status["rss_gb"] is not None and status["rss_gb"] > args.max_rss_gb:
        problems.append(f"resident set {status['rss_gb']:.2f} GB exceeds --max-rss-gb {args.max_rss_gb}")
    if verbose:
        print(f"[guard] gpu used {status['gpu_memory_used_mib']} MiB (limit {args.gpu_guard_mib}), "
              f"free RAM {status['free_ram_gb']} GB (min {args.min_free_ram_gb}), "
              f"rss {status['rss_gb']} GB (cap {args.max_rss_gb})", flush=True)
    for problem in problems:
        print(f"[guard] {problem}")
    if problems:
        raise MachineGuardError("; ".join(problems))
    return status


def check_rss_cap(args) -> None:
    rss = resident_set_gb()
    if rss is None:
        raise MachineGuardError("required telemetry unavailable: rss_gb")
    if rss is not None and rss > args.max_rss_gb:
        raise MemoryError(f"resident set {rss:.2f} GB exceeds --max-rss-gb {args.max_rss_gb}")


class MachineGuardMonitor:
    """Check RAM/RSS every boundary and poll GPU usage at most once per second."""

    def __init__(self, args):
        self.args = args
        self.last_gpu_check = float("-inf")
        self.gpu_usage = None
        self.latest_status = None

    def check(self, force_gpu=False):
        now = time.monotonic()
        if force_gpu or now - self.last_gpu_check >= 1.0:
            self.gpu_usage = gpu_memory_used_mib()
            self.last_gpu_check = now
            if self.gpu_usage is None:
                raise MachineGuardError("required telemetry unavailable: gpu_memory_used_mib")
        self.latest_status = check_machine_guards(self.args, gpu_usage=self.gpu_usage, verbose=force_gpu)
        return self.latest_status


# ------------------------------------------------------------------ model -----------------------
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
    model.to(CPU_DEVICE).eval()
    model.allow_exit = bool(args.allow_exit)
    for _, layer in ternary_bitlinear_layers(model):
        # CPU inference uses the single dense GEMM through w_eff; BitLinear's additive path is two dense
        # GEMMs plus two mask matrices per layer, which doubles both RAM and time on this hardware.
        layer.use_additive_gemm = bool(args.additive_gemm)
    return {"model": model, "config": config, "source": source, "trained": trained}


# ------------------------------------------------------------------ one row ---------------------
def run_one(model, store: TileStore, prompt: torch.Tensor, args, capacity: int, injected: float,
            mode: str, safety_check=None) -> Dict[str, Any]:
    cache = TileCache(store, capacity_tiles=capacity, prefetch=args.prefetch,
                      injected_miss_latency_ms_per_mb=injected)
    if mode == "cached":
        decoder = CachedStreamingDecoder(model, cache, verbose=not args.quiet,
                                         safety_check=safety_check, build_masks=bool(args.additive_gemm))
    else:
        decoder = StreamingDecoder(model, cache, verbose=not args.quiet,
                                   safety_check=safety_check, build_masks=bool(args.additive_gemm))
    try:
        result = decoder.generate(prompt, args.new_tokens)
    finally:
        try:
            cache.clear()
        finally:
            decoder.detach()
    result["mode"] = mode
    result["capacity_tiles"] = capacity
    result["injected_miss_ms_per_mb"] = injected
    result["rss_gb_after"] = resident_set_gb()
    return result


HEADER = (f"  {'mode':>6}  {'prompt':>6}  {'cap':>3}  {'inj ms/MB':>9}  {'tok/s':>8}  {'decode tok/s':>12}  "
          f"{'prefill ms':>10}  {'MB/token':>9}  {'tiles/tok':>9}  {'hit rate':>8}  {'wset':>4}  "
          f"{'miss ms':>8}  {'call H':>6}")


def format_row(result: Dict[str, Any]) -> str:
    decode_tps = result.get("decode_tokens_per_second")
    prefill_ms = result.get("prefill_ms")
    return (f"  {result['mode']:>6}  {result['prompt_tokens']:>6}  {result['capacity_tiles']:>3}  "
            f"{result['injected_miss_ms_per_mb']:>9.2f}  {result['tokens_per_second']:>8.3f}  "
            f"{'-' if decode_tps is None else f'{decode_tps:.3f}':>12}  "
            f"{'-' if prefill_ms is None else f'{prefill_ms:.0f}':>10}  "
            f"{result['mean_mb_per_token']:>9.3f}  {result['mean_tiles_per_token']:>9.2f}  "
            f"{result['hit_rate'] * 100:>7.1f}%  {result['working_set_size']:>4}  "
            f"{result['mean_miss_latency_ms']:>8.2f}  {result['tile_usage_entropy_bits']:>6.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="a general-<tag>-best.pt; random init is used when absent or missing")
    parser.add_argument("--preset", choices=list(PRESETS), default="p512")
    parser.add_argument("--tile-kinds", default="mamba,mamba,mamba,attn")
    parser.add_argument("--mode", choices=["full", "cached", "both"], default="both",
                        help="full: recompute the prefix per token (reference); cached: (hop, tile) state")
    parser.add_argument("--capacity", default="1,2,4", help="comma-separated tile-cache capacities")
    parser.add_argument("--prompt-tokens", default="32", help="comma-separated prompt lengths (prefixes of one prompt)")
    parser.add_argument("--new-tokens", type=int, default=16)
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
    parser.add_argument("--quiet", action="store_true", help="suppress per-token progress lines")
    parser.add_argument("--check-only", action="store_true", help="check machine guards without allocating a model")
    parser.add_argument("--gpu-guard-mib", type=int, default=7000, help="refuse to start above this GPU usage")
    parser.add_argument("--min-free-ram-gb", type=float, default=4.0, help="refuse to start below this free RAM")
    parser.add_argument("--max-rss-gb", type=float, default=6.0, help="stop once this process holds more than this")
    args = parser.parse_args()

    if not 0 < args.gpu_guard_mib <= 7000:
        parser.error("--gpu-guard-mib must be positive and at most 7000")
    if not math.isfinite(args.min_free_ram_gb) or args.min_free_ram_gb < 4.0:
        parser.error("--min-free-ram-gb must be finite and at least 4")
    if not math.isfinite(args.max_rss_gb) or not 0 < args.max_rss_gb <= 6.0:
        parser.error("--max-rss-gb must be finite, positive and at most 6")

    monitor = MachineGuardMonitor(args)
    try:
        guard_status = monitor.check(force_gpu=True)
    except MachineGuardError as error:
        print(f"[guard] refusing to start: {error}", flush=True)
        raise SystemExit(2)
    if args.check_only:
        return

    if os.path.exists(args.output):
        parser.error(f"output already exists; choose a new ledger path: {args.output}")
    try:
        capacities = [int(value) for value in args.capacity.split(",") if value.strip()]
        prompt_lengths = [int(value) for value in args.prompt_tokens.split(",") if value.strip()]
        if not capacities or min(capacities) < 1 or not prompt_lengths or min(prompt_lengths) < 1:
            raise ValueError("capacities and prompt lengths must be nonempty positive integers")
        if args.new_tokens < 1 or args.injected_miss_ms_per_mb < 0:
            raise ValueError("new tokens must be positive and injected latency nonnegative")
    except ValueError as error:
        parser.error(str(error))
    modes = ["full", "cached"] if args.mode == "both" else [args.mode]
    injections = [0.0] if args.no_cold else sorted({0.0, float(args.injected_miss_ms_per_mb)})
    results: List[Dict[str, Any]] = []
    comparisons = []
    completed = False
    stop_reason = None
    built = None
    config = None
    footprint = None
    table = None
    store_dir = None
    temporary_store = None
    exit_code = 0
    try:
        torch.manual_seed(args.seed)
        monitor.check(force_gpu=True)
        built = build_model(args)
        model, config = built["model"], built["config"]
        monitor.check(force_gpu=True)
        if max(prompt_lengths) + args.new_tokens - 1 > config.max_position_embeddings:
            raise ValueError("prompt and generated sequence exceed max_position_embeddings")
        print(f"[model] {built['source']}", flush=True)
        footprint = model.count_parameters()
        print(f"[model] {footprint['total_millions']}M params, tiles={config.tile_kinds}, "
              f"hops={config.max_hops}, allow_exit={model.allow_exit}, "
              f"additive_gemm={bool(args.additive_gemm)}", flush=True)
        if args.store_dir is None:
            temporary_store = tempfile.TemporaryDirectory(prefix="mixture-of-tiles-")
            store_dir = temporary_store.name
        else:
            store_dir = args.store_dir
        monitor.check(force_gpu=True)
        table = TileStore.build(model, store_dir, canonicalize=True, verbose=True)
        monitor.check(force_gpu=True)
        longest = torch.randint(0, config.vocab_size, (1, max(prompt_lengths)),
                                dtype=torch.long, device=CPU_DEVICE)
        with TileStore.open(store_dir) as store:
            print(f"\n{HEADER}", flush=True)
            for prompt_tokens in prompt_lengths:
                prompt = longest[:, :prompt_tokens]
                for capacity in capacities:
                    for injected in injections:
                        paired_results = []
                        for mode in modes:
                            monitor.check(force_gpu=True)
                            print(f"[bench] mode={mode} prompt={prompt_tokens} capacity={capacity} inj={injected}",
                                  flush=True)
                            result = run_one(model, store, prompt, args, capacity, injected, mode,
                                             safety_check=monitor.check)
                            results.append(result)
                            paired_results.append(result)
                            print(format_row(result), flush=True)
                            monitor.check(force_gpu=True)
                        if len(paired_results) == 2:
                            identical = paired_results[0]["generated_ids"] == paired_results[1]["generated_ids"]
                            comparisons.append({"prompt_tokens": prompt_tokens, "capacity_tiles": capacity,
                                                "injected_miss_ms_per_mb": injected, "generated_ids_identical": identical})
                            if not identical:
                                raise RuntimeError("full and cached benchmark rows produced different generated ids")
        completed = True
    except (Exception, KeyboardInterrupt) as error:
        stop_reason = f"{type(error).__name__}: {error}"
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 2
        print(f"[bench] stopped: {stop_reason}; saving completed rows", flush=True)
    finally:
        try:
            payload = {
                "completed": completed,
                "stop_reason": stop_reason,
                "model_source": built["source"] if built else None,
                "trained": built["trained"] if built else None,
                "config": asdict(config) if config else None,
                "footprint": footprint,
                "byte_table": table,
                "protocols": {"full": "greedy batch-1, full recompute per token, no KV cache",
                              "cached": "greedy batch-1, cached (hop, tile) sequence state, one prefill pass then "
                                        "one pass per token; tokens_per_second includes prefill forwards; "
                                        "decode_tokens_per_second excludes prefill"},
                "timing_scope": "model forward durations plus unslept injected latency; excludes guard checks, "
                                "token selection, Python bookkeeping and progress printing",
                "cpu": "CPU only, torch threads 4",
                "injected_latency_is": "accounted, not slept; logical mmap reads include unpacking and do not "
                                       "measure physical cold-disk traffic",
                "machine_guards": guard_status,
                "last_machine_guards": monitor.latest_status,
                "args": vars(args),
                "comparisons": comparisons,
                "rows": [{key: value for key, value in result.items() if key not in ("per_token", "last_logits")}
                         for result in results],
                "per_token": {f"{result['mode']}-p{result['prompt_tokens']}-cap{result['capacity_tiles']}-"
                              f"inj{result['injected_miss_ms_per_mb']}": result["per_token"] for result in results},
            }
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            print(f"\n[bench] wrote {args.output} ({len(results)} completed rows)", flush=True)
        finally:
            if temporary_store is not None:
                temporary_store.cleanup()
                print(f"[bench] removed scratch store {store_dir}", flush=True)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
