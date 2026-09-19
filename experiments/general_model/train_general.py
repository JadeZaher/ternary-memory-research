"""
experiments/general_model/train_general.py: training + held-out read-out for `GeneralRoutedLM`.

Protocol (inherited from research/hardening-2026-09-18-heldout.md sections 5 and 13):
  * data        --data-dir with `train_tokens.pt` + one `val_<set>_tokens.pt` per held-out set, and an
                optional `manifest.json` carrying `vocab_size`, `eos_id` and `val_bytes_per_token`
                (bytes of raw text per token, per set -> bits per byte). `data/clean` works for smoke runs.
  * eval        FIXED seeded batches per set so numbers are comparable across steps and runs; per-set loss,
                ppl and BPB; `mean_bpb_core` over the target sets {tinystories, code, math}.
  * read-out    one trained model, eval only: exit-lambda sweep, random-continuation null bisected to the
                trained policy's mean hops, per-sequence capacity null at the same hops, hop-cap sweep.

Log lines are bracket-tagged so a monitor can grep them: [eval N], [traverse N], [value N], [lambda x],
[controls], [hopcap T], [footprint], and a final `DONE <tag>`.

Examples:
  python experiments/general_model/train_general.py --data-dir data/clean --smoke --batch-size 2 --seq-len 256
  python experiments/general_model/train_general.py --preset p512 --tag g1 --time-budget-h 6 --grad-checkpoint
"""

import os
import sys
import glob
import json
import math
import time
import argparse
import contextlib
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

from experiments.general_model.general_model import GeneralConfig, GeneralRoutedLM

# The sets the footprint/loss target is defined over; any other discovered set is reported separately.
CORE_SETS = ("tinystories", "code", "math")

PRESETS = {
    "p320": dict(hidden_size=320, intermediate_size=1280, num_attention_heads=5, dt_rank=20),
    "p384": dict(hidden_size=384, intermediate_size=1536, num_attention_heads=6, dt_rank=24),
    "p512": dict(hidden_size=512, intermediate_size=2048, num_attention_heads=8, dt_rank=32),
    "p768": dict(hidden_size=768, intermediate_size=3072, num_attention_heads=12, dt_rank=48),
}


def autocast_ctx(device: torch.device, amp_dtype: torch.dtype, enabled: bool):
    """CUDA autocast when it applies; a no-op on CPU (torch.autocast('cuda') probes the device even when disabled)."""
    if device.type != "cuda" or not enabled or not torch.cuda.is_available():
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=amp_dtype)


# ------------------------------------------------------------------ data ----------------------
def load_manifest(data_dir: str) -> Dict[str, Any]:
    """vocab_size / eos_id / val_bytes_per_token when the corpus builder wrote them; safe defaults otherwise."""
    path = os.path.join(data_dir, "manifest.json")
    info: Dict[str, Any] = {"vocab_size": 50257, "eos_id": None, "val_bytes_per_token": {}}
    if os.path.exists(path):
        with open(path) as handle:
            raw = json.load(handle)
        for key in ("vocab_size", "eos_id", "val_bytes_per_token"):
            if raw.get(key) is not None:
                info[key] = raw[key]
        info["manifest"] = {k: v for k, v in raw.items() if not isinstance(v, (dict, list))}
    return info


def load_data(data_dir: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    train = torch.load(os.path.join(data_dir, "train_tokens.pt"), weights_only=True)
    val: Dict[str, torch.Tensor] = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "val_*_tokens.pt"))):
        name = os.path.basename(path)[len("val_"):-len("_tokens.pt")]
        val[name] = torch.load(path, weights_only=True)
    if not val:
        raise FileNotFoundError(f"no val_*_tokens.pt in {data_dir}")
    return train, val


def sample_batch(tokens: torch.Tensor, batch_size: int, seq_len: int,
                 gen: Optional[torch.Generator] = None) -> torch.Tensor:
    max_start = len(tokens) - seq_len - 1
    index = torch.randint(0, max_start, (batch_size,), generator=gen)
    return torch.stack([tokens[i: i + seq_len + 1] for i in index]).long()


def fixed_eval_batches(val: Dict[str, torch.Tensor], batch_size: int, seq_len: int, iters: int,
                       seed: int) -> Dict[str, List[torch.Tensor]]:
    """Same batches at every eval and across runs: a fixed seeded sample per set."""
    out = {}
    for name, tokens in val.items():
        gen = torch.Generator().manual_seed(seed)
        n_iters = min(iters, max(1, (len(tokens) - 1) // (batch_size * seq_len)))
        out[name] = [sample_batch(tokens, batch_size, seq_len, gen) for _ in range(n_iters)]
    return out


# ------------------------------------------------------------------ eval ----------------------
@torch.no_grad()
def evaluate(model, batches, device, amp_dtype, bytes_per_token: Dict[str, float],
             max_loops: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    res: Dict[str, float] = {}
    hops_sum, hops_n = 0.0, 0
    for name, blist in batches.items():
        total, count = 0.0, 0
        for xy in blist:
            xy = xy.to(device)
            with autocast_ctx(device, amp_dtype, True):
                out = model(xy, labels=xy, max_loops=max_loops)
            n = xy.numel() - xy.size(0)
            total += out["ce_loss"].item() * n
            count += n
            hops_sum += out["mean_hops"]
            hops_n += 1
        loss = total / max(1, count)
        res[f"{name}_loss"] = round(loss, 4)
        res[f"{name}_ppl"] = round(math.exp(min(loss, 20.0)), 3)
        if bytes_per_token.get(name):
            res[f"{name}_bpb"] = round(loss / math.log(2) / float(bytes_per_token[name]), 4)
    res["mean_loss"] = round(sum(res[f"{n}_loss"] for n in batches) / len(batches), 4)
    core_loss = [res[f"{n}_loss"] for n in CORE_SETS if f"{n}_loss" in res]
    if core_loss:
        res["mean_loss_core"] = round(sum(core_loss) / len(core_loss), 4)
    core_bpb = [res[f"{n}_bpb"] for n in CORE_SETS if f"{n}_bpb" in res]
    if core_bpb:
        res["mean_bpb_core"] = round(sum(core_bpb) / len(core_bpb), 4)
    res["eval_mean_hops"] = round(hops_sum / max(1, hops_n), 3)
    model.train()
    return res


@torch.no_grad()
def eval_policy(model, batches, device, amp_dtype) -> Dict[str, float]:
    """Loss and mean hops under the model's current exit policy (used by the sweeps and the nulls)."""
    model.eval()
    total, count, hops, n = 0.0, 0, 0.0, 0
    for blist in batches.values():
        for xy in blist:
            xy = xy.to(device)
            with autocast_ctx(device, amp_dtype, True):
                out = model(xy, labels=xy)
            k = xy.numel() - xy.size(0)
            total += out["ce_loss"].item() * k
            count += k
            hops += out["mean_hops"]
            n += 1
    model.train()
    return {"mean_loss": round(total / max(1, count), 4), "mean_hops": round(hops / max(1, n), 3)}


def readout(model, batches, device, amp_dtype, args, bytes_per_token) -> Dict[str, Any]:
    """Eval-time read-out on one trained model: lambda sweep, matched nulls, hop-cap sweep."""
    cfg = model.config
    res: Dict[str, Any] = {"lambda_sweep": {}, "controls": {}, "hopcap_sweep": {}}

    if cfg.exit_mode == "value":
        cfg.exit_policy = "threshold"
        for lam in [float(x) for x in args.lambda_sweep.split(",")]:
            cfg.exit_lambda = lam
            res["lambda_sweep"][str(lam)] = eval_policy(model, batches, device, amp_dtype)
            print(f"  [lambda {lam}] {res['lambda_sweep'][str(lam)]}")
        cfg.exit_lambda = args.exit_lambda
        trained = res["lambda_sweep"].get(str(float(args.exit_lambda)))
        if trained is None:
            trained = eval_policy(model, batches, device, amp_dtype)
        target_hops = trained["mean_hops"]

        # random continuation, per-hop continue probability bisected to the trained policy's mean hops
        cfg.exit_policy = "random"
        probe = {next(iter(batches)): batches[next(iter(batches))][:2]}
        low, high = 0.0, 1.0
        for _ in range(12):
            cfg.random_continue_prob = (low + high) / 2
            measured = eval_policy(model, probe, device, amp_dtype)
            low, high = ((low + high) / 2, high) if measured["mean_hops"] < target_hops else (low, (low + high) / 2)
        res["controls"]["random_matched"] = {"continue_prob": round(cfg.random_continue_prob, 4),
                                             **eval_policy(model, batches, device, amp_dtype)}
        # capacity null: top fraction by predicted gain per sequence, at the same mean hops
        cfg.exit_policy = "capacity"
        cfg.capacity_frac = max(0.05, min(1.0, (target_hops - cfg.min_hops) / max(1, cfg.max_hops - cfg.min_hops)))
        res["controls"]["capacity_matched"] = {"capacity_frac": round(cfg.capacity_frac, 4),
                                               **eval_policy(model, batches, device, amp_dtype)}
        cfg.exit_policy = "threshold"
        res["controls"]["trained_threshold"] = trained
        print(f"  [controls] trained {trained} | random {res['controls']['random_matched']} | "
              f"capacity {res['controls']['capacity_matched']}")

    for hop_cap in range(1, cfg.max_hops + 1):
        res["hopcap_sweep"][f"T{hop_cap}"] = evaluate(model, batches, device, amp_dtype, bytes_per_token,
                                                      max_loops=hop_cap)
        summary = {k: v for k, v in res["hopcap_sweep"][f"T{hop_cap}"].items()
                   if k in ("mean_loss", "mean_bpb_core", "eval_mean_hops")}
        print(f"  [hopcap {hop_cap}] {summary}")
    return res


# ------------------------------------------------------------------ schedule / state ----------
def cosine_lr(step: int, warmup: int, total: int, base: float, minimum: float) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    ratio = min(1.0, (step - warmup) / max(1, total - warmup))
    return minimum + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (base - minimum)


def save_state(path: str, model, opt, gen, step: int, history: List[Dict], best: float,
               config: GeneralConfig, args, footprint: Dict[str, Any]) -> None:
    """Checkpoint carries optimizer, sampler and global RNG state so `--resume` is bit-reproducible."""
    payload = {
        "config": asdict(config), "args": vars(args), "footprint": footprint,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
        "gen_state": gen.get_state(), "torch_rng_state": torch.get_rng_state(),
        "step": step, "history": history, "best": best,
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def load_state(path: str, model, opt=None, gen=None, restore_rng: bool = True) -> Dict[str, Any]:
    # every value stored by `save_state` is a tensor / primitive / dict, so the safe loader suffices
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if opt is not None and checkpoint.get("optimizer_state_dict") is not None:
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
    if gen is not None and checkpoint.get("gen_state") is not None:
        gen.set_state(checkpoint["gen_state"])
    if restore_rng and checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return checkpoint


# ------------------------------------------------------------------ config -------------------
def build_config(args, vocab_size: int) -> GeneralConfig:
    preset = dict(PRESETS[args.preset])
    return GeneralConfig(
        vocab_size=vocab_size,
        # a batch carries seq_len+1 tokens (input and shifted labels share the tensor)
        max_position_embeddings=max(1025, args.seq_len + 1),
        tile_kinds=[k.strip() for k in args.tile_kinds.split(",") if k.strip()],
        routing=args.routing,
        max_hops=args.max_hops, min_hops=args.min_hops,
        exit_mode=args.exit_mode, exit_lambda=args.exit_lambda, explore_prob=args.explore,
        exit_warmup_steps=args.exit_warmup,
        router_cond=args.router_cond, adapter_cond=args.adapter_cond,
        adapter_bank=args.adapter_bank, adapter_rank=args.adapter_rank,
        tile_soft_mix=not args.no_soft_mix,
        ternary_embed=args.ternary_embed,
        deep_sup_weight=args.deep_sup, deep_sup_frac=args.deep_sup_frac,
        hop_cost_weight=args.hop_cost, hop_feature_dropout=args.hop_dropout,
        num_experts=args.num_experts,
        grad_checkpoint=args.grad_checkpoint,
        **preset,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/general")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--preset", choices=list(PRESETS), default="p512")
    ap.add_argument("--tile-kinds", default="mamba,mamba,mamba,attn")
    ap.add_argument("--routing", choices=["free", "fixed"], default="free")
    ap.add_argument("--exit-mode", choices=["value", "none"], default="value")
    ap.add_argument("--exit-lambda", type=float, default=0.02)
    ap.add_argument("--explore", type=float, default=0.3)
    ap.add_argument("--min-hops", type=int, default=2)
    ap.add_argument("--max-hops", type=int, default=8)
    ap.add_argument("--exit-warmup", type=int, default=500)
    ap.add_argument("--router-cond", choices=["state", "path"], default="state")
    ap.add_argument("--adapter-cond", choices=["hop", "path"], default="hop")
    ap.add_argument("--adapter-bank", type=int, default=4)
    ap.add_argument("--adapter-rank", type=int, default=16)
    ap.add_argument("--no-soft-mix", action="store_true", help="disable the per-token mixer/FFN branch weighting")
    ap.add_argument("--ternary-embed", action="store_true", help="ternarise the tied embedding/head matrix")
    ap.add_argument("--num-experts", type=int, default=2)
    ap.add_argument("--deep-sup", type=float, default=0.1)
    ap.add_argument("--deep-sup-frac", type=float, default=0.125)
    ap.add_argument("--hop-cost", type=float, default=0.0)
    ap.add_argument("--hop-dropout", type=float, default=0.0)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--time-budget-h", type=float, default=None,
                    help="measure throughput over 60 steps, then set max_steps so the whole run fits the budget")
    ap.add_argument("--readout-reserve-min", type=float, default=25.0, help="wall minutes reserved for the read-out")
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=16)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--lambda-sweep", default="0,0.005,0.01,0.02,0.05,0.1")
    ap.add_argument("--target-bpb", type=float, default=None)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="10 steps, no checkpoints, read-out still runs")
    ap.add_argument("--output-dir", default="outputs/checkpoints")
    ap.add_argument("--log-dir", default="outputs")
    return ap


# ------------------------------------------------------------------ main ---------------------
def main() -> None:
    args = build_parser().parse_args()
    tag = args.tag or f"{args.preset}-{args.routing}-{args.exit_mode}"
    if args.smoke:
        args.max_steps, args.eval_interval, args.eval_iters = 10, 5, 2
        args.save_interval, args.exit_warmup, args.time_budget_h = 10 ** 9, 2, None

    torch.manual_seed(args.seed)
    want_cuda = args.device != "cpu" and torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if want_cuda else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": torch.float32}[args.amp]
    use_amp = args.amp != "off" and device.type == "cuda"

    info = load_manifest(args.data_dir)
    train_tokens, val_tokens = load_data(args.data_dir)
    bytes_per_token = {k: float(v) for k, v in (info.get("val_bytes_per_token") or {}).items()}
    eval_batches = fixed_eval_batches(val_tokens, args.batch_size, args.seq_len, args.eval_iters, seed=12345)

    config = build_config(args, int(info["vocab_size"]))
    model = GeneralRoutedLM(config).to(device)
    footprint = model.count_parameters()
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len

    print("=" * 96)
    print(f"GeneralRoutedLM [{tag}] preset={args.preset} tiles={config.tile_kinds} routing={config.routing} "
          f"exit={config.exit_mode} hops={config.min_hops}..{config.max_hops} router_cond={config.router_cond} "
          f"adapter_cond={config.adapter_cond} soft_mix={config.tile_soft_mix} ternary_embed={config.ternary_embed}")
    print(f"[footprint] stored_mb_total={footprint['stored_mb_total']} tiles={footprint['tiles_packed_mb']} "
          f"embed={footprint['embed_packed_mb']} adapters={footprint['stored_mb']['adapters']} "
          f"controller={footprint['stored_mb']['controller']} | physical {footprint['total_millions']}M params")
    print(f"data {args.data_dir} vocab {config.vocab_size} | train {len(train_tokens):,} tokens | "
          f"val sets " + ", ".join(f"{k}={len(v):,}" for k, v in val_tokens.items()))
    print(f"tokens/step {tokens_per_step:,} | bpb sets {sorted(bytes_per_token) or 'none (no manifest bytes)'}")
    print("=" * 96)

    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"general-{tag}-log.json")
    latest_path = os.path.join(args.output_dir, f"general-{tag}-latest.pt")
    best_path = os.path.join(args.output_dir, f"general-{tag}-best.pt")

    history: List[Dict] = []
    best = float("inf")
    gen = torch.Generator().manual_seed(args.seed)
    start_step, elapsed_before = 1, 0.0
    if args.resume:
        checkpoint = load_state(latest_path, model, opt, gen)
        history = [h for h in checkpoint.get("history", []) if h["step"] <= checkpoint["step"]]
        best = checkpoint.get("best", min([h["mean_loss"] for h in history], default=float("inf")))
        elapsed_before = history[-1]["elapsed_s"] if history else 0.0
        start_step = checkpoint["step"] + 1
        print(f"[resume] {latest_path} -> step {start_step} (best mean_loss {best:.4f}, {len(history)} eval records)")

    t0 = time.time() - elapsed_before
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    calibrated = args.time_budget_h is None
    step = start_step - 1
    while step < args.max_steps:
        step += 1
        lr = cosine_lr(step, args.warmup, args.max_steps, args.lr, args.min_lr)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        model.allow_exit = step > args.exit_warmup
        loss_acc, ce_acc = 0.0, 0.0
        out: Dict[str, Any] = {}
        for _ in range(args.grad_accum):
            xy = sample_batch(train_tokens, args.batch_size, args.seq_len, gen).to(device)
            with autocast_ctx(device, amp_dtype, use_amp):
                out = model(xy, labels=xy)
                loss = out["loss"] / args.grad_accum
            scaler.scale(loss).backward()
            loss_acc += loss.item()
            ce_acc += out["ce_loss"].item() / args.grad_accum
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        scaler.step(opt)
        scaler.update()

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"step {step:6d} | loss {loss_acc:.4f} ce {ce_acc:.4f} | gnorm {grad_norm:.2f} | lr {lr:.2e} | "
                  f"hops {out['mean_hops']:.2f} | {tokens_per_step * (step - start_step + 1) / max(1e-6, elapsed):,.0f} tok/s | "
                  f"{elapsed / 60:.1f} min")

        if not calibrated and step - start_step + 1 >= 60:
            measured = time.time() - t0
            tok_s = tokens_per_step * (step - start_step + 1) / max(1e-6, measured)
            budget_s = args.time_budget_h * 3600.0 - args.readout_reserve_min * 60.0
            args.max_steps = max(step + 1, int(budget_s * tok_s / tokens_per_step))
            calibrated = True
            print(f"[budget] {tok_s:,.0f} tok/s over {step - start_step + 1} steps -> max_steps {args.max_steps} "
                  f"({args.max_steps * tokens_per_step / 1e6:.1f}M tokens, ~{budget_s / 3600:.2f} h train + "
                  f"{args.readout_reserve_min:.0f} min read-out)")

        if step % args.eval_interval == 0 or step == args.max_steps:
            ev = evaluate(model, eval_batches, device, amp_dtype, bytes_per_token)
            record = {"step": step, "train_loss": round(loss_acc, 4), "train_ce": round(ce_acc, 4), "lr": lr,
                      "grad_norm": round(grad_norm, 3), "elapsed_s": round(time.time() - t0, 1),
                      "tokens_seen": tokens_per_step * step, "train_mean_hops": round(out["mean_hops"], 3),
                      "tile_usage": [round(u, 3) for u in out["tile_usage"]],
                      "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0,
                      **ev}
            history.append(record)
            print(f"  [eval {step}] " + " | ".join(f"{k} {v}" for k, v in ev.items()) +
                  f" | peak VRAM {record['peak_vram_gb']} GB")
            print(f"  [traverse {step}] mean hops {out['mean_hops']:.2f} / {config.max_hops} | "
                  f"tile usage {[round(u, 3) for u in out['tile_usage']]}")
            if config.exit_mode == "value" or config.deep_sup_weight > 0:
                record.update({"deep_sup_loss": round(out["deep_sup_loss"], 4),
                               "value_loss": round(out["value_loss"], 4),
                               "mean_measured_gain": round(out["mean_measured_gain"], 4),
                               "mean_pred_gain": round(out["mean_pred_gain"], 4)})
                print(f"  [value {step}] measured gain {out['mean_measured_gain']:.4f} | "
                      f"predicted {out['mean_pred_gain']:.4f} | value loss {out['value_loss']:.4f} | "
                      f"deep-sup {out['deep_sup_loss']:.3f}")
            if args.target_bpb is not None and "mean_bpb_core" in ev:
                verdict = "PASS" if ev["mean_bpb_core"] <= args.target_bpb else "FAIL"
                record["target_bpb_pass"] = verdict == "PASS"
                print(f"  [target {step}] mean_bpb_core {ev['mean_bpb_core']} vs target {args.target_bpb}: {verdict}")
            if ev["mean_loss"] < best and not args.smoke:
                best = ev["mean_loss"]
                save_state(best_path, model, opt, gen, step, history, best, config, args, footprint)
            with open(log_path, "w") as handle:
                json.dump({"tag": tag, "args": vars(args), "config": asdict(config), "footprint": footprint,
                           "data": info.get("manifest", {}), "history": history}, handle, indent=2)

        if step % args.save_interval == 0 and not args.smoke:
            save_state(latest_path, model, opt, gen, step, history, best, config, args, footprint)

    result = readout(model, eval_batches, device, amp_dtype, args, bytes_per_token)
    final = evaluate(model, eval_batches, device, amp_dtype, bytes_per_token)
    target_check = None
    if args.target_bpb is not None:
        got = final.get("mean_bpb_core")
        target_check = {"target_bpb": args.target_bpb, "mean_bpb_core": got,
                        "pass": bool(got is not None and got <= args.target_bpb)}
        print(f"[target final] {target_check}")
    print(f"[footprint] stored_mb_total={footprint['stored_mb_total']} tiles={footprint['tiles_packed_mb']} "
          f"embed={footprint['embed_packed_mb']} adapters={footprint['stored_mb']['adapters']} "
          f"controller={footprint['stored_mb']['controller']}")
    ledger = {
        "tag": tag, "args": vars(args), "config": asdict(config), "footprint": footprint,
        "data": info.get("manifest", {}), "history": history, "final_eval": final,
        "lambda_sweep": result["lambda_sweep"], "controls": result["controls"],
        "hopcap_sweep": result["hopcap_sweep"],
        "best_mean_val_loss": best if best < float("inf") else final["mean_loss"],
        "target_check": target_check,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0,
    }
    with open(log_path, "w") as handle:
        json.dump(ledger, handle, indent=2)
    print(f"Saved {log_path}")
    print(f"DONE {tag}")


if __name__ == "__main__":
    main()
