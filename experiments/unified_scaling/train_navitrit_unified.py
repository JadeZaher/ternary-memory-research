"""
experiments/unified_scaling/train_navitrit_unified.py: Training + clean evaluation for NaviTrit-Unified.

Protocol (research/hardening-2026-09-18-heldout.md):
  * data      data/clean/*  (document-level split, decontaminated; build with experiments/data/prepare_clean_corpus.py)
  * eval      per-domain held-out perplexity (tinystories / code / math) on FIXED seeded batches, so numbers
              are comparable across steps and across runs; plus a loop-budget sweep T=1..max at the end
  * ablations one flag each: --routing-mode dense|soft|mod, --no-dwp, --num-experts 1, --fp-control,
              --max-loops N. Every routing/DWP claim must be reported against --routing-mode dense --no-dwp
              at the same preset, token budget and seed.

Presets (--preset):
  pilot   d=512,  4 macro-layers x 4 loops  (~50M physical)   -> ablation matrix on an RTX 4060 in hours
  max     d=1024, 4 macro-layers x 6 loops  (~210M physical)  -> needs --grad-checkpoint on 8 GB

Examples:
  python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode dense --no-dwp --tag pilot-dense --max-steps 3000
  python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode mod --tag pilot-mod --max-steps 3000
  python experiments/unified_scaling/train_navitrit_unified.py --preset max --routing-mode mod --grad-checkpoint --tag max-mod --max-steps 12000
"""

import os
import sys
import time
import math
import json
import argparse
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.unified_scaling.navitrit_unified_model import NaviTritUnifiedConfig, NaviTritUnifiedForCausalLM, embedding_weight
from experiments.unified_scaling.navitrit_traverse import NaviTritTraverseForCausalLM, TraverseConfig


PRESETS = {
    "pilot": dict(hidden_size=512, intermediate_size=2048, num_attention_heads=8, num_macro_layers=4, max_loops=4,
                  d_state=16, dt_rank=32, lora_rank=32, use_mamba=False),
    "max": dict(hidden_size=1024, intermediate_size=4096, num_attention_heads=16, num_macro_layers=4, max_loops=6,
                d_state=32, dt_rank=64, lora_rank=64),
}


# ------------------------------------------------------------------ data ----------------------
def load_clean_data(data_dir: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    train = torch.load(os.path.join(data_dir, "train_tokens.pt"), weights_only=True)
    val = {}
    for name in ("tinystories", "code", "math"):
        p = os.path.join(data_dir, f"val_{name}_tokens.pt")
        if os.path.exists(p):
            val[name] = torch.load(p, weights_only=True)
    if not val:
        raise FileNotFoundError(f"No val_*_tokens.pt in {data_dir}; run experiments/data/prepare_clean_corpus.py")
    return train, val


def sample_batch(tokens: torch.Tensor, batch_size: int, seq_len: int, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    max_start = len(tokens) - seq_len - 1
    ix = torch.randint(0, max_start, (batch_size,), generator=gen)
    return torch.stack([tokens[i: i + seq_len + 1] for i in ix]).long()


def fixed_eval_batches(val: Dict[str, torch.Tensor], batch_size: int, seq_len: int, iters: int, seed: int) -> Dict[str, List[torch.Tensor]]:
    out = {}
    for name, toks in val.items():
        gen = torch.Generator().manual_seed(seed)
        n_iters = min(iters, max(1, (len(toks) - 1) // (batch_size * seq_len)))
        out[name] = [sample_batch(toks, batch_size, seq_len, gen) for _ in range(n_iters)]
    return out


# ------------------------------------------------------------------ eval ----------------------
@torch.no_grad()
def evaluate(model, batches: Dict[str, List[torch.Tensor]], device, amp_dtype, max_loops: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    res: Dict[str, float] = {}
    for name, blist in batches.items():
        total, count = 0.0, 0
        for xy in blist:
            xy = xy.to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                out = model(xy, labels=xy, max_loops=max_loops)
            n = xy.numel() - xy.size(0)
            total += out["ce_loss"].item() * n
            count += n
        loss = total / max(1, count)
        res[f"{name}_loss"] = round(loss, 4)
        res[f"{name}_ppl"] = round(math.exp(min(loss, 20.0)), 3)
    res["mean_loss"] = round(sum(res[f"{n}_loss"] for n in batches) / len(batches), 4)
    model.train()
    return res


# ------------------------------------------------------------------ train ---------------------
def cosine_lr(step: int, warmup: int, total: int, base: float, minimum: float) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    ratio = min(1.0, (step - warmup) / max(1, total - warmup))
    return minimum + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (base - minimum)


def build_config(args) -> NaviTritUnifiedConfig:
    kw = dict(PRESETS[args.preset])
    if args.max_loops is not None:
        kw["max_loops"] = args.max_loops
    if args.mamba is not None:
        kw["use_mamba"] = args.mamba
    if args.routing_mode == "traverse":
        kw["use_mamba"] = False
    return NaviTritUnifiedConfig(
        vocab_size=50257, max_position_embeddings=max(1024, args.seq_len), num_experts=args.num_experts,
        use_dwp=not args.no_dwp, routing_mode="dense" if args.routing_mode == "traverse" else args.routing_mode,
        mod_capacity=args.mod_capacity, min_chan_weight=args.min_chan_weight, ternary=not args.fp_control,
        ternary_embed=args.ternary_embed, loop_order_random=args.loop_order_random, tile_drop=args.tile_drop,
        deep_sup_weight=args.deep_sup, deep_sup_frac=args.deep_sup_frac, grad_checkpoint=args.grad_checkpoint, **kw,
    )


def build_model(args, config: NaviTritUnifiedConfig):
    if args.routing_mode != "traverse":
        return NaviTritUnifiedForCausalLM(config)
    tcfg = TraverseConfig(max_hops=args.max_hops, min_hops=args.min_hops, exit_warmup_steps=args.exit_warmup, gate_ste=not args.no_gate_ste,
                          router_cond=args.router_cond, adapter_cond=args.adapter_cond,
                          adapter_bank=args.adapter_bank, adapter_rank=config.lora_rank // 2, hop_cost_weight=args.hop_cost,
                          strain_mode=args.strain, act_strategy=args.act_strategy,
                          exit_mode=args.exit_mode, exit_lambda=args.exit_lambda, deep_sup_weight=args.deep_sup, deep_sup_frac=args.deep_sup_frac,
                          use_liveness=args.liveness, use_gain_proxy=args.gain_proxy, use_coord=args.coord, hop_feature_dropout=args.hop_dropout)
    model = NaviTritTraverseForCausalLM(config, tcfg)
    if args.init_from:
        model.load_dense_checkpoint(args.init_from)
    return model


def forward_with_checkpoint(model, xy, use_ckpt: bool):
    """Gradient checkpointing wraps each macro-layer call; the model's forward is re-implemented here
    only for that purpose (keeps navitrit_unified_model.py free of training-time concerns)."""
    return model(xy, labels=xy)  # checkpointing is handled inside the model (config.grad_checkpoint)
    # legacy path below is unreachable and kept only for reference
    B, S = xy.shape
    positions = torch.arange(0, S, device=xy.device).unsqueeze(0)
    h = F.embedding(xy, embedding_weight(model)) + model.pos_embeddings(positions)
    total_balance = torch.zeros((), device=xy.device)
    for k in range(model.max_loops):
        mod_film, mod_lora = (None, None)
        if model.hypernet is not None:
            mod_film, mod_lora = model.hypernet.get_modulations(k, model.roles[k])
        for layer in model.macro_layers:
            def run(h_in, layer=layer, mod_film=mod_film, mod_lora=mod_lora):
                h_out, bal, _ = layer(h_in, mod_film=mod_film, mod_lora=mod_lora)
                return h_out, bal
            h, bal = grad_checkpoint(run, h, use_reentrant=False)
            total_balance = total_balance + bal
    h = model.final_norm(h)
    logits = F.linear(h, embedding_weight(model))
    ce = F.cross_entropy(logits[..., :-1, :].reshape(-1, model.vocab_size).float(), xy[..., 1:].reshape(-1))
    return {"loss": ce + model.config.balance_loss_weight * total_balance, "ce_loss": ce, "balance_loss": total_balance,
            "avg_w_seq": float("nan"), "avg_w_chan": float("nan"), "avg_w_skip": float("nan")}


@torch.no_grad()
def _eval_policy(model, batches, device, amp_dtype):
    model.eval()
    tot, cnt, hops, n = 0.0, 0, 0.0, 0
    for blist in batches.values():
        for xy in blist:
            xy = xy.to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                out = model(xy, labels=xy)
            k = xy.numel() - xy.size(0)
            tot += out["ce_loss"].item() * k; cnt += k; hops += out["mean_hops"]; n += 1
    return {"mean_loss": round(tot / max(1, cnt), 4), "mean_hops": round(hops / max(1, n), 3)}


def stage2_eval(model, batches, device, amp_dtype, args) -> Dict:
    """Section 13 read-out: lambda sweep, matched random / capacity controls, and the path-diversity probe."""
    import numpy as np
    t = model.tcfg
    res: Dict = {"lambda_sweep": {}, "controls": {}, "diversity": {}}
    t.exit_policy = "threshold"
    for lam in [float(x) for x in args.lambda_sweep.split(",")]:
        t.exit_lambda = lam
        res["lambda_sweep"][str(lam)] = _eval_policy(model, batches, device, amp_dtype)
        print(f"  [lambda {lam}] {res['lambda_sweep'][str(lam)]}")
    t.exit_lambda = args.exit_lambda
    target = res["lambda_sweep"][str(float(args.exit_lambda))]["mean_hops"]
    # random continuation matched to the trained policy's mean hops (bisection on per-hop continue prob)
    t.exit_policy = "random"
    lo, hi = 0.0, 1.0
    for _ in range(12):
        t.random_continue_prob = (lo + hi) / 2
        r = _eval_policy(model, {"tinystories": batches[next(iter(batches))][:2]}, device, amp_dtype)
        lo, hi = ((lo + hi) / 2, hi) if r["mean_hops"] < target else (lo, (lo + hi) / 2)
    res["controls"]["random_matched"] = {"continue_prob": round(t.random_continue_prob, 4), **_eval_policy(model, batches, device, amp_dtype)}
    t.exit_policy = "capacity"
    t.capacity_frac = max(0.05, min(1.0, (target - t.min_hops) / max(1, t.max_hops - t.min_hops)))
    res["controls"]["capacity_matched"] = {"capacity_frac": round(t.capacity_frac, 4), **_eval_policy(model, batches, device, amp_dtype)}
    t.exit_policy = "threshold"
    print(f"  [controls] trained {res['lambda_sweep'][str(float(args.exit_lambda))]} | random {res['controls']['random_matched']} | capacity {res['controls']['capacity_matched']}")
    # path-diversity probe: K forced-random-tile runs per batch; per-token angular spread of final states vs measured gain
    divs, gains = [], []
    for xy in batches[next(iter(batches))][:4]:
        xy = xy.to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            base = model(xy, labels=xy, probe=True)
            model.force_random_tiles = True
            finals = [model(xy, labels=xy, probe=True)["final_hidden"].float() for _ in range(4)]
            model.force_random_tiles = False
        Hs = torch.stack(finals)                                             # [K,B,S,d]
        Hn = F.normalize(Hs, dim=-1)
        sim = torch.einsum("kbsd,lbsd->klbs", Hn, Hn).mean(dim=(0, 1))    # mean pairwise cosine per token
        div = (1.0 - sim)[:, :-1].reshape(-1)
        gain = (base["token_loss_min"] - base["token_loss_final"]).reshape(-1) if base["token_loss_min"] is not None else torch.zeros_like(div)
        divs.append(div.cpu()); gains.append(gain.cpu())
    d = torch.cat(divs).numpy(); g = torch.cat(gains).numpy()
    def _rank(a):
        return np.argsort(np.argsort(a)).astype(np.float64)
    rd, rg = _rank(d), _rank(g)
    rho = float(np.corrcoef(rd, rg)[0, 1]) if len(d) > 2 else 0.0
    z = 0.5 * np.log((1 + rho) / (1 - rho + 1e-12)) * np.sqrt(max(1, len(d) - 3))   # Fisher z, normal approximation
    pval = float(2 * (1 - 0.5 * (1 + np.math.erf(abs(z) / np.sqrt(2))))) if hasattr(np, "math") else float(2 * (1 - 0.5 * (1 + __import__("math").erf(abs(z) / np.sqrt(2)))))
    res["diversity"] = {"spearman_rho": round(float(rho), 4), "p_value": float(pval), "mean_diversity": round(float(d.mean()), 4), "mean_gain_min_to_final": round(float(g.mean()), 4), "n_tokens": int(len(d))}
    print(f"  [diversity] rho {rho:.3f} (p {pval:.2e}) | mean diversity {d.mean():.4f} | mean gain min->final {g.mean():.4f}")
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="pilot")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--data-dir", default="data/clean")
    ap.add_argument("--routing-mode", choices=["dense", "soft", "mod", "traverse"], default="dense")
    # traverse (navitrit_traverse.py): per-token non-monotonic traversal with continuation path state
    ap.add_argument("--max-hops", type=int, default=8)
    ap.add_argument("--min-hops", type=int, default=4)
    ap.add_argument("--exit-warmup", type=int, default=1000, help="steps before EXIT is allowed at all")
    ap.add_argument("--no-gate-ste", action="store_true", help="scale tile output by router prob (arms H-K behaviour)")
    ap.add_argument("--router-cond", choices=["path", "state"], default="path")
    ap.add_argument("--adapter-cond", choices=["path", "hop"], default="path")
    ap.add_argument("--adapter-bank", type=int, default=4)
    ap.add_argument("--hop-cost", type=float, default=0.01)
    ap.add_argument("--strain", choices=["off", "observe", "gate"], default="off", help="strain latent track (report section 10)")
    ap.add_argument("--act-strategy", action="store_true", help="router forwards an activation mixture per edge (report section 12)")
    # stage 2 (report section 13)
    ap.add_argument("--exit-mode", choices=["router", "value"], default="router")
    ap.add_argument("--exit-lambda", type=float, default=0.02)
    ap.add_argument("--liveness", action="store_true")
    ap.add_argument("--gain-proxy", action="store_true")
    ap.add_argument("--coord", action="store_true")
    ap.add_argument("--hop-dropout", type=float, default=0.0)
    ap.add_argument("--lambda-sweep", type=str, default="0,0.005,0.01,0.02,0.05,0.1")
    ap.add_argument("--init-from", type=str, default=None, help="dense-arm checkpoint to warm-start the tiles from")
    ap.add_argument("--mod-capacity", type=float, default=0.5)
    ap.add_argument("--min-chan-weight", type=float, default=0.5)
    ap.add_argument("--no-dwp", action="store_true")
    ap.add_argument("--num-experts", type=int, default=2)
    ap.add_argument("--fp-control", action="store_true", help="ternary=False: matched full-precision control")
    ap.add_argument("--ternary-embed", action="store_true", help="ternarise the tied embedding/head matrix (arm O)")
    ap.add_argument("--max-loops", type=int, default=None)
    ap.add_argument("--mamba", dest="mamba", action="store_true", default=None, help="force the Mamba SSM tile on")
    ap.add_argument("--no-mamba", dest="mamba", action="store_false", help="force the Mamba SSM tile off")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=16)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--loop-order-random", action="store_true", help="stage-1 backbone: permute macro-layer order each loop")
    ap.add_argument("--tile-drop", type=float, default=0.0, help="stage-1 backbone: drop each macro-layer with this prob per loop")
    ap.add_argument("--deep-sup", type=float, default=0.0, help="stage-1 backbone: per-hop readout loss weight")
    ap.add_argument("--deep-sup-frac", type=float, default=0.125)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="10 steps, no checkpoints: throughput + VRAM only")
    ap.add_argument("--output-dir", default="outputs/checkpoints")
    ap.add_argument("--log-dir", default="outputs")
    args = ap.parse_args()

    tag = args.tag or f"{args.preset}-{args.routing_mode}{'-nodwp' if args.no_dwp else ''}{'-fp' if args.fp_control else ''}"
    if args.routing_mode == "traverse" and args.tag is None:
        tag += f"-r{args.router_cond}-a{args.adapter_cond}"
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": torch.float32}[args.amp]
    use_amp = args.amp != "off" and device.type == "cuda"
    if args.smoke:
        args.max_steps, args.eval_interval, args.save_interval, args.eval_iters = 10, 5, 10 ** 9, 2

    train_tokens, val_tokens = load_clean_data(args.data_dir)
    eval_batches = fixed_eval_batches(val_tokens, args.batch_size, args.seq_len, args.eval_iters, seed=12345)
    config = build_config(args)
    model = build_model(args, config).to(device)
    if args.routing_mode == "traverse" and args.gain_proxy:
        counts = torch.bincount(train_tokens[: 5_000_000].long(), minlength=config.vocab_size)
        model.set_proxy_vocab(counts.topk(512).indices)
    sweep_max = args.max_hops if args.routing_mode == "traverse" else config.max_loops
    stats = model.count_parameters()
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    print("=" * 88)
    print(f"NaviTrit-Unified [{tag}] preset={args.preset} routing={args.routing_mode} dwp={config.use_dwp} "
          f"experts={config.num_experts} mamba={config.use_mamba} ternary={config.ternary} order_random={config.loop_order_random} tile_drop={config.tile_drop} deep_sup={config.deep_sup_weight} loops={config.max_loops} (virtual depth {config.virtual_depth})")
    print(f"params {stats['total_millions']}M | super-block packed {stats['superblock_packed_mb']} MB | "
          f"train tokens {len(train_tokens):,} | tokens/step {tokens_per_step:,} | budget {tokens_per_step * args.max_steps / 1e6:.1f}M tokens")
    print(f"val sets: " + ", ".join(f"{k}={len(v):,}" for k, v in val_tokens.items()))
    print("=" * 88)

    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    history: List[Dict] = []
    best = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"navitrit-unified-{tag}-log.json")
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, args.max_steps + 1):
        lr = cosine_lr(step, args.warmup, args.max_steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        if isinstance(model, NaviTritTraverseForCausalLM):
            model.allow_exit = step > args.exit_warmup
        loss_acc, ce_acc = 0.0, 0.0
        for _ in range(args.grad_accum):
            xy = sample_batch(train_tokens, args.batch_size, args.seq_len, gen).to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = forward_with_checkpoint(model, xy, args.grad_checkpoint)
                loss = out["loss"] / args.grad_accum
            scaler.scale(loss).backward()
            loss_acc += loss.item()
            ce_acc += out["ce_loss"].item() / args.grad_accum
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        scaler.step(opt)
        scaler.update()

        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            print(f"step {step:6d} | loss {loss_acc:.4f} ce {ce_acc:.4f} | gnorm {gnorm:.2f} | lr {lr:.2e} | "
                  f"{tokens_per_step * step / el:,.0f} tok/s | {el / 60:.1f} min")

        if step % args.eval_interval == 0 or step == args.max_steps:
            ev = evaluate(model, eval_batches, device, amp_dtype)
            rec = {"step": step, "train_loss": round(loss_acc, 4), "train_ce": round(ce_acc, 4), "lr": lr, "grad_norm": round(gnorm, 3),
                   "elapsed_s": round(time.time() - t0, 1), "tokens_seen": tokens_per_step * step,
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0, **ev}
            if config.routing_mode != "dense":
                rec.update({"w_seq": round(out["avg_w_seq"], 4), "w_chan": round(out["avg_w_chan"], 4), "w_skip": round(out["avg_w_skip"], 4)})
            if "deep_sup_loss" in out:
                rec["deep_sup_loss"] = round(out["deep_sup_loss"], 4)
            if args.routing_mode == "traverse":
                rec.update({"mean_hops": round(out["mean_hops"], 3), "tile_usage": [round(u, 3) for u in out["tile_usage"]]})
                print(f"  [traverse {step}] mean hops {out['mean_hops']:.2f} / {args.max_hops} | tile usage {[round(u, 2) for u in out['tile_usage']]}")
                if args.act_strategy:
                    rec.update({"act_usage": dict(zip(out["act_names"], [round(u, 3) for u in out["act_usage"]]))})
                    print(f"  [act {step}] " + " ".join(f"{n} {u:.2f}" for n, u in zip(out["act_names"], out["act_usage"])))
                if args.exit_mode == "value" or args.deep_sup > 0:
                    rec.update({"deep_sup_loss": round(out["deep_sup_loss"], 4), "value_loss": round(out["value_loss"], 4),
                                "mean_measured_gain": round(out["mean_measured_gain"], 4), "mean_pred_gain": round(out["mean_pred_gain"], 4)})
                    print(f"  [value {step}] measured gain {out['mean_measured_gain']:.4f} | predicted {out['mean_pred_gain']:.4f} | value loss {out['value_loss']:.4f} | deep-sup {out['deep_sup_loss']:.3f}")
                if args.strain != "off":
                    rec.update({"strain_loss": round(out["strain_loss"], 4), "mean_pred_strain": round(out["mean_pred_strain"], 4)})
                    print(f"  [strain {step}] aux loss {out['strain_loss']:.4f} | mean predicted strain {out['mean_pred_strain']:.3f}")
            history.append(rec)
            print(f"  [eval {step}] " + " | ".join(f"{k} {v}" for k, v in ev.items()) + f" | peak VRAM {rec['peak_vram_gb']} GB")
            if ev["mean_loss"] < best and not args.smoke:
                best = ev["mean_loss"]
                torch.save({"config": asdict(config), "model_state_dict": model.state_dict(), "step": step, "eval": ev},
                           os.path.join(args.output_dir, f"navitrit-unified-{tag}-best.pt"))
            with open(log_path, "w") as f:
                json.dump({"tag": tag, "args": vars(args), "config": asdict(config), "params": stats, "history": history}, f, indent=2)

        if step % args.save_interval == 0 and not args.smoke:
            torch.save({"config": asdict(config), "model_state_dict": model.state_dict(), "step": step},
                       os.path.join(args.output_dir, f"navitrit-unified-{tag}-latest.pt"))

    stage2 = {}
    if args.routing_mode == "traverse" and args.exit_mode == "value":
        stage2 = stage2_eval(model, eval_batches, device, amp_dtype, args)
        with open(log_path, "w") as f:
            json.dump({"tag": tag, "args": vars(args), "config": asdict(config), "params": stats, "history": history, "stage2": stage2}, f, indent=2)
    # Test-time compute sweep: does more loops help on held-out data? (the LoopFormer claim, measured cleanly)
    sweep = {}
    for T in range(1, sweep_max + 1):
        sweep[f"T{T}"] = evaluate(model, eval_batches, device, amp_dtype, max_loops=T)
        print(f"  [loop sweep] T={T}: " + " | ".join(f"{k} {v}" for k, v in sweep[f'T{T}'].items() if k.endswith("ppl")))
    with open(log_path, "w") as f:
        json.dump({"tag": tag, "args": vars(args), "config": asdict(config), "params": stats, "history": history,
                   "loop_budget_sweep": sweep, "best_mean_val_loss": best, "stage2": stage2,
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0}, f, indent=2)
    print(f"Saved {log_path}")


if __name__ == "__main__":
    main()
