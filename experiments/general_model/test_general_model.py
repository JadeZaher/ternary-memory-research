"""
experiments/general_model/test_general_model.py: unit tests for `GeneralRoutedLM` (CPU by default so the
suite never competes with a training job; `--cuda` runs the same checks on the GPU with the same tiny config).

Coverage, in the order the mechanisms are risky:
  1 forward/backward and gradient reachability in every routing x exit combination
  2 causality of the gathered-subset dispatch (this is also the proof that Mamba on a subset is causal)
  3 subset equivalence and padding non-leakage
  4 exit warmup, the value threshold, and epsilon exploration
  5 per-tile gradient checkpointing is exact (run with adapters and soft mix ACTIVE: a zero-initialised
    LoRA B contributes no gradient and would hide a closure-capture bug)
  6 stored-byte accounting under `ternary_embed`
  7 probe outputs and the forced-random-tile diversity hook
  8 `--resume` round-trip through the trainer's own save/load helpers
"""

import os
import sys
import math
import shutil
import tempfile
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn.functional as F

from experiments.general_model.general_model import (
    DENSE_BITS,
    TERNARY_BITS,
    GeneralConfig,
    GeneralRoutedLM,
)
from experiments.general_model.train_general import load_state, sample_batch, save_state
from experiments.unified_scaling.navitrit_unified_model import embedding_weight

VOCAB, SEQ = 512, 32


def small_config(**overrides) -> GeneralConfig:
    base = dict(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_attention_heads=4,
        max_position_embeddings=64, tile_kinds=["mamba", "attn"], num_experts=2,
        d_state=8, mamba_expand=2, dt_rank=8, d_conv=4,
        max_hops=4, min_hops=2, exit_warmup_steps=0, path_dim=16, adapter_bank=2, adapter_rank=4,
        explore_prob=0.0, deep_sup_weight=0.1,
    )
    base.update(overrides)
    return GeneralConfig(**base)


def _one_optimizer_step(model, x, lr=1e-2):
    """Moves the zero-initialised LoRA B matrices and FiLM vectors off zero so adapters carry gradient."""
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model(x, labels=x)["loss"].backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


# ------------------------------------------------------------------ 1 -------------------------
def test_forward_backward_and_gradients(device):
    for routing in ("free", "fixed"):
        for exit_mode in ("value", "none"):
            torch.manual_seed(0)
            model = GeneralRoutedLM(small_config(routing=routing, exit_mode=exit_mode)).to(device)
            x = torch.randint(0, VOCAB, (2, SEQ), device=device)
            _one_optimizer_step(model, x)
            out = model(x, labels=x)
            out["loss"].backward()
            assert torch.isfinite(out["loss"]).item(), (routing, exit_mode, out["loss"])
            assert model.tok_embeddings.weight.grad.abs().sum() > 0, "embeddings got no gradient"
            assert model.banks[0].B["q"].grad.abs().sum() > 0, "adapter bank got no gradient"
            assert model.tiles[0].mixer.in_proj.weight.grad.abs().sum() > 0, "Mamba in_proj got no gradient"
            assert model.tiles[1].mixer.q_proj.weight.grad.abs().sum() > 0, "attention q_proj got no gradient"
            if routing == "free":
                assert model.tile_router[-1].weight.grad.abs().sum() > 0, "tile router got no gradient"
            if exit_mode == "value":
                assert model.value_head[-1].weight.grad.abs().sum() > 0, "value head got no gradient"
            print(f"  routing={routing:5s} exit={exit_mode:5s} loss {out['loss'].item():.3f} "
                  f"hops {out['mean_hops']:.2f} usage {[round(u, 2) for u in out['tile_usage']]}")
    print("test_forward_backward_and_gradients PASSED")


# ------------------------------------------------------------------ 2 -------------------------
def test_causality(device):
    """Changing token j must not change the logits at any position < j, in either routing, with exit live."""
    for routing in ("free", "fixed"):
        torch.manual_seed(1)
        model = GeneralRoutedLM(small_config(routing=routing, exit_mode="value")).to(device)
        model.allow_exit = True
        x = torch.randint(0, VOCAB, (1, SEQ), device=device)
        _one_optimizer_step(model, x)              # non-trivial adapters exercise the FiLM/LoRA path
        model.eval()
        cut = 20
        with torch.no_grad():
            y = x.clone()
            y[0, cut:] = torch.randint(0, VOCAB, (SEQ - cut,), device=device)
            lx = model(x)["logits"][0, :cut]
            ly = model(y)["logits"][0, :cut]
        delta = (lx - ly).abs().max().item()
        assert delta < 1e-4, f"{routing}: future token changed past logits by {delta}"
        print(f"  routing={routing:5s} max |delta| over positions < {cut}: {delta:.2e}")
    print("test_causality PASSED")


# ------------------------------------------------------------------ 3 -------------------------
def test_subset_equivalence_and_padding(device):
    """All tokens on one Mamba tile == that tile run on the full sequence; and the padded tail of a
    gathered subset cannot influence the valid rows."""
    torch.manual_seed(2)
    model = GeneralRoutedLM(small_config(exit_mode="none", max_hops=1, min_hops=1)).to(device)
    x = torch.randint(0, VOCAB, (2, SEQ), device=device)
    _one_optimizer_step(model, x)
    model.eval()
    model.force_tile = 0                            # tile 0 is the Mamba tile
    with torch.no_grad():
        got = model(x, max_loops=1)["logits"]
        positions = torch.arange(SEQ, device=device).unsqueeze(0)
        h = F.embedding(x, embedding_weight(model)) + model.pos_embeddings(positions)
        alpha = model._alpha(None, 0, (x.size(0), SEQ))
        valid = torch.ones(x.size(0), SEQ, dtype=torch.bool, device=device)
        delta, _bal, _extras = model._run_tile_impl(0, h, valid, alpha, False)
        want = F.linear(model.final_norm(h + delta), embedding_weight(model))
    gap = (got - want).abs().max().item()
    assert gap < 1e-4, f"dispatch is not an identity when every token selects one tile: {gap}"

    # padding non-leakage: same valid prefix, different padded tail -> identical delta on the valid rows
    with torch.no_grad():
        keep = SEQ // 2
        valid_half = torch.zeros(1, SEQ, dtype=torch.bool, device=device)
        valid_half[0, :keep] = True
        h_a = h[:1].clone()
        h_b = h[:1].clone()
        h_b[0, keep:] = torch.randn_like(h_b[0, keep:]) * 3.0
        delta_a, _, _ = model._run_tile_impl(0, h_a, valid_half, alpha[:1], False)
        delta_b, _, _ = model._run_tile_impl(0, h_b, valid_half, alpha[:1], False)
    leak = (delta_a[0, :keep] - delta_b[0, :keep]).abs().max().item()
    assert leak < 1e-5, f"padded tail leaked into valid rows by {leak}"
    model.force_tile = None
    print(f"test_subset_equivalence_and_padding PASSED (dispatch gap {gap:.2e}, padding leak {leak:.2e})")


# ------------------------------------------------------------------ 4 -------------------------
def test_exit_warmup_threshold_and_explore(device):
    torch.manual_seed(3)
    config = small_config(exit_mode="value", max_hops=4, min_hops=2, exit_warmup_steps=100)
    model = GeneralRoutedLM(config).to(device)
    x = torch.randint(0, VOCAB, (2, SEQ), device=device)

    model.eval()
    with torch.no_grad():
        assert model.allow_exit is False
        assert model(x)["mean_hops"] == 4.0, "exit must be masked during warmup"
        model.value_head[-1].bias.fill_(-100.0)     # predicted gain far below any lambda
        model.allow_exit = True
        hops = model(x)["mean_hops"]
    assert abs(hops - 2.0) < 1e-6, f"negative predicted gain should exit at min_hops, got {hops}"

    model.config.explore_prob = 1.0
    model.train()
    out = model(x, labels=x)
    assert abs(out["mean_hops"] - 4.0) < 1e-6, f"explore_prob=1.0 must keep every token alive, got {out['mean_hops']}"
    print(f"test_exit_warmup_threshold_and_explore PASSED (warmup 4.0, threshold {hops}, explore {out['mean_hops']})")


# ------------------------------------------------------------------ 5 -------------------------
def test_grad_checkpoint_equivalence(device):
    """Loss and every gradient must match the unchecked path. Adapters and the soft branch mix are active:
    a checkpointed callable that captures a grad-carrying tensor by closure instead of taking it as an
    explicit argument inflates gradients on recompute, and zero-initialised adapters would hide it."""
    x = torch.randint(0, VOCAB, (2, SEQ), device=device)

    def build(grad_checkpoint):
        torch.manual_seed(4)
        model = GeneralRoutedLM(small_config(grad_checkpoint=grad_checkpoint, tile_soft_mix=True)).to(device)
        torch.manual_seed(5)
        _one_optimizer_step(model, x)               # LoRA B and FiLM are now non-zero
        model.train()
        torch.manual_seed(6)
        out = model(x, labels=x)
        out["loss"].backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        return out["loss"].item(), grads

    loss_plain, grads_plain = build(False)
    loss_ckpt, grads_ckpt = build(True)
    assert abs(loss_plain - loss_ckpt) < 1e-6, (loss_plain, loss_ckpt)
    assert set(grads_plain) == set(grads_ckpt), set(grads_plain) ^ set(grads_ckpt)
    worst_name, worst = "", 0.0
    for name, g in grads_plain.items():
        diff = (g - grads_ckpt[name]).abs().max().item()
        if diff > worst:
            worst_name, worst = name, diff
    adapter_mass = sum(grads_plain[n].abs().sum().item() for n in grads_plain if n.startswith("banks."))
    assert adapter_mass > 0, "adapter gradients are zero: the test cannot detect a checkpoint bug"
    assert worst < 1e-6, f"gradient mismatch {worst:.3e} at {worst_name}"
    print(f"test_grad_checkpoint_equivalence PASSED (loss {loss_plain:.6f}, worst grad diff {worst:.2e} "
          f"at {worst_name or 'n/a'}, {len(grads_plain)} tensors, adapter grad mass {adapter_mass:.3e})")


# ------------------------------------------------------------------ 6 -------------------------
def test_footprint_accounting(device):
    torch.manual_seed(7)
    dense = GeneralRoutedLM(small_config(ternary_embed=False)).to(device).count_parameters()
    torch.manual_seed(7)
    ternary = GeneralRoutedLM(small_config(ternary_embed=True)).to(device).count_parameters()
    assert dense["params"] == ternary["params"], "ternary_embed must not change parameter counts"
    ratio = ternary["stored_bytes"]["token_embed"] / dense["stored_bytes"]["token_embed"]
    assert abs(ratio - TERNARY_BITS / DENSE_BITS) < 1e-9, ratio
    expected_delta = dense["params"]["token_embed"] * (DENSE_BITS - TERNARY_BITS) / 8.0 / (1024 * 1024)
    got_delta = dense["stored_mb_total"] - ternary["stored_mb_total"]
    assert abs(got_delta - expected_delta) < 1e-3, (got_delta, expected_delta)
    assert dense["accounted_params"] == dense["total_physical_parameters"], \
        (dense["accounted_params"], dense["total_physical_parameters"])
    print(f"test_footprint_accounting PASSED (fp16 embed {dense['stored_mb_total']} MB -> "
          f"ternary embed {ternary['stored_mb_total']} MB, ratio {ratio:.4f})")


# ------------------------------------------------------------------ 7 -------------------------
def test_probe_and_forced_random_tiles(device):
    torch.manual_seed(8)
    model = GeneralRoutedLM(small_config(exit_mode="value", min_hops=2, max_hops=4)).to(device)
    model.allow_exit = True
    x = torch.randint(0, VOCAB, (2, SEQ), device=device)
    model.eval()
    with torch.no_grad():
        out = model(x, labels=x, probe=True)
    assert out["token_loss_final"].shape == (2, SEQ - 1), out["token_loss_final"].shape
    assert out["token_loss_min"] is not None and out["token_loss_min"].shape == (2, SEQ - 1), "probe min-hop loss missing"
    assert out["final_hidden"].shape == (2, SEQ, model.hidden)
    with torch.no_grad():
        base_usage = model(x)["tile_usage"]
        model.force_random_tiles = True
        random_usage = model(x)["tile_usage"]
        model.force_random_tiles = False
    assert base_usage != random_usage, (base_usage, random_usage)
    print(f"test_probe_and_forced_random_tiles PASSED (usage {[round(u, 2) for u in base_usage]} -> "
          f"{[round(u, 2) for u in random_usage]})")


# ------------------------------------------------------------------ 8 -------------------------
def test_resume_round_trip(device):
    """Three steps, save, rebuild, resume, fourth step: the loss must match an uninterrupted four-step run."""
    tokens = torch.randint(0, VOCAB, (4096,), dtype=torch.long)
    config = small_config(exit_mode="value", exit_warmup_steps=0)
    tmpdir = tempfile.mkdtemp(prefix="general-resume-")
    path = os.path.join(tmpdir, "general-test-latest.pt")
    args = argparse.Namespace(seed=0, tag="test")

    def fresh():
        torch.manual_seed(9)
        model = GeneralRoutedLM(config).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(9)
        return model, opt, gen

    def step_once(model, opt, gen):
        xy = sample_batch(tokens, 2, SEQ, gen).to(device)
        out = model(xy, labels=xy)
        out["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return out["loss"].item()

    try:
        torch.manual_seed(100)
        model, opt, gen = fresh()
        uninterrupted = [step_once(model, opt, gen) for _ in range(4)]

        torch.manual_seed(100)
        model, opt, gen = fresh()
        for _ in range(3):
            step_once(model, opt, gen)
        save_state(path, model, opt, gen, 3, [], float("inf"), config, args, {})

        torch.manual_seed(12345)                    # deliberately wrong RNG: load_state must overwrite it
        model2, opt2, gen2 = fresh()
        checkpoint = load_state(path, model2, opt2, gen2)
        assert checkpoint["step"] == 3
        resumed = step_once(model2, opt2, gen2)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    gap = abs(resumed - uninterrupted[3])
    assert gap < 1e-6, f"resumed step-4 loss {resumed} != uninterrupted {uninterrupted[3]} (gap {gap})"
    print(f"test_resume_round_trip PASSED (step 4 loss {resumed:.6f} vs {uninterrupted[3]:.6f})")


TESTS = [
    test_forward_backward_and_gradients,
    test_causality,
    test_subset_equivalence_and_padding,
    test_exit_warmup_threshold_and_explore,
    test_grad_checkpoint_equivalence,
    test_footprint_accounting,
    test_probe_and_forced_random_tiles,
    test_resume_round_trip,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parsed = parser.parse_args()
    dev = torch.device("cuda" if parsed.cuda and torch.cuda.is_available() else "cpu")
    for test in TESTS:
        test(dev)
    print(f"ALL {len(TESTS)} GENERAL MODEL TESTS PASSED on {dev}")
