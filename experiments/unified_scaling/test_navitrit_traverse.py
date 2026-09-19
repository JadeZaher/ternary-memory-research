"""
experiments/unified_scaling/test_navitrit_traverse.py: unit tests for NaviTrit-Traverse (CPU by default so it
never competes with a training job on the GPU; pass --cuda to run on the GPU).

Checks: forward/backward in all four (router_cond, adapter_cond) settings; gradient reaches the tile router,
adapter bank and path cell when history is enabled; tiles are all used at init (no argmax tie collapse);
exit bias drives mean hops to min_hops; causality of the subset attention (a future token cannot change the
logits of a past token).
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

from experiments.unified_scaling.navitrit_unified_model import NaviTritUnifiedConfig
from experiments.unified_scaling.navitrit_traverse import NaviTritTraverseForCausalLM, TraverseConfig


def small_config() -> NaviTritUnifiedConfig:
    return NaviTritUnifiedConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                                 num_macro_layers=4, max_loops=4, use_mamba=False, routing_mode="dense",
                                 max_position_embeddings=64, lora_rank=8)


def test_modes_and_gradients(device):
    for rc, ac in [("path", "path"), ("state", "hop"), ("path", "hop"), ("state", "path")]:
        torch.manual_seed(0)
        m = NaviTritTraverseForCausalLM(small_config(), TraverseConfig(max_hops=6, min_hops=1, exit_warmup_steps=0, path_dim=16, adapter_bank=3, adapter_rank=4, router_cond=rc, adapter_cond=ac)).to(device)
        x = torch.randint(0, 512, (2, 24), device=device)
        # zero-drift init means adapter B = FiLM = 0, so history reaches the path cell only after the first
        # update moves them; take one SGD step, then check gradients on the second pass
        opt = torch.optim.SGD(m.parameters(), lr=1e-2)
        m(x, labels=x)["loss"].backward(); opt.step(); opt.zero_grad()
        out = m(x, labels=x)
        out["loss"].backward()
        assert torch.isfinite(out["loss"]).item()
        assert m.tile_router[-1].weight.grad.abs().sum() > 0, "tile router got no gradient"
        assert m.banks[0].B["q"].grad.abs().sum() > 0, "adapter bank got no gradient"
        if rc == "path" or ac == "path":
            assert m.path_cell.weight_hh.grad is not None and m.path_cell.weight_hh.grad.abs().sum() > 0, "path cell got no gradient with history enabled"
        usage = out["tile_usage"][: m.M]
        assert min(usage) > 0.02, f"a tile is unused at init: {usage}"
        print(f"  router={rc:5s} adapter={ac:4s} loss {out['loss'].item():.3f} hops {out['mean_hops']:.2f} usage {[round(u, 2) for u in usage]}")
    print("test_modes_and_gradients PASSED")


def test_exit_bias(device):
    torch.manual_seed(0)
    m = NaviTritTraverseForCausalLM(small_config(), TraverseConfig(max_hops=6, min_hops=2, exit_warmup_steps=0, path_dim=16, adapter_bank=2, adapter_rank=4)).to(device).eval()
    with torch.no_grad():
        m.tile_router[-1].bias[m.EXIT] = 8.0
        out = m(torch.randint(0, 512, (2, 24), device=device))
    assert abs(out["mean_hops"] - 2.0) < 1e-6, out["mean_hops"]
    print("test_exit_bias PASSED")


def test_causality(device):
    torch.manual_seed(1)
    m = NaviTritTraverseForCausalLM(small_config(), TraverseConfig(max_hops=4, min_hops=1, exit_warmup_steps=0, path_dim=16, adapter_bank=2, adapter_rank=4)).to(device).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "banks" in n and (".B." in n or "film" in n):
                p.normal_(std=0.05)  # make adapters non-trivial so subset attention is exercised
        x = torch.randint(0, 512, (1, 20), device=device)
        y = x.clone(); y[0, 15:] = torch.randint(0, 512, (5,), device=device)
        lx = m(x)["logits"][0, :15]
        ly = m(y)["logits"][0, :15]
    assert torch.allclose(lx, ly, atol=1e-4), (lx - ly).abs().max()
    print("test_causality PASSED")


def test_strain_modes(device):
    torch.manual_seed(0)
    x = torch.randint(0, 512, (2, 24), device=device)
    for mode in ("observe", "gate"):
        m = NaviTritTraverseForCausalLM(small_config(), TraverseConfig(max_hops=6, min_hops=1, exit_warmup_steps=0, path_dim=16, adapter_bank=2, adapter_rank=4, strain_mode=mode)).to(device)
        out = m(x, labels=x); out["loss"].backward()
        assert torch.isfinite(out["loss"]).item() and out["strain_loss"] > 0
        assert m.strain_head[-1].weight.grad.abs().sum() > 0, "strain head got no gradient"
        assert m.strain_in.weight.grad.abs().sum() > 0, "strain features got no gradient into path state"
        print(f"  strain={mode}: loss {out['loss'].item():.3f} strain_loss {out['strain_loss']:.4f} pred {out['mean_pred_strain']:.3f} hops {out['mean_hops']:.2f}")
    # gate: negative scale (high strain -> exit encouraged) must not yield more hops than positive scale
    m.eval()
    with torch.no_grad():
        m.strain_exit_scale.fill_(50.0); hi = m(x)["mean_hops"]
        m.strain_exit_scale.fill_(-50.0); lo = m(x)["mean_hops"]
    assert lo <= hi, (lo, hi)
    print("test_strain_modes PASSED")


def test_act_strategy(device):
    torch.manual_seed(0)
    x = torch.randint(0, 512, (2, 24), device=device)
    m = NaviTritTraverseForCausalLM(small_config(), TraverseConfig(max_hops=6, min_hops=1, exit_warmup_steps=0, path_dim=16, adapter_bank=2, adapter_rank=4, act_strategy=True)).to(device)
    out = m(x, labels=x); out["loss"].backward()
    assert torch.isfinite(out["loss"]).item()
    assert m.strategy_router.weight.grad.abs().sum() > 0, "strategy router got no gradient"
    assert m.offset_embed.weight.grad.abs().sum() > 0, "edge offset embedding got no gradient"
    usage = dict(zip(out["act_names"], out["act_usage"]))
    assert usage["silu"] > 0.9, f"init should be ~pure SiLU, got {usage}"
    # with SiLU dominant at init the model must match the no-strategy model closely
    m2 = NaviTritTraverseForCausalLM(small_config(), TraverseConfig(max_hops=6, min_hops=1, exit_warmup_steps=0, path_dim=16, adapter_bank=2, adapter_rank=4)).to(device)
    m2.load_state_dict({k: v for k, v in m.state_dict().items() if k in m2.state_dict()}, strict=False)
    m.eval(); m2.eval()
    with torch.no_grad():
        d = (m(x)["logits"] - m2(x)["logits"]).abs().max().item()
    assert d < 1.5, d  # 3% non-SiLU mass at init; tiles now run at full magnitude (STE gate) so the gap is larger than with prob-scaled outputs
    print(f"test_act_strategy PASSED (usage {[(k, round(v, 2)) for k, v in usage.items()]}, init logit gap {d:.3f})")


def test_exit_warmup_and_ste(device):
    torch.manual_seed(0)
    x = torch.randint(0, 512, (2, 24), device=device)
    m = NaviTritTraverseForCausalLM(small_config(), TraverseConfig(max_hops=6, min_hops=1, exit_warmup_steps=100, path_dim=16, adapter_bank=2, adapter_rank=4)).to(device).eval()
    with torch.no_grad():
        m.tile_router[-1].bias[m.EXIT] = 8.0
        assert m(x)["mean_hops"] == 6.0, "exit must be masked during warmup"
        m.allow_exit = True
        assert m(x)["mean_hops"] == 1.0
    # STE gate: tile output enters at full magnitude, router still gets gradient
    m.train(); m.allow_exit = False
    out = m(x, labels=x); out["loss"].backward()
    assert m.tile_router[-1].weight.grad.abs().sum() > 0
    print("test_exit_warmup_and_ste PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cuda", action="store_true"); a = ap.parse_args()
    dev = torch.device("cuda" if a.cuda and torch.cuda.is_available() else "cpu")
    test_modes_and_gradients(dev)
    test_exit_bias(dev)
    test_causality(dev)
    test_strain_modes(dev)
    test_act_strategy(dev)
    test_exit_warmup_and_ste(dev)
    print("ALL TRAVERSE TESTS PASSED on", dev)
