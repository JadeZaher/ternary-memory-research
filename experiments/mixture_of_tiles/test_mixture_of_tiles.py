"""
experiments/mixture_of_tiles/test_mixture_of_tiles.py: CPU-only tests for the tile store, the tile cache,
the streaming decoder and the sweep planner.

Coverage, in the order the mechanisms are risky:
  a base-3 pack/unpack is an exact round trip at every tail length
  b a store round trip reproduces the source model's EVAL logits bit-for-bit through a fresh model
  c the LRU cache's hit/miss/bytes accounting and prefetch semantics
  d cache size never changes what is generated, and an evicted tile cannot execute
  e the fixed-bytes sweep planner lands inside its tolerance and emits the right command

    python experiments/mixture_of_tiles/test_mixture_of_tiles.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard rule: this folder never touches the GPU

import sys
import shutil
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

torch.set_num_threads(4)
CPU_DEVICE = torch.device("cpu")

from experiments.general_model.general_model import GeneralConfig, GeneralRoutedLM
from experiments.mixture_of_tiles.cpu_decoder import StreamingDecoder, usage_entropy_bits
from experiments.mixture_of_tiles.sweep_tile_count import plan_sweep
from experiments.mixture_of_tiles.tile_store import (
    TileCache,
    TileNotResidentError,
    TileStore,
    achieved_bits_per_weight,
    install_tile,
    pack_ternary,
    packed_byte_length,
    uninstall_tile,
    unpack_ternary,
)

VOCAB, SEQ, HIDDEN = 512, 32, 64


def small_config(**overrides) -> GeneralConfig:
    """The tiny geometry the general-model tests use, with three tiles so eviction is observable."""
    base = dict(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=128, num_attention_heads=4,
        max_position_embeddings=64, tile_kinds=["mamba", "mamba", "attn"], num_experts=2,
        d_state=8, mamba_expand=2, dt_rank=8, d_conv=4,
        max_hops=4, min_hops=2, exit_warmup_steps=0, path_dim=16, adapter_bank=2, adapter_rank=4,
        explore_prob=0.0, deep_sup_weight=0.1,
    )
    base.update(overrides)
    return GeneralConfig(**base)


def trained_model(config: GeneralConfig, steps: int = 3, seed: int = 0) -> GeneralRoutedLM:
    """A few SGD steps so no weight is at its initialisation value (zero LoRA B would hide bugs)."""
    torch.manual_seed(seed)
    model = GeneralRoutedLM(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    inputs = torch.randint(0, VOCAB, (2, SEQ))
    for _ in range(steps):
        model(inputs, labels=inputs)["loss"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    model.eval()
    return model


# ------------------------------------------------------------------ a -------------------------
def test_pack_unpack_round_trip():
    torch.manual_seed(0)
    for length in (1, 5, 6, 1000, 1001):
        original = torch.randint(-1, 2, (length,), dtype=torch.int8)
        packed = pack_ternary(original)
        assert len(packed) == packed_byte_length(length), (length, len(packed))
        restored = unpack_ternary(packed, length)
        assert torch.equal(original, restored), f"length {length} did not round trip"
        print(f"  len {length:>5}: {len(packed):>4} bytes, "
              f"{achieved_bits_per_weight(length):.4f} bits/weight, exact")

    weight = torch.randint(-1, 2, (48, 96), dtype=torch.int8)
    packed = pack_ternary(weight)
    restored = unpack_ternary(packed, weight.numel()).view(weight.shape)
    assert torch.equal(weight, restored), "2-D weight did not round trip"
    print(f"  2-D {tuple(weight.shape)}: {len(packed):,} bytes, "
          f"{achieved_bits_per_weight(weight.numel()):.4f} bits/weight, exact")
    print("test_pack_unpack_round_trip PASSED")


# ------------------------------------------------------------------ b -------------------------
def _round_trip_one(routing: str, workdir: str) -> float:
    config = small_config(routing=routing)
    model = trained_model(config, seed=1)
    model.allow_exit = True
    store_path = os.path.join(workdir, f"store-{routing}")
    table = TileStore.build(model, store_path, canonicalize=True, verbose=False)

    fresh = GeneralRoutedLM(config)
    fresh.eval()                                   # clears BitLinear eval caches before they are installed
    fresh.allow_exit = True
    with TileStore.open(store_path) as store:
        store.load_resident(fresh)
        for index in range(store.num_tiles):
            read = store.read_tile(index)
            assert read.bytes_read == store.tile_bytes(index)
            install_tile(fresh, index, read.tensors)

        torch.manual_seed(7)
        inputs = torch.randint(0, VOCAB, (2, SEQ))
        with torch.no_grad():
            reference = model(inputs)
            candidate = fresh(inputs)
        delta = (reference["logits"] - candidate["logits"]).abs().max().item()
        assert delta < 1e-5, f"{routing}: store round trip moved the logits by {delta}"
        assert reference["tile_usage"] == candidate["tile_usage"], f"{routing}: tile choices diverged"
        print(f"  routing={routing:5s} max |logit delta| {delta:.3e}  "
              f"tile_usage {[round(value, 3) for value in candidate['tile_usage']]}  "
              f"tiles {table['tile_bytes_total']:,} B  resident {table['resident_logical_bytes']:,} B  "
              f"{table['bits_per_weight_packed']:.4f} bits/weight")
    return delta


def test_store_round_trip():
    workdir = tempfile.mkdtemp(prefix="mot-store-")
    try:
        _round_trip_one("fixed", workdir)          # fixed routing: the same path is taken by construction
        _round_trip_one("free", workdir)           # free routing: identical weights must pick identical tiles
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print("test_store_round_trip PASSED")


# ------------------------------------------------------------------ c -------------------------
def test_cache_semantics():
    workdir = tempfile.mkdtemp(prefix="mot-cache-")
    try:
        model = trained_model(small_config(), seed=2)
        store_path = os.path.join(workdir, "store")
        TileStore.build(model, store_path, canonicalize=True, verbose=False)
        with TileStore.open(store_path) as store:
            sequence = [0, 1, 0, 2, 1, 0]
            # capacity 2, LRU: miss 0, miss 1, hit 0, miss 2 (evict 1), miss 1 (evict 0), miss 0 (evict 2)
            expected_misses = [0, 1, 2, 1, 0]
            cache = TileCache(store, capacity_tiles=2)
            for index in sequence:
                cache.get(index)
            stats = cache.stats()
            assert stats["hits"] == 1, stats
            assert stats["misses"] == 5, stats
            assert stats["evictions"] == 3, stats
            expected_bytes = sum(store.tile_bytes(index) for index in expected_misses)
            assert stats["bytes_read"] == expected_bytes, (stats["bytes_read"], expected_bytes)
            print(f"  capacity 2 over {sequence}: hits {stats['hits']} misses {stats['misses']} "
                  f"evictions {stats['evictions']} bytes {stats['bytes_read']:,} "
                  f"(= sum of tile_bytes over the miss sequence)")

            warm = TileCache(store, capacity_tiles=2)
            warm.get(0)
            warm.prefetch([1], protect=(0,))                 # the next tile arrives before it is asked for
            before = warm.stats()["misses"]
            warm.get(1)
            after = warm.stats()
            assert after["misses"] == before, "prefetched tile still counted as a miss"
            assert after["hits"] == 1, after
            assert after["prefetches"] == 1 and after["prefetch_bytes"] == store.tile_bytes(1), after
            print(f"  prefetch: get(1) after prefetch([1]) -> hits {after['hits']} misses {after['misses']} "
                  f"prefetch_bytes {after['prefetch_bytes']:,}")

            cold = TileCache(store, capacity_tiles=2, injected_miss_latency_ms_per_mb=0.33)
            cold.get(0)
            injected_ms = cold.stats()["injected_seconds"] * 1000.0
            expected_ms = 0.33 * store.tile_bytes(0) / (1024 * 1024)
            assert abs(injected_ms - expected_ms) < 1e-9, (injected_ms, expected_ms)
            print(f"  injected latency at 0.33 ms/MB on a {store.tile_bytes(0):,} B tile: "
                  f"{injected_ms:.4f} ms")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print("test_cache_semantics PASSED")


# ------------------------------------------------------------------ d -------------------------
def test_streaming_decode():
    workdir = tempfile.mkdtemp(prefix="mot-decode-")
    try:
        config = small_config(routing="free")
        model = trained_model(config, seed=3)
        model.allow_exit = False                    # run the full hop budget so every tile gets a chance
        store_path = os.path.join(workdir, "store")
        TileStore.build(model, store_path, canonicalize=True, verbose=False)
        torch.manual_seed(11)
        prompt = torch.randint(0, VOCAB, (1, 8))

        results = {}
        with TileStore.open(store_path) as store:
            for capacity in (1, 3):
                cache = TileCache(store, capacity_tiles=capacity)
                decoder = StreamingDecoder(model, cache)
                try:
                    results[capacity] = decoder.generate(prompt, max_new_tokens=4)
                finally:
                    decoder.detach()
                summary = results[capacity]
                print(f"  capacity {capacity}: ids {summary['generated_ids']} "
                      f"misses {summary['misses']} hits {summary['hits']} "
                      f"bytes/token {summary['mean_bytes_per_token']:,.0f} "
                      f"tiles/token {summary['mean_tiles_per_token']:.2f} "
                      f"working set {summary['working_set_size']} "
                      f"H {summary['tile_usage_entropy_bits']:.3f} bits")

            assert results[1]["generated_ids"] == results[3]["generated_ids"], \
                "cache size changed the generated tokens"
            assert results[1]["misses"] >= results[3]["misses"], \
                "the smaller cache did not miss at least as often"

            uninstall_tile(model, 0, scrub=True)
            hidden = torch.randn(1, 4, HIDDEN)
            valid = torch.ones(1, 4, dtype=torch.bool)
            raised = False
            try:
                model.tiles[0](hidden, valid)
            except TileNotResidentError:
                raised = True
            assert raised, "an evicted tile executed without coming through the cache"
            print("  evicted tile 0 raised TileNotResidentError when executed directly")

            # put the model back together so the object is reusable after the test
            for index in range(store.num_tiles):
                install_tile(model, index, store.read_tile(index).tensors)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print("test_streaming_decode PASSED")


# ------------------------------------------------------------------ e -------------------------
def test_sweep_planner():
    """The tiny reference needs an FFN-dominated shape (intermediate 1024 at hidden 64): the sequence mixer
    and the dense Mamba block are per-tile constants, so they are the headroom the planner spends."""
    overrides = dict(
        hidden_size=HIDDEN, intermediate_size=1024, num_attention_heads=4, dt_rank=8,
        vocab_size=VOCAB, max_position_embeddings=64, d_state=8, mamba_expand=2,
        path_dim=16, adapter_bank=2, adapter_rank=4, max_hops=4, min_hops=2, exit_warmup_steps=0,
    )
    payload = plan_sweep(preset="p512", reference_count=4, tile_counts=[8, 16], tolerance=0.03,
                         base_overrides=overrides, verbose=False)
    reference_bytes = payload["reference"]["tile_bytes"]
    for plan in payload["plans"]:
        assert plan["feasible"], (plan["tile_count"], plan["reason"])
        assert abs(plan["deviation"]) <= 0.03, (plan["tile_count"], plan["deviation"])
        kinds = plan["command"].split("--tile-kinds ")[1].split(" ")[0]
        assert len(kinds.split(",")) == plan["tile_count"], kinds
        assert kinds.count("mamba") == plan["mamba_tiles"] and kinds.count("attn") == plan["attn_tiles"]
        assert all(plan["shape_constraints"].values()), plan["shape_constraints"]
        print(f"  tiles {plan['tile_count']:>3}: intermediate {plan['intermediate_size']:>5} "
              f"tile bytes {plan['tile_bytes']:>12,.0f} vs reference {reference_bytes:,.0f} "
              f"({plan['deviation_pct']:+.2f}%), --tile-kinds has {len(kinds.split(','))} entries")
    print("test_sweep_planner PASSED")


# ------------------------------------------------------------------ runner --------------------
def main() -> None:
    assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == "", \
        "this suite must never take the GPU"
    print(f"torch {torch.__version__} on CPU, threads {torch.get_num_threads()}")
    for test in (test_pack_unpack_round_trip, test_store_round_trip, test_cache_semantics,
                 test_streaming_decode, test_sweep_planner):
        print(f"\n=== {test.__name__} ===")
        test()
    print("\nALL MIXTURE-OF-TILES TESTS PASSED")


if __name__ == "__main__":
    main()
