"""
experiments/mixture_of_tiles/test_mixture_of_tiles.py: CPU-only tests for the tile store, the tile cache,
the streaming decoder and the sweep planner.

Coverage, in the order the mechanisms are risky:
  a base-3 pack/unpack is an exact round trip at every tail length
  b a store round trip reproduces the source model's EVAL logits bit-for-bit through a fresh model
  c the LRU cache's hit/miss/bytes accounting and prefetch semantics
  d cache size never changes what is generated, and an evicted tile cannot execute
  e chunked mixer calls continuing a (hop, tile) stream reproduce one whole-sequence tile forward
  f the state-cached decoder reproduces full recompute: ids identical, logits within 1e-4, with exits,
    repeated and skipped stream visits, weight eviction mid-generation, and the path-conditioned controller
  g the fixed-bytes sweep planner lands inside its tolerance and emits the right command

    python experiments/mixture_of_tiles/test_mixture_of_tiles.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard rule: this folder never touches the GPU

import sys
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

torch.set_num_threads(4)
CPU_DEVICE = torch.device("cpu")

from experiments.general_model.general_model import GeneralConfig, GeneralRoutedLM
from experiments.mixture_of_tiles import bench_cpu_decode as benchmark
from experiments.mixture_of_tiles.cached_decoder import CachedStreamingDecoder
from experiments.mixture_of_tiles.cpu_decoder import StreamingDecoder, usage_entropy_bits
from experiments.mixture_of_tiles.state_cache import hybrid_tile_forward_with_state
from experiments.mixture_of_tiles.sweep_tile_count import plan_sweep
from experiments.mixture_of_tiles.tile_store import (
    TileCache,
    TileNotResidentError,
    TileStore,
    achieved_bits_per_weight,
    install_tile,
    pack_ternary,
    packed_byte_length,
    tile_is_resident,
    ternary_bitlinear_layers,
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
    inputs = torch.randint(0, config.vocab_size, (2, SEQ))
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
def test_state_chunking():
    """A stream fed in chunks (multi-token, single token, multi-token tail) equals one whole-sequence call."""
    config = small_config()
    model = trained_model(config, seed=4)
    torch.manual_seed(13)
    length = 12
    hidden = torch.randn(1, length, HIDDEN)
    valid = torch.ones(1, length, dtype=torch.bool)
    alpha = model._alpha(None, 1, (1, length))
    boundaries = [(0, 5), (5, 6), (6, length)]
    with torch.no_grad():
        for index, tile in enumerate(model.tiles):
            mod_film, mod_lora = model.banks[index].modulations(alpha)
            reference, _balance, _extras = tile(hidden, valid, mod_film, mod_lora)
            state = None
            pieces = []
            for start, stop in boundaries:
                film_part, lora_part = model.banks[index].modulations(alpha[:, start:stop])
                delta, state = hybrid_tile_forward_with_state(tile, hidden[:, start:stop], film_part, lora_part, state)
                pieces.append(delta)
            chunked = torch.cat(pieces, dim=1)
            delta_max = (reference - chunked).abs().max().item()
            assert delta_max < 1e-5, f"tile {index} ({tile.kind}): chunked forward moved the output by {delta_max}"
            assert state.tokens == length, state.tokens
            print(f"  tile {index} ({tile.kind:5s}): chunks {boundaries} max |delta| {delta_max:.3e} "
                  f"stream tokens {state.tokens}")
        pointwise_model = GeneralRoutedLM(small_config(d_conv=1)).to(CPU_DEVICE).eval()
        pointwise_tile = pointwise_model.tiles[0]
        reference, _, _ = pointwise_tile(hidden, valid)
        first, state = hybrid_tile_forward_with_state(pointwise_tile, hidden[:, :1], None, None, None)
        remainder, state = hybrid_tile_forward_with_state(pointwise_tile, hidden[:, 1:], None, None, state)
        delta_max = (reference - torch.cat([first, remainder], dim=1)).abs().max().item()
        assert delta_max < 1e-5, delta_max
        assert state.conv_state.shape[-1] == 0 and state.tokens == length
        print(f"  d_conv=1: empty conv history, single-token then chunk max |delta| {delta_max:.3e}")
    print("test_state_chunking PASSED")


# ------------------------------------------------------------------ f -------------------------
def _reference_generate(model: GeneralRoutedLM, prompt: torch.Tensor, max_new_tokens: int) -> dict:
    """Greedy full recompute through the model itself (the StreamingDecoder protocol, without paging)."""
    ids = prompt.clone()
    generated, last_logits = [], []
    with torch.no_grad():
        prefill_logits = model(ids)["logits"][0].clone()
        mean_hops = 0.0
        for _ in range(max_new_tokens):
            out = model(ids)
            logits = out["logits"][0, -1]
            next_id = int(logits.argmax().item())
            generated.append(next_id)
            last_logits.append(logits.clone())
            mean_hops = float(out["mean_hops"])
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
    return {"generated_ids": generated, "last_logits": last_logits, "prefill_logits": prefill_logits,
            "mean_hops": mean_hops}


def _install_all(model: GeneralRoutedLM, store: TileStore) -> None:
    for index in range(store.num_tiles):
        install_tile(model, index, store.read_tile(index).tensors)


def _compare_cached(model: GeneralRoutedLM, store: TileStore, prompt: torch.Tensor, new_tokens: int,
                    capacity: int, label: str) -> dict:
    """Run the cached decoder against a fresh full-recompute reference and assert ids/logits agree."""
    _install_all(model, store)
    reference = _reference_generate(model, prompt, new_tokens)
    cache = TileCache(store, capacity_tiles=capacity)
    decoder = CachedStreamingDecoder(model, cache)
    try:
        result = decoder.generate(prompt, new_tokens, keep_logits=True)
    finally:
        decoder.detach()
    assert result["generated_ids"] == reference["generated_ids"], (
        f"{label}: cached ids {result['generated_ids']} != reference {reference['generated_ids']}")
    prefill_delta = (result["prefill_logits"] - reference["prefill_logits"]).abs().max().item()
    step_delta = max((cached - full).abs().max().item()
                     for cached, full in zip(result["last_logits"], reference["last_logits"]))
    assert prefill_delta < 1e-4, f"{label}: prefill logits moved by {prefill_delta}"
    assert step_delta < 1e-4, f"{label}: per-token logits moved by {step_delta}"
    print(f"  {label}: ids {result['generated_ids']} prefill max |delta| {prefill_delta:.3e} "
          f"step max |delta| {step_delta:.3e} mean hops ref {reference['mean_hops']:.2f} "
          f"cached {result['decode_mean_hops']:.2f} misses {result['misses']} "
          f"evictions {result['cache_stats']['evictions']} streams {result['num_streams']} "
          f"state {result['state_bytes']:,} B")
    result["reference_mean_hops"] = reference["mean_hops"]
    return result


def test_cached_decode_exactness():
    workdir = tempfile.mkdtemp(prefix="mot-cached-")
    try:
        config = small_config(routing="free")
        model = trained_model(config, seed=5)
        store_path = os.path.join(workdir, "store")
        TileStore.build(model, store_path, canonicalize=True, verbose=False)
        torch.manual_seed(17)
        prompt = torch.randint(0, VOCAB, (1, 12))
        new_tokens = 6

        # exit_lambda at the mean predicted gain: some tokens continue and some exit at every hop past
        # min_hops, so later streams see a strict, changing subset of the tokens (skipped visits).
        model.allow_exit = False
        with torch.no_grad():
            probe = model(prompt)
        model.config.exit_lambda = float(probe["mean_pred_gain"])

        with TileStore.open(store_path) as store:
            for allow_exit in (False, True):
                model.allow_exit = allow_exit
                for capacity in (1, store.num_tiles):
                    label = f"free routing, allow_exit={allow_exit}, capacity {capacity}"
                    result = _compare_cached(model, store, prompt, new_tokens, capacity, label)
                    if allow_exit:
                        assert config.min_hops < result["reference_mean_hops"] < config.max_hops, (
                            f"exit_lambda {config.exit_lambda:.4f} produced no mix of exits: "
                            f"mean hops {result['reference_mean_hops']}")
                    if capacity == 1:
                        assert result["cache_stats"]["evictions"] > 0, "capacity 1 evicted nothing"

            # an evicted tile cannot run through the stateful path either
            uninstall_tile(model, 0, scrub=False)
            assert not tile_is_resident(model, 0)
            raised = False
            try:
                hybrid_tile_forward_with_state(model.tiles[0], torch.randn(1, 2, HIDDEN), None, None, None)
            except TileNotResidentError:
                raised = True
            assert raised, "an evicted tile executed through the stateful path"
            print("  evicted tile 0 raised TileNotResidentError through the stateful path")

            # policies without a causal per-token answer are refused, not approximated
            model.allow_exit = True
            _install_all(model, store)
            decoder = CachedStreamingDecoder(model, TileCache(store, capacity_tiles=store.num_tiles))
            try:
                for setting in ("capacity", "random", "force_random_tiles", "collect_received"):
                    model.config.exit_policy = setting if setting in ("capacity", "random") else "threshold"
                    model.force_random_tiles = setting == "force_random_tiles"
                    model.collect_received = setting == "collect_received"
                    refused = False
                    try:
                        decoder.generate(prompt, 2)
                    except ValueError as error:
                        refused = True
                        print(f"  {setting} refused: {str(error)[:60]}...")
                    assert refused, f"{setting} was not refused"
            finally:
                decoder.detach()
                model.config.exit_policy = "threshold"
                model.force_random_tiles = False
                model.collect_received = False

        # path-conditioned controller (GRU path state per token across hops) through the same decoder
        path_config = small_config(routing="free", router_cond="path", adapter_cond="path")
        path_model = trained_model(path_config, seed=6)
        path_model.allow_exit = False
        path_store = os.path.join(workdir, "store-path")
        TileStore.build(path_model, path_store, canonicalize=True, verbose=False)
        with TileStore.open(path_store) as store:
            _compare_cached(path_model, store, prompt, 4, 2, "path-conditioned routing, capacity 2")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print("test_cached_decode_exactness PASSED")


def test_cached_benchmark_configuration():
    """Check longer dense-GEMM streams, repeated hop visits, and generation reset on tiny CPU geometry."""
    with tempfile.TemporaryDirectory(prefix="mot-benchmark-config-") as workdir:
        config = small_config(vocab_size=64, hidden_size=32, intermediate_size=64, d_state=4,
                              max_position_embeddings=144, tile_kinds=["mamba", "attn"], routing="fixed")
        model = trained_model(config, steps=1, seed=23)
        model.allow_exit = False
        for _, layer in ternary_bitlinear_layers(model):
            layer.use_additive_gemm = False
        store_path = os.path.join(workdir, "store")
        TileStore.build(model, store_path, canonicalize=True, verbose=False)
        torch.manual_seed(24)
        longest = torch.randint(0, config.vocab_size, (1, 128))
        with TileStore.open(store_path) as store:
            for prompt_length in (32, 64, 128):
                result = _compare_cached(model, store, longest[:, :prompt_length], 3, 1,
                                         f"dense GEMM fixed routing, prompt {prompt_length}, capacity 1")
                expected_streams = {f"{hop}:{hop % 2}": prompt_length + 2 for hop in range(config.max_hops)}
                assert result["stream_lengths"] == expected_streams, result["stream_lengths"]
                assert all(not layer.use_additive_gemm for _, layer in ternary_bitlinear_layers(model))

            second_prompt = longest[:, :5].flip(1)
            _install_all(model, store)
            reference = _reference_generate(model, second_prompt, 3)
            decoder = CachedStreamingDecoder(model, TileCache(store, capacity_tiles=1), build_masks=False)
            try:
                decoder.generate(longest[:, :32], 3)
                repeated = decoder.generate(second_prompt, 3, keep_logits=True)
                assert repeated["generated_ids"] == reference["generated_ids"]
                delta = max((cached - full).abs().max().item()
                            for cached, full in zip(repeated["last_logits"], reference["last_logits"]))
                assert delta < 1e-4, delta
                assert repeated["stream_lengths"] == {f"{hop}:{hop % 2}": 7 for hop in range(config.max_hops)}
                print(f"  reused decoder: second prompt 5 tokens, stream lengths all 7, max |delta| {delta:.3e}")
            finally:
                decoder.detach()
    print("test_cached_benchmark_configuration PASSED")


def test_prefetch_latency_accounting():
    """Account only unslept injected latency beyond the measured token time in both decoders."""
    class AccountingModel(torch.nn.Module):
        def __init__(self, cache):
            super().__init__()
            self.tiles = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
            self.config = SimpleNamespace()
            self.use_value = self.allow_exit = self.force_random_tiles = self.collect_received = False
            self.cache = cache

        def _run_tile(self, *arguments):
            raise AssertionError("timing fixture does not execute model tiles")

        def forward(self, ids):
            self.cache.get(0)
            self.cache.prefetch([1], protect=(0,))
            return {"logits": torch.tensor([[[0.0, 1.0]]]), "mean_hops": 1.0}

    store = SimpleNamespace(read_tile=lambda index: SimpleNamespace(
        tensors={}, bytes_read=1024 * 1024, seconds=0.025))
    prompt = torch.tensor([[0]], device=CPU_DEVICE)
    for decoder_type in (StreamingDecoder, CachedStreamingDecoder):
        for injected_ms in (0.0, 2.0):
            cache = TileCache(store, capacity_tiles=2, prefetch=True,
                              injected_miss_latency_ms_per_mb=injected_ms)
            model = AccountingModel(cache)
            safety_check = Mock()
            decoder = decoder_type(model, cache, safety_check=safety_check)
            if decoder_type is CachedStreamingDecoder:
                decoder._forward_chunk = lambda ids, offset: (model(ids)["logits"], 1.0)
            module_name = decoder_type.__module__
            try:
                with patch(f"{module_name}.time.perf_counter", side_effect=[10.0, 10.1, 20.0, 20.1]), \
                        patch("experiments.mixture_of_tiles.tile_store.time.sleep", side_effect=AssertionError("no sleep")):
                    result = decoder.generate(prompt, 2)
                assert safety_check.call_count == 2, safety_check.call_count
                expected_injected_ms = 2 * injected_ms
                assert abs(result["total_injected_ms"] - expected_injected_ms) < 1e-9, result
                assert abs(result["total_wall_ms"] - (200.0 + expected_injected_ms)) < 1e-9, result
                assert result["cache_stats"]["prefetch_read_seconds"] == 0.025
                assert result["cache_stats"]["prefetch_injected_seconds"] == injected_ms / 1000.0

                safety_check.side_effect = [None, RuntimeError("mock safety stop")]
                refused = False
                with patch(f"{module_name}.time.perf_counter", side_effect=[10.0, 10.1]):
                    try:
                        decoder.generate(prompt, 2)
                    except RuntimeError as error:
                        assert str(error) == "mock safety stop"
                        refused = True
                assert refused, "second token bypassed the safety callback"
                assert cache.stats()["hits"] == 0, "second token executed after the guard failed"
                print(f"  {decoder_type.__name__}: mocked 200 ms compute + {expected_injected_ms:.0f} ms "
                      f"injected = {result['total_wall_ms']:.0f} ms; second-token guard stops execution")
            finally:
                decoder.detach()
    print("test_prefetch_latency_accounting PASSED")


def test_machine_guards():
    """Fail closed on unsafe or unavailable mocked telemetry without querying hardware."""
    arguments = SimpleNamespace(gpu_guard_mib=7000, min_free_ram_gb=4.0, max_rss_gb=6.0,
                                ignore_guards=False)
    cases = [(1000, 8.0, 1.0, False), (8000, 8.0, 1.0, True), (1000, 2.0, 1.0, True),
             (1000, 8.0, 7.0, True), (None, 8.0, 1.0, True), (1000, None, 1.0, True),
             (1000, 8.0, None, True)]
    for gpu_usage, free_memory, resident_memory, should_refuse in cases:
        with patch.object(benchmark, "gpu_memory_used_mib", return_value=gpu_usage), \
                patch.object(benchmark, "free_ram_gb", return_value=free_memory), \
                patch.object(benchmark, "resident_set_gb", return_value=resident_memory):
            refused = False
            try:
                benchmark.check_machine_guards(arguments)
            except benchmark.MachineGuardError:
                refused = True
            assert refused == should_refuse, (gpu_usage, free_memory, resident_memory, refused)
    with patch.object(benchmark, "gpu_memory_used_mib", side_effect=[1000, 8000]) as gpu_probe, \
            patch.object(benchmark, "free_ram_gb", return_value=8.0), \
            patch.object(benchmark, "resident_set_gb", return_value=1.0), \
            patch.object(benchmark.time, "monotonic", side_effect=[0.0, 0.1, 1.1]):
        monitor = benchmark.MachineGuardMonitor(arguments)
        monitor.check()
        monitor.check()
        assert gpu_probe.call_count == 1, "GPU telemetry was not throttled"
        refused = False
        try:
            monitor.check()
        except benchmark.MachineGuardError:
            refused = True
        assert refused and gpu_probe.call_count == 2, "rising GPU usage was not rechecked"
    with patch.object(benchmark, "gpu_memory_used_mib", return_value=1000), \
            patch.object(benchmark, "free_ram_gb", side_effect=[8.0, 2.0]), \
            patch.object(benchmark, "resident_set_gb", return_value=1.0), \
            patch.object(benchmark.time, "monotonic", side_effect=[0.0, 0.1]):
        monitor = benchmark.MachineGuardMonitor(arguments)
        monitor.check()
        refused = False
        try:
            monitor.check()
        except benchmark.MachineGuardError:
            refused = True
        assert refused, "RAM was not rechecked between GPU polls"
    print("  mocked telemetry: healthy allowed; high GPU, low RAM, high RSS and each unavailable metric refused")
    print("  monitored telemetry: GPU rise caught after refresh; RAM decline caught between GPU polls")
    print("test_machine_guards PASSED")


# ------------------------------------------------------------------ g -------------------------
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
                 test_streaming_decode, test_state_chunking, test_cached_decode_exactness,
                 test_cached_benchmark_configuration, test_prefetch_latency_accounting,
                 test_machine_guards, test_sweep_planner):
        print(f"\n=== {test.__name__} ===")
        test()
    print("\nALL MIXTURE-OF-TILES TESTS PASSED")


if __name__ == "__main__":
    main()
