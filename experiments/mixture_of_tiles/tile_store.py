"""
experiments/mixture_of_tiles/tile_store.py: on-disk ternary tile library + a RAM tile cache.

Turns a `GeneralRoutedLM` into the artifact the hardware thesis assumes: one `tiles.bin` on NVMe holding
every routed tile at ~1.6 bits/weight, an `index.json` describing where each tensor lives, and a
`resident.pt` holding everything that is always in DRAM (embeddings, adapter banks, controller).
Rationale and the flash-resident mapping live in `experiments/mixture_of_tiles/AGENTS.md`.

Nothing here re-implements the quantiser: the ternary snapshot comes out of `BitLinear`'s own eval cache
(`experiments/bitlinear.py`), and `install_tile` writes that snapshot straight back into the cache.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard rule: this folder never touches the GPU

import sys
import json
import mmap
import time
import math
from collections import OrderedDict
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
CPU_DEVICE = torch.device("cpu")

from experiments.bitlinear import BitLinear, prepare_additive_masks

TRITS_PER_BYTE = 5                       # 3^5 = 243 < 256, so five ternary weights fit in one byte
BITS_PER_TERNARY_WEIGHT = 8.0 / TRITS_PER_BYTE   # 1.6
BASE3_POWERS = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32)
STORE_FORMAT = "mixture_of_tiles/v1"
MEGABYTE = 1024 * 1024


# ------------------------------------------------------------------ base-3 packing --------------
def packed_byte_length(numel: int) -> int:
    """Bytes needed for `numel` trits at five trits per byte (the tail byte is zero-padded)."""
    return (numel + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE


def pack_ternary(ternary_tensor: torch.Tensor) -> bytes:
    """Pack a tensor whose values are all in {-1, 0, +1} into base-3 bytes, five weights per byte."""
    flat = ternary_tensor.reshape(-1).to(torch.int8)
    if flat.numel() and int(flat.abs().max()) > 1:
        raise ValueError("pack_ternary expects values in {-1, 0, +1}")
    pad = (-flat.numel()) % TRITS_PER_BYTE
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.int8)])
    digits = (flat.to(torch.int32) + 1).view(-1, TRITS_PER_BYTE)     # {-1,0,1} -> {0,1,2}
    codes = (digits * BASE3_POWERS).sum(dim=1).to(torch.uint8)
    return codes.numpy().tobytes()


def unpack_ternary(packed: bytes, numel: int) -> torch.Tensor:
    """Inverse of `pack_ternary`; returns a flat int8 tensor of exactly `numel` values."""
    codes = np.frombuffer(packed, dtype=np.uint8, count=packed_byte_length(numel))
    return _unpack_codes(codes, numel)


def _unpack_codes(codes: np.ndarray, numel: int) -> torch.Tensor:
    """Base-3 digit extraction from a uint8 code array (the hot path shared by file reads)."""
    codes = np.ascontiguousarray(codes)
    if not codes.flags.writeable:                    # torch.from_numpy refuses a read-only buffer
        codes = codes.copy()
    remainder = torch.from_numpy(codes).to(torch.int32)
    out = torch.empty(remainder.numel(), TRITS_PER_BYTE, dtype=torch.int8)
    for digit in range(TRITS_PER_BYTE):
        out[:, digit] = (remainder % 3).to(torch.int8) - 1
        remainder = remainder // 3
    return out.reshape(-1)[:numel]


def achieved_bits_per_weight(numel: int) -> float:
    """Bits per weight actually spent, tail padding included."""
    return packed_byte_length(numel) * 8.0 / max(1, numel)


# ------------------------------------------------------------------ BitLinear plumbing ----------
def ternary_bitlinear_layers(module: nn.Module) -> List[Tuple[str, BitLinear]]:
    """Every ternary `BitLinear` under `module`, by dotted name."""
    return [(name, sub) for name, sub in module.named_modules()
            if isinstance(sub, BitLinear) and sub.ternary]


def dequantize_ternary(w_tilde: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Rebuild the effective weight `w_tilde * gamma` with BitLinear's own block layout."""
    if gamma.dim() > 0 and gamma.numel() > 1:
        block_size = w_tilde.numel() // gamma.numel()
        return (w_tilde.reshape(-1, block_size) * gamma).view(w_tilde.shape)
    return w_tilde * gamma


def install_bitlinear_snapshot(layer: BitLinear, w_tilde: torch.Tensor, gamma: torch.Tensor,
                               dtype: torch.dtype = torch.float32, build_masks: Optional[bool] = None) -> None:
    """Write a stored ternary snapshot into `layer`'s eval cache instead of re-quantising the master weight."""
    w_tilde = w_tilde.to(torch.float32)
    gamma = gamma.to(torch.float32)
    if build_masks is None:
        build_masks = bool(layer.use_additive_gemm)
    cache: Dict[str, object] = {
        "dtype": dtype,
        "w_eff": dequantize_ternary(w_tilde, gamma).to(dtype),
        "w_tilde": w_tilde,
        "gamma": gamma,
        "m_pos": None, "m_neg": None, "gamma_scalar": None,
    }
    if build_masks:
        m_pos, m_neg, gamma_scalar = prepare_additive_masks(w_tilde, gamma, dtype)
        cache.update(m_pos=m_pos, m_neg=m_neg, gamma_scalar=gamma_scalar)
    layer._eval_cache = cache
    layer.last_w_tilde = w_tilde
    layer.last_gamma = gamma


def canonicalize_to_store_precision(model: nn.Module) -> None:
    """Round every float parameter/buffer to fp16 and back, in place, so the in-memory model equals what
    `TileStore.build` writes. Without this the store is an fp16 approximation of the model and a round-trip
    can only be checked to ~1e-3; with it the check is exact. Caller must be in eval mode afterwards."""
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            if tensor.is_floating_point():
                tensor.data.copy_(tensor.data.to(torch.float16).to(tensor.dtype))
    for _, layer in ternary_bitlinear_layers(model):
        layer._eval_cache = None                    # force re-quantisation from the canonicalised masters


def _dense_state(tile: nn.Module) -> List[Tuple[str, torch.Tensor]]:
    """Tile tensors that are NOT ternary BitLinear weights: norms, Mamba conv1d/x_proj/dt_proj/A_log/D,
    expert router, soft-mix gate, and any bias."""
    ternary_weight_names = {f"{name}.weight" if name else "weight"
                            for name, _ in ternary_bitlinear_layers(tile)}
    return [(name, tensor) for name, tensor in tile.state_dict().items()
            if name not in ternary_weight_names]


# ------------------------------------------------------------------ residency guard -------------
class TileNotResidentError(RuntimeError):
    """Raised when a tile is executed while its weights are not paged in."""


def _residency_pre_hook(tile: nn.Module, _inputs):
    if not getattr(tile, "tile_is_resident", True):
        raise TileNotResidentError(
            f"tile {getattr(tile, 'tile_index', '?')} executed while evicted; it never came through the cache")


def enable_residency_guard(model: nn.Module) -> None:
    """Make an evicted tile raise instead of silently computing with scrubbed weights."""
    for index, tile in enumerate(model.tiles):
        tile.tile_index = index
        if not hasattr(tile, "tile_is_resident"):
            tile.tile_is_resident = True
        if not getattr(tile, "_residency_guard_installed", False):
            tile.register_forward_pre_hook(_residency_pre_hook)
            tile._residency_guard_installed = True


def tile_is_resident(model: nn.Module, index: int) -> bool:
    return bool(getattr(model.tiles[index], "tile_is_resident", True))


def install_tile(model: nn.Module, index: int, tensors: Dict[str, torch.Tensor],
                 build_masks: Optional[bool] = None) -> None:
    """Page a loaded tile into the model so its EVAL forward matches the model the store was built from."""
    tile = model.tiles[index]
    dense: Dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if name.endswith(".w_tilde") or name.endswith(".gamma"):
            continue
        dense[name] = tensor
    if dense:
        tile.load_state_dict(dense, strict=False)
    for name, layer in ternary_bitlinear_layers(tile):
        install_bitlinear_snapshot(layer, tensors[f"{name}.w_tilde"], tensors[f"{name}.gamma"],
                                   build_masks=build_masks)
    tile.tile_is_resident = True


def uninstall_tile(model: nn.Module, index: int, scrub: bool = False) -> None:
    """Evict a tile. The residency flag is what the guard checks; `scrub` additionally zeroes the tensors so
    a bypass of the guard produces obviously-wrong output (it costs a full pass over the tile, so the
    benchmark leaves it off and relies on the flag)."""
    tile = model.tiles[index]
    tile.tile_is_resident = False
    for _, layer in ternary_bitlinear_layers(tile):
        layer._eval_cache = None
        layer.last_w_tilde = None
        layer.last_gamma = None
    if scrub:
        with torch.no_grad():
            for tensor in list(tile.parameters()) + list(tile.buffers()):
                tensor.data.zero_()


# ------------------------------------------------------------------ the store -------------------
class TileRead(NamedTuple):
    """One page-in: the tensors, the bytes pulled off disk, and how long the pull took."""
    tensors: Dict[str, torch.Tensor]
    bytes_read: int
    seconds: float


class TileStore:
    """`tiles.bin` + `index.json` + `resident.pt`. Open with `TileStore.open(path)`."""

    def __init__(self, path: str, index: Dict[str, Any]):
        self.path = path
        self.index = index
        self._handle = open(os.path.join(path, "tiles.bin"), "rb")
        self._map = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)

    # -------------------------------------------------------------- build -----------------------
    @staticmethod
    def build(model: nn.Module, path: str, canonicalize: bool = True, dummy_seq_len: int = 8,
              verbose: bool = True) -> Dict[str, Any]:
        """Write the tile library for `model` under `path` and return (and optionally print) the byte table."""
        os.makedirs(path, exist_ok=True)
        model.eval()
        if canonicalize:
            canonicalize_to_store_precision(model)

        TileStore._populate_eval_caches(model, dummy_seq_len)

        tile_records: List[Dict[str, Any]] = []
        total_ternary_weights = 0
        total_packed_bytes = 0
        total_gamma_bytes = 0
        total_dense_bytes = 0
        blob_offset = 0
        with open(os.path.join(path, "tiles.bin"), "wb") as binary:
            for index, tile in enumerate(model.tiles):
                tensor_records: List[Dict[str, Any]] = []
                inner_offset = 0
                chunks: List[bytes] = []
                ram_bytes = 0

                for name, layer in ternary_bitlinear_layers(tile):
                    cache = layer._eval_cache
                    w_tilde = cache["w_tilde"].detach().to(torch.float32)
                    gamma = cache["gamma"].detach().to(torch.float32)
                    gamma = gamma.to(torch.float16).to(torch.float32)    # the store holds fp16 scales
                    install_bitlinear_snapshot(layer, w_tilde, gamma)    # keep the source model == the store
                    packed = pack_ternary(w_tilde.to(torch.int8))
                    gamma_bytes = gamma.to(torch.float16).numpy().tobytes()
                    tensor_records.append({
                        "name": name, "role": "ternary", "shape": list(w_tilde.shape),
                        "numel": int(w_tilde.numel()),
                        "packing": "base3x5", "bits_per_weight": achieved_bits_per_weight(w_tilde.numel()),
                        "offset": inner_offset, "length": len(packed),
                        "gamma_offset": inner_offset + len(packed), "gamma_length": len(gamma_bytes),
                        "gamma_shape": list(gamma.shape), "gamma_numel": int(gamma.numel()),
                        "gamma_dtype": "float16",
                    })
                    chunks.append(packed)
                    chunks.append(gamma_bytes)
                    inner_offset += len(packed) + len(gamma_bytes)
                    total_ternary_weights += int(w_tilde.numel())
                    total_packed_bytes += len(packed)
                    total_gamma_bytes += len(gamma_bytes)
                    ram_bytes += int(w_tilde.numel()) * 4

                for name, tensor in _dense_state(tile):
                    stored = tensor.detach().to(torch.float16) if tensor.is_floating_point() else tensor.detach()
                    raw = stored.contiguous().numpy().tobytes()
                    tensor_records.append({
                        "name": name, "role": "dense", "shape": list(tensor.shape),
                        "numel": int(tensor.numel()), "dtype": str(stored.dtype).replace("torch.", ""),
                        "offset": inner_offset, "length": len(raw),
                    })
                    chunks.append(raw)
                    inner_offset += len(raw)
                    total_dense_bytes += len(raw)
                    ram_bytes += int(tensor.numel()) * 4

                blob = b"".join(chunks)
                binary.write(blob)
                tile_records.append({
                    "index": index, "kind": getattr(tile, "kind", "?"),
                    "offset": blob_offset, "length": len(blob),
                    "ram_bytes": ram_bytes, "tensors": tensor_records,
                })
                blob_offset += len(blob)

        resident_state = {name: (tensor.to(torch.float16) if tensor.is_floating_point() else tensor)
                          for name, tensor in model.state_dict().items() if not name.startswith("tiles.")}
        resident_logical_bytes = sum(t.numel() * t.element_size() for t in resident_state.values())
        torch.save(resident_state, os.path.join(path, "resident.pt"))

        index_payload = {
            "format": STORE_FORMAT,
            "config": asdict(model.config),
            "canonicalized": bool(canonicalize),
            "tiles": tile_records,
            "resident": {
                "file": "resident.pt",
                "dtype": "float16",
                "logical_bytes": int(resident_logical_bytes),
                "file_bytes": int(os.path.getsize(os.path.join(path, "resident.pt"))),
                "keys": sorted(resident_state.keys()),
            },
        }
        with open(os.path.join(path, "index.json"), "w", encoding="utf-8") as handle:
            json.dump(index_payload, handle, indent=2)

        table = {
            "path": path,
            "num_tiles": len(tile_records),
            "per_tile_bytes": [record["length"] for record in tile_records],
            "per_tile_kind": [record["kind"] for record in tile_records],
            "per_tile_ram_bytes": [record["ram_bytes"] for record in tile_records],
            "tile_bytes_total": int(blob_offset),
            "ternary_weights": int(total_ternary_weights),
            "ternary_packed_bytes": int(total_packed_bytes),
            "gamma_bytes": int(total_gamma_bytes),
            "tile_dense_bytes": int(total_dense_bytes),
            "bits_per_weight_packed": (total_packed_bytes * 8.0 / max(1, total_ternary_weights)),
            "bits_per_weight_with_gamma": ((total_packed_bytes + total_gamma_bytes) * 8.0
                                           / max(1, total_ternary_weights)),
            "resident_logical_bytes": int(resident_logical_bytes),
            "resident_file_bytes": int(index_payload["resident"]["file_bytes"]),
        }
        table["total_stored_bytes"] = table["tile_bytes_total"] + table["resident_logical_bytes"]
        if verbose:
            print_byte_table(table)
        return table

    @staticmethod
    def _populate_eval_caches(model: nn.Module, seq_len: int) -> None:
        """One eval forward per tile on a dummy input, so every BitLinear holds its ternary snapshot.
        `force_tile` is used because free routing would leave unvisited tiles without a cache."""
        saved_force, saved_exit = model.force_tile, model.allow_exit
        seq_len = max(2, min(seq_len, model.config.max_position_embeddings))
        dummy = torch.zeros(1, seq_len, dtype=torch.long)
        try:
            model.allow_exit = False
            with torch.no_grad():
                for index in range(len(model.tiles)):
                    model.force_tile = index
                    model(dummy, max_loops=1)
        finally:
            model.force_tile, model.allow_exit = saved_force, saved_exit
        for _, layer in ternary_bitlinear_layers(model.tiles):
            if layer._eval_cache is None:                    # belt and braces: use BitLinear's own quantiser
                w_tilde, gamma = layer.get_ternary_weights()
                install_bitlinear_snapshot(layer, w_tilde, gamma)

    # -------------------------------------------------------------- open / read -----------------
    @classmethod
    def open(cls, path: str) -> "TileStore":
        with open(os.path.join(path, "index.json"), "r", encoding="utf-8") as handle:
            index = json.load(handle)
        if index.get("format") != STORE_FORMAT:
            raise ValueError(f"unexpected store format {index.get('format')!r}")
        return cls(path, index)

    @property
    def num_tiles(self) -> int:
        return len(self.index["tiles"])

    def tile_bytes(self, index: int) -> int:
        return int(self.index["tiles"][index]["length"])

    def tile_ram_bytes(self, index: int) -> int:
        """Float32 bytes the tile occupies once unpacked (what the DRAM cache actually holds)."""
        return int(self.index["tiles"][index]["ram_bytes"])

    def _slice(self, base: int, offset: int, length: int) -> np.ndarray:
        """Copy a byte range out of the mapping. The copy is the page-in, so it is the metered read."""
        return np.frombuffer(self._map, dtype=np.uint8, count=length, offset=base + offset).copy()

    def read_tile(self, index: int) -> TileRead:
        """Page tile `index` in: base-3 unpack to float32 ternary, fp16 scales and dense tensors to float32."""
        record = self.index["tiles"][index]
        base = int(record["offset"])
        started = time.perf_counter()
        tensors: Dict[str, torch.Tensor] = {}
        for entry in record["tensors"]:
            if entry["role"] == "ternary":
                codes = self._slice(base, entry["offset"], entry["length"])
                w_tilde = _unpack_codes(codes, int(entry["numel"])).to(torch.float32)
                tensors[f"{entry['name']}.w_tilde"] = w_tilde.view(*entry["shape"])
                gamma_raw = self._slice(base, entry["gamma_offset"], entry["gamma_length"])
                gamma = torch.from_numpy(gamma_raw.view(np.float16)).to(torch.float32)
                tensors[f"{entry['name']}.gamma"] = gamma.view(*entry["gamma_shape"])
            else:
                raw = self._slice(base, entry["offset"], entry["length"])
                numpy_dtype = np.dtype(entry["dtype"])
                flat = torch.from_numpy(np.ascontiguousarray(raw.view(numpy_dtype)))
                if flat.is_floating_point():
                    flat = flat.to(torch.float32)
                tensors[entry["name"]] = flat.view(*entry["shape"]) if entry["shape"] else flat.view(())
        return TileRead(tensors, int(record["length"]), time.perf_counter() - started)

    def load_resident(self, model: nn.Module) -> None:
        """Restore everything that is never paged: embeddings, adapter banks, controller, final norm."""
        state = torch.load(os.path.join(self.path, "resident.pt"), map_location="cpu", weights_only=True)
        float32_state = {name: (tensor.to(torch.float32) if tensor.is_floating_point() else tensor)
                         for name, tensor in state.items()}
        model.load_state_dict(float32_state, strict=False)

    def close(self) -> None:
        self._map.close()
        self._handle.close()

    def __enter__(self) -> "TileStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def print_byte_table(table: Dict[str, Any]) -> None:
    """The headline of a build: what a tile costs on disk and what is pinned in DRAM."""
    print(f"[store] {table['path']}")
    print(f"  {'tile':>4}  {'kind':>5}  {'disk bytes':>12}  {'disk MB':>8}  {'ram MB (fp32)':>14}")
    for index, (length, kind, ram) in enumerate(zip(table["per_tile_bytes"], table["per_tile_kind"],
                                                    table["per_tile_ram_bytes"])):
        print(f"  {index:>4}  {kind:>5}  {length:>12,}  {length / MEGABYTE:>8.3f}  {ram / MEGABYTE:>14.3f}")
    print(f"  tile bytes total      {table['tile_bytes_total']:>14,}  "
          f"({table['tile_bytes_total'] / MEGABYTE:.3f} MB)")
    print(f"    ternary packed      {table['ternary_packed_bytes']:>14,}  "
          f"({table['ternary_weights']:,} weights)")
    print(f"    gamma (fp16)        {table['gamma_bytes']:>14,}")
    print(f"    tile dense (fp16)   {table['tile_dense_bytes']:>14,}")
    print(f"  resident bytes        {table['resident_logical_bytes']:>14,}  "
          f"({table['resident_logical_bytes'] / MEGABYTE:.3f} MB, file "
          f"{table['resident_file_bytes'] / MEGABYTE:.3f} MB)")
    print(f"  bits/weight packed    {table['bits_per_weight_packed']:.4f}  "
          f"(with gamma {table['bits_per_weight_with_gamma']:.4f})")


# ------------------------------------------------------------------ the cache -------------------
class TileCache:
    """LRU of paged-in tiles. `on_evict` mirrors an eviction into the model so the model can only ever
    compute with tiles that came through this cache."""

    def __init__(self, store: TileStore, capacity_tiles: int, prefetch: bool = False,
                 injected_miss_latency_ms_per_mb: float = 0.0,
                 on_evict: Optional[Callable[[int], None]] = None,
                 sleep_injected: bool = False):
        if capacity_tiles < 1:
            raise ValueError("capacity_tiles must be >= 1")
        self.store = store
        self.capacity_tiles = int(capacity_tiles)
        self.prefetch_enabled = bool(prefetch)
        self.injected_miss_latency_ms_per_mb = float(injected_miss_latency_ms_per_mb)
        self.sleep_injected = bool(sleep_injected)
        self.on_evict = on_evict
        self._entries: "OrderedDict[int, Dict[str, torch.Tensor]]" = OrderedDict()
        self.reset_stats()

    # -------------------------------------------------------------- stats -----------------------
    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.prefetches = 0
        self.prefetch_bytes = 0
        self.bytes_read = 0                 # demand misses only (prefetch is counted separately)
        self.read_seconds = 0.0             # real mmap + unpack time of demand misses
        self.injected_seconds = 0.0         # modelled cold-NVMe time of demand misses
        self.prefetch_seconds = 0.0
        self.miss_latencies_ms: List[float] = []

    def stats(self) -> Dict[str, Any]:
        requests = self.hits + self.misses
        return {
            "capacity_tiles": self.capacity_tiles,
            "hits": self.hits, "misses": self.misses, "requests": requests,
            "hit_rate": (self.hits / requests) if requests else 0.0,
            "evictions": self.evictions,
            "prefetches": self.prefetches, "prefetch_bytes": self.prefetch_bytes,
            "bytes_read": self.bytes_read,
            "bytes_read_total": self.bytes_read + self.prefetch_bytes,
            "read_seconds": self.read_seconds,
            "injected_seconds": self.injected_seconds,
            "prefetch_seconds": self.prefetch_seconds,
            "total_read_seconds": self.read_seconds + self.injected_seconds + self.prefetch_seconds,
            "mean_miss_latency_ms": (sum(self.miss_latencies_ms) / len(self.miss_latencies_ms))
                                    if self.miss_latencies_ms else 0.0,
            "resident_tiles": list(self._entries.keys()),
        }

    # -------------------------------------------------------------- core ------------------------
    def _injected_seconds_for(self, byte_count: int) -> float:
        return self.injected_miss_latency_ms_per_mb * (byte_count / MEGABYTE) / 1000.0

    def _load(self, index: int) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
        read = self.store.read_tile(index)
        injected = self._injected_seconds_for(read.bytes_read)
        if injected and self.sleep_injected:
            time.sleep(injected)
        return read.tensors, read.bytes_read, read.seconds, injected

    def _evict_until(self, room_for: int, protect: Sequence[int] = ()) -> None:
        protected = set(protect)
        while len(self._entries) + room_for > self.capacity_tiles:
            victim = None
            for candidate in self._entries:                  # OrderedDict iterates least-recent first
                if candidate not in protected:
                    victim = candidate
                    break
            if victim is None:
                return
            self._entries.pop(victim)
            self.evictions += 1
            if self.on_evict is not None:
                self.on_evict(victim)

    def get(self, index: int) -> Dict[str, torch.Tensor]:
        """Demand-fetch tile `index`, recording hit/miss, bytes and latency."""
        if index in self._entries:
            self.hits += 1
            self._entries.move_to_end(index)
            return self._entries[index]
        self.misses += 1
        tensors, byte_count, seconds, injected = self._load(index)
        self.bytes_read += byte_count
        self.read_seconds += seconds
        self.injected_seconds += injected
        self.miss_latencies_ms.append((seconds + injected) * 1000.0)
        self._evict_until(1, protect=())
        self._entries[index] = tensors
        self._entries.move_to_end(index)
        return tensors

    def prefetch(self, indices: Iterable[int], protect: Sequence[int] = ()) -> List[int]:
        """Load tiles into the cache without counting them as a use. Never evicts anything in `protect`,
        and stops when the only remaining victims are protected."""
        loaded: List[int] = []
        protected = set(protect)
        for index in indices:
            if index in self._entries:
                continue
            evictable = [key for key in self._entries if key not in protected]
            if len(self._entries) >= self.capacity_tiles and not evictable:
                break
            tensors, byte_count, seconds, injected = self._load(index)
            self.prefetches += 1
            self.prefetch_bytes += byte_count
            self.prefetch_seconds += seconds + injected
            self._evict_until(1, protect=protected)
            self._entries[index] = tensors
            self._entries.move_to_end(index)
            protected.add(index)                      # a just-prefetched tile is not its own victim
            loaded.append(index)
        return loaded

    def contains(self, index: int) -> bool:
        return index in self._entries

    def clear(self) -> None:
        for index in list(self._entries.keys()):
            self._entries.pop(index)
            if self.on_evict is not None:
                self.on_evict(index)


__all__ = [
    "BITS_PER_TERNARY_WEIGHT", "TRITS_PER_BYTE", "TileCache", "TileNotResidentError", "TileRead",
    "TileStore", "achieved_bits_per_weight", "canonicalize_to_store_precision", "dequantize_ternary",
    "enable_residency_guard", "install_bitlinear_snapshot", "install_tile", "pack_ternary",
    "packed_byte_length", "print_byte_table", "ternary_bitlinear_layers", "tile_is_resident",
    "uninstall_tile", "unpack_ternary",
]
