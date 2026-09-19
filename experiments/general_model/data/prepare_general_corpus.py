"""
experiments/general_model/data/prepare_general_corpus.py: general-model pretraining corpus (~300M tokens).

Scales experiments/data/prepare_clean_corpus.py from 40M tokens of story/code/math to a 300M-token
four-domain corpus under a purpose-trained 16k byte-level BPE. Helpers (ngram_hashes,
dedupe_documents, take_tokens, load_gsm8k, load_csn_python, load_tinystories, EOT) are imported from
that module, never copied. Rationale, sources and design notes: see experiments/general_model/data/AGENTS.md.

Stages:
  fetch      download/stream the raw sources into data/raw/
  tokenizer  train the 16k byte-level BPE on a stratified TRAIN-only sample -> data/general/tokenizer.json
  build      dedupe, encode, budget, leak-filter, shuffle, write data/general/*.pt + manifest.json
  verify     reload artefacts and re-prove the quality bar

Run:  python experiments/general_model/data/prepare_general_corpus.py --stage all
"""

import argparse
import glob
import hashlib
import json
import os
import random
import shutil
import sys
import time
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.data.prepare_clean_corpus import (  # noqa: E402
    EOT,
    dedupe_documents,
    format_gsm8k,  # noqa: F401  (re-exported for callers of this module)
    load_csn_python,
    load_gsm8k,
    load_tinystories,
    ngram_hashes,
    take_tokens,
)

SOURCES = ("fineweb", "tinystories", "code", "math")
DEFAULT_MIX = "fineweb:0.50,tinystories:0.25,code:0.17,math:0.08"
VOCAB_SIZE = 16384
NGRAM = 16
MIN_CHARS = {"fineweb": 200, "tinystories": 50, "code": 40, "math": 0}
BYTES_PER_TOKEN_GUESS = 4.3  # only used to size the raw FineWeb pull

_TIMINGS: Dict[str, float] = {}
_TIMINGS_PATH: Optional[str] = None  # set by resolve_paths; stages may run in separate processes


# ------------------------------------------------------------------ small utilities ------------
def _log(msg: str) -> None:
    print(msg, flush=True)


def _mb(n: int) -> str:
    return f"{n / 1024 ** 2:,.1f} MB"


def _doc_key(d: str) -> bytes:
    # identical to the key used inside dedupe_documents, so stream dedupe and batch dedupe agree
    return hashlib.blake2b(" ".join(d.split()).encode("utf-8"), digest_size=16).digest()


def _batched(it: Iterable[str], n: int) -> Iterator[List[str]]:
    buf: List[str] = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


class _Stage:
    """Context manager that records wall time per stage."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        _log(f"\n=== {self.name} ===")
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        _TIMINGS[self.name] = round(dt, 1)
        _persist_timings()
        _log(f"=== {self.name}: {dt / 60:.1f} min ===")
        return False


def _persist_timings() -> None:
    """Accumulate stage wall times on disk so stages run as separate processes still report."""
    if not _TIMINGS_PATH:
        return
    os.makedirs(os.path.dirname(_TIMINGS_PATH), exist_ok=True)
    prev = {}
    if os.path.exists(_TIMINGS_PATH):
        try:
            prev = json.load(open(_TIMINGS_PATH))
        except json.JSONDecodeError:
            prev = {}
    prev.update(_TIMINGS)
    _TIMINGS.update(prev)
    with open(_TIMINGS_PATH, "w") as f:
        json.dump(prev, f, indent=2)


class StreamDeduper:
    """Global exact-duplicate removal over a document stream, reusing dedupe_documents per batch."""

    def __init__(self) -> None:
        self.seen: set = set()
        self.removed = 0
        self.kept = 0

    def filter(self, docs: Iterable[str], batch: int = 2048) -> Iterator[str]:
        for chunk in _batched(docs, batch):
            uniq, dups = dedupe_documents(chunk)
            self.removed += dups
            for d in uniq:
                k = _doc_key(d)
                if k in self.seen:
                    self.removed += 1
                    continue
                self.seen.add(k)
                self.kept += 1
                yield d

    def contains(self, d: str) -> bool:
        return _doc_key(d) in self.seen

    def release(self) -> None:
        self.seen = set()


# ------------------------------------------------------------------ source iterators -----------
def _iter_jsonl_text(paths: List[str]) -> Iterator[str]:
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)["text"]


def _iter_eot_documents(path: str, min_chars: int, chunk_chars: int = 1 << 24) -> Iterator[str]:
    """Split a multi-GB <|endoftext|>-separated file without holding it in memory."""
    buf = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(EOT)
            buf = parts.pop()
            for p in parts:
                p = p.strip()
                if len(p) > min_chars:
                    yield p
    # the trailing remainder is dropped: TinyStoriesV2-GPT4-train.txt ends mid-story, exactly as
    # load_tinystories assumes with docs[:-1]


def _iter_csn_parquet(paths: List[str], min_chars: int) -> Iterator[str]:
    import pyarrow.parquet as pq

    for p in paths:
        pf = pq.ParquetFile(p)
        names = pf.schema_arrow.names
        col = "whole_func_string" if "whole_func_string" in names else "func_code_string"
        for batch in pf.iter_batches(batch_size=2048, columns=[col]):
            for s in batch.column(0).to_pylist():
                if s and len(s) > min_chars:
                    yield s


def csn_train_parquets(args) -> List[str]:
    return sorted(glob.glob(os.path.join(args.csn_dir, "train-*.parquet")))


def fineweb_train_shards(args) -> List[str]:
    return sorted(glob.glob(os.path.join(args.fineweb_dir, "train-*.jsonl")))


def train_document_iter(name: str, args) -> Iterator[str]:
    """Streamed TRAIN documents for one source (never materialises a multi-GB list)."""
    if name == "fineweb":
        return _iter_jsonl_text(fineweb_train_shards(args))
    if name == "tinystories":
        return _iter_eot_documents(args.tinystories_train, MIN_CHARS["tinystories"])
    if name == "code":
        return _iter_csn_parquet(csn_train_parquets(args), MIN_CHARS["code"])
    if name == "math":
        return iter(load_gsm8k(args.gsm8k_train, args.gsm8k_test)[0])
    raise KeyError(name)


def heldout_documents(name: str, args) -> List[str]:
    """Author-provided held-out documents for one source (all four fit comfortably in memory)."""
    if name == "fineweb":
        return list(_iter_jsonl_text([os.path.join(args.fineweb_dir, "heldout-00000.jsonl")]))
    if name == "tinystories":
        return load_tinystories(args.tinystories_valid, args.tinystories_valid)[1]
    if name == "code":
        return load_csn_python(args.csn_test, args.csn_test)[1]
    if name == "math":
        return load_gsm8k(args.gsm8k_train, args.gsm8k_test)[1]
    raise KeyError(name)


# ------------------------------------------------------------------ stage: fetch ----------------
def fetch_fineweb(args) -> None:
    marker = os.path.join(args.fineweb_dir, "fetch_complete.json")
    if os.path.exists(marker) and not args.force:
        _log(f"  fineweb: already fetched ({json.load(open(marker))['train_bytes']:,} train bytes)")
        return
    from datasets import load_dataset

    os.makedirs(args.fineweb_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(args.fineweb_dir, "*.jsonl")):
        os.remove(stale)

    train_budget = int(args.fineweb_train_mb * 1024 ** 2)
    hold_budget = int(args.fineweb_heldout_mb * 1024 ** 2)
    shard_budget = int(args.fineweb_shard_mb * 1024 ** 2)
    _log(f"  fineweb: streaming {args.fineweb_repo}:{args.fineweb_config} "
         f"for {_mb(train_budget)} train + {_mb(hold_budget)} held-out")

    ds = load_dataset(args.fineweb_repo, args.fineweb_config, split="train", streaming=True)
    stats = {"train_docs": 0, "train_bytes": 0, "heldout_docs": 0, "heldout_bytes": 0,
             "docs_seen": 0, "docs_dropped_short": 0, "shards": []}
    t0 = time.time()
    shard_idx, shard_bytes = 0, 0
    f = open(os.path.join(args.fineweb_dir, f"train-{shard_idx:05d}.jsonl"), "w", encoding="utf-8")
    phase = "train"
    try:
        for rec in ds:
            stats["docs_seen"] += 1
            text = rec["text"]
            if len(text) < MIN_CHARS["fineweb"]:
                stats["docs_dropped_short"] += 1
                continue
            nbytes = len(text.encode("utf-8"))
            line = json.dumps({"id": rec["id"], "url": rec.get("url", ""), "text": text},
                              ensure_ascii=False)
            f.write(line + "\n")
            if phase == "train":
                stats["train_docs"] += 1
                stats["train_bytes"] += nbytes
                shard_bytes += nbytes
                if shard_bytes >= shard_budget:
                    f.close()
                    stats["shards"].append(f"train-{shard_idx:05d}.jsonl")
                    shard_idx += 1
                    shard_bytes = 0
                    f = open(os.path.join(args.fineweb_dir, f"train-{shard_idx:05d}.jsonl"),
                             "w", encoding="utf-8")
                if stats["train_bytes"] >= train_budget:
                    # the held-out slice is a strictly later, document-disjoint slice of the same stream
                    f.close()
                    stats["shards"].append(f"train-{shard_idx:05d}.jsonl")
                    phase = "heldout"
                    f = open(os.path.join(args.fineweb_dir, "heldout-00000.jsonl"), "w", encoding="utf-8")
                    _log(f"    train slice done: {stats['train_docs']:,} docs / "
                         f"{_mb(stats['train_bytes'])} in {(time.time() - t0) / 60:.1f} min")
                elif stats["train_docs"] % 20000 == 0:
                    _log(f"    {stats['train_docs']:,} docs / {_mb(stats['train_bytes'])} "
                         f"({(time.time() - t0) / 60:.1f} min)")
            else:
                stats["heldout_docs"] += 1
                stats["heldout_bytes"] += nbytes
                if stats["heldout_bytes"] >= hold_budget:
                    break
    finally:
        if not f.closed:
            f.close()
    stats["seconds"] = round(time.time() - t0, 1)
    with open(marker, "w") as m:
        json.dump(stats, m, indent=2)
    _log(f"  fineweb: {stats['train_docs']:,} train docs ({_mb(stats['train_bytes'])}) + "
         f"{stats['heldout_docs']:,} held-out docs ({_mb(stats['heldout_bytes'])})")


def _hf_download_to(repo_id: str, filename: str, dest: str, dl_dir: str, expect_size: Optional[int] = None) -> None:
    from huggingface_hub import hf_hub_download

    if os.path.exists(dest) and (expect_size is None or os.path.getsize(dest) == expect_size):
        _log(f"  {os.path.basename(dest)}: present ({_mb(os.path.getsize(dest))})")
        return
    os.makedirs(dl_dir, exist_ok=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    _log(f"  downloading {repo_id}:{filename} -> {dest}")
    p = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=dl_dir)
    os.replace(p, dest)  # move, never copy: these files are up to 2.2 GB
    _log(f"  {os.path.basename(dest)}: {_mb(os.path.getsize(dest))}")


def fetch_tinystories(args) -> None:
    _hf_download_to("roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt",
                    args.tinystories_train, args.hf_dl_dir, expect_size=2227753162)


def fetch_csn(args) -> None:
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_files("code-search-net/code_search_net", repo_type="dataset")
             if f.startswith("python/train-") and f.endswith(".parquet")]
    if not files:
        raise RuntimeError("no python train parquet found in code-search-net/code_search_net")
    for fn in sorted(files):
        _hf_download_to("code-search-net/code_search_net", fn,
                        os.path.join(args.csn_dir, os.path.basename(fn)), args.hf_dl_dir)


def stage_fetch(args) -> None:
    with _Stage("fetch"):
        fetch_fineweb(args)
        fetch_tinystories(args)
        fetch_csn(args)
        for p in (args.tinystories_valid, args.csn_test, args.gsm8k_train, args.gsm8k_test):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        _log("  gsm8k + held-out files present")
        if os.path.isdir(args.hf_dl_dir):
            shutil.rmtree(args.hf_dl_dir, ignore_errors=True)


# ------------------------------------------------------------------ stage: tokenizer ------------
def _write_tokenizer_sample(name: str, args, budget_bytes: int, path: str) -> int:
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for d in train_document_iter(name, args):
            f.write(d)
            f.write("\n")
            written += len(d.encode("utf-8")) + 1
            if written >= budget_bytes:
                break
    return written


def bytes_per_token_report(args, sample_dir: str, sample_bytes: int) -> Dict[str, Dict[str, float]]:
    """bytes/token for the new tokenizer vs GPT-2 on the same per-source sample."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    new_tok = Tokenizer.from_file(args.tokenizer_path)
    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    report: Dict[str, Dict[str, float]] = {}
    for name in SOURCES:
        p = os.path.join(sample_dir, f"{name}.txt")
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            text = f.read(sample_bytes)
        nbytes = len(text.encode("utf-8"))
        pieces = [text[i:i + 100_000] for i in range(0, len(text), 100_000)]
        n_new = sum(len(e.ids) for e in new_tok.encode_batch(pieces))
        n_gpt2 = sum(len(ids) for ids in gpt2(pieces, add_special_tokens=False)["input_ids"])
        report[name] = {
            "sample_bytes": nbytes,
            "new_tokens": n_new,
            "gpt2_tokens": n_gpt2,
            "new_bytes_per_token": round(nbytes / max(1, n_new), 4),
            "gpt2_bytes_per_token": round(nbytes / max(1, n_gpt2), 4),
            "compression_vs_gpt2": round((nbytes / max(1, n_new)) / (nbytes / max(1, n_gpt2)), 4),
        }
    return report


def stage_tokenizer(args) -> Dict:
    with _Stage("tokenizer"):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

        os.makedirs(args.out_dir, exist_ok=True)
        sample_dir = os.path.join(args.cache_dir, "tokmix")
        os.makedirs(sample_dir, exist_ok=True)
        mix = parse_mix(args.mix)
        total = int(args.tokenizer_sample_mb * 1024 ** 2)

        sample_files, sample_sizes = [], {}
        for name in SOURCES:
            path = os.path.join(sample_dir, f"{name}.txt")
            want = int(mix[name] * total)
            if os.path.exists(path) and os.path.getsize(path) >= 0.99 * want and not args.force:
                got = os.path.getsize(path)
            else:
                got = _write_tokenizer_sample(name, args, want, path)
            sample_sizes[name] = got
            sample_files.append(path)
            _log(f"  sample {name:12s} {_mb(got):>10s} (target {_mb(want)})")

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.post_processor = processors.ByteLevel(trim_offsets=True)
        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=[EOT],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes -> never OOV
            show_progress=False,
        )
        _log(f"  training byte-level BPE vocab={VOCAB_SIZE} on {_mb(sum(sample_sizes.values()))}")
        tok.train(sample_files, trainer)
        tok.save(args.tokenizer_path)
        eos_id = tok.token_to_id(EOT)
        _log(f"  saved {args.tokenizer_path} | vocab {tok.get_vocab_size()} | eos_id {eos_id}")

        report = bytes_per_token_report(args, sample_dir, int(args.bpt_sample_mb * 1024 ** 2))
        out = {"vocab_size": tok.get_vocab_size(), "eos_id": eos_id,
               "sample_bytes": sample_sizes, "bytes_per_token": report}
        with open(os.path.join(args.out_dir, "tokenizer_report.json"), "w") as f:
            json.dump(out, f, indent=2)
        _log(f"  {'source':12s} {'new B/tok':>10s} {'gpt2 B/tok':>11s} {'ratio':>7s}")
        for name, r in report.items():
            _log(f"  {name:12s} {r['new_bytes_per_token']:10.3f} {r['gpt2_bytes_per_token']:11.3f} "
                 f"{r['compression_vs_gpt2']:7.3f}")
        return out


# ------------------------------------------------------------------ budgeting -------------------
def parse_mix(spec: str) -> Dict[str, float]:
    return {k: float(v) for k, v in (kv.split(":") for kv in spec.split(","))}


def resolve_budgets(mix: Dict[str, float], total: int, available: Dict[str, int]) -> Dict[str, int]:
    """mix*total per source; any shortfall is redistributed proportionally over sources with headroom."""
    budget = {k: min(available[k], int(round(v * total))) for k, v in mix.items()}
    for _ in range(16):
        short = total - sum(budget.values())
        if short <= 0:
            break
        room = {k: available[k] - budget[k] for k in mix if available[k] - budget[k] > 0}
        if not room:
            break
        wsum = sum(mix[k] for k in room)
        added = 0
        for k in room:
            add = min(room[k], int(short * mix[k] / wsum))
            budget[k] += add
            added += add
        if added == 0:
            break
    return budget


def take_indices_by_tokens(lengths: np.ndarray, order: np.ndarray, budget: int) -> Tuple[List[int], int]:
    """Index-level analogue of take_tokens: walk shuffled doc ids until the token budget is met."""
    out, n = [], 0
    for i in order:
        if n >= budget:
            break
        out.append(int(i))
        n += int(lengths[i])
    return out, n


# ------------------------------------------------------------------ encoding --------------------
def encode_train_to_cache(tok, docs: Iterable[str], cache_path: str, eos_id: int,
                          cap_tokens: int, batch: int = 512, label: str = "") -> Tuple[np.ndarray, bool]:
    """Encode a document stream to a uint16 .bin + per-doc lengths; stop at cap_tokens."""
    lengths: List[int] = []
    total = 0
    exhausted = True
    t0 = time.time()
    with open(cache_path, "wb") as f:
        for chunk in _batched(docs, batch):
            encs = tok.encode_batch(chunk)
            arrs = []
            for e in encs:
                ids = e.ids
                arrs.append(np.asarray(ids + [eos_id], dtype=np.uint16))
                lengths.append(len(ids) + 1)
            block = np.concatenate(arrs)
            block.tofile(f)
            total += int(block.size)
            if total >= cap_tokens:
                exhausted = False
                break
            if len(lengths) % 100_000 < batch:
                _log(f"    {label}: {total / 1e6:.1f}M tokens, {len(lengths):,} docs "
                     f"({time.time() - t0:.0f}s)")
    return np.asarray(lengths, dtype=np.int64), exhausted


def encode_heldout(tok, docs: List[str], eos_id: int, cap_tokens: int) -> Tuple[List[str], List[np.ndarray]]:
    kept_text: List[str] = []
    kept_tok: List[np.ndarray] = []
    total = 0
    for chunk in _batched(docs, 512):
        for text, e in zip(chunk, tok.encode_batch(chunk)):
            a = np.asarray(e.ids + [eos_id], dtype=np.int32)
            kept_text.append(text)
            kept_tok.append(a)
            total += len(a)
        if total >= cap_tokens:
            break
    return kept_text, kept_tok


# ------------------------------------------------------------------ 16-gram leak index ----------
def build_gram_buckets(stream: np.ndarray, bucket_dir: str, n_buckets: int, chunk: int) -> int:
    """Hash every 16-gram of the train stream into n_buckets on-disk files (chunked: RAM stays ~1 GB)."""
    os.makedirs(bucket_dir, exist_ok=True)
    for p in glob.glob(os.path.join(bucket_dir, "b*.bin")):
        os.remove(p)
    files = [open(os.path.join(bucket_dir, f"b{i:02d}.bin"), "wb") for i in range(n_buckets)]
    total = len(stream)
    written = 0
    try:
        start = 0
        while start < total:
            end = min(total, start + chunk)
            seg = np.asarray(stream[start:min(total, end + NGRAM - 1)], dtype=np.int64)
            h = ngram_hashes(seg, NGRAM)
            if len(h):
                b = (h & (n_buckets - 1)).astype(np.int8)
                for i in range(n_buckets):
                    part = h[b == i]
                    part.tofile(files[i])
                    written += len(part)
            del seg, h
            _log(f"    grams {min(end, total) / 1e6:.0f}M/{total / 1e6:.0f}M tokens")
            start = end
    finally:
        for f in files:
            f.close()
    return written


def bucket_membership(bucket_dir: str, n_buckets: int, vg: np.ndarray) -> Tuple[np.ndarray, int]:
    """Exact membership of val 16-grams in the train gram set + exact global unique gram count."""
    hit = np.zeros(len(vg), dtype=bool)
    vb = (vg & (n_buckets - 1)).astype(np.int8)
    unique_total = 0
    for i in range(n_buckets):
        tb = np.fromfile(os.path.join(bucket_dir, f"b{i:02d}.bin"), dtype=np.int64)
        if len(tb) == 0:
            continue
        tb = np.unique(tb)  # sorted -> membership via searchsorted, as in prepare_clean_corpus
        unique_total += len(tb)
        m = vb == i
        sub = vg[m]
        if len(sub):
            pos = np.searchsorted(tb, sub)
            pos[pos == len(tb)] = 0
            hit[m] = tb[pos] == sub
        del tb
    return hit, unique_total


# ------------------------------------------------------------------ stage: build ----------------
def stage_build(args) -> Dict:
    from tokenizers import Tokenizer

    with _Stage("build") as stage:
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)
        os.makedirs(args.meta_dir, exist_ok=True)
        rng = random.Random(args.seed)
        nprng = np.random.default_rng(args.seed)
        tok = Tokenizer.from_file(args.tokenizer_path)
        eos_id = tok.token_to_id(EOT)
        mix = parse_mix(args.mix)
        manifest: Dict = {
            "tokenizer": os.path.relpath(args.tokenizer_path, REPO_ROOT).replace("\\", "/"),
            "vocab_size": tok.get_vocab_size(),
            "eos_id": eos_id,
            "seed": args.seed,
            "ngram": NGRAM,
            "leak_drop_threshold": args.leak_drop_threshold,
            "sources": {},
        }

        # -- 1. encode train + held-out per source ------------------------------------------------
        cache: Dict[str, Dict] = {}
        val_text: Dict[str, List[str]] = {}
        val_tok: Dict[str, List[np.ndarray]] = {}
        for name in SOURCES:
            t0 = time.time()
            cap = int(mix[name] * args.train_tokens * args.encode_headroom)
            bin_path = os.path.join(args.cache_dir, f"{name}.u16.bin")
            len_path = os.path.join(args.cache_dir, f"{name}.lengths.npy")
            dd = StreamDeduper()
            _log(f"  [{name}] encoding train (cap {cap / 1e6:.0f}M tokens)")
            if os.path.exists(len_path) and os.path.exists(bin_path) and not args.force:
                # resumed run: the train key set is gone, so the val exact-match filter below is
                # skipped -- the 16-gram filter subsumes it for any document of >= 16 tokens
                lengths = np.load(len_path)
                stats = json.load(open(os.path.join(args.cache_dir, f"{name}.meta.json")))
                _log(f"    cached: {int(lengths.sum()) / 1e6:.1f}M tokens, {len(lengths):,} docs")
            else:
                lengths, exhausted = encode_train_to_cache(
                    tok, dd.filter(train_document_iter(name, args)), bin_path, eos_id, cap, label=name)
                stats = {"exhausted": exhausted, "dups": dd.removed, "docs": int(len(lengths)),
                         "tokens": int(lengths.sum())}
                np.save(len_path, lengths)
                with open(os.path.join(args.cache_dir, f"{name}.meta.json"), "w") as f:
                    json.dump(stats, f)
            _log(f"    train {int(lengths.sum()) / 1e6:.1f}M tokens / {len(lengths):,} docs "
                 f"| dup removed {stats.get('dups', 0):,} | source exhausted {stats['exhausted']} "
                 f"| {time.time() - t0:.0f}s")

            # held-out: author-provided split, deduped and stripped of any exact train document
            hv = heldout_documents(name, args)
            hv, hv_dups = dedupe_documents(hv)
            n_before = len(hv)
            if dd.seen:
                hv = [d for d in hv if not dd.contains(d)]
            dd.release()
            rng.shuffle(hv)
            vt, va = encode_heldout(tok, hv, eos_id, int(args.val_tokens_per_source * args.val_surplus))
            val_text[name], val_tok[name] = vt, va
            cache[name] = {"lengths": lengths, "bin": bin_path, "exhausted": stats["exhausted"],
                           "dups": stats.get("dups", 0), "val_dups": hv_dups,
                           "val_docs_available": len(hv),
                           "val_exact_train_matches": n_before - len(hv)}
            _log(f"    val   {sum(len(a) for a in va):,} tokens / {len(va):,} docs "
                 f"| dup removed {hv_dups} | exact-train matches removed {n_before - len(hv)}")

        # -- 2. budget + redistribute -------------------------------------------------------------
        available = {n: int(cache[n]["lengths"].sum()) for n in SOURCES}
        budget = resolve_budgets(mix, args.train_tokens, available)
        _log("\n  budgets (tokens):")
        for n in SOURCES:
            _log(f"    {n:12s} requested {int(mix[n] * args.train_tokens):>12,} | "
                 f"available {available[n]:>12,} | granted {budget[n]:>12,}")

        # -- 3. select + shuffle documents --------------------------------------------------------
        selected: List[Tuple[str, int]] = []
        realised: Dict[str, int] = {}
        for name in SOURCES:
            lengths = cache[name]["lengths"]
            order = nprng.permutation(len(lengths))
            idx, got = take_indices_by_tokens(lengths, order, budget[name])
            realised[name] = got
            cache[name]["offsets"] = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
            cache[name]["selected"] = idx
            selected.extend((name, i) for i in idx)
        rng.shuffle(selected)  # interleave domains before concatenation
        _log(f"  selected {len(selected):,} documents, {sum(realised.values()) / 1e6:.1f}M tokens")

        # -- 4. write the train stream (uint16 cache -> int32 tensor) ----------------------------
        train_bin = os.path.join(args.cache_dir, "train_stream.u16.bin")
        mms = {n: np.memmap(cache[n]["bin"], dtype=np.uint16, mode="r") for n in SOURCES}
        buf: List[np.ndarray] = []
        buf_n = 0
        written = 0
        with open(train_bin, "wb") as f:
            for name, i in selected:
                off = cache[name]["offsets"]
                buf.append(np.asarray(mms[name][off[i]:off[i + 1]]))
                buf_n += int(off[i + 1] - off[i])
                if buf_n >= 4_000_000:
                    np.concatenate(buf).tofile(f)
                    written += buf_n
                    buf, buf_n = [], 0
            if buf:
                np.concatenate(buf).tofile(f)
                written += buf_n
        del mms
        train_stream = np.memmap(train_bin, dtype=np.uint16, mode="r")
        _log(f"  train stream: {len(train_stream):,} tokens -> {train_bin}")

        # -- 5. 16-gram leak filter ---------------------------------------------------------------
        _log("\n  hashing train 16-grams into buckets...")
        bucket_dir = os.path.join(args.cache_dir, "grams")
        n_grams = build_gram_buckets(train_stream, bucket_dir, args.gram_buckets, args.gram_chunk_tokens)
        vg_parts, vg_slices = [], {}
        cursor = 0
        for name in SOURCES:
            spans = []
            for a in val_tok[name]:
                g = ngram_hashes(a, NGRAM)
                vg_parts.append(g)
                spans.append((cursor, cursor + len(g)))
                cursor += len(g)
            vg_slices[name] = spans
        vg = np.concatenate(vg_parts) if vg_parts else np.zeros(0, dtype=np.int64)
        del vg_parts
        _log(f"  testing {len(vg):,} held-out 16-grams against {n_grams:,} train 16-grams")
        hit, unique_grams = bucket_membership(bucket_dir, args.gram_buckets, vg)

        val_bpt: Dict[str, float] = {}
        gpt2_bpt: Dict[str, float] = {}
        from transformers import AutoTokenizer
        gpt2 = AutoTokenizer.from_pretrained("gpt2")
        for name in SOURCES:
            kept_tok, kept_text, per_doc = [], [], []
            dropped = 0
            for (s, e), a, t in zip(vg_slices[name], val_tok[name], val_text[name]):
                if e > s:
                    h = hit[s:e]
                    if h.mean() >= args.leak_drop_threshold:
                        dropped += 1
                        continue
                    per_doc.append((int(h.sum()), int(e - s)))
                else:
                    per_doc.append((0, 0))
                kept_tok.append(a)
                kept_text.append(t)
            final_tok = take_tokens(kept_tok, args.val_tokens_per_source)
            n_final = len(final_tok)
            final_text = kept_text[:n_final]
            leak_hits = sum(h for h, _ in per_doc[:n_final])
            leak_total = sum(t for _, t in per_doc[:n_final])
            stream = np.concatenate(final_tok) if final_tok else np.zeros(0, dtype=np.int32)
            nbytes = sum(len(t.encode("utf-8")) for t in final_text)
            torch.save(torch.from_numpy(stream.astype(np.int32)),
                       os.path.join(args.out_dir, f"val_{name}_tokens.pt"))
            np.save(os.path.join(args.meta_dir, f"val_{name}_doc_lengths.npy"),
                    np.asarray([len(a) for a in final_tok], dtype=np.int64))
            with open(os.path.join(args.meta_dir, f"val_{name}_docs.json"), "w", encoding="utf-8") as f:
                json.dump(final_text[:args.meta_docs_kept], f)
            g2 = sum(len(ids) + 1 for ids in gpt2(final_text, add_special_tokens=False)["input_ids"]) \
                if final_text else 1
            val_bpt[name] = round(nbytes / max(1, len(stream)), 4)
            gpt2_bpt[name] = round(nbytes / max(1, g2), 4)
            manifest["sources"][name] = {
                "train_docs_available": int(len(cache[name]["lengths"])),
                "train_docs_used": len(cache[name]["selected"]),
                "train_source_exhausted": bool(cache[name]["exhausted"]),
                "train_exact_duplicates_removed": int(cache[name]["dups"]),
                "train_tokens": int(realised[name]),
                "train_tokens_available": int(available[name]),
                "val_docs_available": int(cache[name]["val_docs_available"]),
                "val_docs_encoded": len(val_tok[name]),
                "val_docs_kept": n_final,
                "val_docs_dropped_for_leak": dropped,
                "val_exact_duplicates_removed": int(cache[name]["val_dups"]),
                "val_exact_train_matches_removed": int(cache[name]["val_exact_train_matches"]),
                "val_tokens": int(len(stream)),
                "val_bytes": int(nbytes),
                "val_residual_16gram_leak": round(leak_hits / max(1, leak_total), 5),
            }
            _log(f"  {name:12s} val {len(stream):>8,} tok | dropped {dropped:>4d} docs | "
                 f"residual leak {leak_hits / max(1, leak_total) * 100:6.2f}% | "
                 f"{val_bpt[name]:.2f} B/tok (gpt2 {gpt2_bpt[name]:.2f})")

        # -- 6. save train tensor + manifest ------------------------------------------------------
        _log("\n  saving train_tokens.pt ...")
        train_i32 = np.asarray(train_stream, dtype=np.int32)
        torch.save(torch.from_numpy(train_i32), os.path.join(args.out_dir, "train_tokens.pt"))
        n_train = int(len(train_i32))
        del train_i32

        probe = load_gsm8k(args.gsm8k_train, args.gsm8k_test)[2]
        with open(os.path.join(args.out_dir, "gsm8k_test_probe.json"), "w") as f:
            json.dump(probe[:300], f, indent=1)

        realised_frac = {n: round(realised[n] / max(1, sum(realised.values())), 4) for n in SOURCES}
        manifest.update({
            "train_tokens_total": n_train,
            "train_documents": len(selected),
            "train_unique_16gram_ratio": round(unique_grams / max(1, n_train), 4),
            "train_16grams_total": int(n_grams),
            "train_16grams_unique": int(unique_grams),
            "mix": {"requested": mix, "realised": realised_frac,
                    "requested_tokens": {n: int(mix[n] * args.train_tokens) for n in SOURCES},
                    "realised_tokens": {n: int(realised[n]) for n in SOURCES}},
            "val_bytes_per_token": val_bpt,
            "gpt2_bytes_per_token": gpt2_bpt,
            "outputs": {
                "train": "data/general/train_tokens.pt",
                **{f"val_{n}": f"data/general/val_{n}_tokens.pt" for n in SOURCES},
                "probe": "data/general/gsm8k_test_probe.json",
            },
        })
        tok_report = os.path.join(args.out_dir, "tokenizer_report.json")
        if os.path.exists(tok_report):
            manifest["tokenizer_bytes_per_token"] = json.load(open(tok_report))["bytes_per_token"]
        _TIMINGS["build"] = round(time.time() - stage.t0, 1)
        _persist_timings()
        manifest["timings_sec"] = dict(_TIMINGS)
        with open(args.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest


# ------------------------------------------------------------------ stage: verify ---------------
def stage_verify(args, bucket_dir: Optional[str] = None) -> Dict:
    from tokenizers import Tokenizer

    with _Stage("verify") as stage:
        manifest = json.load(open(args.manifest_path))
        result: Dict = {"checks": {}}

        # (a) every .pt loads and matches the manifest
        train = torch.load(os.path.join(args.out_dir, "train_tokens.pt"), weights_only=True)
        ok = (train.dtype == torch.int32 and train.dim() == 1
              and len(train) == manifest["train_tokens_total"])
        result["checks"]["train_tokens_pt"] = {
            "ok": bool(ok), "len": int(len(train)), "dtype": str(train.dtype),
            "manifest_len": manifest["train_tokens_total"],
            "max_id": int(train.max()), "vocab_size": manifest["vocab_size"]}
        _log(f"  (a) train_tokens.pt len {len(train):,} dtype {train.dtype} "
             f"max_id {int(train.max())} -> {'OK' if ok else 'MISMATCH'}")
        vals = {}
        for name in SOURCES:
            v = torch.load(os.path.join(args.out_dir, f"val_{name}_tokens.pt"), weights_only=True)
            vok = (v.dtype == torch.int32 and len(v) == manifest["sources"][name]["val_tokens"])
            vals[name] = v
            result["checks"][f"val_{name}_pt"] = {"ok": bool(vok), "len": int(len(v)),
                                                  "manifest_len": manifest["sources"][name]["val_tokens"]}
            _log(f"      val_{name}_tokens.pt len {len(v):,} -> {'OK' if vok else 'MISMATCH'}")

        # (b) residual leak per set, straight from the manifest
        result["checks"]["residual_leak"] = {n: manifest["sources"][n]["val_residual_16gram_leak"]
                                             for n in SOURCES}
        _log("  (b) residual 16-gram leak: " + ", ".join(
            f"{n} {manifest['sources'][n]['val_residual_16gram_leak'] * 100:.2f}%" for n in SOURCES))

        # (c) spot-check 20 random held-out documents against the train 16-gram index
        own_buckets = bucket_dir is None or not glob.glob(os.path.join(bucket_dir, "b*.bin"))
        if own_buckets:
            bucket_dir = os.path.join(args.cache_dir, "verify_grams")
            _log("  (c) rebuilding train 16-gram index for the spot check...")
            build_gram_buckets(train.numpy(), bucket_dir, args.gram_buckets, args.gram_chunk_tokens)
        rng = random.Random(args.seed + 7)
        spot: List[Dict] = []
        vg_parts, meta = [], []
        cursor = 0
        for name in SOURCES:
            lens = np.load(os.path.join(args.meta_dir, f"val_{name}_doc_lengths.npy"))
            if len(lens) == 0:
                continue
            offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
            picks = rng.sample(range(len(lens)), min(5, len(lens)))
            v = vals[name].numpy().astype(np.int64)
            for p in picks:
                doc = v[offs[p]:offs[p + 1]]
                g = ngram_hashes(doc, NGRAM)
                vg_parts.append(g)
                meta.append((name, int(p), cursor, cursor + len(g)))
                cursor += len(g)
        vg = np.concatenate(vg_parts) if vg_parts else np.zeros(0, dtype=np.int64)
        hit, _ = bucket_membership(bucket_dir, args.gram_buckets, vg)
        worst = 0.0
        for name, p, s, e in meta:
            frac = float(hit[s:e].mean()) if e > s else 0.0
            worst = max(worst, frac)
            spot.append({"set": name, "doc": p, "leaked_16gram_fraction": round(frac, 5)})
        result["checks"]["spot_check_20_docs"] = {
            "n": len(spot), "max_leaked_fraction": round(worst, 5),
            "threshold": args.leak_drop_threshold,
            "ok": bool(worst < args.leak_drop_threshold), "docs": spot}
        _log(f"  (c) spot check {len(spot)} held-out docs: max leaked 16-gram fraction "
             f"{worst * 100:.2f}% (< {args.leak_drop_threshold * 100:.0f}% required) -> "
             f"{'OK' if worst < args.leak_drop_threshold else 'FAIL'}")
        if own_buckets:
            shutil.rmtree(bucket_dir, ignore_errors=True)

        # (d) tokenizer round-trips a code and a story sample exactly
        tok = Tokenizer.from_file(args.tokenizer_path)
        rt = {}
        for name, path in (("code", os.path.join(args.meta_dir, "val_code_docs.json")),
                           ("tinystories", os.path.join(args.meta_dir, "val_tinystories_docs.json"))):
            sample = json.load(open(path, encoding="utf-8"))[0]
            dec = tok.decode(tok.encode(sample).ids)
            rt[name] = {"ok": dec == sample, "chars": len(sample)}
            _log(f"  (d) round-trip {name}: {'exact' if dec == sample else 'MISMATCH'} "
                 f"({len(sample)} chars)")
        result["checks"]["tokenizer_round_trip"] = rt

        result["ok"] = bool(
            result["checks"]["train_tokens_pt"]["ok"]
            and all(result["checks"][f"val_{n}_pt"]["ok"] for n in SOURCES)
            and result["checks"]["spot_check_20_docs"]["ok"]
            and all(v["ok"] for v in rt.values()))
        manifest["verification"] = result
        _TIMINGS["verify"] = round(time.time() - stage.t0, 1)
        _persist_timings()
        manifest["timings_sec"] = dict(_TIMINGS)
        with open(args.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        _log(f"  verification: {'PASS' if result['ok'] else 'FAIL'}")
        return result


# ------------------------------------------------------------------ summary ---------------------
def print_summary(manifest: Dict) -> None:
    m = manifest
    _log("\n" + "=" * 104)
    _log(f"data/general  |  {m['train_tokens_total']:,} train tokens  |  vocab {m['vocab_size']}  "
         f"|  eos_id {m['eos_id']}  |  unique 16-gram ratio {m['train_unique_16gram_ratio']}")
    _log("=" * 104)
    head = (f"{'source':12s} {'train tok':>13s} {'mix req':>8s} {'mix real':>9s} {'val tok':>9s} "
            f"{'drop':>5s} {'leak%':>7s} {'B/tok':>7s} {'gpt2':>7s}")
    _log(head)
    _log("-" * 104)
    for n in SOURCES:
        s = m["sources"][n]
        _log(f"{n:12s} {s['train_tokens']:>13,} {m['mix']['requested'][n]:>8.3f} "
             f"{m['mix']['realised'][n]:>9.3f} {s['val_tokens']:>9,} "
             f"{s['val_docs_dropped_for_leak']:>5d} {s['val_residual_16gram_leak'] * 100:>7.2f} "
             f"{m['val_bytes_per_token'][n]:>7.2f} {m['gpt2_bytes_per_token'][n]:>7.2f}")
    _log("-" * 104)
    if "tokenizer_bytes_per_token" in m:
        _log(f"{'tokenizer':12s} bytes/token on 2 MB train samples (new vs gpt2):")
        for n, r in m["tokenizer_bytes_per_token"].items():
            _log(f"{'  ' + n:12s} new {r['new_bytes_per_token']:.3f}  gpt2 {r['gpt2_bytes_per_token']:.3f}  "
                 f"ratio {r['compression_vs_gpt2']:.3f}")
    _log(f"\ntimings (s): {m.get('timings_sec', {})}")
    _log("=" * 104)


# ------------------------------------------------------------------ cli -------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["fetch", "tokenizer", "build", "verify", "all"])
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "general"))
    ap.add_argument("--raw-dir", default=os.path.join(REPO_ROOT, "data", "raw"))
    ap.add_argument("--train-tokens", type=int, default=300_000_000)
    ap.add_argument("--val-tokens-per-source", type=int, default=400_000)
    ap.add_argument("--mix", default=DEFAULT_MIX)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--leak-drop-threshold", type=float, default=0.20)
    ap.add_argument("--fineweb-repo", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--fineweb-config", default="sample-10BT")
    ap.add_argument("--fineweb-train-mb", type=float, default=740.0)
    ap.add_argument("--fineweb-heldout-mb", type=float, default=8.0)
    ap.add_argument("--fineweb-shard-mb", type=float, default=64.0)
    ap.add_argument("--tokenizer-sample-mb", type=float, default=200.0)
    ap.add_argument("--bpt-sample-mb", type=float, default=2.0)
    ap.add_argument("--encode-headroom", type=float, default=1.30)
    ap.add_argument("--val-surplus", type=float, default=3.0)
    ap.add_argument("--gram-buckets", type=int, default=16)
    ap.add_argument("--gram-chunk-tokens", type=int, default=25_000_000)
    ap.add_argument("--meta-docs-kept", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json to build with (default: <out-dir>/tokenizer.json)")
    ap.add_argument("--force", action="store_true", help="ignore fetch/encode caches")
    ap.add_argument("--keep-cache", action="store_true", help="keep data/general/_cache after build")
    return ap


def resolve_paths(args) -> None:
    global _TIMINGS_PATH
    args.out_dir = os.path.abspath(args.out_dir)
    args.raw_dir = os.path.abspath(args.raw_dir)
    args.cache_dir = os.path.join(args.out_dir, "_cache")
    args.meta_dir = os.path.join(args.out_dir, "_meta")
    args.tokenizer_path = os.path.abspath(args.tokenizer) if getattr(args, "tokenizer", None)         else os.path.join(args.out_dir, "tokenizer.json")
    args.manifest_path = os.path.join(args.out_dir, "manifest.json")
    args.fineweb_dir = os.path.join(args.raw_dir, "fineweb_edu")
    args.hf_dl_dir = os.path.join(args.raw_dir, "_hf_download")
    args.tinystories_train = os.path.join(args.raw_dir, "tinystories_v2_train_full.txt")
    args.tinystories_valid = os.path.join(REPO_ROOT, "data", "tinystories_valid.txt")
    args.csn_dir = os.path.join(args.raw_dir, "csn", "python")
    args.csn_test = os.path.join(args.csn_dir, "test-00000-of-00001.parquet")
    args.gsm8k_train = os.path.join(args.raw_dir, "gsm8k", "main", "train-00000-of-00001.parquet")
    args.gsm8k_test = os.path.join(args.raw_dir, "gsm8k", "main", "test-00000-of-00001.parquet")
    _TIMINGS_PATH = os.path.join(args.meta_dir, "timings.json")


def main() -> None:
    args = build_argparser().parse_args()
    resolve_paths(args)
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.threads))  # leave cores for the GPU job
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    t0 = time.time()
    if args.stage in ("fetch", "all"):
        stage_fetch(args)
    if args.stage in ("tokenizer", "all"):
        stage_tokenizer(args)
    if args.stage in ("build", "all"):
        stage_build(args)
    if args.stage in ("verify", "all"):
        stage_verify(args, bucket_dir=os.path.join(args.cache_dir, "grams"))
    if args.stage in ("build", "all") and not args.keep_cache:
        shutil.rmtree(args.cache_dir, ignore_errors=True)
        _log(f"removed cache {args.cache_dir}")
    if os.path.exists(args.manifest_path):
        print_summary(json.load(open(args.manifest_path)))
    _log(f"total wall time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
