"""
experiments/frontier_scaling/eval_heldout_clean.py: Decontaminated held-out evaluation of the headline checkpoints.

Why this exists (see research/hardening-2026-09-18-heldout.md):
  Every reported "val PPL" for Gates 15-19 is measured on the tail slice of an interleaved corpus whose
  math / code / agro segments are generated from a handful of templates. The Gemini benchmark prompts are
  drawn from those same templates. This script measures the same checkpoints on data that cannot have
  been memorised:
    A. the corpus-tail split the deep dives used            (reproduces the claimed number)
    B. real TinyStories validation text (data/tinystories_valid.txt)
    C. hand-written Python functions not present in any training template
    D. hand-written arithmetic word problems with names / items / structure outside the generator lists
Outputs: outputs/heldout-clean-eval.json
"""

import os
import sys
import math
import json
import argparse
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.frontier_scaling.loopformer_model import LoopFormerForCausalLM
from experiments.frontier_scaling.looped_dwp_model import LoopedDWPForCausalLM
from experiments.frontier_scaling.navitrit_scale_model import NaviTritScaleForCausalLM, get_scale_config


# ---------------------------------------------------------------- probes (decontaminated) ----
CLEAN_PYTHON = [
    "def running_median(values):\n    import heapq\n    low, high, out = [], [], []\n    for v in values:\n        heapq.heappush(low, -v)\n        heapq.heappush(high, -heapq.heappop(low))\n        if len(high) > len(low):\n            heapq.heappush(low, -heapq.heappop(high))\n        out.append(-low[0] if len(low) > len(high) else (-low[0] + high[0]) / 2)\n    return out\n",
    "def rle_encode(s):\n    if not s:\n        return ''\n    parts = []\n    prev, count = s[0], 1\n    for ch in s[1:]:\n        if ch == prev:\n            count += 1\n        else:\n            parts.append(f'{prev}{count}')\n            prev, count = ch, 1\n    parts.append(f'{prev}{count}')\n    return ''.join(parts)\n",
    "def topological_order(graph):\n    indeg = {u: 0 for u in graph}\n    for u in graph:\n        for v in graph[u]:\n            indeg[v] = indeg.get(v, 0) + 1\n    queue = [u for u in indeg if indeg[u] == 0]\n    order = []\n    while queue:\n        u = queue.pop()\n        order.append(u)\n        for v in graph.get(u, []):\n            indeg[v] -= 1\n            if indeg[v] == 0:\n                queue.append(v)\n    return order\n",
    "class LRUCache:\n    def __init__(self, capacity):\n        from collections import OrderedDict\n        self.capacity = capacity\n        self.store = OrderedDict()\n\n    def get(self, key):\n        if key not in self.store:\n            return None\n        self.store.move_to_end(key)\n        return self.store[key]\n\n    def put(self, key, value):\n        self.store[key] = value\n        self.store.move_to_end(key)\n        if len(self.store) > self.capacity:\n            self.store.popitem(last=False)\n",
    "def matrix_transpose(m):\n    rows = len(m)\n    cols = len(m[0]) if rows else 0\n    return [[m[r][c] for r in range(rows)] for c in range(cols)]\n",
    "def is_balanced(expr):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for ch in expr:\n        if ch in '([{':\n            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack.pop() != pairs[ch]:\n                return False\n    return not stack\n",
    "def moving_average(xs, window):\n    total = 0.0\n    result = []\n    for i, x in enumerate(xs):\n        total += x\n        if i >= window:\n            total -= xs[i - window]\n        if i >= window - 1:\n            result.append(total / window)\n    return result\n",
    "def dedupe_preserve_order(items):\n    seen = set()\n    out = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            out.append(item)\n    return out\n",
]

CLEAN_MATH = [
    "Problem: Priya baked 42 muffins. She packed 15 into boxes and her neighbour took 9 more. How many muffins does Priya still have?\nReasoning: Start with 42. After packing 15, 42 - 15 = 27 remain. After the neighbour takes 9, 27 - 9 = 18 remain.\nAnswer: 18\n",
    "Problem: A train has 8 carriages with 36 seats each. 197 seats are taken. How many seats are empty?\nReasoning: Total seats are 8 * 36 = 288. Empty seats are 288 - 197 = 91.\nAnswer: 91\n",
    "Problem: Kenji saves 12 dollars every week. After 7 weeks he spends 30 dollars on a game. How much money does he have left?\nReasoning: He saves 12 * 7 = 84 dollars. After spending 30, 84 - 30 = 54 dollars remain.\nAnswer: 54\n",
    "Problem: A farmer plants 5 rows of 14 seedlings. 11 seedlings wilt. How many seedlings survive?\nReasoning: Planted 5 * 14 = 70 seedlings. Survivors are 70 - 11 = 59.\nAnswer: 59\n",
    "Problem: Marta reads 23 pages on Monday and twice as many on Tuesday. How many pages did she read in total?\nReasoning: Tuesday is 2 * 23 = 46 pages. Total is 23 + 46 = 69 pages.\nAnswer: 69\n",
    "Problem: A shop sells pencils in packs of 6. Dev buys 4 packs and gives away 7 pencils. How many pencils does he keep?\nReasoning: He buys 4 * 6 = 24 pencils. After giving away 7, 24 - 7 = 17 remain.\nAnswer: 17\n",
    "Problem: There are 63 chairs in a hall. 27 are stacked away and 18 more are moved outside. How many chairs stay in the hall?\nReasoning: 63 - 27 = 36 remain after stacking. 36 - 18 = 18 remain after moving.\nAnswer: 18\n",
    "Problem: Sofia has 3 jars with 25 buttons each. She uses 41 buttons for a craft. How many buttons are left?\nReasoning: She has 3 * 25 = 75 buttons. 75 - 41 = 34 remain.\nAnswer: 34\n",
]


# ---------------------------------------------------------------- helpers -----------------
def ngram_hashes(t: np.ndarray, n: int) -> np.ndarray:
    t = t.astype(np.int64)
    h = np.zeros(len(t) - n + 1, dtype=np.int64)
    for i in range(n):
        h = h * 1000003 + t[i: len(t) - n + 1 + i]
    return h


def leak_fraction(probe: torch.Tensor, train: torch.Tensor, n: int = 16) -> float:
    """Fraction of probe n-grams that occur verbatim anywhere in train."""
    if len(probe) < n:
        return 0.0
    train_set = np.unique(ngram_hashes(train.numpy(), n))
    return float(np.isin(ngram_hashes(probe.numpy(), n), train_set).mean())


def chunk_batches(tokens: torch.Tensor, seq_len: int, batch_size: int, max_batches: int):
    n = (len(tokens) - 1) // seq_len
    n = min(n, max_batches * batch_size)
    xs = torch.stack([tokens[i * seq_len: i * seq_len + seq_len] for i in range(n)]).long()
    ys = torch.stack([tokens[i * seq_len + 1: i * seq_len + seq_len + 1] for i in range(n)]).long()
    for b in range(0, n, batch_size):
        yield xs[b: b + batch_size], ys[b: b + batch_size]


@torch.no_grad()
def ppl_on(forward_fn, tokens: torch.Tensor, seq_len: int, batch_size: int, max_batches: int, device) -> Dict[str, float]:
    total, count = 0.0, 0
    for x, y in chunk_batches(tokens, seq_len, batch_size, max_batches):
        x, y = x.to(device), y.to(device)
        logits = forward_fn(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1, V).float(), y.reshape(-1), reduction="sum")
        total += loss.item()
        count += y.numel()
    mean = total / max(1, count)
    return {"loss": round(mean, 4), "ppl": round(math.exp(min(mean, 20.0)), 3), "tokens": count}


def pad_probe(tok, texts: List[str], seq_len: int) -> torch.Tensor:
    """Concatenate probe texts with EOT separators so they can be scored as one stream."""
    ids: List[int] = []
    for t in texts:
        ids.extend(tok.encode(t))
        ids.append(tok.eos_token_id)
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------- model loaders ------------
def load_loopformer(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = LoopFormerForCausalLM(ck["config"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    budget = ck["config"].default_recursions
    return lambda x: model(x, recursion_budget=budget)["logits"], ck["config"].max_position_embeddings


def load_looped_dwp(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = LoopedDWPForCausalLM(ck["config"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    budget = ck["config"].default_recursions
    return lambda x: model(x, recursion_budget=budget)["logits"], ck["config"].max_position_embeddings


def load_navitrit_100m(path, device):
    cfg = get_scale_config("100m", max_hops=6, use_scale_adaptive=True)
    model = NaviTritScaleForCausalLM(cfg).to(device)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [navitrit-100m] missing={len(missing)} unexpected={len(unexpected)} keys: {missing[:4]} {unexpected[:4]}")
    model.eval()
    # evaluate_arm in train_scale.py scores with temperature=0.7 and drops the last position
    return lambda x: model(x, temperature=0.7)["logits"], cfg.max_position_embeddings


# ---------------------------------------------------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--valid-tokens", type=int, default=120_000)
    ap.add_argument("--out", type=str, default="outputs/heldout-clean-eval.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("gpt2")

    # --- data
    corpus = torch.load("data/multicorpus_10m.pt", weights_only=True)
    corpus_train, corpus_tail = corpus[: int(0.9 * len(corpus))], corpus[int(0.9 * len(corpus)):]
    with open("data/tinystories_valid.txt", "r", encoding="utf-8") as f:
        valid_text = f.read()
    ts_valid = torch.tensor(tok.encode(valid_text[: args.valid_tokens * 5])[: args.valid_tokens], dtype=torch.long)
    py_probe = pad_probe(tok, CLEAN_PYTHON, args.seq_len)
    math_probe = pad_probe(tok, CLEAN_MATH, args.seq_len)

    clean_sets = {}
    for name in ("tinystories", "code", "math"):
        cp = os.path.join("data/clean", f"val_{name}_tokens.pt")
        if os.path.exists(cp):
            clean_sets[name] = torch.load(cp, weights_only=True).long()

    leaks = {
        "corpus_tail_16gram_in_train": round(leak_fraction(corpus_tail[:200_000], corpus_train), 4),
        "tinystories_valid_16gram_in_train": round(leak_fraction(ts_valid, corpus_train), 4),
        "clean_python_16gram_in_train": round(leak_fraction(py_probe, corpus_train), 4),
        "clean_math_16gram_in_train": round(leak_fraction(math_probe, corpus_train), 4),
    }
    print("Leakage (fraction of 16-grams found verbatim in the 90% train slice):")
    for k, v in leaks.items():
        print(f"  {k:40s} {v*100:6.1f}%")

    checkpoints = {
        "navitrit-100m-step10000": ("outputs/checkpoints/navitrit-100m-step10000.pt", load_navitrit_100m, 1.15),
        "navitrit-100m-loopformer": ("outputs/checkpoints/navitrit-100m-loopformer.pt", load_loopformer, 1.12),
        "navitrit-100m-looped-dwp": ("outputs/checkpoints/navitrit-100m-looped-dwp.pt", load_looped_dwp, 1.12),
    }
    results = {"leakage": leaks, "checkpoints": {}}
    for name, (path, loader, claimed) in checkpoints.items():
        if not os.path.exists(path):
            print(f"[skip] {path} missing")
            continue
        print(f"\n=== {name} (claimed PPL {claimed}) ===")
        fwd, max_pos = loader(path, device)
        seq_len = min(args.seq_len, max_pos)
        r = {
            "claimed_ppl": claimed,
            "A_corpus_tail": ppl_on(fwd, corpus_tail, seq_len, args.batch_size, args.max_batches, device),
            "B_tinystories_valid": ppl_on(fwd, ts_valid, seq_len, args.batch_size, args.max_batches, device),
            "C_clean_python": ppl_on(fwd, py_probe, seq_len, args.batch_size, args.max_batches, device),
            "D_clean_math": ppl_on(fwd, math_probe, seq_len, args.batch_size, args.max_batches, device),
        }
        for set_name, toks in clean_sets.items():  # same held-out sets the unified pilot arms are scored on
            r[f"E_dataclean_{set_name}"] = ppl_on(fwd, toks, seq_len, args.batch_size, args.max_batches, device)
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"  {k:22s} loss {v['loss']:.3f}  ppl {v['ppl']:8.2f}  ({v['tokens']} tokens)")
        results["checkpoints"][name] = r
        del fwd
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
