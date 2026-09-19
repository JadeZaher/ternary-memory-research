# `experiments/general_model/data/` — general-model pretraining corpus

## What this directory produces

`prepare_general_corpus.py` builds `data/general/`: a ~300M-token, four-domain, decontaminated
pretraining corpus under a purpose-trained 16,384-entry byte-level BPE, plus four independent
held-out sets and a manifest that records every number a trainer or reviewer needs.

| artefact | contents |
| --- | --- |
| `data/general/train_tokens.pt` | 1-D `torch.int32`, the shuffled train stream |
| `data/general/val_{fineweb,tinystories,code,math}_tokens.pt` | 1-D `torch.int32`, ~400k tokens each |
| `data/general/tokenizer.json` | HF `tokenizers` byte-level BPE, vocab 16384, `<\|endoftext\|>` = `eos_id` |
| `data/general/gsm8k_test_probe.json` | 300 `{question, answer}` pairs for generative math probing |
| `data/general/manifest.json` | mix, per-source stats, leak numbers, bytes/token, verification record |
| `data/general/tokenizer_report.json` | bytes/token per source, new tokenizer vs GPT-2 |
| `data/general/_meta/` | per-set held-out document lengths + a few sample docs, for `--stage verify` |

`train_tokenizer.py` is a thin CLI over the tokenizer stage so the vocabulary can be rebuilt without
re-running the corpus build.

## Why these sources

Four domains, each with an **author-provided** held-out split, so the split is a property of the
dataset rather than a slice of our own stream:

| source | train | held-out | why it is here |
| --- | --- | --- | --- |
| `fineweb` | `HuggingFaceFW/fineweb-edu`, `sample-10BT`, streamed | a strictly later, document-disjoint slice of the same stream | general English; the only source with enough real text to carry half of a 300M-token budget |
| `tinystories` | full `TinyStoriesV2-GPT4-train.txt` (2.2 GB) | `data/tinystories_valid.txt` | simple narrative with a clean author split; cheap signal for small models |
| `code` | CodeSearchNet python `train` parquet | python `test` parquet | structured, long-range-dependency text; a different tokenizer regime from prose |
| `math` | GSM8K `main/train` | `main/test` (also the probe) | the only source with a checkable answer, so a probe can measure reasoning rather than perplexity |

FineWeb is taken **by byte count** (~740 MB ≈ 170M GPT-2-equivalent tokens) and the exact token
count is whatever the new tokenizer produces. The raw text pulled from the stream is written to
`data/raw/fineweb_edu/*.jsonl` with each document's `id` and `url`, so the build is reproducible
offline and the provenance of every train document is recoverable.

## Why document-level splits and a 16-gram filter

The corpora this replaces (`prepare_multicorpus.py` and friends) split a single generated stream by
position: 96-99% of "validation" 16-grams occurred verbatim in training, so every perplexity
measured on them was a memorisation score (`outputs/heldout-clean-eval.json`). Two guarantees stop
that from recurring, both inherited from `experiments/data/prepare_clean_corpus.py`:

1. **Document-level splits.** Held-out documents are whole documents from the dataset author's own
   split (or, for FineWeb, a later disjoint slice of the stream), never a tail of the train stream.
2. **A 16-gram decontamination pass.** Every 16-gram of the finished train stream is hashed; a
   held-out document sharing >= 20% of its 16-grams with train is dropped, and the *residual* leak
   of the surviving documents is recorded per set in the manifest. 16 tokens is long enough that a
   collision means copied text rather than a common phrase, and short enough to catch near-duplicates
   that exact-hash dedupe misses.

Exact-duplicate removal runs first, within each split, and any held-out document that is byte-identical
(after whitespace normalisation) to a train document is removed before the n-gram pass.

## Why the mix is 0.50 / 0.25 / 0.17 / 0.08

Web text dominates because it is the only domain that generalises; stories buy fluency cheaply at
small scale; code supplies long structured dependencies; math is capped by availability, not by
preference. GSM8K holds only ~1M tokens, far under its 8% share of 300M, so the pipeline
**redistributes the shortfall proportionally** to the remaining sources and records both the
requested and the realised mix. Any source may do this; math simply always does. Documents are
shuffled with seed 1234 before concatenation so domains interleave rather than arriving in blocks.

## Why a new 16k byte-level BPE

GPT-2's 50,257-entry vocabulary spends ~19% of a small model's parameters on embeddings and was
fitted to 2019 WebText, not to this mix. A 16,384-entry byte-level BPE trained on a stratified
200 MB sample of **train text only** (never held-out text — that would leak) is matched to the four
domains and shrinks the embedding table ~3x. Byte-level with the full 256-byte initial alphabet
means no token is ever out-of-vocabulary and decoding round-trips exactly, which the verify stage
checks on a code and a story sample. The cost is compression: a smaller vocabulary emits more tokens
per byte, so `tokenizer_report.json` records bytes/token for both tokenizers on the same 2 MB sample
per source, and the manifest records `val_bytes_per_token` per held-out set so the trainer can convert
nats/token into bits/byte — the only cross-tokenizer-comparable loss unit.

## What the first build measured, and what to watch for

300,001,859 train tokens over 733,745 documents; realised mix 0.542 / 0.271 / 0.184 / 0.0035.
GSM8K holds only 1.05M tokens, so math lands at 0.35% instead of 8% and the other three absorb the
shortfall — **the math held-out set is a probe, not a domain the model is trained on at scale**.

Per-document 16-gram overlap with train, measured over every document in the final held-out sets:

| set | docs | mean | median | p99 | max | docs >= 10% |
| --- | --- | --- | --- | --- | --- | --- |
| fineweb | 332 | 0.33% | 0.00% | 7.7% | 19.1% | 2 |
| tinystories | 2038 | 2.95% | 1.57% | 15.8% | 19.6% | 138 |
| code | 1287 | 0.38% | 0.00% | 10.4% | 19.9% | 15 |
| math | 1319 | 0.03% | 0.00% | 0.0% | 15.0% | 2 |

No held-out document anywhere exceeds 19.9%, so none is a copy of a train document — the tail is
shared *phrases*, not shared documents. Two caveats follow from the shape of that tail:

- **TinyStories is intrinsically repetitive.** It is generated from a small vocabulary with formulaic
  sentences, so 16-grams recur across unrelated stories. Its 2.84% residual is a property of the
  corpus, not a decontamination failure, but it does mean `val_tinystories` loss is slightly
  optimistic; prefer `val_fineweb` (median overlap 0.00%) as the headline held-out number.
- **The 20% threshold leaves a thin band just under it** (28 TinyStories and 7 code documents between
  15% and 20%). Lowering the threshold would discard a large slice of TinyStories for little gain;
  the distribution is published here instead so the number can be interpreted rather than trusted.

## Memory and streaming notes

The build is written to run beside a GPU job on a machine with only a few GB of free RAM:

- Documents are **streamed** from disk (JSONL shards, a chunked `<|endoftext|>` splitter for the
  2.2 GB TinyStories file, `iter_batches` for parquet); no source is ever a multi-GB list of strings.
- Encoded tokens land in a `uint16` cache (`data/general/_cache/`) with a per-document length array.
  Shuffling and budgeting then operate on **document indices**, not token arrays.
- 300M 16-gram hashes are 2.4 GB of `int64`, too much to sort in memory here, so the hash pass is
  chunked (25M tokens at a time) and partitioned into 16 on-disk buckets by low hash bits. Because
  equal hashes always land in the same bucket, per-bucket `np.unique` gives the **exact** global
  unique-16-gram count and an exact membership test, at a ~400 MB peak.
- `_cache/` is deleted after a successful build (`--keep-cache` to retain it for a re-run).

## Running it

```bash
python experiments/general_model/data/prepare_general_corpus.py --stage all      # fetch -> tokenizer -> build -> verify
python experiments/general_model/data/prepare_general_corpus.py --stage build    # re-mix from cached encodes
python experiments/general_model/data/train_tokenizer.py                         # vocabulary only
```

`--stage verify` reloads the artefacts and re-proves the quality bar: every `.pt` loads with the
length the manifest claims, residual leak is reported per set, 20 random held-out documents are
re-searched against a freshly rebuilt train 16-gram index, and the tokenizer round-trips exactly.
The result is written back into `manifest.json` under `verification`.

Helpers are **imported** from `experiments/data/prepare_clean_corpus.py` (`ngram_hashes`,
`dedupe_documents`, `take_tokens`, `load_gsm8k`, `load_csn_python`, `load_tinystories`, `EOT`) and
that file is never edited; this pipeline is its scaled-up sibling, not a fork.
