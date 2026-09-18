# Ternary Memory Research

Non-monotonic, history-conditioned traversal of a stationary ternary weight block: can a small set of
{−1, 0, +1} weights that never leave on-chip memory yield more function per stored bit, and less compute
per token, when each token chooses its own path through the block and reinterprets the weights along the way?

**Author:** Ahmed Zaher · **Status:** active research, single-GPU (RTX 4060, 8 GB) · **License:** MIT (code), CC BY 4.0 (research documents)

---

## What is established

| result | evidence | where |
|---|---|---|
| Multiplication-free ternary GEMM matches dense to 1.2e-6 | `outputs/arithmetic-upgrade-test.json` | `research/deep-dives/01-*` |
| Layer-bypass / early-exit routing gives 2.2x / 4.0x measured latency on a 135M ternary model | `outputs/bitroute-inference-benchmark.json` | `research/deep-dives/02-*` |
| Weight-tied recurrent ternary solver with fixed-point forcing recovers FP32 solve rate on Sudoku | `outputs/flowtrit-*.json` (leak-free protocol, 2026-09-16) | `research/deep-dives/03-*` |
| A parallel log-space selective scan reproduces the sequential Mamba recurrence to 1e-8, 17x faster | `experiments/mamba/test_ternary_mamba.py` | `research/hardening-2026-09-18-heldout.md` §7 |
| On a decontaminated 37.8M-token corpus, per-loop low-rank weight modulation (DWP) of a tied ternary block improves held-out loss by 0.11 nats over the tied baseline, and differentiable soft gating adds a further 0.07 | `outputs/navitrit-unified-{A,B,C}-*-log.json` | `research/hardening-2026-09-18-heldout.md` §9 |
| At 24.6M training tokens, 2 loops beat 4 loops at half the compute; Mixture-of-Depths token skipping at 50% capacity does not recover that gap | `outputs/navitrit-unified-{D,E}-*-log.json` | same, §9 |

## What was retracted (2026-09-18)

Every perplexity and LLM-judge score reported for Gates 15–23 (`research/deep-dives/10-*` through `23-*`)
was measured on a templated synthetic corpus whose "validation" split is 93–99% verbatim in training, and
the judge prompts were training templates. Re-scored on unseen text, those checkpoints reach perplexity
730–2156 on Python and 4000+ on arithmetic (`outputs/heldout-clean-eval.json`). The documents are kept,
each with a status banner, as a record of the work and of the failure mode. The full audit, the
decontaminated protocol, and the replacement experiments are in
**[`research/hardening-2026-09-18-heldout.md`](research/hardening-2026-09-18-heldout.md)**.

## The current experiment

`experiments/unified_scaling/` holds the models under test, one training script, and one evaluation protocol:

- `navitrit_unified_model.py`: a 4-macro-layer ternary block (attention + dual-expert SwiGLU, optional
  Mamba) applied for T loops, with per-loop FiLM/LoRA modulation and three routing modes (`dense`, `soft`,
  `mod`).
- `navitrit_traverse.py`: the thesis model. Each token carries a continuation-style path state; at every
  hop it picks one tile or EXIT (per-token, non-monotonic), the path state selects a mixture from a bank of
  adapters that reinterpret the tile's weights, an optional strain latent predicts how unresolved the token
  is and gates its exit, and an optional selector forwards a gate activation chosen from the edge taken.
  Every mechanism has an ablation switch.
- `train_navitrit_unified.py`: fixed seeded held-out batches per domain (TinyStoriesV2 valid, CodeSearchNet
  Python test, GSM8K test), a loop-budget sweep at the end, one flag per ablation.

Ablation arms A–Q and E1 are listed in the hardening report §5, §8, §10–12; each writes
`outputs/navitrit-unified-<tag>-log.json`. Results are reported as deltas against the matched dense baseline
at equal measured tile evaluations per token, never as absolute records.

## Reproduce

```bash
pip install torch transformers pyarrow numpy huggingface_hub
python experiments/data/prepare_clean_corpus.py            # downloads to data/raw, builds data/clean (document-level split, leak-filtered)
python experiments/unified_scaling/test_navitrit_unified.py
python experiments/unified_scaling/test_navitrit_traverse.py
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --grad-checkpoint --routing-mode dense --no-dwp --max-loops 2 --tag E-dense2 --max-steps 3000
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode traverse --max-hops 8 --router-cond path --adapter-cond path --tag H-traverse-path --max-steps 3000
```

Checkpoints and corpora are not versioned (`.gitignore`); the JSON ledgers in `outputs/` are the published
evidence and every number in the documents is traceable to one of them.

## Layout

```
experiments/            models, trainers, tests (bitlinear.py is the ternary STE primitive)
experiments/unified_scaling/   current experiment (see above)
experiments/data/       corpus builders; prepare_clean_corpus.py is the only one to use
research/               reports; deep-dives/01-09 established, 10-23 retracted with banners
research/hardening-2026-09-18-heldout.md   audit, protocol, live results
outputs/                JSON ledgers (evidence); checkpoints/ ignored
```

## Acknowledgements

This project's research discipline, in particular the separation of proposal, evidence and review that the
2026-09-18 audit applied to its own results, follows the orientation and claim/evidence protocol of
[agentprivacy.org](https://agentprivacy.org/). The initial ecosystem survey run under that protocol is
`research/agentprivacy-orientation.md`. Their Boolean/ring geometry is not used as evidence for any ternary
result here.

## Citation

See `CITATION.cff`. If you use the decontamination protocol or the traversal model, please cite the
repository and the hardening report rather than the retracted deep dives.
