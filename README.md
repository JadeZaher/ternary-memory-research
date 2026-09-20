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
| Night 1 of the general-model track (16k tokenizer, ~300M-token general corpus, 41.3M training tokens each): a 3-Mamba + 1-attention ternary hybrid with free per-token routing and a value exit reaches 2.007 mean core bits/byte at 2.9 tile applications per token, against 1.436 for arm C's fixed attention-only recipe at 16; the curves cross between 8 and 12 applications. The value exit beats matched random continuation by 0.15 nats on the hybrid. Under the pre-registered rule the routed hybrid does not qualify | `outputs/general-G1-full-log.json`, `outputs/navitrit-unified-G-anchor-log.json` | `research/general-model-track.md` §6 |
| A per-token value-based exit (a head predicting the next hop's gain in nats, trained on measured gain) is a working adaptive-depth mechanism: 3.50 held-out loss at 3.2 tile applications per token, beating matched random continuation by 0.03 nats and per-sequence capacity routing by 0.04. Path-history conditioning adds nothing measurable (rho 0.027 between path diversity and gain). The non-monotonic traversal thesis is not supported at this scale | `outputs/navitrit-unified-{S1-backbone,S2-traverse}-log.json` | same, §13.1 |

## The research journey

This repository is the working record of the project, not a curated paper. The deep dives in
`research/deep-dives/` are numbered in the order the work happened and none of them was published before
this release. Dives 01–09 (ternary arithmetic, bypass routing, recurrent solvers, hardware synthesis) hold
up. Dives 10–23 explored scaling, graph routing, looped blocks and dynamic weight parameterization on a
synthetic multi-domain corpus; a self-audit on 2026-09-18 found that the corpus was templated, its
validation split was 93–99% verbatim in training, and the judge prompts were training templates, so the
perplexities and scores in those documents measure memorisation rather than language ability
(`outputs/heldout-clean-eval.json`). Those dives are kept intact, each with a status banner, because the
architectural ideas in them are what the current experiment tests, and because the failure mode is itself
a finding worth recording. The audit, the decontaminated protocol, and the replacement experiments are in
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

Ablation arms A–Q and E1 are listed in the hardening report §5, §8, §10–12, and the condensed two-stage
experiment that closed the traversal question is §13; each writes
`outputs/navitrit-unified-<tag>-log.json`. Results are reported as deltas against the matched dense baseline
at equal measured tile evaluations per token, never as absolute records.

## The general-model track (2026-09-19 onward)

`experiments/general_model/` works backwards from the results above to the smallest stored model that reaches
arm C's held-out quality, under three fixed requirements: ternary weights, Mamba sequence mixing, per-token
non-monotonic routing. Decisions, assumptions and the pre-registered rule are in
[`research/general-model-track.md`](research/general-model-track.md); the data pipeline (16k tokenizer,
FineWeb-Edu + TinyStories + CodeSearchNet + GSM8K, ~300M decontaminated tokens) is `experiments/general_model/data/`.
Quality is compared in bits per byte, the only tokenizer-independent unit; ledgers are `outputs/general-<tag>-log.json`.

## Reproduce

```bash
pip install torch transformers pyarrow numpy huggingface_hub
python experiments/data/prepare_clean_corpus.py            # downloads to data/raw, builds data/clean (document-level split, leak-filtered)
python experiments/unified_scaling/test_navitrit_unified.py
python experiments/unified_scaling/test_navitrit_traverse.py
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --grad-checkpoint --routing-mode dense --no-dwp --max-loops 2 --tag E-dense2 --max-steps 3000
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --routing-mode traverse --max-hops 8 --router-cond path --adapter-cond path --tag H-traverse-path --max-steps 3000
# condensed two-stage experiment (report section 13):
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --max-steps 3000 --grad-checkpoint --routing-mode dense --no-dwp --loop-order-random --tile-drop 0.25 --deep-sup 0.1 --deep-sup-frac 0.125 --tag S1-backbone
python experiments/unified_scaling/train_navitrit_unified.py --preset pilot --max-steps 3000 --grad-checkpoint --routing-mode traverse --max-hops 8 --min-hops 2 --exit-warmup 500 --exit-mode value --exit-lambda 0.02 --explore 0.3 --deep-sup 0.1 --deep-sup-frac 0.125 --liveness --gain-proxy --coord --hop-dropout 0.3 --act-strategy --router-cond path --adapter-cond path --init-from outputs/checkpoints/navitrit-unified-S1-backbone-best.pt --tag S2-traverse
```
Any run can be continued from its `-latest.pt` with `--resume` (optimizer and sampler state are checkpointed).

Checkpoints and corpora are not versioned (`.gitignore`); the JSON ledgers in `outputs/` are the published
evidence and every number in the documents is traceable to one of them.

## Layout

```
experiments/            models, trainers, tests (bitlinear.py is the ternary STE primitive)
experiments/unified_scaling/   current experiment (see above)
experiments/data/       corpus builders; prepare_clean_corpus.py is the only one to use
research/               reports; deep-dives/ in chronological order (10-23 carry status banners)
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
repository and the hardening report; the earlier deep dives are working notes, not results.
