# Repository status (corrected 2026-09-16)

The note that previously lived here said this folder was a stale copy of
`C:\Users\atooz\Programming\trits-memory-inference` and was safe to delete. That is no
longer true and must not be acted on.

- `trits-memory-inference` (git-versioned) carries the operator-mathematics line of work:
  per-block dispatch theorem, table space, expressiveness proof, task demo.
- **This folder** carries the entire model line of work and nothing else does: `bitlinear.py`,
  `tristate_router.py`, `bitroute_model.py`, `flowtrit_model.py`, `flowroute_model.py`, the
  trainers, benchmarks, ledgers under `outputs/`, and the conductor registry. None of these
  files exist in the other repository (checked with `diff -rq` on 2026-09-16).
- This folder is **not under version control**. Every result here lives only on this disk.

Recommended next step for the owner: `git init` here (or move `experiments/`, `outputs/`,
`research/` into the versioned repo) before further work. Nothing has been deleted.
