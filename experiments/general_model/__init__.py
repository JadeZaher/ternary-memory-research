"""
experiments/general_model: hybrid Mamba/attention ternary tiles with per-token routing and a value-based exit.

Track objective: the smallest stored-byte footprint that reaches a held-out loss / BPB target.
Module notes and the rationale behind every config toggle live in `experiments/general_model/AGENTS.md`.
"""

from experiments.general_model.general_model import (
    DENSE_BITS,
    TERNARY_BITS,
    GeneralConfig,
    GeneralRoutedLM,
    HybridTile,
)

__all__ = ["GeneralConfig", "GeneralRoutedLM", "HybridTile", "TERNARY_BITS", "DENSE_BITS"]
