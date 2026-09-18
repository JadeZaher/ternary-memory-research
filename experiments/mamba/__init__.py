"""
experiments/mamba: Dedicated package for Ternary Selective State Space Models (SSMs).

Gate 20 (Phase II Track E: Hybrid Mamba-Samba-Jamba SSM):
- Ternary Selective State Space Layer (TernaryMambaBlock)
- Contractive Zero-Order Hold (ZOH) Discretization (|A_bar| < 1.0)
- Hybrid Mamba-Attention Language Models (HybridMambaForCausalLM)
- Compact Hybrid KV-Cache with 4x-8x Memory Compression
"""

from experiments.mamba.ternary_mamba_block import MambaConfig, TernaryMambaBlock
from experiments.mamba.hybrid_mamba_model import (
    HybridMambaConfig,
    HybridMambaKVCache,
    HybridMambaForCausalLM,
)

__all__ = [
    "MambaConfig",
    "TernaryMambaBlock",
    "HybridMambaConfig",
    "HybridMambaKVCache",
    "HybridMambaForCausalLM",
]
