"""
continuation_simulator.py: Simulates layer-by-layer autoregressive token generation,
continuation states, early exits, and memory traffic under different precision formats.
Writes results to outputs/continuation-simulation.json.
Uses standard library only.
"""

import json
import math
import random
from pathlib import Path

SEED = 20260915
random.seed(SEED)

class MemoryModelSimulator:
    def __init__(self, num_layers=16, weights_per_layer=8_000_000, hidden_dim=2048):
        """
        Simulate a 128M parameter transformer model (16 layers, ~8M weights per layer).
        hidden_dim = 2048.
        """
        self.num_layers = num_layers
        self.weights_per_layer = weights_per_layer
        self.hidden_dim = hidden_dim

        # Precision bits per weight
        self.bpw_fp16 = 16.0
        self.bpw_int4 = 4.0
        self.bpw_tq1_0 = 1.6875  # 5 trits per byte + FP16 scale per 256 elements
        self.bpw_mask_sign = 1.45  # Mask + sign at density ~0.40 + 0.063 rank directory

        # Descriptor region sizes (resident in fast cache)
        # 1 exit head per layer: linear probe hidden_dim -> 1 scalar confidence (FP16 = 2 bytes)
        self.exit_head_bytes_per_layer = hidden_dim * 2
        # Total resident descriptors across all layers:
        self.total_resident_D_bytes = (
            (self.weights_per_layer / 256 * 2) * num_layers +  # scales
            self.exit_head_bytes_per_layer * num_layers         # exit heads
        )

    def simulate_token_generation(self, num_tokens=200):
        """
        Simulate generating tokens across varying difficulty distributions:
        - 60% easy tokens: exit between layer 2 and 5
        - 25% medium tokens: exit between layer 6 and 11
        - 15% hard tokens: execute all 16 layers
        """
        records = []
        bytes_streamed = {
            "fp16_full": 0,
            "int4_full": 0,
            "tq1_0_full": 0,
            "tq1_0_dynamic_exit": 0,
            "mask_sign_dynamic_exit": 0
        }

        layer_exit_counts = [0] * (self.num_layers + 1)

        for _ in range(num_tokens):
            r = random.random()
            if r < 0.60:
                exit_layer = random.randint(2, 5)
            elif r < 0.85:
                exit_layer = random.randint(6, 11)
            else:
                exit_layer = self.num_layers

            layer_exit_counts[exit_layer] += 1

            # Full passes without early exit
            bytes_fp16 = self.num_layers * (self.weights_per_layer * self.bpw_fp16 / 8)
            bytes_int4 = self.num_layers * (self.weights_per_layer * self.bpw_int4 / 8)
            bytes_tq1_0_full = self.num_layers * (self.weights_per_layer * self.bpw_tq1_0 / 8)

            # Dynamic passes with early exit
            bytes_tq1_0_dyn = exit_layer * (self.weights_per_layer * self.bpw_tq1_0 / 8)
            bytes_ms_dyn = exit_layer * (self.weights_per_layer * self.bpw_mask_sign / 8)

            bytes_streamed["fp16_full"] += bytes_fp16
            bytes_streamed["int4_full"] += bytes_int4
            bytes_streamed["tq1_0_full"] += bytes_tq1_0_full
            bytes_streamed["tq1_0_dynamic_exit"] += bytes_tq1_0_dyn
            bytes_streamed["mask_sign_dynamic_exit"] += bytes_ms_dyn

        avg_exit_layer = sum(idx * count for idx, count in enumerate(layer_exit_counts)) / num_tokens

        # Compression and bandwidth ratios against baseline FP16
        baseline = bytes_streamed["fp16_full"]
        ratios = {
            k: round(baseline / v, 2) for k, v in bytes_streamed.items()
        }

        return {
            "model_config": {
                "num_layers": self.num_layers,
                "weights_per_layer": self.weights_per_layer,
                "total_parameters": self.num_layers * self.weights_per_layer,
                "resident_descriptor_D_kb": round(self.total_resident_D_bytes / 1024, 2)
            },
            "tokens_simulated": num_tokens,
            "avg_exit_layer": round(avg_exit_layer, 2),
            "exit_layer_distribution": layer_exit_counts,
            "total_megabytes_streamed": {
                k: round(v / (1024 * 1024), 2) for k, v in bytes_streamed.items()
            },
            "speedup_bandwidth_multipliers_vs_fp16": ratios
        }

def main():
    sim = MemoryModelSimulator(num_layers=16, weights_per_layer=8_000_000, hidden_dim=2048)
    results = sim.simulate_token_generation(num_tokens=500)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "continuation-simulation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Continuation simulation complete.")
    print(f" - Average exit layer: {results['avg_exit_layer']} / {results['model_config']['num_layers']}")
    print(f" - Total MB streamed (FP16): {results['total_megabytes_streamed']['fp16_full']} MB")
    print(f" - Total MB streamed (TQ1_0 Dynamic Exit): {results['total_megabytes_streamed']['tq1_0_dynamic_exit']} MB")
    print(f" - Bandwidth reduction vs FP16: {results['speedup_bandwidth_multipliers_vs_fp16']['tq1_0_dynamic_exit']}x")

if __name__ == "__main__":
    main()
