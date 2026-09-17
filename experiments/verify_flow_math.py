"""
verify_flow_math.py: Exhaustive and empirical verification of mathematical claims
for ternary accumulators, fixed-point contraction, and early-exit thresholds.
Writes outputs to outputs/flow-math-verification.json.
Uses standard library only.
"""

import json
import math
import random
from pathlib import Path

SEED = 20260915
random.seed(SEED)

def check_accumulator_bounds():
    """Verify that integer accumulation cannot overflow for specified dimensions and int8 inputs."""
    dimensions = [128, 512, 1024, 2048, 4096]
    x_max = 127  # signed int8 maximum
    results = []

    for N in dimensions:
        # Worst-case: all weights are +1 (or -1) and all inputs are +127
        max_accum_theoretical = N * x_max
        required_bits = 1 + math.ceil(math.log2(max_accum_theoretical))
        fits_int32 = required_bits <= 31  # 1 sign bit + 31 magnitude bits

        # Randomized trial: sample 500 ternary vectors and int8 inputs
        max_observed = 0
        for _ in range(500):
            w = [random.choice([-1, 0, 1]) for _ in range(N)]
            x = [random.randint(-128, 127) for _ in range(N)]
            accum = sum(wi * xi for wi, xi in zip(w, x))
            max_observed = max(max_observed, abs(accum))

        results.append({
            "dimension": N,
            "max_theoretical": max_accum_theoretical,
            "required_bits": required_bits,
            "fits_int32": fits_int32,
            "max_observed": max_observed,
            "observed_le_theoretical": max_observed <= max_accum_theoretical
        })

    all_pass = all(r["fits_int32"] and r["observed_le_theoretical"] for r in results)
    return {
        "name": "accumulator_bounds_check",
        "passed": all_pass,
        "details": results
    }

def softmax(vec):
    """Compute numerically stable softmax."""
    max_v = max(vec)
    exps = [math.exp(v - max_v) for v in vec]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def vector_diff_norm_inf(v1, v2):
    """Compute infinity norm (max absolute difference) between two vectors."""
    return max(abs(a - b) for a, b in zip(v1, v2))

def check_fixed_point_contraction():
    """Verify that a contractive recurrent map exhibits monotonic contraction and early exit reliability."""
    dim = 16
    trials = 50
    success_count = 0
    trajectories = []

    for trial_idx in range(trials):
        # Generate a scaled transition matrix with spectral norm < 1
        scale = 0.6  # strictly contractive
        W = [[(random.uniform(-1, 1) * scale) / dim for _ in range(dim)] for _ in range(dim)]
        b = [random.uniform(-0.5, 0.5) for _ in range(dim)]

        # Recurrence: s^{(k+1)} = softmax(W * s^{(k)} + b)
        s = [1.0 / dim] * dim  # uniform start
        diffs = []
        converged_step = None
        eps = 1e-4

        for k in range(1, 25):
            # Compute W * s + b
            lin = [sum(W[i][j] * s[j] for j in range(dim)) + b[i] for i in range(dim)]
            s_next = softmax(lin)
            diff = vector_diff_norm_inf(s_next, s)
            diffs.append(diff)
            s = s_next

            if diff < eps and converged_step is None:
                converged_step = k

        # Check if difference drops over time
        is_contractive = diffs[-1] < diffs[0]
        if is_contractive and converged_step is not None and converged_step < 20:
            success_count += 1

        if trial_idx < 3:
            trajectories.append({
                "trial": trial_idx,
                "converged_step": converged_step,
                "initial_diff": diffs[0],
                "final_diff": diffs[-1]
            })

    return {
        "name": "fixed_point_contraction_check",
        "passed": success_count == trials,
        "success_rate": success_count / trials,
        "sample_trajectories": trajectories
    }

def check_ternary_recurrent_stability():
    """Verify that ternary quantized weights maintain contractive fixed points when appropriately scaled."""
    dim = 16
    trials = 50
    stable_count = 0

    for _ in range(trials):
        # Generate ternary matrix {-1, 0, 1}
        # To maintain contractive dynamics on dim=16, scale factor gamma < 1 / sqrt(dim)
        scale = 0.5 / math.sqrt(dim)
        W_ternary = [[random.choice([-1, 0, 1]) for _ in range(dim)] for _ in range(dim)]
        b = [random.uniform(-0.2, 0.2) for _ in range(dim)]

        s = [1.0 / dim] * dim
        diffs = []
        for k in range(20):
            lin = [sum(scale * W_ternary[i][j] * s[j] for j in range(dim)) + b[i] for i in range(dim)]
            s_next = softmax(lin)
            diff = vector_diff_norm_inf(s_next, s)
            diffs.append(diff)
            s = s_next

        if diffs[-1] < 1e-3:
            stable_count += 1

    return {
        "name": "ternary_recurrent_stability_check",
        "passed": stable_count >= 48,  # at least 96% stable
        "stable_trials": stable_count,
        "total_trials": trials
    }

def main():
    checks = [
        check_accumulator_bounds(),
        check_fixed_point_contraction(),
        check_ternary_recurrent_stability()
    ]

    all_passed = all(c["passed"] for c in checks)
    output_data = {
        "suite": "flow_math_verification",
        "seed": SEED,
        "all_passed": all_passed,
        "checks": checks
    }

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "flow-math-verification.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Flow math verification complete. All passed: {all_passed}")
    for c in checks:
        print(f" - {c['name']}: {'PASSED' if c['passed'] else 'FAILED'}")

if __name__ == "__main__":
    main()
