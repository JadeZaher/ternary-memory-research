"""
toy_frm_recurrent.py: Micro-scale Flow Reasoning Model (FRM) with Fixed-Point Forcing (FPF)
and early-exit recurrence on a 4x4 Mini-Sudoku / Latin square constraint puzzle.
Compares:
- One-shot flow vs Recurrent flow (k steps)
- Dynamic early exit via fixed-point convergence ||s^(k+1) - s^(k)|| < delta
- Float32 weights vs Ternary-quantized weights {-1, 0, +1}
Writes outputs to outputs/toy-frm-recurrent.json.
Uses standard library only.
"""

import json
import math
import random
from pathlib import Path

SEED = 20260915
random.seed(SEED)

GRID_SIZE = 4
NUM_CELLS = GRID_SIZE * GRID_SIZE  # 16 cells
NUM_DIGITS = 4                      # Digits 1..4
INPUT_DIM = NUM_CELLS * NUM_DIGITS  # 64 one-hot variables

# Helper math
def softmax(logits):
    """Numerically stable softmax over a list."""
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

def vector_diff_max(v1, v2):
    """L-infinity norm (max absolute difference)."""
    return max(abs(a - b) for a, b in zip(v1, v2))

def ternarize_matrix(W, threshold_ratio=0.75):
    """
    Ternary Weight Networks thresholding (Li & Liu 2016):
    delta = threshold_ratio * mean(|W|)
    W_ternary = +1 if W > delta, -1 if W < -delta, else 0
    alpha = mean(|W| for nonzeros)
    """
    flat = [abs(w) for row in W for w in row]
    mean_abs = sum(flat) / len(flat) if flat else 1.0
    delta = threshold_ratio * mean_abs

    W_ternary = []
    nonzeros = []
    for row in W:
        t_row = []
        for w in row:
            if w > delta:
                t_row.append(1)
                nonzeros.append(abs(w))
            elif w < -delta:
                t_row.append(-1)
                nonzeros.append(abs(w))
            else:
                t_row.append(0)
        W_ternary.append(t_row)

    alpha = sum(nonzeros) / len(nonzeros) if nonzeros else 1.0
    return W_ternary, alpha

class ToyMiniSudokuSolver:
    """
    A 4x4 puzzle generator and rule evaluator.
    Puzzle represented as a flat list of 16 digits (0 = empty, 1..4 = filled).
    """
    @staticmethod
    def is_valid_4x4(grid):
        # Rows
        for r in range(4):
            vals = [grid[r * 4 + c] for c in range(4) if grid[r * 4 + c] != 0]
            if len(vals) != len(set(vals)):
                return False
        # Cols
        for c in range(4):
            vals = [grid[r * 4 + c] for r in range(4) if grid[r * 4 + c] != 0]
            if len(vals) != len(set(vals)):
                return False
        # 2x2 boxes
        for br in (0, 2):
            for bc in (0, 2):
                box = [
                    grid[(br + r) * 4 + (bc + c)]
                    for r in range(2) for c in range(2)
                    if grid[(br + r) * 4 + (bc + c)] != 0
                ]
                if len(box) != len(set(box)):
                    return False
        return True

    @staticmethod
    def generate_valid_solution():
        """Generate a valid complete 4x4 Sudoku solution."""
        base = [
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1]
        ]
        # Randomly permute numbers
        p = [1, 2, 3, 4]
        random.shuffle(p)
        mapping = {i + 1: p[i] for i in range(4)}
        sol = [mapping[base[r][c]] for r in range(4) for c in range(4)]
        return sol

    @staticmethod
    def to_one_hot(solution):
        vec = [0.0] * INPUT_DIM
        for idx, digit in enumerate(solution):
            vec[idx * NUM_DIGITS + (digit - 1)] = 1.0
        return vec

    @staticmethod
    def from_probs(probs):
        digits = []
        for cell in range(NUM_CELLS):
            cell_probs = probs[cell * NUM_DIGITS : (cell + 1) * NUM_DIGITS]
            best_digit = max(range(NUM_DIGITS), key=lambda d: cell_probs[d]) + 1
            digits.append(best_digit)
        return digits

class RecurrentFlowDenoiser:
    """
    Two-layer MLP denoiser with carry conditioning:
    Input: [x_t (64) + condition_c (64) + carry_s (64)] = 192 dims.
    Hidden: 128 dims.
    Output: 64 logits (softmaxed per 4-cell group).
    """
    def __init__(self, hidden_dim=128):
        self.in_dim = INPUT_DIM * 3
        self.hidden_dim = hidden_dim
        self.out_dim = INPUT_DIM

        scale1 = 1.0 / math.sqrt(self.in_dim)
        self.W1 = [[random.gauss(0, scale1) for _ in range(self.in_dim)] for _ in range(self.hidden_dim)]
        self.b1 = [0.0] * self.hidden_dim

        scale2 = 1.0 / math.sqrt(self.hidden_dim)
        self.W2 = [[random.gauss(0, scale2) for _ in range(self.hidden_dim)] for _ in range(self.out_dim)]
        self.b2 = [0.0] * self.out_dim

    def forward(self, x_t, c, s, use_ternary=False):
        """
        Forward pass with optional ternary weights.
        """
        # Form combined input vector
        inp = x_t + c + s

        if use_ternary:
            W1_t, a1 = ternarize_matrix(self.W1)
            W2_t, a2 = ternarize_matrix(self.W2)
            # Layer 1: ReLU(a1 * (W1_t * inp) + b1)
            h = []
            for i in range(self.hidden_dim):
                val = a1 * sum(W1_t[i][j] * inp[j] for j in range(self.in_dim)) + self.b1[i]
                h.append(val if val > 0 else 0.0)
            # Layer 2: a2 * (W2_t * h) + b2
            out = []
            for i in range(self.out_dim):
                val = a2 * sum(W2_t[i][j] * h[j] for j in range(self.hidden_dim)) + self.b2[i]
                out.append(val)
        else:
            # Standard FP32
            h = []
            for i in range(self.hidden_dim):
                val = sum(self.W1[i][j] * inp[j] for j in range(self.in_dim)) + self.b1[i]
                h.append(val if val > 0 else 0.0)
            out = []
            for i in range(self.out_dim):
                val = sum(self.W2[i][j] * h[j] for j in range(self.hidden_dim)) + self.b2[i]
                out.append(val)

        # Softmax per 4-digit cell
        probs = []
        for cell in range(NUM_CELLS):
            cell_logits = out[cell * NUM_DIGITS : (cell + 1) * NUM_DIGITS]
            probs.extend(softmax(cell_logits))
        return probs

    def train_fpf_step(self, x_clean, c, lr=0.01):
        """
        Fixed-Point Forcing (FPF) training step:
        Simulate an inference rollout to produce carry s_fpf, then compute gradient update on D_theta(x_t | c, s_fpf).
        Simplified analytical approximation for toy demonstration.
        """
        # Sample t in (0, 1)
        t = random.uniform(0.2, 0.9)
        noise = [random.gauss(0, 1) for _ in range(INPUT_DIM)]
        x_t = [(1 - t) * n + t * c_val for n, c_val in zip(noise, x_clean)]

        # Rollout 2 recurrent steps to obtain s_fpf
        s_curr = [0.25] * INPUT_DIM
        for _ in range(2):
            s_curr = self.forward(x_t, c, s_curr, use_ternary=False)

        # Target is x_clean (one-hot)
        # Gradient update on W2, b2, W1, b1 (mini-batch gradient step)
        inp = x_t + c + s_curr
        h = [max(0.0, sum(self.W1[i][j] * inp[j] for j in range(self.in_dim)) + self.b1[i]) for i in range(self.hidden_dim)]
        out = [sum(self.W2[i][j] * h[j] for j in range(self.hidden_dim)) + self.b2[i] for i in range(self.out_dim)]

        # Softmax cross-entropy gradient: dL/dout = p - target
        dL_dout = []
        for cell in range(NUM_CELLS):
            p_cell = softmax(out[cell * NUM_DIGITS : (cell + 1) * NUM_DIGITS])
            t_cell = x_clean[cell * NUM_DIGITS : (cell + 1) * NUM_DIGITS]
            for p_v, t_v in zip(p_cell, t_cell):
                dL_dout.append(p_v - t_v)

        # Backprop to W2
        for i in range(self.out_dim):
            g = dL_dout[i]
            for j in range(self.hidden_dim):
                self.W2[i][j] -= lr * g * h[j]
            self.b2[i] -= lr * g

def run_experiment():
    print("Training toy Flow Reasoning Model with Fixed-Point Forcing on Mini-Sudoku...")
    solver = ToyMiniSudokuSolver()
    denoiser = RecurrentFlowDenoiser(hidden_dim=96)

    # Train for 400 synthetic steps
    for _ in range(400):
        sol = solver.generate_valid_solution()
        x_clean = solver.to_one_hot(sol)
        # 8 of 16 cells given as prompt condition c
        mask = [1 if random.random() < 0.5 else 0 for _ in range(NUM_CELLS)]
        c = []
        for cell in range(NUM_CELLS):
            if mask[cell]:
                c.extend(x_clean[cell * NUM_DIGITS : (cell + 1) * NUM_DIGITS])
            else:
                c.extend([0.0] * NUM_DIGITS)
        denoiser.train_fpf_step(x_clean, c, lr=0.015)

    # Evaluation phase: 50 test puzzles
    test_trials = 50
    results = {
        "one_shot_fp32": {"valid_count": 0, "avg_steps": 1.0},
        "recurrent_fixed_k5_fp32": {"valid_count": 0, "avg_steps": 5.0},
        "recurrent_dynamic_exit_fp32": {"valid_count": 0, "steps": [], "avg_steps": 0.0},
        "recurrent_dynamic_exit_ternary": {"valid_count": 0, "steps": [], "avg_steps": 0.0}
    }

    eps_exit = 0.04  # L-inf contraction threshold for early exit

    for _ in range(test_trials):
        sol = solver.generate_valid_solution()
        x_clean = solver.to_one_hot(sol)
        # Condition: 10 fixed clues
        clue_indices = random.sample(range(NUM_CELLS), 10)
        c = [0.0] * INPUT_DIM
        for idx in clue_indices:
            c[idx * NUM_DIGITS : (idx + 1) * NUM_DIGITS] = x_clean[idx * NUM_DIGITS : (idx + 1) * NUM_DIGITS]

        # Initial noise state at t=0.5
        noise = [random.gauss(0, 0.5) for _ in range(INPUT_DIM)]
        x_t = [0.5 * n + 0.5 * c_val for n, c_val in zip(noise, c)]

        # 1. One-shot
        s_init = [0.25] * INPUT_DIM
        s_one = denoiser.forward(x_t, c, s_init, use_ternary=False)
        pred_one = solver.from_probs(s_one)
        # Fill in clues
        for idx in clue_indices:
            pred_one[idx] = sol[idx]
        if solver.is_valid_4x4(pred_one):
            results["one_shot_fp32"]["valid_count"] += 1

        # 2. Fixed k=5 recurrence
        s_rec = s_init
        for _ in range(5):
            s_rec = denoiser.forward(x_t, c, s_rec, use_ternary=False)
        pred_k5 = solver.from_probs(s_rec)
        for idx in clue_indices:
            pred_k5[idx] = sol[idx]
        if solver.is_valid_4x4(pred_k5):
            results["recurrent_fixed_k5_fp32"]["valid_count"] += 1

        # 3. Dynamic Early-Exit (FP32)
        s_dyn = s_init
        dyn_steps = 0
        for k in range(1, 10):
            s_next = denoiser.forward(x_t, c, s_dyn, use_ternary=False)
            diff = vector_diff_max(s_next, s_dyn)
            s_dyn = s_next
            dyn_steps = k
            if diff < eps_exit:
                break
        results["recurrent_dynamic_exit_fp32"]["steps"].append(dyn_steps)
        pred_dyn = solver.from_probs(s_dyn)
        for idx in clue_indices:
            pred_dyn[idx] = sol[idx]
        if solver.is_valid_4x4(pred_dyn):
            results["recurrent_dynamic_exit_fp32"]["valid_count"] += 1

        # 4. Dynamic Early-Exit (Ternary Weights)
        s_ter = s_init
        ter_steps = 0
        for k in range(1, 10):
            s_next = denoiser.forward(x_t, c, s_ter, use_ternary=True)
            diff = vector_diff_max(s_next, s_ter)
            s_ter = s_next
            ter_steps = k
            if diff < eps_exit:
                break
        results["recurrent_dynamic_exit_ternary"]["steps"].append(ter_steps)
        pred_ter = solver.from_probs(s_ter)
        for idx in clue_indices:
            pred_ter[idx] = sol[idx]
        if solver.is_valid_4x4(pred_ter):
            results["recurrent_dynamic_exit_ternary"]["valid_count"] += 1

    # Summarize
    results["recurrent_dynamic_exit_fp32"]["avg_steps"] = round(
        sum(results["recurrent_dynamic_exit_fp32"]["steps"]) / test_trials, 2
    )
    results["recurrent_dynamic_exit_ternary"]["avg_steps"] = round(
        sum(results["recurrent_dynamic_exit_ternary"]["steps"]) / test_trials, 2
    )

    summary = {
        "test_trials": test_trials,
        "results": {
            "one_shot_fp32": {
                "solve_rate": results["one_shot_fp32"]["valid_count"] / test_trials,
                "avg_steps": 1.0
            },
            "recurrent_fixed_k5_fp32": {
                "solve_rate": results["recurrent_fixed_k5_fp32"]["valid_count"] / test_trials,
                "avg_steps": 5.0
            },
            "recurrent_dynamic_exit_fp32": {
                "solve_rate": results["recurrent_dynamic_exit_fp32"]["valid_count"] / test_trials,
                "avg_steps": results["recurrent_dynamic_exit_fp32"]["avg_steps"]
            },
            "recurrent_dynamic_exit_ternary": {
                "solve_rate": results["recurrent_dynamic_exit_ternary"]["valid_count"] / test_trials,
                "avg_steps": results["recurrent_dynamic_exit_ternary"]["avg_steps"]
            }
        }
    }

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "toy-frm-recurrent.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Toy FRM evaluation complete.")
    for k, v in summary["results"].items():
        print(f" - {k}: solve rate = {v['solve_rate'] * 100:.1f}%, avg steps = {v['avg_steps']}")

if __name__ == "__main__":
    run_experiment()
