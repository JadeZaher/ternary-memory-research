"""
experiments/sudoku9x9_dataset.py: 9x9 Sudoku Generator, Solver, and Benchmark Dataset.

Provides:
1. Constraint validation for 9x9 boards (9 rows, 9 columns, nine 3x3 blocks).
2. Backtracking solver to generate valid canonical 9x9 completed grids.
3. Puzzle generator creating puzzles across 3 difficulty tiers:
   - Easy:   32-36 clues (fewest empty cells: 45-49)
   - Medium: 26-31 clues (empty cells: 50-55)
   - Hard:   17-25 clues (empty cells: 56-64, near minimal 17-clue limit)
4. Tensor encoding into [81, 9] one-hot tensors suitable for FlowTrit.
"""

import random
from typing import Dict, List, Optional, Tuple
import torch


def is_valid_9x9(board: List[List[int]], row: int, col: int, val: int) -> bool:
    """Checks if placing val at (row, col) violates row, column, or 3x3 box constraints."""
    for c in range(9):
        if board[row][c] == val:
            return False
    for r in range(9):
        if board[r][col] == val:
            return False
    br = (row // 3) * 3
    bc = (col // 3) * 3
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if board[r][c] == val:
                return False
    return True


def solve_9x9_backtracking(board: List[List[int]]) -> bool:
    """Solves 9x9 Sudoku in place using recursive backtracking."""
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                nums = list(range(1, 10))
                random.shuffle(nums)
                for val in nums:
                    if is_valid_9x9(board, r, col=c, val=val):
                        board[r][c] = val
                        if solve_9x9_backtracking(board):
                            return True
                        board[r][c] = 0
                return False
    return True


def generate_valid_solution_9x9() -> List[List[int]]:
    """Generates a complete, randomly filled valid 9x9 Sudoku board."""
    board = [[0] * 9 for _ in range(9)]
    # Fill diagonal 3x3 boxes first (independent) for fast random generation
    for i in range(0, 9, 3):
        nums = list(range(1, 10))
        random.shuffle(nums)
        for r in range(3):
            for c in range(3):
                board[i + r][i + c] = nums.pop()
    solve_9x9_backtracking(board)
    return board


def create_puzzle_9x9(solution: List[List[int]], tier: str = "easy") -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """
    Creates a puzzle from a solution by removing numbers based on difficulty tier.
    Returns:
        puzzle: 9x9 board with 0 for empty cells
        clues: list of (row, col) coordinates of given cells
    """
    if tier == "easy":
        target_clues = random.randint(32, 36)
    elif tier == "medium":
        target_clues = random.randint(26, 31)
    else:  # hard
        target_clues = random.randint(18, 25)

    puzzle = [row[:] for row in solution]
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)

    to_remove = 81 - target_clues
    for i in range(to_remove):
        r, c = cells[i]
        puzzle[r][c] = 0

    clues = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] != 0]
    return puzzle, clues


def encode_board_to_tensor(board: List[List[int]]) -> torch.Tensor:
    """
    Encodes 9x9 board into one-hot tensor of shape [81, 9].
    0 maps to all-zeros vector. Values 1..9 map to index 0..8.
    """
    tensor = torch.zeros(81, 9, dtype=torch.float32)
    flat_idx = 0
    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val > 0:
                tensor[flat_idx, val - 1] = 1.0
            flat_idx += 1
    return tensor


def generate_sudoku9x9_benchmark(
    num_train: int = 150,
    num_test_per_tier: int = 20,
) -> Dict[str, any]:
    """
    Generates a structured 9x9 benchmark dataset with train and stratified test tiers.
    """
    print(f"[+] Generating {num_train} training 9x9 solutions...")
    train_solutions = [generate_valid_solution_9x9() for _ in range(num_train)]

    test_puzzles = {"easy": [], "medium": [], "hard": []}
    for tier in ["easy", "medium", "hard"]:
        print(f"[+] Generating {num_test_per_tier} {tier} test 9x9 puzzles...")
        for _ in range(num_test_per_tier):
            sol = generate_valid_solution_9x9()
            puz, clues = create_puzzle_9x9(sol, tier=tier)
            test_puzzles[tier].append({
                "solution": encode_board_to_tensor(sol),
                "puzzle": encode_board_to_tensor(puz),
                "clue_indices": [r * 9 + c for (r, c) in clues],
                "num_clues": len(clues),
            })

    train_tensors = [encode_board_to_tensor(s) for s in train_solutions]

    return {
        "train_solutions": train_tensors,
        "test_puzzles": test_puzzles,
        "num_train": num_train,
        "num_test_per_tier": num_test_per_tier,
    }


if __name__ == "__main__":
    bm = generate_sudoku9x9_benchmark(num_train=5, num_test_per_tier=2)
    print("Benchmark generation verified successfully!")
    print("Sample puzzle tensor shape:", bm["test_puzzles"]["easy"][0]["puzzle"].shape)
