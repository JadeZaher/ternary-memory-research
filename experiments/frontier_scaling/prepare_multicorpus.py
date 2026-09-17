"""
experiments/frontier_scaling/prepare_multicorpus.py: Multi-Corpus Pretraining Dataset Builder.

Gate 15 (Track D-4): Builds a balanced 10M-token training cache combining:
1. TinyStories (6.0M tokens, 60%): Fluent narrative syntax & entity persistence.
2. GSM8K / Multi-step Math (2.0M tokens, 20%): Multi-hop relational arithmetic.
3. Python Algorithmic Code (2.0M tokens, 20%): Structured indentation & bracket binding.

Saves unified 1D token tensor to `data/multicorpus_10m.pt`.
"""

import os
import sys
import random
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def generate_math_problems(num_samples: int = 15000) -> str:
    """Generates synthetic multi-step arithmetic word problems with chain-of-thought."""
    names = ["Lily", "Tom", "Sam", "Emma", "Jack", "Olivia", "Leo", "Mia", "Noah", "Ava", "Lucas", "Zoe"]
    items = ["apples", "pencils", "stickers", "marbles", "candies", "toy cars", "crayons", "erasers", "books"]
    verbs_loss = ["gave", "lost", "shared", "dropped", "sold", "donated"]
    verbs_gain = ["bought", "found", "received", "earned", "collected", "picked"]

    lines = []
    for _ in range(num_samples):
        n1 = random.choice(names)
        n2 = random.choice([n for n in names if n != n1])
        n3 = random.choice([n for n in names if n not in (n1, n2)])
        item = random.choice(items)

        start = random.randint(15, 60)
        loss1 = random.randint(2, 10)
        loss2 = random.randint(2, 10)
        gain = random.randint(5, 25)

        v_loss1 = random.choice(verbs_loss)
        v_loss2 = random.choice(verbs_loss)
        v_gain = random.choice(verbs_gain)

        step1 = start - loss1
        step2 = step1 - loss2
        final = step2 + gain

        q = (
            f"Problem: {n1} had {start} {item}. {n1} {v_loss1} {loss1} {item} to {n2} "
            f"and then {v_loss2} {loss2} {item} to {n3}. Later, {n1} {v_gain} {gain} more {item}. "
            f"How many {item} does {n1} have in the end?\n"
            f"Reasoning: Let's track the count step by step.\n"
            f"Step 1: {n1} begins with {start} {item}.\n"
            f"Step 2: After {v_loss1} {loss1} to {n2}, {start} - {loss1} = {step1} {item} remain.\n"
            f"Step 3: After {v_loss2} {loss2} to {n3}, {step1} - {loss2} = {step2} {item} remain.\n"
            f"Step 4: Then {n1} {v_gain} {gain} {item}, so {step2} + {gain} = {final} {item}.\n"
            f"Final Answer: {n1} has {final} {item}. #### {final}\n\n"
        )
        lines.append(q)
    return "".join(lines)


def generate_python_code(num_samples: int = 8000) -> str:
    """Generates clean, algorithmic Python snippets with structured logic."""
    templates = [
        """def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Test binary search
data = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
found_index = binary_search(data, 23)
assert found_index == 5
""",
        """def merge_sorted_arrays(left, right):
    result = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

a = [1, 3, 5, 7]
b = [2, 4, 6, 8]
merged = merge_sorted_arrays(a, b)
assert merged == [1, 2, 3, 4, 5, 6, 7, 8]
""",
        """class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self.is_empty():
            return self._items.pop()
        raise IndexError("pop from empty stack")

    def peek(self):
        if not self.is_empty():
            return self._items[-1]
        return None

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)

s = Stack()
s.push(10)
s.push(20)
assert s.pop() == 20
assert s.peek() == 10
""",
        """def fibonacci_memo(n, memo=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

assert fibonacci_memo(0) == 0
assert fibonacci_memo(1) == 1
assert fibonacci_memo(6) == 8
assert fibonacci_memo(10) == 55
""",
        """def count_word_frequencies(text):
    words = text.lower().split()
    freq = {}
    for word in words:
        clean_word = word.strip(".,!?:;")
        if clean_word:
            freq[clean_word] = freq.get(clean_word, 0) + 1
    return freq

sample_text = "The cat saw another cat and the little dog."
counts = count_word_frequencies(sample_text)
assert counts["cat"] == 2
assert counts["the"] == 2
"""
    ]

    snippets = []
    for _ in range(num_samples):
        snippet = random.choice(templates)
        # Add random variations
        salt = random.randint(100, 999)
        modified = snippet.replace("data", f"data_{salt}").replace("result", f"res_{salt}")
        snippets.append(modified + "\n\n")
    return "".join(snippets)


def build_multicorpus(
    tinystories_path: str = "data/tinystories_tokens.pt",
    output_path: str = "data/multicorpus_10m.pt",
    total_tokens: int = 10_000_000,
):
    print(f"=== Building Multi-Corpus Pretraining Dataset ({total_tokens:,} tokens) ===")
    
    # 1. Load TinyStories tokens
    print(f"[1/4] Loading TinyStories tokens from {tinystories_path}...")
    if not os.path.exists(tinystories_path):
        raise FileNotFoundError(f"Missing {tinystories_path}!")
    ts_all = torch.load(tinystories_path, weights_only=True)
    target_ts = int(0.60 * total_tokens)  # 6.0M tokens
    ts_tokens = ts_all[:target_ts]
    print(f"  TinyStories slice: {len(ts_tokens):,} tokens ({len(ts_tokens)/total_tokens*100:.1f}%)")

    # 2. Tokenize GSM8K / Multi-step Math
    print("[2/4] Generating & tokenizing GSM8K multi-step arithmetic problems...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    math_text = generate_math_problems(num_samples=25000)
    math_enc = tokenizer.encode(math_text, return_tensors="pt")[0]
    target_math = int(0.20 * total_tokens)  # 2.0M tokens
    math_tokens = math_enc[:target_math]
    print(f"  Math slice: {len(math_tokens):,} tokens ({len(math_tokens)/total_tokens*100:.1f}%)")

    # 3. Tokenize Python Code
    print("[3/4] Generating & tokenizing algorithmic Python code...")
    code_text = generate_python_code(num_samples=18000)
    code_enc = tokenizer.encode(code_text, return_tensors="pt")[0]
    target_code = total_tokens - len(ts_tokens) - len(math_tokens)  # Remaining ~2.0M tokens
    code_tokens = code_enc[:target_code]
    print(f"  Code slice: {len(code_tokens):,} tokens ({len(code_tokens)/total_tokens*100:.1f}%)")

    # 4. Interleave & Combine
    print("[4/4] Interleaving multi-corpus segments and saving...")
    chunk_size = 2048
    chunks = []
    
    i_ts = 0
    i_math = 0
    i_code = 0

    while i_ts < len(ts_tokens) or i_math < len(math_tokens) or i_code < len(code_tokens):
        # 3 chunks of TinyStories
        for _ in range(3):
            if i_ts < len(ts_tokens):
                chunks.append(ts_tokens[i_ts : i_ts + chunk_size])
                i_ts += chunk_size
        # 1 chunk of Math
        if i_math < len(math_tokens):
            chunks.append(math_tokens[i_math : i_math + chunk_size])
            i_math += chunk_size
        # 1 chunk of Code
        if i_code < len(code_tokens):
            chunks.append(code_tokens[i_code : i_code + chunk_size])
            i_code += chunk_size

    multicorpus = torch.cat(chunks)[:total_tokens]
    print(f"Final tensor shape: {multicorpus.shape}, total tokens: {len(multicorpus):,}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(multicorpus, output_path)
    print(f">>> Successfully saved multi-corpus cache to {output_path} ({os.path.getsize(output_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    build_multicorpus()
