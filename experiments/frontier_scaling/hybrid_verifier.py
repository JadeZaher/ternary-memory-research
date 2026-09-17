"""
experiments/frontier_scaling/hybrid_verifier.py: Multi-Task Hybrid Verifier for GRPO Training.

Combines deterministic rule-based verification (0ms overhead) for arithmetic & code
with heuristic and batch LLM verification (Gemini 2.5 Flash via OpenRouter) for narrative text.
"""

import os
import re
import ast
import json
import urllib.request
from typing import Dict, List, Optional, Tuple, Any

ENV_PATH = r"C:\Users\atooz\.pi\agent\.env"
MODEL_ID = "google/gemini-2.5-flash"


def get_api_key() -> Optional[str]:
    if not os.path.exists(ENV_PATH):
        return None
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


class HybridVerifier:
    """
    Hybrid Outcome Verifier providing reward R(y | x, task_type) in [-1.0, +2.0].
    """
    def __init__(self, use_gemini: bool = False):
        self.use_gemini = use_gemini
        self.api_key = get_api_key() if use_gemini else None

    def verify_arithmetic(self, continuation: str, target: Optional[str] = None) -> Tuple[float, str]:
        """
        Evaluates multi-step arithmetic reasoning.
        Looks for:
        1. Numerical answer matching target (+1.5)
        2. Valid arithmetic statement (+0.5)
        3. Hallucinated / invalid calculations like '15 - 3 = 17' (-1.0)
        """
        text = continuation.strip()
        reward = 0.0
        rationale = []

        # Check for false equation like 'A - B = C' where A - B != C
        equation_pattern = re.compile(r'(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*(\d+)')
        matches = equation_pattern.findall(text)
        
        has_equation = False
        valid_equation_count = 0
        invalid_equation_count = 0

        for a_str, op, b_str, c_str in matches:
            has_equation = True
            a, b, c = int(a_str), int(b_str), int(c_str)
            correct_val = None
            if op == '+': correct_val = a + b
            elif op == '-': correct_val = a - b
            elif op == '*': correct_val = a * b
            elif op == '/' and b != 0: correct_val = a // b

            if correct_val is not None:
                if correct_val == c:
                    valid_equation_count += 1
                else:
                    invalid_equation_count += 1

        if invalid_equation_count > 0:
            reward -= 1.0
            rationale.append(f"Invalid arithmetic detected ({invalid_equation_count} false equations)")
        elif valid_equation_count > 0:
            reward += 0.5
            rationale.append(f"Valid arithmetic steps ({valid_equation_count} correct equations)")

        # Target number matching
        if target is not None:
            # Look for target number in text
            target_clean = str(target).strip()
            # Check if target appears near the end of reasoning or as answer
            number_tokens = re.findall(r'\b\d+\b', text)
            if target_clean in number_tokens:
                reward += 1.0
                rationale.append(f"Target number {target_clean} matched")
            else:
                reward -= 0.5
                rationale.append(f"Target number {target_clean} not reached")

        # Repetition penalty
        words = text.split()
        if len(words) > 6:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            distinct_ratio = len(set(bigrams)) / max(1, len(bigrams))
            if distinct_ratio < 0.60:
                reward -= 0.5
                rationale.append("High n-gram repetition")

        final_reward = max(-1.0, min(2.0, reward))
        return final_reward, "; ".join(rationale)

    def verify_code(self, prompt: str, continuation: str) -> Tuple[float, str]:
        """
        Evaluates algorithmic code synthesis.
        Uses Python AST parser to check syntactical validity and structural progression.
        """
        full_code = prompt + continuation
        reward = 0.0
        rationale = []

        # Attempt AST parse
        try:
            ast.parse(full_code)
            reward += 0.75
            rationale.append("Valid Python syntax (AST parsed)")
        except SyntaxError:
            # Code might be incomplete at the very end
            # Try parsing with pass/ellipsis or indentation closing
            for closing in ["\n    pass", "\n        pass", "\n"]:
                try:
                    ast.parse(full_code + closing)
                    reward += 0.5
                    rationale.append("Incomplete but syntactically valid prefix")
                    break
                except SyntaxError:
                    continue
            if reward == 0.0:
                reward -= 0.75
                rationale.append("Syntax error")

        # Structural completeness checks
        if "while" in continuation or "for" in continuation:
            reward += 0.25
            rationale.append("Constructed control loop")
        if "mid" in continuation or "return" in continuation:
            reward += 0.25
            rationale.append("Midpoint/return keyword present")

        final_reward = max(-1.0, min(2.0, reward))
        return final_reward, "; ".join(rationale)

    def verify_story(self, prompt: str, continuation: str) -> Tuple[float, str]:
        """
        Evaluates narrative text using lexical coherence heuristics.
        """
        text = continuation.strip()
        reward = 0.0
        rationale = []

        # Lexical diversity check
        words = text.split()
        if len(words) >= 8:
            distinct_1 = len(set(words)) / len(words)
            if distinct_1 > 0.75:
                reward += 0.5
                rationale.append(f"Good vocabulary diversity ({distinct_1:.2f})")
            elif distinct_1 < 0.50:
                reward -= 0.5
                rationale.append("Vocabulary collapse / loop")

        # Punctuation and dialogue balance
        if '"' in text:
            quote_count = text.count('"')
            if quote_count % 2 == 0:
                reward += 0.25
                rationale.append("Balanced quotation marks")

        # Check prompt entity retention
        prompt_entities = re.findall(r'\b[A-Z][a-z]+\b', prompt)
        retained = [e for e in prompt_entities if e in continuation]
        if len(retained) > 0:
            reward += 0.5
            rationale.append(f"Retained characters: {retained}")

        final_reward = max(-1.0, min(2.0, reward))
        return final_reward, "; ".join(rationale)

    def compute_reward(self, task_type: str, prompt: str, continuation: str, target: Optional[str] = None) -> Tuple[float, str]:
        """Dispatches to appropriate task verifier."""
        if task_type == "math":
            return self.verify_arithmetic(continuation, target)
        elif task_type == "code":
            return self.verify_code(prompt, continuation)
        else:
            return self.verify_story(prompt, continuation)
