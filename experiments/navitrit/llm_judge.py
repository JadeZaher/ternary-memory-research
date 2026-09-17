"""
experiments/navitrit/llm_judge.py: Local LLM-as-a-Judge Engine.

Provides automated narrative coherence and fluency evaluation for NaviTrit:
1. Local TinyLlama-1.1B-Chat model running in FP16 on CUDA (zero external API calls).
2. Structured Rubric Evaluation:
     - Narrative Coherence & Flow
     - Common-Sense Realism & World Consistency
     - Grammatical Closure & Syntactic Validity
   Emits scalar score S in [1, 10] with reasoning.
3. Intrinsic Token Likelihood Reward:
     Computes conditional cross-entropy of continuation given the prompt.
4. Combined Multi-Signal Reward:
     R in [0, 1] balancing rubric score with statistical language fluency.
"""

import argparse
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMJudge:
    """
    Local LLM Judge wrapper for evaluating narrative coherence.
    Uses TinyLlama-1.1B-Chat-v1.0 running locally on CUDA.
    """
    def __init__(
        self,
        model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: Optional[torch.device] = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[LLMJudge] Loading {model_id} onto {self.device} ({torch_dtype})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map={"": self.device} if self.device.type == "cuda" else None,
            local_files_only=True,
        )
        self.model.eval()
        print(f"[LLMJudge] Successfully initialized on {self.device}.")

    def compute_conditional_loss(self, prompt: str, continuation: str) -> float:
        """
        Computes conditional cross-entropy loss of the continuation given the prompt.
        Lower loss indicates higher natural language fluency under the judge's prior.
        """
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)

        if len(cont_ids) == 0:
            return 10.0

        full_ids = torch.tensor([prompt_ids + cont_ids], device=self.device)
        prompt_len = len(prompt_ids)
        total_len = full_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits[0, prompt_len - 1 : total_len - 1, :]
            target_ids = full_ids[0, prompt_len:total_len]
            loss = F.cross_entropy(logits, target_ids)

        return float(loss.item())

    def score_rubric(self, prompt: str, continuation: str) -> Tuple[int, str]:
        """
        Queries the judge with a structured rubric prompt and extracts score S in [1, 10].
        """
        system_msg = (
            "You are an expert literary evaluator assessing children's story coherence.\n"
            "Score the story continuation on a scale of 1 to 10 for:\n"
            "1. Narrative Coherence: Does it make logical sense as a continuation?\n"
            "2. Grammar & Syntax: Are sentences grammatically sound and complete?\n"
            "3. Common-Sense Realism: Do characters and objects behave reasonably?\n"
            "Respond strictly in this format:\n"
            "SCORE: <integer 1 to 10>\n"
            "REASON: <one sentence>"
        )

        user_msg = (
            f'Story Beginning: "{prompt.strip()}"\n'
            f'Continuation: "{continuation.strip()}"'
        )

        chat = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_text = self.tokenizer.decode(output_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Regex extraction of SCORE: <int>
        score = 5
        reason = "Default extraction"
        match = re.search(r"SCORE:\s*(\d+)", gen_text, re.IGNORECASE)
        if match:
            try:
                raw_score = int(match.group(1))
                score = max(1, min(10, raw_score))
            except ValueError:
                score = 5
        else:
            # Fallback: search for any standalone integer 1-10
            int_match = re.search(r"\b([1-9]|10)\b", gen_text)
            if int_match:
                score = int(int_match.group(1))

        reason_match = re.search(r"REASON:\s*(.*)", gen_text, re.IGNORECASE)
        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            reason = gen_text.strip().replace("\n", " ")[:80]

        return score, reason

    def evaluate_candidate(
        self,
        prompt: str,
        continuation: str,
        alpha_rubric: float = 0.6,
    ) -> Dict:
        """
        Combines rubric evaluation score with conditional fluency reward.
        Returns:
            dict containing score, conditional_loss, combined_reward in [0, 1], and reason.
        """
        score, reason = self.score_rubric(prompt, continuation)
        cond_loss = self.compute_conditional_loss(prompt, continuation)

        # Normalize score from [1, 10] -> [0, 1]
        norm_score = (score - 1.0) / 9.0

        # Transform loss into fluency reward: exp(-loss / 3.0) in [0, 1]
        fluency_reward = float(torch.exp(torch.tensor(-min(10.0, max(1.0, cond_loss)) / 3.0)).item())

        # Combined multi-signal reward
        combined_reward = alpha_rubric * norm_score + (1.0 - alpha_rubric) * fluency_reward

        return {
            "score": score,
            "norm_score": round(norm_score, 4),
            "cond_loss": round(cond_loss, 4),
            "fluency_reward": round(fluency_reward, 4),
            "combined_reward": round(combined_reward, 4),
            "reason": reason,
        }

    def compare_pair(
        self,
        prompt: str,
        cont_a: str,
        cont_b: str,
    ) -> Dict:
        """
        Direct pairwise head-to-head comparison between two candidate continuations.
        """
        system_msg = (
            "You are an expert literary judge for children's stories.\n"
            "Compare Story A and Story B as continuations of the prompt.\n"
            "Choose the continuation that is more coherent, grammatical, and logical.\n"
            "Respond strictly in this format:\n"
            "WINNER: <A or B>\n"
            "REASON: <one sentence>"
        )

        user_msg = (
            f'Prompt: "{prompt.strip()}"\n\n'
            f'Story A: "{cont_a.strip()}"\n\n'
            f'Story B: "{cont_b.strip()}"'
        )

        chat = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_text = self.tokenizer.decode(output_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        winner = "TIE"
        match = re.search(r"WINNER:\s*([AB])", gen_text, re.IGNORECASE)
        if match:
            winner = match.group(1).upper()
        else:
            if "Story A" in gen_text or "winner is A" in gen_text.lower():
                winner = "A"
            elif "Story B" in gen_text or "winner is B" in gen_text.lower():
                winner = "B"

        reason_match = re.search(r"REASON:\s*(.*)", gen_text, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else gen_text.strip()[:80]

        return {
            "winner": winner,
            "reason": reason,
            "raw_judge_text": gen_text.strip(),
        }


def run_judge_test():
    """Self-contained test verifying LLMJudge on test sentences."""
    print("=" * 80)
    print("Testing LLMJudge (TinyLlama-1.1B-Chat-v1.0)...")
    print("=" * 80)

    judge = LLMJudge()

    prompt = "Once upon a time, there was a little dog named Spot."
    
    # Coherent story
    good_cont = " Spot loved to play with his friends in the sunny park. One day, he found a red ball."
    # Incoherent / non-sequitur continuation
    bad_cont = " tree wanted to eat the ball. Lily shouted dirt and jumped upside the shoe."

    print(f"\nPrompt: '{prompt}'")
    
    print("\n--- Evaluating Good Continuation ---")
    good_eval = judge.evaluate_candidate(prompt, good_cont)
    print(f"Good Cont:  \"{good_cont}\"")
    print(f"Score:      {good_eval['score']}/10 (Reward: {good_eval['combined_reward']:.4f}, Loss: {good_eval['cond_loss']:.2f})")
    print(f"Reason:     {good_eval['reason']}")

    print("\n--- Evaluating Bad Continuation ---")
    bad_eval = judge.evaluate_candidate(prompt, bad_cont)
    print(f"Bad Cont:   \"{bad_cont}\"")
    print(f"Score:      {bad_eval['score']}/10 (Reward: {bad_eval['combined_reward']:.4f}, Loss: {bad_eval['cond_loss']:.2f})")
    print(f"Reason:     {bad_eval['reason']}")

    assert good_eval["score"] >= bad_eval["score"], "Good continuation should score higher than bad continuation!"
    assert good_eval["combined_reward"] > bad_eval["combined_reward"], "Good reward must exceed bad reward!"

    print("\n--- Testing Pairwise Comparison ---")
    pair_res = judge.compare_pair(prompt, good_cont, bad_cont)
    print(f"Pairwise Winner: {pair_res['winner']} (Expected: A)")
    print(f"Judge Reason:    {pair_res['reason']}")
    assert pair_res["winner"] == "A", f"Expected winner 'A', got '{pair_res['winner']}'"

    print("\n[LLMJUDGE TEST PASSED SUCCESSFULLY]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", default=True)
    args = parser.parse_args()

    if args.test:
        run_judge_test()
