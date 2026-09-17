"""
experiments/navitrit/audit_judge_coherence.py: Comparative LLM-as-a-Judge Audit.

Rigorously evaluates the impact of LLM Judge GRPO alignment on NaviTrit:
Compares:
1. NaviTrit Hardened (2,500 steps, Attention Diversity Regularization)
2. NaviTrit Judge-Aligned (Trained via GRPO with Local TinyLlama-1.1B-Chat Judge)

Measures:
- Mean Rubric Score (1 to 10 on Narrative Logic, Commonsense, Grammar)
- Judge Perplexity & Conditional Cross-Entropy
- Head-to-Head Pairwise Win Rate (Judge blind pairwise choices: A vs B)
- Generated Story Continuations and Explanations
- Ledger: outputs/judge-alignment-audit.json
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from transformers import AutoTokenizer

from experiments.navitrit.navitrit_model import NaviTritConfig, NaviTritForCausalLM
from experiments.navitrit.llm_judge import LLMJudge

EVAL_PROMPTS = [
    "Once upon a time, there was a little dog named Spot.",
    "One sunny day, Lily and her brother found a",
    "Tim wanted to build a big tower with his",
    "The little bird looked out of the nest and saw",
    "Mia had a magical red paintbrush that could",
    "Sam the puppy went to the park and met",
    "Two little kittens found a mysterious basket of",
    "A happy frog sat by the cool blue pond and",
    "On a windy morning, Lucy's yellow kite flew up into",
    "The friendly dolphin swam up to the wooden boat and",
]


def generate_story(
    model: NaviTritForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 35,
    device: torch.device = torch.device("cuda:0"),
    temperature: float = 0.7,
    top_k: int = 40,
) -> Tuple[str, List[int]]:
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    curr_ids = input_ids.clone()
    trajectory_nodes = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(curr_ids)
            logits = out.logits[0, -1, :]
            trajectory_nodes.append(out.trajectory_nodes)

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
                    logits[indices_to_remove] = -float("Inf")
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            next_id = int(next_token.item())
            next_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
            curr_ids = torch.cat([curr_ids, next_tensor], dim=-1)

            if next_id == tokenizer.eos_token_id:
                break

    full_text = tokenizer.decode(curr_ids[0].tolist(), skip_special_tokens=True)
    cont_text = tokenizer.decode(curr_ids[0][input_ids.shape[1]:].tolist(), skip_special_tokens=True)
    return cont_text, trajectory_nodes


def run_audit(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"Comparative LLM-as-a-Judge Audit: Hardened vs Judge-Aligned ({device})")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = NaviTritConfig(
        vocab_size=50257,
        hidden_size=384,
        intermediate_size=1024,
        num_layers=3,
        num_attention_heads=6,
        max_position_embeddings=512,
        block_size=256,
        max_hops=6,
        d_nav_route=128,
        ternary=True,
    )

    def load_model(path: str) -> NaviTritForCausalLM:
        m = NaviTritForCausalLM(config).to(device)
        raw = torch.load(path, map_location=device, weights_only=True)
        sd = raw["state_dict"] if "state_dict" in raw else raw
        m.load_state_dict(sd)
        m.eval()
        return m

    # 1. Load Hardened Model
    p_hardened = args.hardened_ckpt
    print(f"[1/3] Loading Hardened Model from {p_hardened}...")
    model_hardened = load_model(p_hardened)

    # 2. Load Judge-Aligned Model
    p_aligned = args.aligned_ckpt
    print(f"[2/3] Loading Judge-Aligned Model from {p_aligned}...")
    model_aligned = load_model(p_aligned)

    # 3. Load LLM Judge
    print("\n[3/3] Initializing LLM Judge (TinyLlama-1.1B-Chat)...")
    judge = LLMJudge(device=device, torch_dtype=torch.float16)

    print(f"\n---> Auditing {len(EVAL_PROMPTS)} test prompts...")
    
    hardened_scores = []
    hardened_losses = []
    hardened_rewards = []

    aligned_scores = []
    aligned_losses = []
    aligned_rewards = []

    pairwise_wins = {"HARDENED": 0, "ALIGNED": 0, "TIE": 0}
    prompt_comparisons = []

    for idx, prompt in enumerate(EVAL_PROMPTS):
        print(f"\n[{idx+1}/{len(EVAL_PROMPTS)}] Prompt: '{prompt}'")

        # Generate from Hardened
        cont_hard, traj_hard = generate_story(model_hardened, tokenizer, prompt, max_new_tokens=args.max_new_tokens, device=device)
        eval_hard = judge.evaluate_candidate(prompt, cont_hard)

        # Generate from Aligned
        cont_align, traj_align = generate_story(model_aligned, tokenizer, prompt, max_new_tokens=args.max_new_tokens, device=device)
        eval_align = judge.evaluate_candidate(prompt, cont_align)

        # Head-to-head pairwise judgment
        pair_res = judge.compare_pair(prompt, cont_hard, cont_align)
        winner_tag = "HARDENED" if pair_res["winner"] == "A" else ("ALIGNED" if pair_res["winner"] == "B" else "TIE")
        pairwise_wins[winner_tag] += 1

        hardened_scores.append(eval_hard["score"])
        hardened_losses.append(eval_hard["cond_loss"])
        hardened_rewards.append(eval_hard["combined_reward"])

        aligned_scores.append(eval_align["score"])
        aligned_losses.append(eval_align["cond_loss"])
        aligned_rewards.append(eval_align["combined_reward"])

        print(f"  [Hardened] Score: {eval_hard['score']}/10 | Loss: {eval_hard['cond_loss']:.2f} | R: {eval_hard['combined_reward']:.3f}")
        print(f"             Output: \"{cont_hard.strip()[:80]}\"")
        print(f"  [Aligned]  Score: {eval_align['score']}/10 | Loss: {eval_align['cond_loss']:.2f} | R: {eval_align['combined_reward']:.3f}")
        print(f"             Output: \"{cont_align.strip()[:80]}\"")
        print(f"  --> Winner: {winner_tag} | Reason: {pair_res['reason']}")

        prompt_comparisons.append({
            "prompt_idx": idx + 1,
            "prompt": prompt,
            "hardened": {
                "continuation": cont_hard.strip(),
                "score": eval_hard["score"],
                "cond_loss": eval_hard["cond_loss"],
                "combined_reward": eval_hard["combined_reward"],
                "reason": eval_hard["reason"],
            },
            "aligned": {
                "continuation": cont_align.strip(),
                "score": eval_align["score"],
                "cond_loss": eval_align["cond_loss"],
                "combined_reward": eval_align["combined_reward"],
                "reason": eval_align["reason"],
            },
            "pairwise_winner": winner_tag,
            "judge_explanation": pair_res["reason"],
        })

    # Summary calculations
    mean_h_score = round(sum(hardened_scores) / len(hardened_scores), 2)
    mean_a_score = round(sum(aligned_scores) / len(aligned_scores), 2)
    mean_h_loss = round(sum(hardened_losses) / len(hardened_losses), 3)
    mean_a_loss = round(sum(aligned_losses) / len(aligned_losses), 3)
    mean_h_reward = round(sum(hardened_rewards) / len(hardened_rewards), 3)
    mean_a_reward = round(sum(aligned_rewards) / len(aligned_rewards), 3)
    
    total_valid = pairwise_wins["HARDENED"] + pairwise_wins["ALIGNED"]
    align_win_rate = round(pairwise_wins["ALIGNED"] / max(1, total_valid) * 100, 1)

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY & HEAD-TO-HEAD COMPARISON:")
    print("=" * 80)
    print(f"  Hardened NaviTrit:     Mean Score: {mean_h_score}/10 | Judge Loss: {mean_h_loss:.2f} | Reward: {mean_h_reward:.3f}")
    print(f"  Judge-Aligned NaviTrit: Mean Score: {mean_a_score}/10 | Judge Loss: {mean_a_loss:.2f} | Reward: {mean_a_reward:.3f}")
    print(f"  Score Improvement:     +{round(mean_a_score - mean_h_score, 2)} points")
    print(f"  Judge Loss Reduction:  {round(mean_a_loss - mean_h_loss, 3)} (lower is better)")
    print(f"  Head-to-Head Win Rate: Aligned Wins: {pairwise_wins['ALIGNED']} | Hardened Wins: {pairwise_wins['HARDENED']} | Ties: {pairwise_wins['TIE']}")
    print(f"  Aligned Win Rate:      {align_win_rate}%")

    results = {
        "status": "COMPLETED",
        "benchmark": "Head-to-Head LLM-as-a-Judge Audit (Hardened vs Judge-Aligned NaviTrit)",
        "judge_model": "TinyLlama-1.1B-Chat-v1.0",
        "evaluated_prompts_count": len(EVAL_PROMPTS),
        "hardened_metrics": {
            "mean_rubric_score": mean_h_score,
            "mean_judge_loss": mean_h_loss,
            "mean_combined_reward": mean_h_reward,
        },
        "aligned_metrics": {
            "mean_rubric_score": mean_a_score,
            "mean_judge_loss": mean_a_loss,
            "mean_combined_reward": mean_a_reward,
        },
        "pairwise_head_to_head": {
            "aligned_wins": pairwise_wins["ALIGNED"],
            "hardened_wins": pairwise_wins["HARDENED"],
            "ties": pairwise_wins["TIE"],
            "aligned_win_rate_percent": align_win_rate,
        },
        "detailed_prompts": prompt_comparisons,
    }

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full audit ledger saved to {args.json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardened-ckpt", type=str, default="outputs/checkpoints/navitrit-flow-hardened.pt")
    parser.add_argument("--aligned-ckpt", type=str, default="outputs/checkpoints/navitrit-judge-aligned.pt")
    parser.add_argument("--max-new-tokens", type=int, default=35)
    parser.add_argument("--json", type=str, default="outputs/judge-alignment-audit.json")
    args = parser.parse_args()

    run_audit(args)


if __name__ == "__main__":
    main()
