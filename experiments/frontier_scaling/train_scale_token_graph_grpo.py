"""
experiments/frontier_scaling/train_scale_token_graph_grpo.py: Token-Level Graph GRPO Training Engine (DTRNet).

Gate 18 (Track H) Implementation:
1. Warm-starts from Gate 16-C Tree GRPO Checkpoint (outputs/checkpoints/navitrit-100m-tree-grpo.pt).
2. 100% Freezes ternary language backbone (124.15M parameters: embed_tokens, attn_tiles, ffn_tiles, lm_head).
3. Trains Token-Level Routing & Gating Head (DTRNet) + Neural ODE Velocity + Collapse Operator.
4. Evaluates outcome rewards via HybridVerifier across Python Code, GSM8K Math, and TinyStories.
5. Optimizes Group Relative Policy Optimization (GRPO, K=4 rollouts) with KL divergence constraint.
6. Emits outputs/checkpoints/navitrit-100m-token-graph-grpo.pt and performance ledger.
"""

import os
import sys
import re
import math
import copy
import time
import random
import argparse
import json
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.frontier_scaling.navitrit_scale_model import get_scale_config
from experiments.frontier_scaling.navitrit_token_graph_model import (
    NaviTritTokenGraphForCausalLM,
    NaviTritScaleConfig,
)
from experiments.frontier_scaling.hybrid_verifier import HybridVerifier


def make_math_prompt() -> Dict[str, Any]:
    names = ["Olivia", "Liam", "Lily", "Tom", "Sam", "Emma", "Jack", "Leo", "Mia", "Noah"]
    items = ["apples", "pencils", "stickers", "cookies", "candies", "toy cars", "marbles"]
    n1, n2 = random.sample(names, 2)
    item = random.choice(items)
    
    start = random.randint(12, 40)
    loss = random.randint(3, 10)
    final = start - loss
    
    prompt = (
        f"Problem: {n1} had {start} {item}. {n1} gave {loss} {item} to {n2}. How many {item} remain?\n"
        f"Reasoning:"
    )
    return {
        "type": "math",
        "prompt": prompt,
        "target": str(final),
    }


def make_code_prompt() -> Dict[str, Any]:
    templates = [
        {
            "prompt": "def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n",
            "target": "while",
        },
        {
            "prompt": "def factorial(n):\n    if n <= 1:\n        return 1\n",
            "target": "return",
        },
        {
            "prompt": "def is_even(number):\n    return",
            "target": "number % 2 == 0",
        },
        {
            "prompt": "def get_length(s):\n    count = 0\n    for char in s:\n",
            "target": "count +=",
        },
    ]
    item = random.choice(templates)
    return {
        "type": "code",
        "prompt": item["prompt"],
        "target": item["target"],
    }


def make_story_prompt() -> Dict[str, Any]:
    templates = [
        {"prompt": "One sunny day, Lily and her brother found a big, red ball.", "target": "Lily"},
        {"prompt": "Once upon a time, there was a little dog named Spot.", "target": "Spot"},
        {"prompt": "Tim wanted to build a tall tower with his colorful blocks.", "target": "Tim"},
        {"prompt": "Mia had a magic paintbrush that made drawings come alive.", "target": "Mia"},
        {"prompt": "In a cozy kitchen, Oliver the cat was looking for some warm milk.", "target": "Oliver"},
    ]
    item = random.choice(templates)
    return {
        "type": "story",
        "prompt": item["prompt"],
        "target": item["target"],
    }


def sample_prompt(p_math: float = 0.40, p_code: float = 0.35) -> Dict[str, Any]:
    r = random.random()
    if r < p_math:
        return make_math_prompt()
    elif r < p_math + p_code:
        return make_code_prompt()
    else:
        return make_story_prompt()


def rollout_token_graph_continuation(
    model: NaviTritTokenGraphForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 25,
    temperature: float = 1.0,
    traversal_depth: int = 3,
    ref_controller: Optional[nn.Module] = None,
    bypass_thresh: float = 0.20,
) -> Dict[str, Any]:
    """Generates a rollout trajectory while tracking log-probs for token-graph policy gradients."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    curr_ids = input_ids.clone()

    # Step 1: Plan token trajectory across depth D
    history_pairs, history_gates, history_masks, traj_log_prob, slots, kl_loss, entropy = model.plan_token_trajectory(
        input_ids,
        temperature=temperature,
        traversal_depth=traversal_depth,
        ref_controller=ref_controller,
        return_diagnostics=True,
        bypass_thresh=bypass_thresh,
    )

    flat_nodes = []
    for p in history_pairs:
        flat_nodes.extend([p[0], p[1]])

    # Calculate attention bypass fraction
    total_masks = sum(m.sum().item() for m in history_masks)
    total_tokens = sum(m.numel() for m in history_masks)
    bypass_rate = total_masks / max(1, total_tokens)

    # Step 2: Autoregressive token generation along planned token trajectory
    for step in range(max_new_tokens):
        with torch.no_grad():
            logits = model.forward_token_trajectory(curr_ids, history_pairs, history_gates, history_masks, slots=slots)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id or curr_ids.shape[1] > 256:
                break

    continuation_text = tokenizer.decode(curr_ids[0][input_ids.size(1):])
    return {
        "continuation": continuation_text,
        "trajectory_log_prob": traj_log_prob.mean(),
        "tree_pairs": history_pairs,
        "sample_nodes": flat_nodes,
        "reasoning_visited": (model.node_reasoning in flat_nodes),
        "attention_bypass_rate": bypass_rate,
        "kl_loss": kl_loss,
        "entropy": entropy,
    }


def train_token_graph_grpo(
    model_size: str = "100m",
    steps: int = 300,
    rollouts_k: int = 4,
    traversal_depth: int = 3,
    lr: float = 0.0001,
    init_checkpoint: str = "outputs/checkpoints/navitrit-100m-tree-grpo.pt",
    output_checkpoint: str = "outputs/checkpoints/navitrit-100m-token-graph-grpo.pt",
    output_json: str = "outputs/navitrit-100m-token-graph-grpo-results.json",
    seed: int = 20260917,
):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"=== Starting Token Graph (DTRNet) GRPO Alignment for [{model_size.upper()}] on {device} ===")
    print(f"Depth: {traversal_depth} | Rollouts per Prompt: K={rollouts_k} | Total Steps: {steps}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    cfg = get_scale_config(model_size)
    model = NaviTritTokenGraphForCausalLM(cfg).to(device)

    # Warm start from Gate 16-C tree checkpoint or trained backbone
    if os.path.exists(init_checkpoint):
        model.load_from_pretrained_backbone(init_checkpoint, device)
    else:
        print(f"Warning: init_checkpoint {init_checkpoint} not found; initializing cold.")

    # 100% Freeze the ternary language backbone (preserves pre-trained representation)
    frozen_count = 0
    trainable_count = 0
    for name, p in model.named_parameters():
        if any(b in name for b in ["embed_tokens", "embed_positions", "attn_tiles", "attn_norms", "ffn_tiles", "ffn_norms", "lm_head", "final_norm", "hop_mod"]):
            p.requires_grad = False
            frozen_count += p.numel()
        else:
            p.requires_grad = True
            trainable_count += p.numel()

    print(f"Parameter Partition: {frozen_count:,} Frozen Backbone (100% Ternary Intact) | {trainable_count:,} Trainable Routing/Latent Parameters")

    # Create frozen reference controller for KL divergence regularization
    ref_controller = copy.deepcopy(model.token_controller)
    for p in ref_controller.parameters():
        p.requires_grad = False
    ref_controller.eval()

    verifier = HybridVerifier(use_gemini=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    start_time = time.time()
    history_rewards = []
    task_rewards: Dict[str, List[float]] = {"math": [], "code": [], "story": []}
    reasoning_visitation_history = []
    bypass_rate_history = []

    print("\n--- Running Token Graph GRPO Policy Optimization Loop ---")
    model.train()

    for step in range(1, steps + 1):
        prompt_info = sample_prompt()
        p_type = prompt_info["type"]
        prompt_text = prompt_info["prompt"]
        target = prompt_info["target"]

        rollout_records = []
        rewards = []

        # Temperature annealing: 1.0 down to 0.7
        temp = max(0.7, 1.0 - 0.3 * (step / steps))

        # Sample K rollouts
        for k in range(rollouts_k):
            res = rollout_token_graph_continuation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                device=device,
                max_new_tokens=25,
                temperature=temp,
                traversal_depth=traversal_depth,
                ref_controller=ref_controller,
                bypass_thresh=0.25,
            )
            reward, rationale = verifier.compute_reward(
                p_type, prompt_text, res["continuation"], target
            )
            # DTRNet Efficiency Reward: reward policies that maintain accuracy while maximizing attention bypass
            if reward > 0.0:
                reward = reward + 0.20 * res["attention_bypass_rate"]

            rewards.append(reward)
            res["reward"] = reward
            res["rationale"] = rationale
            rollout_records.append(res)

        # Group Relative Advantages
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        mean_r = rewards_tensor.mean().item()
        std_r = rewards_tensor.std().item() + 1e-6
        advantages = (rewards_tensor - mean_r) / std_r

        # GRPO Policy Loss
        policy_loss = torch.tensor(0.0, device=device)
        for k in range(rollouts_k):
            adv_k = advantages[k].detach()
            log_prob_k = rollout_records[k]["trajectory_log_prob"]
            policy_loss = policy_loss - adv_k * log_prob_k

        policy_loss = policy_loss / rollouts_k

        # Regularization: KL to reference policy and entropy bonus
        mean_kl = torch.stack([r["kl_loss"] for r in rollout_records]).mean()
        mean_entropy = torch.stack([r["entropy"] for r in rollout_records]).mean()

        total_loss = policy_loss + 0.10 * mean_kl - 0.01 * mean_entropy

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
        optimizer.step()

        # Telemetry
        history_rewards.append(mean_r)
        task_rewards[p_type].append(mean_r)
        any_reasoning = any(r["reasoning_visited"] for r in rollout_records)
        reasoning_visitation_history.append(1 if any_reasoning else 0)
        avg_bypass = sum(r["attention_bypass_rate"] for r in rollout_records) / len(rollout_records)
        bypass_rate_history.append(avg_bypass)

        if step % 50 == 0 or step == steps:
            elapsed = time.time() - start_time
            recent_r = sum(history_rewards[-50:]) / min(50, len(history_rewards))
            recent_reason = sum(reasoning_visitation_history[-50:]) / min(50, len(reasoning_visitation_history))
            recent_bypass = sum(bypass_rate_history[-50:]) / min(50, len(bypass_rate_history))
            best_sample = max(rollout_records, key=lambda x: x["reward"])
            print(
                f"  Step {step}/{steps} ({elapsed:.1f}s) | Task: {p_type.upper():5s} | "
                f"Reward Mean: {recent_r:+.2f} | Attn Bypass: {recent_bypass*100:.1f}% | "
                f"Node 24 Rate: {recent_reason*100:.1f}% | Best R: {best_sample['reward']:+.2f}"
            )
            if p_type == "math":
                print(f"    Math Continuation: {best_sample['continuation'][:75]}...")
            elif p_type == "code":
                print(f"    Code Continuation: {repr(best_sample['continuation'][:60])}...")

    total_time = time.time() - start_time
    print(f"\nToken Graph GRPO Alignment finished in {total_time:.2f}s ({total_time/60:.2f} min).")

    # Save aligned checkpoint
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint)
    print(f"Token Graph Checkpoint saved to: {output_checkpoint}")

    # Compile results
    results = {
        "model_size": model_size,
        "steps": steps,
        "rollouts_k": rollouts_k,
        "traversal_depth": traversal_depth,
        "elapsed_seconds": total_time,
        "mean_final_reward": sum(history_rewards[-100:]) / min(100, len(history_rewards)),
        "task_rewards": {k: sum(v[-50:]) / max(1, len(v[-50:])) for k, v in task_rewards.items() if len(v) > 0},
        "reasoning_core_visitation_rate": sum(reasoning_visitation_history[-100:]) / min(100, len(reasoning_visitation_history)),
        "mean_attention_bypass_rate": sum(bypass_rate_history[-100:]) / min(100, len(bypass_rate_history)),
        "frozen_backbone_parameters": frozen_count,
        "trainable_router_parameters": trainable_count,
        "output_checkpoint": output_checkpoint,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Training summary saved to: {output_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NaviTrit Token Graph GRPO Training")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--init-checkpoint", type=str, default="outputs/checkpoints/navitrit-100m-tree-grpo.pt")
    parser.add_argument("--output-checkpoint", type=str, default="outputs/checkpoints/navitrit-100m-token-graph-grpo.pt")
    parser.add_argument("--output-json", type=str, default="outputs/navitrit-100m-token-graph-grpo-results.json")
    args = parser.parse_args()

    train_token_graph_grpo(
        steps=args.steps,
        rollouts_k=args.rollouts,
        traversal_depth=args.depth,
        lr=args.lr,
        init_checkpoint=args.init_checkpoint,
        output_checkpoint=args.output_checkpoint,
        output_json=args.output_json,
    )
