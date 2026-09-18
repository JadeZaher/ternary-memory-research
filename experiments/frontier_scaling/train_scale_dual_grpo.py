"""
experiments/frontier_scaling/train_scale_dual_grpo.py: GRPO Training Engine for Hierarchical Dual-Controller NaviTrit-100M.

Combines Strategy B (Outcome-Driven GRPO) with Strategy D (Hierarchical Flow Planner + Local Dynamic Navigation):
1. Loads pretrained 100M backbone (outputs/checkpoints/navitrit-100m-trained.pt).
2. 100% Freezes ternary language backbone (embed_tokens, attn_tiles, ffn_tiles, lm_head) to prevent representation drift.
3. Anchors navigation policy to frozen reference controller pi_ref via KL divergence and entropy bonus.
4. Directly supervises Reasoning Core (Node 24) on arithmetic deduction pairs.
5. Generates K=4 rollouts per prompt across Math, Code, and Narrative tasks.
6. Evaluates outcome rewards via HybridVerifier (checking exact arithmetic, Python AST, and character retention).
7. Computes Group Relative Advantage A_k = (R_k - mean(R)) / (std(R) + eps).
8. Updates GlobalFlowPlanner, LocalFlowController, PersistentEntityRegisters, and ReasoningCore.
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
from experiments.frontier_scaling.navitrit_dual_model import (
    NaviTritDualForCausalLM,
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
    ground_truth_target = str(final)
    ground_truth_text = (
        f" Let's track the count step by step.\n"
        f"Step 1: {n1} begins with {start} {item}.\n"
        f"Step 2: After gave {loss} to {n2}, {start} - {loss} = {final} {item} remain."
    )
    return {
        "type": "math",
        "prompt": prompt,
        "target": ground_truth_target,
        "full_text": prompt + ground_truth_text,
    }


def make_code_prompt() -> Dict[str, Any]:
    templates = [
        {
            "prompt": "def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n",
            "target": "while",
            "full_text": "def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n",
        },
        {
            "prompt": "def factorial(n):\n    if n <= 1:\n        return 1\n",
            "target": "return",
            "full_text": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n",
        },
        {
            "prompt": "def is_even(number):\n    return",
            "target": "number % 2 == 0",
            "full_text": "def is_even(number):\n    return number % 2 == 0\n",
        },
    ]
    item = random.choice(templates)
    return {
        "type": "code",
        "prompt": item["prompt"],
        "target": item["target"],
        "full_text": item["full_text"],
    }


def make_story_prompt() -> Dict[str, Any]:
    templates = [
        {"prompt": "One sunny day, Lily and her brother found a big, red ball.", "target": "Lily"},
        {"prompt": "Once upon a time, there was a little dog named Spot.", "target": "Spot"},
        {"prompt": "Tim wanted to build a tall tower with his colorful blocks.", "target": "Tim"},
        {"prompt": "Mia had a magic paintbrush that made drawings come alive.", "target": "Mia"},
    ]
    item = random.choice(templates)
    return {
        "type": "story",
        "prompt": item["prompt"],
        "target": item["target"],
        "full_text": item["prompt"] + " They played happily and took care of their toys.",
    }


def sample_prompt(p_math: float = 0.50, p_code: float = 0.25) -> Dict[str, Any]:
    r = random.random()
    if r < p_math:
        return make_math_prompt()
    elif r < p_math + p_code:
        return make_code_prompt()
    else:
        return make_story_prompt()


def rollout_continuation(
    model: NaviTritDualForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 30,
    temperature: float = 1.0,
    enforce_reasoning: bool = False,
    ref_controller: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Generates a rollout trajectory while tracking log-probs for policy gradients."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    curr_ids = input_ids.clone()
    
    total_trajectory_log_prob = torch.tensor(0.0, device=device)
    nodes_visited = set()
    sample_trajectory = []
    
    # Step 1: Plan global trajectory and record log-probability for policy gradient
    is_math = ("Problem:" in prompt or "Reasoning:" in prompt) or enforce_reasoning
    trajectory, trajectory_log_prob, slots, kl_loss, entropy = model.plan_trajectory(
        input_ids,
        temperature=temperature,
        is_math=is_math,
        ref_controller=ref_controller,
        return_diagnostics=True,
    )

    # Step 2: Fast autoregressive token generation along planned trajectory
    for step in range(max_new_tokens):
        with torch.no_grad():
            logits = model.forward_trajectory(curr_ids, trajectory, slots=slots)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id or curr_ids.shape[1] > 256:
                break

    continuation_text = tokenizer.decode(curr_ids[0][input_ids.size(1):])
    return {
        "continuation": continuation_text,
        "trajectory_log_prob": trajectory_log_prob.mean(),
        "sample_trajectory": trajectory,
        "nodes_visited": set(trajectory),
        "reasoning_visited": (model.node_reasoning in trajectory),
        "kl_loss": kl_loss,
        "entropy": entropy,
    }


def train_grpo(
    model_size: str = "100m",
    steps: int = 600,
    rollouts_k: int = 4,
    lr_router: float = 0.0005,
    lr_reasoning: float = 0.0005,
    init_checkpoint: str = "outputs/checkpoints/navitrit-100m-trained.pt",
    output_checkpoint: str = "outputs/checkpoints/navitrit-100m-dual-grpo.pt",
    output_json: str = "outputs/navitrit-100m-dual-grpo-results.json",
    seed: int = 20260917,
):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"=== Starting GRPO Alignment for NaviTrit-Dual [{model_size.upper()}] on {device} ===")
    print(f"Rollouts per Prompt: K={rollouts_k} | Total Steps: {steps}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    cfg = get_scale_config(model_size)
    model = NaviTritDualForCausalLM(cfg).to(device)

    # Warm start from pretrained 100M checkpoint
    if os.path.exists(init_checkpoint):
        model.load_from_pretrained_backbone(init_checkpoint, device)
    else:
        print(f"Warning: init_checkpoint {init_checkpoint} not found; initializing cold.")

    # 100% Freeze the ternary language backbone (preserves 8.5/10 code and grammar fluency)
    frozen_count = 0
    for name, p in model.named_parameters():
        if any(b in name for b in ["embed_tokens", "embed_positions", "attn_tiles", "attn_norms", "ffn_tiles", "ffn_norms", "lm_head", "final_norm", "hop_mod"]):
            p.requires_grad = False
            frozen_count += p.numel()
    print(f"Backbone Frozen: {frozen_count / 1e6:.2f}M parameters permanently locked.")

    # Create frozen reference controller for KL divergence regularization
    ref_controller = copy.deepcopy(model.controller)
    for p in ref_controller.parameters():
        p.requires_grad = False
    ref_controller.eval()

    verifier = HybridVerifier(use_gemini=False)

    # Decoupled optimizers:
    # 1. Router optimizer (Global Planner, Local Controller, Entity Registers)
    router_params = (
        list(model.global_planner.parameters())
        + list(model.controller.parameters())
        + list(model.entity_registers.parameters())
    )
    optimizer_router = torch.optim.AdamW(router_params, lr=lr_router, weight_decay=0.01)

    # 2. Reasoning Core optimizer (Node 24)
    reasoning_params = list(model.reasoning_core.parameters())
    optimizer_reasoning = torch.optim.AdamW(reasoning_params, lr=lr_reasoning, weight_decay=0.01)

    start_time = time.time()
    history_rewards = []
    reasoning_visitations_history = []
    
    print("\n--- Running GRPO Policy Alignment Loop ---")
    model.train()

    for step in range(1, steps + 1):
        prompt_info = sample_prompt()
        p_type = prompt_info["type"]
        prompt_text = prompt_info["prompt"]
        target = prompt_info["target"]

        rollout_records = []
        rewards = []

        # Temperature schedule: start at 1.0, anneal to 0.7
        temp = max(0.7, 1.0 - 0.3 * (step / steps))

        # Sample K rollouts
        for k in range(rollouts_k):
            # For math problems, test with and without prior reasoning enforcement
            enforce = (p_type == "math" and k >= 2 and step < steps // 2)
            res = rollout_continuation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                device=device,
                max_new_tokens=25,
                temperature=temp,
                enforce_reasoning=enforce,
                ref_controller=ref_controller,
            )
            reward, rationale = verifier.compute_reward(
                p_type, prompt_text, res["continuation"], target
            )
            rewards.append(reward)
            res["reward"] = reward
            res["rationale"] = rationale
            rollout_records.append(res)

        # Compute Group Relative Advantages
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        mean_r = rewards_tensor.mean().item()
        std_r = rewards_tensor.std().item() + 1e-6
        advantages = (rewards_tensor - mean_r) / std_r

        # Compute GRPO Policy Loss
        policy_loss = torch.tensor(0.0, device=device)
        for k in range(rollouts_k):
            adv_k = advantages[k].detach()
            log_prob_k = rollout_records[k]["trajectory_log_prob"]
            policy_loss = policy_loss - adv_k * log_prob_k

        policy_loss = policy_loss / rollouts_k

        # KL divergence and entropy regularization from rollouts
        mean_kl = torch.stack([r["kl_loss"] for r in rollout_records]).mean()
        mean_entropy = torch.stack([r["entropy"] for r in rollout_records]).mean()

        total_router_loss = policy_loss + 0.05 * mean_kl - 0.02 * mean_entropy

        optimizer_router.zero_grad()
        total_router_loss.backward()
        torch.nn.utils.clip_grad_norm_(router_params, 1.0)
        optimizer_router.step()

        # Direct arithmetic supervision for Node 24 (Reasoning Core)
        if p_type == "math":
            enc_full = tokenizer.encode(prompt_info["full_text"], return_tensors="pt").to(device)
            if enc_full.shape[1] > 128:
                enc_full = enc_full[:, :128]

            # Route through reasoning trajectory: Attention -> Reasoning Core -> FFN
            math_traj = [0, 1, 10, 11, model.node_reasoning, 18, 19]
            logits_math = model.forward_trajectory(enc_full[:, :-1], math_traj)
            labels_math = enc_full[:, 1:].contiguous()

            # Supervise specifically on the reasoning/solution tokens
            prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
            prompt_len = prompt_ids.shape[1]

            loss_mask = torch.zeros_like(labels_math, dtype=torch.bool)
            if prompt_len - 1 < labels_math.shape[1]:
                loss_mask[:, prompt_len - 1:] = True

            if loss_mask.sum() > 0:
                loss_arith = F.cross_entropy(logits_math[loss_mask], labels_math[loss_mask])
            else:
                loss_arith = torch.tensor(0.0, device=device)

            # Contraction loss for Reasoning Core
            h_sample = model.embed_tokens(enc_full)
            _, core_fpf = model.execute_tile(model.node_reasoning, h_sample, t=2)
            loss_reasoning = loss_arith + 0.1 * core_fpf

            optimizer_reasoning.zero_grad()
            loss_reasoning.backward()
            torch.nn.utils.clip_grad_norm_(reasoning_params, 1.0)
            optimizer_reasoning.step()

        # Telemetry tracking
        any_reasoning = any(r["reasoning_visited"] for r in rollout_records)
        history_rewards.append(mean_r)
        reasoning_visitations_history.append(1 if any_reasoning else 0)

        if step % 50 == 0 or step == steps:
            elapsed = time.time() - start_time
            recent_r = sum(history_rewards[-50:]) / min(50, len(history_rewards))
            recent_reason = sum(reasoning_visitations_history[-50:]) / min(50, len(reasoning_visitations_history))
            best_sample = max(rollout_records, key=lambda x: x["reward"])
            print(
                f"  Step {step}/{steps} ({elapsed:.1f}s) | Task: {p_type.upper():5s} | "
                f"Reward Mean: {recent_r:+.2f} | Node 24 Rate: {recent_reason*100:.1f}% | "
                f"Traj: {best_sample['sample_trajectory']} | Best R: {best_sample['reward']:+.2f}"
            )
            if p_type == "math":
                print(f"    Math Continuation: {best_sample['continuation'][:75]}...")

    total_time = time.time() - start_time
    print(f"\nGRPO Alignment finished in {total_time:.2f}s ({total_time/60:.2f} min).")

    # Save Checkpoint
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint)
    print(f"Saved aligned checkpoint to {output_checkpoint}")

    # Compile Summary
    summary = {
        "status": "COMPLETED",
        "gate": "Gate 16-B (Dual-Controller GRPO Alignment)",
        "model_size": model_size,
        "total_parameters": model.count_parameters(),
        "total_steps": steps,
        "rollouts_k": rollouts_k,
        "wall_time_s": round(total_time, 2),
        "mean_final_reward": round(sum(history_rewards[-100:]) / 100.0, 3),
        "node_24_visitation_rate": round(sum(reasoning_visitations_history[-100:]) / 100.0, 3),
        "checkpoints": {
            "init_checkpoint": init_checkpoint,
            "output_checkpoint": output_checkpoint,
        },
    }
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Results written to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Alignment for NaviTrit-Dual")
    parser.add_argument("--model_size", type=str, default="100m", choices=["10m", "100m"])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--lr_router", type=float, default=0.0005)
    parser.add_argument("--lr_reasoning", type=float, default=0.0005)
    parser.add_argument("--init_checkpoint", type=str, default="outputs/checkpoints/navitrit-100m-trained.pt")
    parser.add_argument("--save_checkpoint", type=str, default="outputs/checkpoints/navitrit-100m-dual-grpo.pt")
    parser.add_argument("--json", type=str, default="outputs/navitrit-100m-dual-grpo-results.json")
    args = parser.parse_args()

    train_grpo(
        model_size=args.model_size,
        steps=args.steps,
        rollouts_k=args.rollouts,
        lr_router=args.lr_router,
        lr_reasoning=args.lr_reasoning,
        init_checkpoint=args.init_checkpoint,
        output_checkpoint=args.save_checkpoint,
        output_json=args.json,
    )
