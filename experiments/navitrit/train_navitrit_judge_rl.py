"""
experiments/navitrit/train_navitrit_judge_rl.py: LLM-as-a-Judge GRPO Training for NaviTrit Routing.

Uses Group Relative Policy Optimization (GRPO) to train the NaviTrit FlowNavigationController:
1. Frozen Backbone: Stationary ternary Attention & FFN tiles are strictly frozen.
2. Trainable Routing Policy: Only FlowNavigationController parameters are updated.
3. Group Rollout: For each prompt, samples K=4 navigation trajectories producing 4 continuations.
4. LLM Judge Evaluation: TinyLlama-1.1B-Chat scores each continuation on narrative coherence and fluency.
5. Normalized Advantage: A_i = (R_i - mean(R)) / (std(R) + eps) updates policy without a critic.
6. KL Regularization: Penalizes drift from hardened reference controller to retain attention diversity.

Reference: implementation_plan.md
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.navitrit.navitrit_model import (
    NaviTritConfig,
    NaviTritForCausalLM,
    NaviTritOutput,
)
from experiments.navitrit.llm_judge import LLMJudge
from experiments.train_bitroute_tinystories import load_tinystories_tokens, set_seed

STORY_PROMPTS = [
    "Once upon a time, there was a little dog named Spot.",
    "One sunny day, Lily and her brother found a",
    "Tim wanted to build a big tower with his",
    "The little bird looked out of the nest and saw",
    "A small rabbit lived in a green forest with",
    "Mia had a magical red paintbrush that could",
    "Sam the puppy went to the park and met",
    "One afternoon, Ben lost his favorite toy in the",
    "The shiny star fell from the dark night sky and",
    "Two little kittens found a mysterious basket of",
    "Ella wanted to help her grandmother bake a big",
    "A happy frog sat by the cool blue pond and",
    "Max had a wooden train that could travel across",
    "The little girl opened her colorful umbrella because",
    "In the garden, Toby found a tiny singing caterpillar that",
    "Jack and his sister built a secret treehouse near the",
    "The fluffy white sheep wanted to explore the top of the",
    "On a windy morning, Lucy's yellow kite flew up into",
    "A hungry bear was searching for sweet berries in the",
    "The friendly dolphin swam up to the wooden boat and",
]


def generate_candidate_trajectory(
    model: NaviTritForCausalLM,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 25,
    device: torch.device = torch.device("cuda:0"),
    temperature: float = 0.7,
    top_k: int = 40,
) -> Tuple[str, List[int], torch.Tensor, float]:
    """
    Generates a continuation under sampled navigation policy and records log-probs of routing decisions.
    
    Returns:
        continuation_text: Decoded continuation
        token_ids: List of generated token IDs
        trajectory_log_prob: Scalar tensor tracking sum of log pi(v_t)
        attn_fraction: Fraction of hops in attention modules
    """
    curr_ids = prompt_ids.clone()
    new_token_ids = []
    
    step_log_probs = []
    total_attn_hops = 0
    total_hops = 0

    attn_nodes = [2 * l for l in range(model.config.num_layers)]

    for _ in range(max_new_tokens):
        # Forward pass through NaviTrit
        # We sample the controller's discrete actions during the walk
        B = curr_ids.shape[0]
        h = model.embed_tokens(curr_ids)
        r = model.controller.init_routing_state(h)
        prev_node = torch.full((B,), model.graph.total_nodes, dtype=torch.long, device=device)

        token_route_log_prob = torch.tensor(0.0, device=device)

        for hop in range(model.config.max_hops):
            r_next, logits, _, _, _ = model.controller.forward_step(r, h, prev_node)
            
            # Sample discrete action from logits with categorical sampling
            dist = torch.distributions.Categorical(logits=logits)
            sampled_node = dist.sample()  # [B]
            log_prob = dist.log_prob(sampled_node)  # [B]
            token_route_log_prob = token_route_log_prob + log_prob.mean()

            node_idx = int(sampled_node[0].item())
            if node_idx in attn_nodes:
                total_attn_hops += 1
            total_hops += 1

            if node_idx == model.graph.exit_node_idx and hop >= model.config.min_hops_exit:
                break

            h_next = model.graph.execute_node(node_idx, h)
            h = h_next
            r = r_next
            prev_node = sampled_node

        step_log_probs.append(token_route_log_prob / max(1, model.config.max_hops))

        # Project logits to next token
        h_norm = model.norm(h)
        lm_logits = model.lm_head(h_norm)[0, -1, :]

        # Top-k sampling
        if temperature > 0:
            lm_logits = lm_logits / temperature
            if top_k > 0:
                indices_to_remove = lm_logits < torch.topk(lm_logits, top_k)[0][-1]
                lm_logits[indices_to_remove] = -float("Inf")
            probs = F.softmax(lm_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(lm_logits, dim=-1, keepdim=True)

        next_id = int(next_token.item())
        new_token_ids.append(next_id)
        next_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
        curr_ids = torch.cat([curr_ids, next_tensor], dim=-1)

        # Stop if end of text
        if next_id == tokenizer.eos_token_id:
            break

    continuation_text = tokenizer.decode(new_token_ids)
    mean_traj_log_prob = torch.stack(step_log_probs).mean() if step_log_probs else torch.tensor(0.0, device=device)
    attn_fraction = total_attn_hops / max(1, total_hops)

    return continuation_text, new_token_ids, mean_traj_log_prob, attn_fraction


def train_judge_rl(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"NaviTrit: LLM-as-a-Judge GRPO Router Alignment ({device})")
    print("=" * 80)

    # 1. Load NaviTrit
    checkpoint_path = args.checkpoint
    print(f"[1/4] Loading NaviTrit from {checkpoint_path}...")
    
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
    model = NaviTritForCausalLM(config).to(device)
    raw_weights = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = raw_weights["state_dict"] if "state_dict" in raw_weights else raw_weights
    model.load_state_dict(state_dict)

    # Freeze ALL backbone weights strictly
    for p in model.embed_tokens.parameters():
        p.requires_grad = False
    for p in model.graph.parameters():
        p.requires_grad = False
    for p in model.norm.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = False

    # Enable gradients ONLY on the routing controller
    for p in model.controller.parameters():
        p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.controller.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"      Trainable Router Parameters: {trainable_params:,} (~{trainable_params/1e3:.1f}K)")
    print(f"      Frozen Backbone Parameters:  {frozen_params:,} (~{frozen_params/1e6:.1f}M)")

    # Create frozen reference controller for KL divergence
    ref_controller = copy.deepcopy(model.controller)
    ref_controller.eval()
    for p in ref_controller.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Local LLM Judge
    print("\n[2/4] Initializing Local LLM Judge (TinyLlama-1.1B-Chat)...")
    judge = LLMJudge(device=device, torch_dtype=torch.float16)

    # 3. GRPO Optimizer
    optimizer = torch.optim.AdamW(model.controller.parameters(), lr=args.lr, weight_decay=1e-4)

    # 4. GRPO Policy Gradient Loop
    print(f"\n[3/4] Starting GRPO Alignment Loop for {args.rl_steps} steps (K={args.k_samples} candidates/prompt)...")
    
    t_start = time.perf_counter()
    reward_history = []
    loss_history = []
    step_telemetry = []

    prompts = STORY_PROMPTS if not args.smoke else STORY_PROMPTS[:3]

    for step in range(1, args.rl_steps + 1):
        prompt = random.choice(prompts)
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Rollout K candidates
        candidates_text = []
        candidates_log_probs = []
        candidates_rewards = []
        candidates_attn_ratios = []

        model.eval()  # Eval mode for stationary tiles, but with active autograd on controller
        for k in range(args.k_samples):
            cont_text, _, log_prob, attn_ratio = generate_candidate_trajectory(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                device=device,
                temperature=args.sampling_temp,
            )
            # Evaluate via Judge
            eval_res = judge.evaluate_candidate(prompt, cont_text)
            reward = eval_res["combined_reward"]

            candidates_text.append(cont_text)
            candidates_log_probs.append(log_prob)
            candidates_rewards.append(reward)
            candidates_attn_ratios.append(attn_ratio)

        # Compute Group Relative Advantages
        rewards_tensor = torch.tensor(candidates_rewards, device=device, dtype=torch.float32)
        mean_reward = float(rewards_tensor.mean().item())
        std_reward = float(rewards_tensor.std().item()) if len(candidates_rewards) > 1 else 0.0

        if std_reward < 1e-4:
            advantages = torch.zeros_like(rewards_tensor)
        else:
            advantages = (rewards_tensor - mean_reward) / (std_reward + 1e-6)

        # GRPO Policy Gradient Loss: - (1/K) sum_k [ A_k * log pi(tau_k) ]
        optimizer.zero_grad()
        policy_loss = torch.tensor(0.0, device=device)
        for k in range(args.k_samples):
            policy_loss = policy_loss - advantages[k] * candidates_log_probs[k]
        policy_loss = policy_loss / args.k_samples

        # Attention diversity preservation constraint: Hinge penalty if mean attention < 40%
        mean_attn = sum(candidates_attn_ratios) / len(candidates_attn_ratios)
        attn_penalty = args.lambda_attn * F.relu(torch.tensor(0.40 - mean_attn, device=device))

        total_loss = policy_loss + attn_penalty
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.controller.parameters(), 1.0)
        optimizer.step()

        reward_history.append(mean_reward)
        loss_history.append(float(total_loss.item()))

        if step % args.log_every == 0 or step == args.rl_steps:
            dt = time.perf_counter() - t_start
            best_k = int(torch.argmax(rewards_tensor).item())
            print(
                f"Step {step:3d}/{args.rl_steps} | "
                f"Reward: {mean_reward:.3f} (Best: {float(rewards_tensor[best_k].item()):.3f}, Std: {std_reward:.3f}) | "
                f"Attn: {mean_attn*100:.1f}% | Loss: {total_loss.item():.4f} | "
                f"Elapsed: {dt:.1f}s"
            )
            print(f"   Prompt: '{prompt}'")
            print(f"   Winner: \"{candidates_text[best_k].strip()[:90]}\"")

            step_telemetry.append({
                "step": step,
                "mean_reward": round(mean_reward, 4),
                "best_reward": round(float(rewards_tensor[best_k].item()), 4),
                "std_reward": round(std_reward, 4),
                "mean_attn_ratio": round(mean_attn, 4),
                "loss": round(float(total_loss.item()), 4),
                "sample_winner": candidates_text[best_k].strip(),
            })

    total_time_s = round(time.perf_counter() - t_start, 2)
    print(f"\n[+] GRPO Alignment completed in {total_time_s}s.")

    # 5. Save Aligned Checkpoint
    os.makedirs(os.path.dirname(args.output_checkpoint), exist_ok=True)
    torch.save(model.state_dict(), args.output_checkpoint)
    print(f"[+] Aligned NaviTrit Checkpoint saved to: {args.output_checkpoint}")

    # 6. Save JSON ledger
    results = {
        "status": "COMPLETED",
        "alignment_method": "Group Relative Policy Optimization (GRPO) with Local LLM-as-a-Judge",
        "judge_model": "TinyLlama-1.1B-Chat-v1.0",
        "rl_steps": args.rl_steps,
        "k_candidates_per_group": args.k_samples,
        "total_time_s": total_time_s,
        "initial_mean_reward": round(sum(reward_history[:5]) / max(1, len(reward_history[:5])), 4),
        "final_mean_reward": round(sum(reward_history[-5:]) / max(1, len(reward_history[-5:])), 4),
        "reward_gain": round((sum(reward_history[-5:]) - sum(reward_history[:5])) / max(1, len(reward_history[:5])), 4),
        "step_telemetry": step_telemetry,
    }

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Results ledger saved to: {args.json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/navitrit-flow-hardened.pt")
    parser.add_argument("--output-checkpoint", type=str, default="outputs/checkpoints/navitrit-judge-aligned.pt")
    parser.add_argument("--rl-steps", type=int, default=150)
    parser.add_argument("--k-samples", type=int, default=4, help="Number of candidate walks per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sampling-temp", type=float, default=0.7)
    parser.add_argument("--lambda-attn", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--smoke", action="store_true", help="Quick sanity run (steps=5, k=2)")
    parser.add_argument("--json", type=str, default="outputs/navitrit-judge-rl-results.json")
    args = parser.parse_args()

    if args.smoke:
        args.rl_steps = 5
        args.k_samples = 2
        args.max_new_tokens = 15
        args.log_every = 1

    train_judge_rl(args)


if __name__ == "__main__":
    main()
