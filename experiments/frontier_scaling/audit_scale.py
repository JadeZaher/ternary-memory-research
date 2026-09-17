"""
experiments/frontier_scaling/audit_scale.py: Multi-Corpus Generation & Routing Audit for NaviTrit-10M.

Evaluates:
1. Narrative story prompts (Lily and the red ball, Spot the dog).
2. Multi-step arithmetic reasoning word problems (GSM8K style).
3. Algorithmic Python code completion.
4. Trajectory analysis: Does the model route to the Reasoning Core more frequently on math/code than on stories?
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.frontier_scaling.navitrit_scale_model import (
    NaviTritScaleConfig,
    NaviTritScaleForCausalLM,
    get_scale_config,
)


def audit_generation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(os.path.dirname(__file__), "../../outputs/checkpoints/navitrit-10m-trained.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found yet.")
        return

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    cfg = get_scale_config("10m")
    model = NaviTritScaleForCausalLM(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    prompts = [
        {"type": "story", "prompt": "One sunny day, Lily and her brother found a big, red ball."},
        {"type": "story", "prompt": "Once upon a time, there was a little dog named Spot."},
        {"type": "math", "prompt": "Problem: Olivia had 15 apples. She gave 4 apples to Liam. How many apples remain?\nReasoning:"},
        {"type": "code", "prompt": "def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n"},
    ]

    print("\n=== NaviTrit-10M Multi-Corpus Generation & Routing Audit ===")
    results = []

    for item in prompts:
        prompt_text = item["prompt"]
        prompt_type = item["type"]
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

        curr_ids = input_ids.clone()
        trajectory_log = []

        with torch.no_grad():
            for _ in range(35):
                # Run model
                out = model(curr_ids, temperature=0.7)
                logits = out["logits"][:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_token], dim=-1)
                trajectory_log.append(out["trajectory"])

        continuation = tokenizer.decode(curr_ids[0][input_ids.size(1):])
        print(f"\n[{prompt_type.upper()}] Prompt: {prompt_text}")
        print(f"Continuation: {continuation}")
        print(f"Sample Hops Trajectory: {trajectory_log[0]}")

        results.append({
            "type": prompt_type,
            "prompt": prompt_text,
            "continuation": continuation,
            "trajectory": trajectory_log[0],
        })

    out_file = os.path.join(os.path.dirname(__file__), "../../outputs/navitrit-10m-generation-audit.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAudit results saved to {out_file}")


if __name__ == "__main__":
    audit_generation()
