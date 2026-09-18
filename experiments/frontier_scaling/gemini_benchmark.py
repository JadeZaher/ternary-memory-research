"""
experiments/frontier_scaling/gemini_benchmark.py: Robust LLM-as-a-Judge Benchmark using Gemini 2.5 Flash via OpenRouter.

Evaluates NaviTrit-100M generated outputs across:
1. Narrative Coherence & Entity Consistency
2. Mathematical Reasoning & Arithmetic Correctness
3. Algorithmic Code Synthesis & Syntax Validity

Reads OPENROUTER_API_KEY from C:\\Users\\atooz\\.pi\\agent\\.env without printing secrets.
"""

import os
import sys
import json
import time
import urllib.request
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.frontier_scaling.navitrit_scale_model import (
    NaviTritScaleConfig,
    NaviTritScaleForCausalLM,
    get_scale_config,
)

ENV_PATH = r"C:\Users\atooz\.pi\agent\.env"
MODEL_ID = "google/gemini-2.5-flash"


def get_api_key() -> str:
    if not os.path.exists(ENV_PATH):
        raise FileNotFoundError(f"Environment file not found at {ENV_PATH}")
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    raise ValueError("OPENROUTER_API_KEY not found in .env file")


def call_gemini_judge(api_key: str, prompt: str, system_instruction: str) -> Dict[str, Any]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ternary-research",
        "X-Title": "Ternary NaviTrit Gemini Benchmark",
    }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(url, headers=headers, data=json.dumps(payload).encode("utf-8"))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                # Parse JSON from model response
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Strip markdown block if wrapped
                    clean = content.strip()
                    if clean.startswith("```json"):
                        clean = clean[7:]
                    if clean.endswith("```"):
                        clean = clean[:-3]
                    return json.loads(clean.strip())
        except Exception as e:
            time.sleep(2 * (attempt + 1))
            if attempt == 2:
                return {"error": str(e), "score": 1.0, "verdict": "ERROR", "diagnosis": f"API request failed: {e}"}


def generate_completions(
    model: NaviTritScaleForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[Dict[str, str]],
    device: torch.device,
    max_tokens: int = 40,
    static_routes: Optional[Dict[str, List[int]]] = None,
) -> List[Dict[str, Any]]:
    from experiments.frontier_scaling.macro_route_search import forward_static_route
    model.eval()
    completions = []
    with torch.no_grad():
        for item in prompts:
            p_type = item["type"]
            prompt_text = item["prompt"]
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
            curr_ids = input_ids.clone()
            trajectories = []

            is_math = (p_type == "math")
            route_for_task = static_routes.get(p_type) if static_routes else None

            for _ in range(max_tokens):
                if route_for_task is not None:
                    logits = forward_static_route(model, curr_ids, route_for_task)[:, -1, :]
                    trajectories.append(route_for_task)
                elif hasattr(model, "super_block"):
                    out = model(curr_ids, recursion_budget=4)
                    logits = out["logits"][:, -1, :]
                    hops = out.get("recursion_budget", out.get("hop_budget", 4))
                    trajectories.append([f"recursion_{k}" for k in range(hops)])
                elif hasattr(model, "global_planner"):
                    out = model(curr_ids, temperature=0.7, is_math=is_math)
                    logits = out["logits"][:, -1, :]
                    if "traversal_graph" in out:
                        trajectories.append(out["traversal_graph"]["directed_edges"])
                    elif "tree_pairs" in out:
                        trajectories.append(out["tree_pairs"])
                    else:
                        trajectories.append(out.get("trajectory", []))
                else:
                    out = model(curr_ids, temperature=0.7)
                    logits = out["logits"][:, -1, :]
                    trajectories.append(out.get("trajectory", []))

                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_token], dim=-1)

            gen_text = tokenizer.decode(curr_ids[0][input_ids.size(1):])
            completions.append({
                "type": p_type,
                "prompt": prompt_text,
                "continuation": gen_text,
                "trajectory_sample": trajectories[0] if len(trajectories) > 0 else [],
                "all_trajectories": trajectories,
            })
    return completions


def run_benchmark(
    checkpoints: List[Dict[str, str]],
    output_json: str = "outputs/navitrit-100m-gemini-benchmark.json",
):
    api_key = get_api_key()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Gemini 2.5 Flash Benchmark on {device} ===")
    print(f"Judge Model: {MODEL_ID} via OpenRouter")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    test_prompts = [
        {
            "type": "math",
            "prompt": "Problem: Olivia had 15 apples. She gave 4 apples to Liam. How many apples remain?\nReasoning:",
            "target": "11",
        },
        {
            "type": "math",
            "prompt": "Problem: A bakery bakes 24 cookies. They sell 10 in the morning and 6 in the afternoon. How many cookies are left?\nReasoning:",
            "target": "8",
        },
        {
            "type": "story",
            "prompt": "One sunny day, Lily and her brother found a big, red ball.",
            "target": "Coherent narrative about Lily, brother, and the ball",
        },
        {
            "type": "story",
            "prompt": "Once upon a time, there was a little dog named Spot.",
            "target": "Coherent narrative maintaining Spot the dog identity",
        },
        {
            "type": "code",
            "prompt": "def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n",
            "target": "Syntactically correct binary search loop",
        },
    ]

    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                benchmark_results = json.load(f)
        except Exception:
            benchmark_results = {
                "judge_model": MODEL_ID,
                "date": "2026-09-17",
                "checkpoints_evaluated": {},
            }
    else:
        benchmark_results = {
            "judge_model": MODEL_ID,
            "date": "2026-09-17",
            "checkpoints_evaluated": {},
        }

    judge_system_instruction = (
        "You are an expert AI evaluator judging the output quality of small-to-medium language models.\n"
        "Score the completion on a scale of 1.0 to 10.0.\n"
        "Return a strict JSON object with the following fields:\n"
        "{\n"
        '  "score": <float 1.0 to 10.0>,\n'
        '  "verdict": "<CORRECT|PARTIALLY_CORRECT|CONFUSED|INCORRECT>",\n'
        '  "entity_consistency": <float 1.0 to 10.0>,\n'
        '  "syntactic_validity": <float 1.0 to 10.0>,\n'
        '  "mathematical_correctness": <float 1.0 to 10.0 or null if not math>,\n'
        '  "arithmetic_calculation_performed": "<e.g. 15 - 3 = 17 or None>",\n'
        '  "hallucination_detected": <true|false>,\n'
        '  "diagnosis": "<detailed critique of the error or strength>",\n'
        '  "remedy_recommendation": "<how architecture/training can fix this>"\n'
        "}"
    )

    for ckpt_info in checkpoints:
        ckpt_name = ckpt_info["name"]
        ckpt_path = ckpt_info["path"]
        model_size = ckpt_info.get("model_size", "100m")

        if not os.path.exists(ckpt_path):
            print(f"Skipping {ckpt_name}: file {ckpt_path} not found.")
            continue

        print(f"\n--- Loading and Evaluating Checkpoint: {ckpt_name} ({model_size}) ---")
        if "ifmor" in ckpt_name:
            from experiments.frontier_scaling.navitrit_ifmor_model import (
                IFMoRConfig,
                NaviTritIFMoRForCausalLM,
            )
            if_cfg = IFMoRConfig(
                vocab_size=50257,
                hidden_size=768,
                intermediate_size=4096,
                num_attention_heads=12,
                d_state=64,
                max_position_embeddings=512,
                max_loops=6,
                default_loops=4,
            )
            model = NaviTritIFMoRForCausalLM(if_cfg).to(device)
        elif "looped-dwp" in ckpt_name:
            from experiments.frontier_scaling.looped_dwp_model import (
                LoopedDWPConfig,
                LoopedDWPForCausalLM,
            )
            dwp_cfg = LoopedDWPConfig(
                vocab_size=50257,
                hidden_size=768,
                intermediate_size=2048,
                num_attention_heads=12,
                max_position_embeddings=512,
                max_recursions=8,
                default_recursions=4,
                d_context=64,
                lora_rank=32,
                lora_alpha=32.0,
            )
            model = LoopedDWPForCausalLM(dwp_cfg).to(device)
            raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(raw_ckpt.get("model_state_dict", raw_ckpt))
        elif "condprog" in ckpt_name:
            from experiments.frontier_scaling.conditional_program_model import (
                ConditionalProgramConfig,
                NaviTritConditionalProgramForCausalLM,
            )
            cp_cfg = ConditionalProgramConfig(
                vocab_size=50257,
                hidden_size=768,
                intermediate_size=2048,
                num_attention_heads=12,
                max_position_embeddings=512,
                max_hops=8,
                default_hops=4,
                d_context=64,
                lora_rank=16,
                lora_alpha=16.0,
            )
            model = NaviTritConditionalProgramForCausalLM(cp_cfg).to(device)
            raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(raw_ckpt.get("model_state_dict", raw_ckpt))
        elif "loopformer" in ckpt_name:
            from experiments.frontier_scaling.loopformer_model import (
                LoopFormerConfig,
                LoopFormerForCausalLM,
            )
            lf_cfg = LoopFormerConfig(
                vocab_size=50257,
                hidden_size=768,
                intermediate_size=2048,
                num_attention_heads=12,
                max_position_embeddings=512,
                max_recursions=8,
                default_recursions=4,
            )
            model = LoopFormerForCausalLM(lf_cfg).to(device)
            raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(raw_ckpt.get("model_state_dict", raw_ckpt))
        elif "token-graph" in ckpt_name:
            from experiments.frontier_scaling.navitrit_token_graph_model import NaviTritTokenGraphForCausalLM
            model = NaviTritTokenGraphForCausalLM(cfg).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        elif "graph" in ckpt_name:
            from experiments.frontier_scaling.navitrit_graph_model import NaviTritGraphForCausalLM
            model = NaviTritGraphForCausalLM(cfg).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        elif "tree" in ckpt_name:
            from experiments.frontier_scaling.navitrit_tree_model import NaviTritTreeForCausalLM
            model = NaviTritTreeForCausalLM(cfg).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        elif "dual" in ckpt_name:
            from experiments.frontier_scaling.navitrit_dual_model import NaviTritDualForCausalLM
            model = NaviTritDualForCausalLM(cfg).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        else:
            model = NaviTritScaleForCausalLM(cfg).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        static_routes = ckpt_info.get("static_routes")
        if static_routes is None and "macro-static" in ckpt_name:
            macro_json = "outputs/macro-routes.json"
            if os.path.exists(macro_json):
                with open(macro_json, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                    static_routes = m_data.get("discovered_routes", {})
                print(f"  Loaded MACRO Static Routes: {static_routes}")

        completions = generate_completions(model, tokenizer, test_prompts, device, max_tokens=45, static_routes=static_routes)

        eval_records = []
        domain_scores = {"math": [], "story": [], "code": []}

        for comp in completions:
            p_type = comp["type"]
            prompt = comp["prompt"]
            continuation = comp["continuation"]
            full_text = prompt + continuation

            judge_prompt = (
                f"Evaluation Task Type: {p_type.upper()}\n"
                f"Prompt:\n\"\"\"{prompt}\"\"\"\n\n"
                f"Model Generated Continuation:\n\"\"\"{continuation}\"\"\"\n\n"
                f"Full Output:\n\"\"\"{full_text}\"\"\"\n\n"
                f"Please evaluate this completion thoroughly against the rubric."
            )

            print(f"  Evaluating [{p_type.upper()}]: {prompt[:40]}...")
            judge_res = call_gemini_judge(api_key, judge_prompt, judge_system_instruction)

            score = float(judge_res.get("score", 1.0))
            domain_scores[p_type].append(score)

            record = {
                "type": p_type,
                "prompt": prompt,
                "continuation": continuation,
                "trajectory_sample": comp["trajectory_sample"],
                "judge_evaluation": judge_res,
            }
            eval_records.append(record)
            diag = judge_res.get("diagnosis", "")
            print(f"    -> Score: {score}/10 | Verdict: {judge_res.get('verdict')} | Diagnosis: {diag[:80]}...")

        # Averages
        avg_math = sum(domain_scores["math"]) / max(1, len(domain_scores["math"]))
        avg_story = sum(domain_scores["story"]) / max(1, len(domain_scores["story"]))
        avg_code = sum(domain_scores["code"]) / max(1, len(domain_scores["code"]))
        overall_avg = (avg_math + avg_story + avg_code) / 3.0

        benchmark_results["checkpoints_evaluated"][ckpt_name] = {
            "checkpoint_path": ckpt_path,
            "model_size": model_size,
            "overall_score": round(overall_avg, 2),
            "math_score": round(avg_math, 2),
            "story_score": round(avg_story, 2),
            "code_score": round(avg_code, 2),
            "records": eval_records,
        }

        # Free GPU memory between checkpoints
        del model
        torch.cuda.empty_cache()

    # Save to JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\n==========================================")
    print(f"Gemini Benchmark Complete! Results written to: {output_json}")
    for name, res in benchmark_results["checkpoints_evaluated"].items():
        print(f"  Checkpoint [{name}]: Overall: {res['overall_score']}/10 | Math: {res['math_score']}/10 | Story: {res['story_score']}/10 | Code: {res['code_score']}/10")
    print(f"==========================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Filter specific checkpoint by name")
    args = parser.parse_args()

    checkpoints_to_eval = [
        {
            "name": "navitrit-100m-step10000",
            "path": "outputs/checkpoints/navitrit-100m-step10000.pt",
            "model_size": "100m",
        },
        {
            "name": "navitrit-100m-tree-grpo",
            "path": "outputs/checkpoints/navitrit-100m-tree-grpo.pt",
            "model_size": "100m",
        },
        {
            "name": "navitrit-100m-graph-traversal",
            "path": "outputs/checkpoints/navitrit-100m-tree-grpo.pt",
            "model_size": "100m",
        },
        {
            "name": "navitrit-100m-token-graph-grpo",
            "path": "outputs/checkpoints/navitrit-100m-token-graph-grpo.pt",
            "model_size": "100m",
        },
        {
            "name": "navitrit-100m-loopformer",
            "path": "outputs/checkpoints/navitrit-100m-loopformer.pt",
            "model_size": "100m",
        },
        {
            "name": "navitrit-100m-ifmor",
            "path": "outputs/checkpoints/navitrit-100m-ifmor.pt",
            "model_size": "100m",
        },
        {
            "name": "navitrit-100m-condprog",
            "path": "outputs/checkpoints/navitrit-100m-condprog.pt",
            "model_size": "100m",
        },
        {
            "name": "navitrit-100m-looped-dwp",
            "path": "outputs/checkpoints/navitrit-100m-looped-dwp.pt",
            "model_size": "100m",
        },
    ]
    if args.model:
        checkpoints_to_eval = [c for c in checkpoints_to_eval if args.model in c["name"]]
    run_benchmark(checkpoints_to_eval)
