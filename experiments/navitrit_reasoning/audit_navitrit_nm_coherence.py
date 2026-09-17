"""
experiments/navitrit_reasoning/audit_navitrit_nm_coherence.py: Comparative Entity Persistence & Linguistic Coherence Audit.

Compares:
1. Base Hardened NaviTrit (outputs/checkpoints/navitrit-flow-hardened.pt)
2. New NaviTrit-NM with Hop Modulation & Reasoning Core (outputs/checkpoints/navitrit-nm-trained.pt)

Specifically tests diagnostic prompts for:
- Entity Persistence (e.g. red ball staying a ball, not becoming a stick)
- Spatial / Physical Grounding (e.g. kites in the sky, not in houses)
- Distinct-1 and Distinct-2 lexical diversity
- 3-gram phrase repetition
- Local TinyLlama-1.1B-Chat Judge scores and conditional perplexity

Output ledger: outputs/navitrit-nm-coherence-audit.json
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.navitrit.navitrit_model import (
    NaviTritConfig,
    NaviTritForCausalLM,
)
from experiments.navitrit_reasoning.navitrit_nm_model import (
    NaviTritNMConfig,
    NaviTritNMForCausalLM,
)
from experiments.navitrit.llm_judge import LLMJudge


DIAGNOSTIC_PROMPTS = [
    {
        "id": "entity_ball",
        "prompt": "One sunny day, Lily and her brother found a big, red ball.",
        "target_entity": "ball",
        "contradictory_entities": ["stick", "car", "rock", "box"],
    },
    {
        "id": "spatial_kite",
        "prompt": "The boy flew his kite high up in the",
        "target_entity": "sky",
        "contradictory_entities": ["house", "room", "bed"],
    },
    {
        "id": "character_spot",
        "prompt": "Once upon a time, there was a little dog named Spot.",
        "target_entity": "dog",
        "contradictory_entities": ["cat", "bird", "fish"],
    },
    {
        "id": "intent_tower",
        "prompt": "Tim wanted to build a big tower with his",
        "target_entity": "blocks",
        "contradictory_entities": ["car", "food"],
    },
    {
        "id": "biological_bird",
        "prompt": "The little bird looked out of the nest and saw",
        "target_entity": "tree",
        "contradictory_entities": ["car", "water"],
    },
]


def compute_diversity_metrics(token_ids: List[int]) -> Dict[str, float]:
    """Computes Distinct-1, Distinct-2, and 3-gram repetition rates."""
    if len(token_ids) == 0:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_3": 0.0}
    
    unigrams = token_ids
    bigrams = list(zip(token_ids[:-1], token_ids[1:])) if len(token_ids) > 1 else []
    trigrams = list(zip(token_ids[:-2], token_ids[1:-1], token_ids[2:])) if len(token_ids) > 2 else []
    
    distinct_1 = len(set(unigrams)) / len(unigrams)
    distinct_2 = len(set(bigrams)) / max(1, len(bigrams))
    
    rep_3 = 0.0
    if len(trigrams) > 0:
        rep_3 = 1.0 - (len(set(trigrams)) / len(trigrams))
        
    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "repetition_3": round(rep_3, 4),
    }


def generate_continuation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.7,
    top_k: int = 40,
    device: str = "cuda:0",
) -> Tuple[str, List[int], List[List[int]]]:
    """Autoregressive generation with temperature sampling."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_tokens = []
    trajectories = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids)
            logits = out.logits[:, -1, :] / max(1e-5, temperature)
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_tokens.append(next_token.item())
            
            if hasattr(out, "trajectory"):
                step_nodes = [step.selected_node for step in out.trajectory]
            elif hasattr(out, "trajectory_nodes"):
                step_nodes = out.trajectory_nodes
            else:
                step_nodes = []
            trajectories.append(step_nodes)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    continuation_text = tokenizer.decode(generated_tokens)
    return continuation_text, generated_tokens, trajectories


def check_entity_persistence(text: str, target: str, contradictions: List[str]) -> Dict:
    """Checks whether the text preserves the target entity without contradictory object switches."""
    text_lower = text.lower()
    has_target = target.lower() in text_lower
    found_contradictions = [c for c in contradictions if c.lower() in text_lower]
    
    score = 1.0 if has_target and len(found_contradictions) == 0 else (0.5 if has_target else 0.0)
    return {
        "preserved_target": has_target,
        "found_contradictions": found_contradictions,
        "persistence_score": score,
    }


def run_coherence_audit():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Comparative Coherence & Entity Persistence Audit on {device} ===")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Checkpoint paths
    base_ckpt_path = os.path.join(os.path.dirname(__file__), "../../outputs/checkpoints/navitrit-flow-hardened.pt")
    nm_ckpt_path = os.path.join(os.path.dirname(__file__), "../../outputs/checkpoints/navitrit-nm-trained.pt")
    
    assert os.path.isfile(base_ckpt_path), f"Missing base checkpoint: {base_ckpt_path}"
    assert os.path.isfile(nm_ckpt_path), f"Missing NM checkpoint: {nm_ckpt_path}"
    
    # 1. Load Base Hardened NaviTrit
    print("[1/3] Loading Base Hardened NaviTrit...")
    base_state = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    if isinstance(base_state, dict) and "model_state_dict" in base_state:
        base_state = base_state["model_state_dict"]
    base_config = NaviTritConfig(
        vocab_size=50257,
        hidden_size=384,
        intermediate_size=1024,
        num_layers=3,
        num_attention_heads=6,
        max_position_embeddings=512,
        max_hops=6,
        ternary=True,
    )
    base_model = NaviTritForCausalLM(base_config).to(device)
    base_model.load_state_dict(base_state)
    base_model.eval()
    
    # 2. Load New NaviTrit-NM
    print("[2/3] Loading NaviTrit-NM (with Hop Modulation & Reasoning Core)...")
    nm_data = torch.load(nm_ckpt_path, map_location=device, weights_only=False)
    nm_config = nm_data["config"]
    nm_model = NaviTritNMForCausalLM(nm_config).to(device)
    nm_model.load_state_dict(nm_data["model_state_dict"])
    nm_model.eval()
    
    # 3. Load Local LLM Judge
    print("[3/3] Initializing Local TinyLlama Judge...")
    try:
        judge = LLMJudge(device=device)
        has_judge = True
    except Exception as e:
        print(f"Notice: Local judge unavailable ({e}); proceeding with entity metrics.")
        judge = None
        has_judge = False
        
    audit_results = []
    base_diversity_accum = []
    nm_diversity_accum = []
    base_persistence_accum = []
    nm_persistence_accum = []
    
    print("\nEvaluating Diagnostic Prompts...")
    for idx, item in enumerate(DIAGNOSTIC_PROMPTS):
        prompt = item["prompt"]
        target = item["target_entity"]
        contradictions = item["contradictory_entities"]
        
        print(f"\n--- Diagnostic Prompt {idx+1}: '{prompt}' ---")
        
        # Generate from Base NaviTrit
        base_cont, base_tokens, base_trajs = generate_continuation(base_model, tokenizer, prompt, device=device)
        base_div = compute_diversity_metrics(base_tokens)
        base_persist = check_entity_persistence(base_cont, target, contradictions)
        base_diversity_accum.append(base_div)
        base_persistence_accum.append(base_persist["persistence_score"])
        
        print(f"  [Base NaviTrit]: {base_cont.strip()[:100]}...")
        print(f"    Distinct-1: {base_div['distinct_1']} | Target '{target}' Preserved: {base_persist['preserved_target']} | Contradictions: {base_persist['found_contradictions']}")
        
        # Generate from NaviTrit-NM
        nm_cont, nm_tokens, nm_trajs = generate_continuation(nm_model, tokenizer, prompt, device=device)
        nm_div = compute_diversity_metrics(nm_tokens)
        nm_persist = check_entity_persistence(nm_cont, target, contradictions)
        nm_diversity_accum.append(nm_div)
        nm_persistence_accum.append(nm_persist["persistence_score"])
        
        print(f"  [NaviTrit-NM]  : {nm_cont.strip()[:100]}...")
        print(f"    Distinct-1: {nm_div['distinct_1']} | Target '{target}' Preserved: {nm_persist['preserved_target']} | Contradictions: {nm_persist['found_contradictions']}")
        
        # Judge Scoring
        base_judge_score = 0.0
        base_judge_loss = 0.0
        nm_judge_score = 0.0
        nm_judge_loss = 0.0
        
        if has_judge:
            base_eval = judge.evaluate_candidate(prompt, base_cont)
            base_judge_score = base_eval["score"]
            base_judge_loss = base_eval["cond_loss"]
            
            nm_eval = judge.evaluate_candidate(prompt, nm_cont)
            nm_judge_score = nm_eval["score"]
            nm_judge_loss = nm_eval["cond_loss"]
            print(f"    Judge Score: Base={base_judge_score}/10 (loss {base_judge_loss:.2f}) vs NM={nm_judge_score}/10 (loss {nm_judge_loss:.2f})")
            
        audit_results.append({
            "prompt_id": item["id"],
            "prompt": prompt,
            "target_entity": target,
            "base_navitrit": {
                "continuation": base_cont,
                "diversity": base_div,
                "persistence": base_persist,
                "judge_score": base_judge_score,
                "judge_loss": round(base_judge_loss, 4),
            },
            "navitrit_nm": {
                "continuation": nm_cont,
                "diversity": nm_div,
                "persistence": nm_persist,
                "judge_score": nm_judge_score,
                "judge_loss": round(nm_judge_loss, 4),
            },
            "winner_persistence": "NM" if nm_persist["persistence_score"] > base_persist["persistence_score"] else ("BASE" if base_persist["persistence_score"] > nm_persist["persistence_score"] else "TIE"),
        })
        
    mean_base_persist = sum(base_persistence_accum) / len(base_persistence_accum)
    mean_nm_persist = sum(nm_persistence_accum) / len(nm_persistence_accum)
    
    mean_base_d1 = sum(d["distinct_1"] for d in base_diversity_accum) / len(base_diversity_accum)
    mean_nm_d1 = sum(d["distinct_1"] for d in nm_diversity_accum) / len(nm_diversity_accum)
    
    summary = {
        "status": "COMPLETED",
        "benchmark": "Comparative Entity Persistence & Linguistic Coherence Audit",
        "models": {
            "base_navitrit": {
                "mean_entity_persistence": round(mean_base_persist, 4),
                "mean_distinct_1": round(mean_base_d1, 4),
            },
            "navitrit_nm": {
                "mean_entity_persistence": round(mean_nm_persist, 4),
                "mean_distinct_1": round(mean_nm_d1, 4),
            },
        },
        "persistence_improvement_pct": round((mean_nm_persist - mean_base_persist) * 100.0, 2),
        "detailed_prompts": audit_results,
    }
    
    out_json = os.path.join(os.path.dirname(__file__), "../../outputs/navitrit-nm-coherence-audit.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved coherence audit ledger to {out_json}")
    print(f"\nFinal Entity Persistence: Base NaviTrit={mean_base_persist*100:.1f}% vs NaviTrit-NM={mean_nm_persist*100:.1f}%")
    return summary


if __name__ == "__main__":
    run_coherence_audit()
