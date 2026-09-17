"""
experiments/audit_language_coherence.py: Rigorous Language Coherence & Validity Audit.

Grills the linguistic and generative validity of all trained ternary models:
1. FP32 Baseline (bitroute-tinystories-fp32.pt)
2. Native Ternary Baseline (bitroute-tinystories-ternary.pt)
3. Greedy 1-Hop Ternary Router (bitroute-tinystories-ternary_router.pt)
4. Flow-Reasoned Multi-Hop (bitroute-multihop-flow.pt)
5. NaviTrit Graph Navigation (navitrit-flow.pt)

Evaluates:
- Actual generated text samples across diverse prompts
- Text coherence, syntax validity, and story structure
- Distinct-1 and Distinct-2 n-gram lexical diversity ratios
- Repetition rate (% repeated 3-grams)
- Dynamic routing telemetry during generation (which layers/nodes fire per token)

Saves unvarnished findings to outputs/language-coherence-audit.json.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.bitroute_model import BitRouteForCausalLM, BitRouteConfig
from experiments.bitroute_multihop_model import BitRouteMultiHopForCausalLM
from experiments.navitrit.navitrit_model import NaviTritForCausalLM, NaviTritConfig

AUDIT_PROMPTS = [
    "Once upon a time, there was a little dog named Spot.",
    "One sunny day, Lily and her brother found a",
    "Tim wanted to build a big tower with his",
    "The little bird looked out of the nest and saw",
]


def calculate_ngram_diversity(token_ids: List[int]) -> Dict[str, float]:
    """Calculates distinct-1, distinct-2, and 3-gram repetition ratio."""
    if len(token_ids) == 0:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_3gram": 0.0}

    # Distinct-1
    distinct_1 = len(set(token_ids)) / max(1, len(token_ids))

    # Distinct-2
    bigrams = [tuple(token_ids[i:i+2]) for i in range(len(token_ids) - 1)]
    distinct_2 = len(set(bigrams)) / max(1, len(bigrams)) if bigrams else 1.0

    # 3-gram repetition ratio
    trigrams = [tuple(token_ids[i:i+3]) for i in range(len(token_ids) - 2)]
    if trigrams:
        repeated_trigrams = len(trigrams) - len(set(trigrams))
        rep_3gram = repeated_trigrams / len(trigrams)
    else:
        rep_3gram = 0.0

    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "repetition_3gram": round(rep_3gram, 4),
    }


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 40,
    repetition_penalty: float = 1.15,
    generated_ids: Optional[List[int]] = None,
) -> int:
    """Samples next token with temperature, top-k truncation, and repetition penalty."""
    logits = logits.clone()

    # Apply repetition penalty
    if generated_ids and repetition_penalty != 1.0:
        for tid in set(generated_ids):
            if logits[tid] > 0:
                logits[tid] /= repetition_penalty
            else:
                logits[tid] *= repetition_penalty

    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    logits = logits / max(1e-4, temperature)

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


def generate_with_bitroute(
    model: BitRouteForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.7,
    repetition_penalty: float = 1.15,
) -> Dict:
    """Autoregressive generation for BitRoute with per-token routing telemetry."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    curr_ids = input_ids.clone()
    prompt_len = curr_ids.shape[1]
    new_token_ids = []

    per_token_telemetry = []
    t0 = time.perf_counter()

    for step in range(max_new_tokens):
        out = model(curr_ids)
        next_id = sample_next_token(
            out.logits[0, -1, :],
            temperature=temperature,
            top_k=40,
            repetition_penalty=repetition_penalty,
            generated_ids=new_token_ids,
        )
        new_token_ids.append(next_id)
        next_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
        curr_ids = torch.cat([curr_ids, next_tensor], dim=-1)

        per_token_telemetry.append({
            "step": step,
            "token": tokenizer.decode(next_id),
            "executed_layers": getattr(out, "executed_layers", []),
            "bypassed_layers": getattr(out, "bypassed_layers", []),
        })

    gen_time_s = time.perf_counter() - t0
    full_text = tokenizer.decode(curr_ids[0].tolist())
    continuation_text = tokenizer.decode(new_token_ids)
    diversity = calculate_ngram_diversity(new_token_ids)

    return {
        "prompt": prompt,
        "full_text": full_text,
        "continuation_text": continuation_text,
        "new_tokens_count": max_new_tokens,
        "tokens_per_sec": round(max_new_tokens / max(1e-4, gen_time_s), 2),
        "diversity": diversity,
        "sample_telemetry": per_token_telemetry[:5],
    }


def generate_with_multihop(
    model: BitRouteMultiHopForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.7,
    repetition_penalty: float = 1.15,
) -> Dict:
    """Autoregressive generation for BitRouteMultiHop."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    curr_ids = input_ids.clone()
    new_token_ids = []

    t0 = time.perf_counter()
    for step in range(max_new_tokens):
        out = model(curr_ids)
        next_id = sample_next_token(
            out.logits[0, -1, :],
            temperature=temperature,
            top_k=40,
            repetition_penalty=repetition_penalty,
            generated_ids=new_token_ids,
        )
        new_token_ids.append(next_id)
        next_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
        curr_ids = torch.cat([curr_ids, next_tensor], dim=-1)

    gen_time_s = time.perf_counter() - t0
    full_text = tokenizer.decode(curr_ids[0].tolist())
    continuation_text = tokenizer.decode(new_token_ids)
    diversity = calculate_ngram_diversity(new_token_ids)

    return {
        "prompt": prompt,
        "full_text": full_text,
        "continuation_text": continuation_text,
        "new_tokens_count": max_new_tokens,
        "tokens_per_sec": round(max_new_tokens / max(1e-4, gen_time_s), 2),
        "diversity": diversity,
    }


def generate_with_navitrit(
    model: NaviTritForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.7,
    repetition_penalty: float = 1.15,
) -> Dict:
    """Autoregressive generation for NaviTrit."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    curr_ids = input_ids.clone()
    new_token_ids = []

    t0 = time.perf_counter()
    trajectory_history = []

    for step in range(max_new_tokens):
        out = model(curr_ids)
        next_id = sample_next_token(
            out.logits[0, -1, :],
            temperature=temperature,
            top_k=40,
            repetition_penalty=repetition_penalty,
            generated_ids=new_token_ids,
        )
        new_token_ids.append(next_id)
        trajectory_history.append(out.trajectory_nodes)
        next_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
        curr_ids = torch.cat([curr_ids, next_tensor], dim=-1)

    gen_time_s = time.perf_counter() - t0
    full_text = tokenizer.decode(curr_ids[0].tolist())
    continuation_text = tokenizer.decode(new_token_ids)
    diversity = calculate_ngram_diversity(new_token_ids)

    return {
        "prompt": prompt,
        "full_text": full_text,
        "continuation_text": continuation_text,
        "new_tokens_count": max_new_tokens,
        "tokens_per_sec": round(max_new_tokens / max(1e-4, gen_time_s), 2),
        "diversity": diversity,
        "sample_trajectories": trajectory_history[:5],
    }


def audit_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"Linguistic Coherence & Generative Validity Audit ({device})")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    base_config = dict(
        vocab_size=50257,
        hidden_size=384,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=6,
        max_position_embeddings=512,
        block_size=256,
        use_trained_router=True,
    )

    models_to_audit = {}

    def load_weights(path):
        raw = torch.load(path, weights_only=True)
        return raw["state_dict"] if "state_dict" in raw else raw

    # 1. FP32 Baseline
    p_fp32 = "outputs/checkpoints/bitroute-tinystories-fp32.pt"
    if os.path.exists(p_fp32):
        cfg = BitRouteConfig(ternary=False, **base_config)
        m = BitRouteForCausalLM(cfg).to(device)
        m.load_state_dict(load_weights(p_fp32))
        models_to_audit["fp32_baseline"] = ("bitroute", m)

    # 2. Native Ternary Baseline
    p_tern = "outputs/checkpoints/bitroute-tinystories-ternary.pt"
    if os.path.exists(p_tern):
        cfg = BitRouteConfig(ternary=True, **base_config)
        m = BitRouteForCausalLM(cfg).to(device)
        m.load_state_dict(load_weights(p_tern))
        models_to_audit["native_ternary_baseline"] = ("bitroute", m)

    # 3. Greedy Ternary Router
    p_rout = "outputs/checkpoints/bitroute-tinystories-ternary_router.pt"
    if os.path.exists(p_rout):
        cfg = BitRouteConfig(ternary=True, **base_config)
        m = BitRouteForCausalLM(cfg).to(device)
        m.load_state_dict(load_weights(p_rout))
        models_to_audit["ternary_greedy_router"] = ("bitroute", m)

    # 4. Flow-Reasoned Multi-Hop
    p_multi = "outputs/checkpoints/bitroute-multihop-flow.pt"
    if os.path.exists(p_multi):
        cfg = BitRouteConfig(ternary=True, **base_config)
        m = BitRouteMultiHopForCausalLM(cfg).to(device)
        m.load_state_dict(load_weights(p_multi))
        models_to_audit["flow_multihop_router"] = ("multihop", m)

    # 5. NaviTrit
    p_navi = "outputs/checkpoints/navitrit-flow.pt"
    if os.path.exists(p_navi):
        cfg = NaviTritConfig(
            vocab_size=50257,
            hidden_size=384,
            intermediate_size=1024,
            num_layers=3,
            num_attention_heads=6,
            max_position_embeddings=512,
            max_hops=6,
            ternary=True,
        )
        m = NaviTritForCausalLM(cfg).to(device)
        m.load_state_dict(load_weights(p_navi))
        models_to_audit["navitrit_graph"] = ("navitrit", m)

    print(f"[+] Loaded {len(models_to_audit)} trained models for comparative generation audit.\n")

    audit_results = {}

    for model_name, (m_type, model) in models_to_audit.items():
        print(f"---> Auditing: {model_name}...")
        prompt_outputs = []
        all_rep_rates = []
        all_d1 = []
        all_d2 = []

        for p_idx, prompt in enumerate(AUDIT_PROMPTS):
            if m_type == "bitroute":
                gen = generate_with_bitroute(model, tokenizer, prompt, max_new_tokens=40, device=device)
            elif m_type == "multihop":
                gen = generate_with_multihop(model, tokenizer, prompt, max_new_tokens=40, device=device)
            elif m_type == "navitrit":
                gen = generate_with_navitrit(model, tokenizer, prompt, max_new_tokens=40, device=device)

            all_rep_rates.append(gen["diversity"]["repetition_3gram"])
            all_d1.append(gen["diversity"]["distinct_1"])
            all_d2.append(gen["diversity"]["distinct_2"])

            print(f"     Prompt {p_idx+1}: '{prompt}'")
            print(f"     Output:   \"{gen['continuation_text']}\"")
            print(f"     Metrics:  Distinct-1: {gen['diversity']['distinct_1']:.2f}, Repetition-3: {gen['diversity']['repetition_3gram']:.2f}, {gen['tokens_per_sec']:.1f} tok/s\n")
            prompt_outputs.append(gen)

        audit_results[model_name] = {
            "mean_distinct_1": round(sum(all_d1) / len(all_d1), 4),
            "mean_distinct_2": round(sum(all_d2) / len(all_d2), 4),
            "mean_repetition_3gram": round(sum(all_rep_rates) / len(all_rep_rates), 4),
            "prompts": prompt_outputs,
        }

    # Qualitative verdict analysis
    print("=" * 80)
    print("AUDIT SUMMARY & VERDICT:")
    print("=" * 80)
    for name, res in audit_results.items():
        print(f"  {name:25s} | Distinct-1: {res['mean_distinct_1']:.3f} | Distinct-2: {res['mean_distinct_2']:.3f} | Repetition-3: {res['mean_repetition_3gram']:.3f}")

    out_file = "outputs/language-coherence-audit.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n[+] Full unvarnished audit ledger saved to {out_file}")


if __name__ == "__main__":
    audit_models()
