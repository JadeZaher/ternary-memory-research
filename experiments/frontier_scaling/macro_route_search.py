"""
experiments/frontier_scaling/macro_route_search.py: Gate 17 (Track I) MACRO Offline Route Search.

Cross-Entropy Method (CEM) search over discrete path space P = V^T of the frozen 124.15M
ternary backbone (navitrit-100m-trained.pt).

Extracts static task sub-graphs:
  - r*_code   : Optimal tile sequence for Python Algorithmic Code AST
  - r*_math   : Optimal tile sequence for Multi-step GSM8K Arithmetic (exploring Node 24 Reasoning Core)
  - r*_story  : Optimal tile sequence for TinyStories narrative coherence
  - r*_multi  : Balanced multi-corpus route

Zero weight modification. Zero runtime gating overhead.
"""

import os
import sys
import math
import time
import json
import random
import argparse
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.frontier_scaling.navitrit_scale_model import (
    NaviTritScaleConfig,
    NaviTritScaleForCausalLM,
    get_scale_config,
)
from experiments.frontier_scaling.prepare_multicorpus import (
    generate_math_problems,
    generate_python_code,
)


class TaskTokenDataset(Dataset):
    """Dataset of fixed-length token chunks for discrete route evaluation."""
    def __init__(self, tokens: torch.Tensor, seq_len: int = 256):
        self.seq_len = seq_len
        self.num_samples = (len(tokens) - 1) // seq_len
        self.tokens = tokens[: self.num_samples * seq_len + 1]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len].long()
        y = self.tokens[start + 1 : start + self.seq_len + 1].long()
        return x, y


def build_validation_data(
    tokenizer: AutoTokenizer,
    tinystories_valid_path: str = "data/tinystories_valid.txt",
    seq_len: int = 256,
    target_tokens_per_task: int = 25000,
) -> Dict[str, DataLoader]:
    """Prepares clean, distinct validation batches for Python Code, GSM8K Math, and TinyStories."""
    print("=== Preparing Task-Specific Validation Datasets for MACRO Search ===")
    loaders = {}

    # 1. Python Code validation
    print("[1/3] Generating & tokenizing Python Code validation set...")
    random.seed(42)
    code_text = generate_python_code(num_samples=200)
    code_tokens = tokenizer(code_text, return_tensors="pt", truncation=False)["input_ids"][0][:target_tokens_per_task]
    code_ds = TaskTokenDataset(code_tokens, seq_len=seq_len)
    loaders["code"] = DataLoader(code_ds, batch_size=4, shuffle=False)
    print(f"  Python Code: {len(code_tokens):,} tokens ({len(code_ds)} batches)")

    # 2. GSM8K / Multi-step Math validation
    print("[2/3] Generating & tokenizing GSM8K Multi-step Math validation set...")
    random.seed(42)
    math_text = generate_math_problems(num_samples=250)
    math_tokens = tokenizer(math_text, return_tensors="pt", truncation=False)["input_ids"][0][:target_tokens_per_task]
    math_ds = TaskTokenDataset(math_tokens, seq_len=seq_len)
    loaders["math"] = DataLoader(math_ds, batch_size=4, shuffle=False)
    print(f"  GSM8K Math: {len(math_tokens):,} tokens ({len(math_ds)} batches)")

    # 3. TinyStories validation
    print("[3/3] Tokenizing TinyStories validation set...")
    if os.path.exists(tinystories_valid_path):
        with open(tinystories_valid_path, "r", encoding="utf-8") as f:
            story_lines = [f.readline() for _ in range(400)]
        story_text = "\n".join(story_lines)
        story_tokens = tokenizer(story_text, return_tensors="pt", truncation=False)["input_ids"][0][:target_tokens_per_task]
    else:
        # Fallback to multicorpus slice
        print("  Notice: tinystories_valid_path not found, using slice from multicorpus.")
        mc = torch.load("data/multicorpus_10m.pt", weights_only=True)
        story_tokens = mc[:target_tokens_per_task]
    story_ds = TaskTokenDataset(story_tokens, seq_len=seq_len)
    loaders["story"] = DataLoader(story_ds, batch_size=4, shuffle=False)
    print(f"  TinyStories: {len(story_tokens):,} tokens ({len(story_ds)} batches)")

    return loaders


def forward_static_route(
    model: NaviTritScaleForCausalLM,
    input_ids: torch.Tensor,
    route: List[int],
) -> torch.Tensor:
    """Executes an exact static sequence of stationary tiles through the frozen backbone."""
    B, S = input_ids.size()
    device = input_ids.device
    pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
    h = model.embed_tokens(input_ids) + model.embed_positions(pos)

    for t, node_idx in enumerate(route):
        h, _ = model.execute_tile(node_idx, h, t)

    h_norm = model.final_norm(h)
    logits = model.lm_head(h_norm)
    return logits


@torch.no_grad()
def evaluate_static_route(
    model: NaviTritScaleForCausalLM,
    loader: DataLoader,
    route: List[int],
    device: torch.device,
    max_batches: int = 12,
) -> Tuple[float, float]:
    """Computes cross-entropy loss and perplexity for a candidate static route."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for b_idx, (x, y) in enumerate(loader):
        if b_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = forward_static_route(model, x, route)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    return avg_loss, ppl


@torch.no_grad()
def evaluate_dynamic_baseline(
    model: NaviTritScaleForCausalLM,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 12,
) -> Tuple[float, float, List[int]]:
    """Evaluates the base model using its learned dynamic controller."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    sample_trajectory = []

    for b_idx, (x, y) in enumerate(loader):
        if b_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x, temperature=0.7)
        logits = out["logits"]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
        if len(sample_trajectory) == 0 and "trajectory" in out:
            sample_trajectory = out["trajectory"]

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    return avg_loss, ppl, sample_trajectory


@torch.no_grad()
def extract_task_traversal_graph(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_math: bool = False,
    max_batches: int = 15,
) -> Dict[str, Any]:
    """Evaluates the Graph-of-Traversals model and computes empirical edge weight scores."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_directed_edges = []
    cumulative_node_scores = {}
    total_pairs = []

    for b_idx, (x, y) in enumerate(loader):
        if b_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x, temperature=0.7, is_math=is_math)
        logits = out["logits"]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()

        tg = out.get("traversal_graph", {})
        if b_idx == 0 and "directed_edges" in tg:
            all_directed_edges = tg["directed_edges"]
            total_pairs = out.get("tree_pairs", [])
        if "node_scores" in tg:
            for k, v in tg["node_scores"].items():
                cumulative_node_scores[k] = round(cumulative_node_scores.get(k, 0.0) + v, 4)

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))

    sum_scores = sum(cumulative_node_scores.values()) or 1.0
    normalized_scores = {k: round(v / sum_scores, 4) for k, v in sorted(cumulative_node_scores.items())}

    return {
        "loss": round(avg_loss, 4),
        "ppl": round(ppl, 2),
        "traversal_pairs": total_pairs,
        "directed_edges": all_directed_edges,
        "node_weight_scores": normalized_scores,
    }


class CrossEntropyMethodRouteSearcher:
    """
    Cross-Entropy Method (CEM) for Discrete Route Discovery on Stationary Language Graphs.
    
    Searches discrete space P = V^T where |V| = 25 operational nodes:
      Nodes 2l     : Attention tiles (0, 2, 4, ..., 22)
      Nodes 2l + 1 : FFN tiles (1, 3, 5, ..., 23)
      Node 24      : Reasoning Core
      
    Maintains:
      pi_0 in Delta^{|V|-1}     : Initial hop probability distribution
      P in [0, 1]^{|V| x |V|}   : First-order Markov transition matrix
    """
    def __init__(
        self,
        num_nodes: int = 25,
        max_hops: int = 6,
        population_size: int = 64,
        elite_size: int = 8,
        alpha: float = 0.30,
        smoothing: float = 1e-3,
        bipartite: bool = True,
    ):
        self.num_nodes = num_nodes
        self.max_hops = max_hops
        self.population_size = population_size
        self.elite_size = elite_size
        self.alpha = alpha
        self.smoothing = smoothing
        self.bipartite = bipartite

        self.num_layers = (num_nodes - 1) // 2
        self.seq_nodes = [2 * l for l in range(self.num_layers)] + [num_nodes - 1]
        self.chan_nodes = [2 * l + 1 for l in range(self.num_layers)]

        # Initialize distributions
        if self.bipartite:
            self.pi_0 = torch.zeros(num_nodes)
            self.pi_0[self.seq_nodes] = 1.0 / len(self.seq_nodes)

            self.P = torch.zeros(num_nodes, num_nodes)
            for u in self.seq_nodes:
                self.P[u, self.chan_nodes] = 1.0 / len(self.chan_nodes)
            for u in self.chan_nodes:
                self.P[u, self.seq_nodes] = 1.0 / len(self.seq_nodes)
        else:
            self.pi_0 = torch.full((num_nodes,), 1.0 / num_nodes)
            self.P = torch.full((num_nodes, num_nodes), 1.0 / num_nodes)

    def sample_routes(self, best_route: Optional[List[int]] = None) -> List[List[int]]:
        """Samples M candidate routes from the parameterized Markov distribution."""
        routes = []
        for m in range(self.population_size):
            # Elitism: Retain current best route as first candidate
            if m == 0 and best_route is not None:
                routes.append(list(best_route))
                continue

            # Sample first hop
            r0 = int(torch.multinomial(self.pi_0, 1).item())
            route = [r0]

            # Sample remaining T-1 hops from transition matrix
            for _ in range(1, self.max_hops):
                curr_node = route[-1]
                trans_dist = self.P[curr_node]
                next_node = int(torch.multinomial(trans_dist, 1).item())
                route.append(next_node)

            routes.append(route)
        return routes

    def update_distribution(self, elite_routes: List[List[int]]):
        """Updates (pi_0, P) towards the elite empirical distribution with smoothing."""
        # 1. Update initial hop distribution pi_0
        if self.bipartite:
            counts_pi0 = torch.zeros(self.num_nodes)
            counts_pi0[self.seq_nodes] = self.smoothing
            for r in elite_routes:
                counts_pi0[r[0]] += 1.0
            target_pi0 = counts_pi0 / counts_pi0.sum()
            self.pi_0 = (1.0 - self.alpha) * self.pi_0 + self.alpha * target_pi0

            # 2. Update transition matrix P
            counts_P = torch.zeros(self.num_nodes, self.num_nodes)
            for u in self.seq_nodes:
                counts_P[u, self.chan_nodes] = self.smoothing
            for u in self.chan_nodes:
                counts_P[u, self.seq_nodes] = self.smoothing

            for r in elite_routes:
                for t in range(len(r) - 1):
                    u, v = r[t], r[t + 1]
                    counts_P[u, v] += 1.0

            # Row-normalize
            row_sums = counts_P.sum(dim=-1, keepdim=True)
            row_sums[row_sums == 0] = 1.0
            target_P = counts_P / row_sums
            self.P = (1.0 - self.alpha) * self.P + self.alpha * target_P
        else:
            counts_pi0 = torch.full((self.num_nodes,), self.smoothing)
            for r in elite_routes:
                counts_pi0[r[0]] += 1.0
            target_pi0 = counts_pi0 / counts_pi0.sum()
            self.pi_0 = (1.0 - self.alpha) * self.pi_0 + self.alpha * target_pi0

            counts_P = torch.full((self.num_nodes, self.num_nodes), self.smoothing)
            for r in elite_routes:
                for t in range(len(r) - 1):
                    u, v = r[t], r[t + 1]
                    counts_P[u, v] += 1.0

            target_P = counts_P / counts_P.sum(dim=-1, keepdim=True)
            self.P = (1.0 - self.alpha) * self.P + self.alpha * target_P

    def compute_entropy(self) -> float:
        """Computes the average transition distribution entropy."""
        entropy_rows = -(self.P * torch.log(self.P + 1e-12)).sum(dim=-1)
        return entropy_rows.mean().item()


def run_macro_search_for_task(
    task_name: str,
    model: NaviTritScaleForCausalLM,
    loader: DataLoader,
    device: torch.device,
    num_nodes: int = 25,
    max_hops: int = 6,
    num_iterations: int = 15,
    population_size: int = 64,
    elite_size: int = 8,
    search_batches: int = 8,
    eval_batches: int = 20,
    bipartite: bool = True,
) -> Dict[str, Any]:
    """Executes CEM route discovery for a single task."""
    print(f"\n=======================================================")
    print(f"Starting MACRO Route Search: Task = [{task_name.upper()}] (Bipartite Sequence-Channel Mixing = {bipartite})")
    print(f"Discrete Path Space: {num_nodes}^{max_hops} = {num_nodes**max_hops:,} candidate trajectories")
    print(f"Iterations: {num_iterations} | Population: {population_size} | Elites: {elite_size} | Batches/Eval: {search_batches}")
    print(f"=======================================================")

    searcher = CrossEntropyMethodRouteSearcher(
        num_nodes=num_nodes,
        max_hops=max_hops,
        population_size=population_size,
        elite_size=elite_size,
        bipartite=bipartite,
    )

    best_route = None
    best_loss = float("inf")
    best_ppl = float("inf")
    history = []

    start_time = time.time()

    for it in range(1, num_iterations + 1):
        it_start = time.time()
        candidates = searcher.sample_routes(best_route)

        scored_candidates = []
        for r in candidates:
            loss, ppl = evaluate_static_route(model, loader, r, device, max_batches=search_batches)
            scored_candidates.append((loss, ppl, r))

        # Sort by loss ascending
        scored_candidates.sort(key=lambda x: x[0])

        # Best in this generation
        gen_best_loss, gen_best_ppl, gen_best_route = scored_candidates[0]
        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_ppl = gen_best_ppl
            best_route = gen_best_route

        # Select top Ke elites
        elites = [cand[2] for cand in scored_candidates[:elite_size]]
        elite_mean_loss = sum(cand[0] for cand in scored_candidates[:elite_size]) / elite_size

        # Update distribution
        searcher.update_distribution(elites)
        entropy = searcher.compute_entropy()

        elapsed_it = time.time() - it_start
        print(f"  [Gen {it:02d}/{num_iterations:02d}] Best Loss: {best_loss:.4f} (PPL {best_ppl:.2f}) | Elite Mean: {elite_mean_loss:.4f} | Entropy: {entropy:.3f} | Best Route: {best_route} ({elapsed_it:.1f}s)")

        history.append({
            "iteration": it,
            "best_loss": round(best_loss, 4),
            "best_ppl": round(best_ppl, 2),
            "elite_mean_loss": round(elite_mean_loss, 4),
            "distribution_entropy": round(entropy, 4),
            "best_route": best_route,
        })

    # Final thorough evaluation of best route on full eval_batches
    eval_loss, eval_ppl = evaluate_static_route(model, loader, best_route, device, max_batches=eval_batches)
    total_time = time.time() - start_time
    print(f">>> Task [{task_name.upper()}] Search Complete in {total_time:.1f}s!")
    print(f"    Optimal Route r*_{task_name}: {best_route}")
    print(f"    Full Validation Loss: {eval_loss:.4f} | Perplexity: {eval_ppl:.2f}")

    return {
        "task": task_name,
        "optimal_route": best_route,
        "optimal_loss": round(eval_loss, 4),
        "optimal_ppl": round(eval_ppl, 2),
        "search_time_sec": round(total_time, 2),
        "iterations": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Gate 17 Track I: MACRO Offline Route Search")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/navitrit-100m-trained.pt")
    parser.add_argument("--model_size", type=str, default="100m")
    parser.add_argument("--max_hops", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--population_size", type=int, default=64)
    parser.add_argument("--elite_size", type=int, default=8)
    parser.add_argument("--search_batches", type=int, default=8)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--bipartite", action=argparse.BooleanOptionalAction, default=True, help="Enforce alternating Sequence-Channel mixing invariant")
    parser.add_argument("--output_json", type=str, default="outputs/macro-routes.json")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Gate 17: MACRO Offline Route Search on {device} ===")
    print(f"Backbone Checkpoint: {args.checkpoint}")
    print(f"Bipartite Sequence-Channel Invariant: {args.bipartite}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Backbone checkpoint not found: {args.checkpoint}")

    # Load frozen backbone
    cfg = get_scale_config(args.model_size, max_hops=args.max_hops)
    model = NaviTritScaleForCausalLM(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    # Verify zero training / freeze parameters
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    loaders = build_validation_data(tokenizer, seq_len=cfg.block_size)

    # Operational tile set: 12 Attn + 12 FFN + 1 Reasoning Core = 25 tiles
    num_operational_nodes = 2 * cfg.num_layers + 1  # 25 nodes (indices 0..24)

    # 1. Monotonic Baseline: [0, 1, 2, 3, 4, 5]
    mono_route = [t % (2 * cfg.num_layers) for t in range(args.max_hops)]
    print(f"\n--- Evaluating Monotonic Baseline Route: {mono_route} ---")
    mono_results = {}
    for task_name, loader in loaders.items():
        m_loss, m_ppl = evaluate_static_route(model, loader, mono_route, device)
        mono_results[task_name] = {"loss": round(m_loss, 4), "ppl": round(m_ppl, 2)}
        print(f"  Monotonic [{task_name.upper()}]: Loss = {m_loss:.4f} | PPL = {m_ppl:.2f}")

    # 2. Dynamic Router Baseline (Trained controller in navitrit-100m-trained.pt)
    print(f"\n--- Evaluating Base Dynamic Controller Baseline ---")
    dynamic_results = {}
    for task_name, loader in loaders.items():
        d_loss, d_ppl, d_traj = evaluate_dynamic_baseline(model, loader, device)
        dynamic_results[task_name] = {
            "loss": round(d_loss, 4),
            "ppl": round(d_ppl, 2),
            "sample_trajectory": d_traj,
        }
        print(f"  Base Dynamic [{task_name.upper()}]: Loss = {d_loss:.4f} | PPL = {d_ppl:.2f} | Trajectory: {d_traj}")

    # 3. Execute CEM Search for each task
    task_searches = {}
    optimal_routes = {}
    for task_name in ["code", "math", "story"]:
        loader = loaders[task_name]
        res = run_macro_search_for_task(
            task_name=task_name,
            model=model,
            loader=loader,
            device=device,
            num_nodes=num_operational_nodes,
            max_hops=args.max_hops,
            num_iterations=args.iterations,
            population_size=args.population_size,
            elite_size=args.elite_size,
            search_batches=args.search_batches,
            eval_batches=args.eval_batches,
            bipartite=args.bipartite,
        )
        task_searches[task_name] = res
        optimal_routes[task_name] = res["optimal_route"]

    # 4. Summary & Comparison Table
    print(f"\n=======================================================")
    print(f"              GATE 17 MACRO SEARCH SUMMARY             ")
    print(f"=======================================================")
    print(f"| Task   | Monotonic PPL | Base Dynamic PPL | MACRO Static PPL | Optimal Route r* |")
    print(f"|--------|---------------|------------------|------------------|------------------|")
    for task_name in ["code", "math", "story"]:
        m_p = mono_results[task_name]["ppl"]
        d_p = dynamic_results[task_name]["ppl"]
        s_p = task_searches[task_name]["optimal_ppl"]
        r_opt = optimal_routes[task_name]
        print(f"| {task_name:6s} | {m_p:13.2f} | {d_p:16.2f} | {s_p:16.2f} | {str(r_opt):16s} |")
    print(f"=======================================================")

    # 5. Extract Scored Graphs of Traversals using NaviTritGraphForCausalLM
    print(f"\n--- Extracting Task Graphs-of-Traversals with Scored Weights ---")
    tree_ckpt = "outputs/checkpoints/navitrit-100m-tree-grpo.pt"
    task_traversal_graphs = {}
    if os.path.exists(tree_ckpt):
        from experiments.frontier_scaling.navitrit_graph_model import NaviTritGraphForCausalLM
        graph_model = NaviTritGraphForCausalLM(cfg).to(device)
        graph_model.load_state_dict(torch.load(tree_ckpt, map_location=device, weights_only=True))
        graph_model.eval()

        for task_name, loader in loaders.items():
            is_math = (task_name == "math")
            tg_res = extract_task_traversal_graph(graph_model, loader, device, is_math=is_math, max_batches=args.eval_batches)
            task_traversal_graphs[task_name] = tg_res
            print(f"  Graph-of-Traversals [{task_name.upper()}]: Loss = {tg_res['loss']:.4f} | PPL = {tg_res['ppl']:.2f}")
            print(f"    Pairs: {tg_res['traversal_pairs']}")
            print(f"    Node Weights: {tg_res['node_weight_scores']}")
        del graph_model
        torch.cuda.empty_cache()

    # 6. Export results
    output_data = {
        "gate": "Gate 17",
        "track": "Phase II Track I: MACRO Offline Search & Graph of Traversals",
        "date": "2026-09-17",
        "backbone_checkpoint": args.checkpoint,
        "tree_backbone_checkpoint": tree_ckpt,
        "model_size": args.model_size,
        "max_hops": args.max_hops,
        "population_size": args.population_size,
        "elite_size": args.elite_size,
        "iterations": args.iterations,
        "monotonic_baseline": mono_results,
        "base_dynamic_baseline": dynamic_results,
        "discovered_routes": optimal_routes,
        "task_searches": task_searches,
        "task_traversal_graphs": task_traversal_graphs,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n>>> Full MACRO route & Graph-of-Traversals ledger written to: {args.output_json}")


if __name__ == "__main__":
    main()
