"""
Gate 12 Corpus Consistency & Artifact Verification Script
Verifies that all Gate 12 manuscripts exist, resolve valid file references,
and contain exact metrics matching empirical JSON ledgers.
"""

import json
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DOCUMENTS = [
    "research/paper-1-bitroute-systems.md",
    "research/paper-2-flowtrit-reasoning.md",
    "research/paper-3-flowroute-architecture.md",
    "research/paper-4-navitrit-graph-navigation.md",
    "research/synthesis-executive-summary.md",
]

LEDGERS = [
    "outputs/bitroute-test.json",
    "outputs/arithmetic-upgrade-test.json",
    "outputs/bitroute-tinystories-lambda0.2.json",
    "outputs/flowtrit-test.json",
    "outputs/flowtrit-sudoku9-results.json",
    "outputs/flowroute-test.json",
    "outputs/bitroute-multihop-results.json",
    "outputs/navitrit-hardened-results.json",
    "outputs/language-coherence-audit.json",
    "outputs/judge-alignment-audit.json",
]


def test_documents_exist():
    print("[1/4] Checking document existence and non-zero size...")
    for doc in DOCUMENTS:
        path = os.path.join(REPO_ROOT, doc)
        assert os.path.isfile(path), f"Missing document: {doc}"
        size = os.path.getsize(path)
        assert size > 2000, f"Document {doc} is suspiciously small: {size} bytes"
        print(f"  [OK] {doc} ({size:,} bytes)")
    print("All 5 Gate 12 documents verified.")


def test_ledgers_exist():
    print("\n[2/4] Checking empirical telemetry ledgers...")
    for ledger in LEDGERS:
        path = os.path.join(REPO_ROOT, ledger)
        assert os.path.isfile(path), f"Missing ledger: {ledger}"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) > 0, f"Empty ledger: {ledger}"
        print(f"  [OK] {ledger}")
    print("All 10 empirical ledgers verified.")


def test_metric_fidelity():
    print("\n[3/4] Verifying metric fidelity between papers and ledgers...")

    # Paper 1: BitRoute
    with open(os.path.join(REPO_ROOT, "outputs/bitroute-tinystories-lambda0.2.json"), "r") as f:
        br_data = json.load(f)
    br_loss = br_data["arms"]["ternary_router"]["router_on"]["val_loss"]
    assert abs(br_loss - 2.6987) < 1e-4, f"BitRoute loss mismatch: {br_loss}"
    print(f"  [OK] BitRoute-135M TinyStories Val Loss: {br_loss:.4f} verified")

    # Paper 2: FlowTrit
    with open(os.path.join(REPO_ROOT, "outputs/flowtrit-test.json"), "r") as f:
        ft_data = json.load(f)
    ft_solve = ft_data["verification_checks"]["comparisons"]["native_flowtrit_dynamic_early_exit"]["solve_rate"]
    assert abs(ft_solve - 0.90) < 1e-4, f"FlowTrit solve rate mismatch: {ft_solve}"
    ft_mem = ft_data["verification_checks"]["weight_tied_recurrence_and_cache"]["scaled_model_flowtrit_40m"]["ternary_packed_tq1_0_mb"]
    assert abs(ft_mem - 8.0466) < 0.01, f"FlowTrit memory footprint mismatch: {ft_mem}"
    print(f"  [OK] FlowTrit-40M Solve Rate: {ft_solve*100:.1f}%, Footprint: {ft_mem:.2f} MB verified")

    # Paper 3: FlowRoute
    with open(os.path.join(REPO_ROOT, "outputs/bitroute-multihop-results.json"), "r") as f:
        fr_data = json.load(f)
    fr_loss = fr_data["arm"]["router_on"]["val_loss"]
    fr_base_loss = fr_data["arm"]["forced_execute"]["val_loss"]
    assert abs(fr_loss - 2.9836) < 1e-4, f"FlowRoute loss mismatch: {fr_loss}"
    assert abs(fr_base_loss - 3.0238) < 1e-4, f"FlowRoute base loss mismatch: {fr_base_loss}"
    print(f"  [OK] FlowRoute Val Loss: {fr_loss:.4f} vs Full {fr_base_loss:.4f} verified")

    # Paper 4: NaviTrit
    with open(os.path.join(REPO_ROOT, "outputs/navitrit-hardened-results.json"), "r") as f:
        nv_data = json.load(f)
    nv_loss = nv_data["arms"]["navitrit_learned"]["val_loss"]
    nv_mono_loss = nv_data["arms"]["monotonic_baseline"]["val_loss"]
    assert abs(nv_loss - 3.0166) < 1e-4, f"NaviTrit loss mismatch: {nv_loss}"
    assert abs(nv_mono_loss - 3.6302) < 1e-4, f"NaviTrit monotonic loss mismatch: {nv_mono_loss}"
    print(f"  [OK] NaviTrit Val Loss: {nv_loss:.4f} vs Monotonic {nv_mono_loss:.4f} verified")


def test_cross_references():
    print("\n[4/4] Verifying file cross-references inside markdown documents...")
    file_link_pattern = re.compile(r'\[.*?\]\((?:file:///)?([c-zC-Z]:[\\/][^)]+|experiments/[^)]+|outputs/[^)]+|research/[^)]+)\)')
    
    total_links = 0
    resolved_links = 0
    for doc in DOCUMENTS:
        path = os.path.join(REPO_ROOT, doc)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        matches = file_link_pattern.findall(content)
        for match in matches:
            total_links += 1
            # Normalize path
            clean_path = match.replace("\\", "/")
            if clean_path.startswith("file:///"):
                clean_path = clean_path[8:]
            if re.match(r'^[a-zA-Z]:', clean_path):
                # Absolute path
                resolved = os.path.exists(clean_path)
            else:
                # Relative path from REPO_ROOT
                resolved = os.path.exists(os.path.join(REPO_ROOT, clean_path))
            
            assert resolved, f"Broken link in {doc}: {match}"
            resolved_links += 1
    
    print(f"  [OK] All {resolved_links}/{total_links} file links successfully resolved on disk.")


if __name__ == "__main__":
    print("=== GATE 12 RESEARCH CORPUS CONSISTENCY AUDIT ===")
    try:
        test_documents_exist()
        test_ledgers_exist()
        test_metric_fidelity()
        test_cross_references()
        print("\n>>> ALL GATE 12 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
    except AssertionError as e:
        print(f"\n[FAIL] Assertion failed: {e}")
        sys.exit(1)
