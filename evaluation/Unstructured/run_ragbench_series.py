"""
evaluation/Unstructured/run_ragbench_series.py
===============================================
Runs all T-series configs against RAGBench sequentially,
then prints a comparison table identical to the SQuAD analysis.

Usage:
    python evaluation/Unstructured/run_ragbench_series.py
    python evaluation/Unstructured/run_ragbench_series.py --configs T5A T3A C1
    python evaluation/Unstructured/run_ragbench_series.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
RUNNER      = SCRIPT_DIR / "run_ragbench_eval.py"
RESULTS_DIR = SCRIPT_DIR / "results"

DEFAULT_CONFIGS = ["T2A", "T2B", "T3A", "T3B", "T4A", "T4B", "T5A", "T5B"]


def print_comparison_table(configs: list[str]) -> None:
    """Print the same summary table format as the SQuAD analysis."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*110}")
    print(f"{'Config':<8} {'GoldHit':>8} {'AnsRel':>8} {'Faith':>8} {'CtxPrec':>8} {'CtxRec':>8} "
          f"{'NotFound':>9} | params")
    print(f"{'='*110}")

    for cfg_name in configs:
        raw_path   = RESULTS_DIR / f"ragbench_{cfg_name}_rag_results.json"
        score_path = RESULTS_DIR / f"ragbench_{cfg_name}_ragas_scores.json"

        if not raw_path.exists() or not score_path.exists():
            print(f"{cfg_name:<8} {'(no results)':>50}")
            continue

        rag    = json.loads(raw_path.read_text())
        scores = json.loads(score_path.read_text())
        params = scores["parameters"]
        overall = scores["overall"]

        gold_results = [r for r in rag if r.get("gold_hit") is not None]
        gold_hit = sum(r["gold_hit"] for r in gold_results) / len(gold_results) if gold_results else 0
        not_found = sum(1 for r in rag if "not found" in r.get("answer","").lower())
        n = len(rag)

        param_str = (
            f"reranker={params['reranker']} top_k={params['top_k']} "
            f"fetch_k={params.get('fetch_k','?')} pool={params.get('rerank_pool','?')}"
        )

        print(
            f"{cfg_name:<8} {gold_hit:>8.3f} "
            f"{overall.get('answer_relevancy',0):>8.4f} "
            f"{overall.get('faithfulness',0):>8.4f} "
            f"{overall.get('context_precision',0):>8.4f} "
            f"{overall.get('context_recall',0):>8.4f} "
            f"{not_found:>6}/{n:<3} | {param_str}"
        )

    print(f"{'='*110}")

    # Per-subset breakdown for available configs
    print(f"\nPer-subset context_precision:")
    header_printed = False
    for cfg_name in configs:
        score_path = RESULTS_DIR / f"ragbench_{cfg_name}_ragas_scores.json"
        if not score_path.exists():
            continue
        scores = json.loads(score_path.read_text())
        per_sub = scores.get("per_subset", {})
        if not per_sub:
            continue
        if not header_printed:
            subsets = sorted(per_sub.keys())
            print(f"  {'Config':<8} " + " ".join(f"{s:<12}" for s in subsets))
            print(f"  {'-'*8} " + " ".join(f"{'-'*12}" for _ in subsets))
            header_printed = True
        vals = " ".join(
            f"{per_sub.get(s, {}).get('context_precision', 0):<12.4f}"
            for s in subsets
        )
        print(f"  {cfg_name:<8} {vals}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip configs that already have result files")
    args = parser.parse_args()

    summary = []

    for cfg_name in args.configs:
        raw_path = RESULTS_DIR / f"ragbench_{cfg_name}_rag_results.json"
        score_path = RESULTS_DIR / f"ragbench_{cfg_name}_ragas_scores.json"

        if args.skip_existing and raw_path.exists() and score_path.exists():
            print(f"[SKIP] ragbench_{cfg_name} — results already exist")
            summary.append((cfg_name, "SKIPPED", 0))
            continue

        print(f"\n{'#'*65}")
        print(f"  Starting ragbench_{cfg_name}  "
              f"({args.configs.index(cfg_name)+1}/{len(args.configs)})")
        print(f"{'#'*65}\n")

        start = time.time()
        result = subprocess.run(
            [sys.executable, str(RUNNER), cfg_name],
            check=False,
        )
        elapsed = time.time() - start
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        summary.append((cfg_name, status, elapsed))
        print(f"\n[ragbench_{cfg_name}] {status}  —  {elapsed/60:.1f} min")

    print(f"\n{'='*65}")
    print("  T-SERIES RAGBENCH RUN COMPLETE")
    print(f"{'='*65}")
    for name, status, elapsed in summary:
        if elapsed > 0:
            print(f"  {name:<6}  {status:<25}  {elapsed/60:.1f} min")
        else:
            print(f"  {name:<6}  {status}")
    print(f"{'='*65}\n")

    # Print comparison table
    print_comparison_table(args.configs)


if __name__ == "__main__":
    main()
