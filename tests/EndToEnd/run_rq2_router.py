"""
run_rq2_router.py
─────────────────────────────────────────────────────────────────────────────
RQ2 — Router Classification Accuracy

Tests the Zero-Shot LLM Router Agent's ability to correctly classify queries
into sql, text, or hybrid routes across the 90-query classification set:
  • SQL-001  … SQL-030   (30 sql   queries)
  • TXT-001  … TXT-030   (30 text  queries)
  • HYB-001  … HYB-030   (30 hybrid queries)

Metrics computed
─────────────────
  • Per-class accuracy  (sql / text / hybrid)
  • Overall accuracy    (target ≥ 80% = 72/90)
  • Confusion matrix    (3 × 3)
  • Most common misclassifications

Outputs
────────
  results/rq2_results.json      — per-query results
  results/rq2_summary.json      — accuracy scores + confusion matrix
─────────────────────────────────────────────────────────────────────────────
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from config import BENCHMARK_FILE, RESULTS_DIR, INTER_QUERY_DELAY_SEC, VERBOSE
from system_adapter import call_system

CLASSES = ["sql", "text", "hybrid"]
TARGET_OVERALL_ACCURACY = 0.80   # 72/90

# ─────────────────────────────────────────────────────────────────────────────

def load_rq2_queries() -> list[dict]:
    """Return SQL + TXT + HYB queries (the 90-query classification set)."""
    data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    return [q for q in data if q["qid"].split("-")[0] in ("SQL", "TXT", "HYB")]


def run_classification(queries: list[dict]) -> list[dict]:
    results = []
    total = len(queries)

    print(f"\n{'='*70}")
    print(f"  RQ2 Router Classification — {total} queries")
    print(f"{'='*70}")

    for idx, q in enumerate(queries, 1):
        qid    = q["qid"]
        query  = q["query"]
        truth  = q["ground_truth_route"]

        if VERBOSE:
            print(f"\n[{idx:>2}/{total}] {qid} [{truth}]: {query[:65]}…")

        response = call_system(query)

        predicted = (response["router_decision"] or "").strip().lower()
        correct   = predicted == truth

        record = {
            "qid":                qid,
            "query":              query,
            "ground_truth_route": truth,
            "predicted_route":    predicted,
            "correct":            correct,
            "latency_ms":         response["latency_ms"],
            "error":              response["error"],
            "answer_snippet":     (response["answer"] or "")[:200],
            "timestamp":          datetime.utcnow().isoformat(),
        }

        if VERBOSE:
            icon = "✓" if correct else "✗"
            arrow = f"expected={truth}  predicted={predicted or '(none)'}"
            print(f"       {icon} {arrow}  |  {response['latency_ms']:.0f} ms")
            if response["error"]:
                print(f"       ERROR: {response['error']}")

        results.append(record)

        if idx < total:
            time.sleep(INTER_QUERY_DELAY_SEC)

    return results


def build_confusion_matrix(results: list[dict]) -> dict:
    """
    Returns a dict-of-dicts: matrix[true][predicted] = count.
    e.g. matrix["sql"]["hybrid"] = number of sql queries routed to hybrid.
    """
    matrix = {t: {p: 0 for p in CLASSES + ["other"]} for t in CLASSES}
    for r in results:
        truth = r["ground_truth_route"]
        pred  = r["predicted_route"] if r["predicted_route"] in CLASSES else "other"
        if truth in CLASSES:
            matrix[truth][pred] += 1
    return matrix


def build_summary(results: list[dict]) -> dict:
    per_class = {}
    for cls in CLASSES:
        cls_results = [r for r in results if r["ground_truth_route"] == cls]
        correct     = sum(1 for r in cls_results if r["correct"])
        total       = len(cls_results)
        per_class[cls] = {
            "total":    total,
            "correct":  correct,
            "accuracy": round(correct / total, 4) if total else 0,
        }

    total_correct = sum(v["correct"] for v in per_class.values())
    total_queries = len(results)
    overall_acc   = round(total_correct / total_queries, 4) if total_queries else 0
    passes        = overall_acc >= TARGET_OVERALL_ACCURACY

    confusion = build_confusion_matrix(results)

    # Most common errors
    errors = [
        {"qid": r["qid"], "truth": r["ground_truth_route"], "predicted": r["predicted_route"]}
        for r in results if not r["correct"]
    ]

    # Latency stats
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else None

    return {
        "rq":                 "RQ2 — Router Classification Accuracy",
        "run_timestamp":      datetime.utcnow().isoformat(),
        "total_queries":      total_queries,
        "total_correct":      total_correct,
        "overall_accuracy":   overall_acc,
        "target_accuracy":    TARGET_OVERALL_ACCURACY,
        "passes_threshold":   passes,
        "verdict": (
            f"✓ PASS — {overall_acc:.1%} accuracy ({total_correct}/{total_queries})"
            if passes else
            f"✗ FAIL — {overall_acc:.1%} accuracy ({total_correct}/{total_queries}), "
            f"need ≥ {TARGET_OVERALL_ACCURACY:.0%}"
        ),
        "per_class_accuracy": per_class,
        "confusion_matrix":   confusion,
        "misclassified":      errors,
        "avg_latency_ms":     avg_latency,
    }


def print_confusion_matrix(matrix: dict):
    print("\n  Confusion Matrix (rows=truth, cols=predicted):")
    header = f"  {'truth \\ pred':>15}  " + "  ".join(f"{c:>8}" for c in CLASSES + ["other"])
    print(header)
    print("  " + "-" * (len(header) - 2))
    for truth in CLASSES:
        row = f"  {truth:>15}  " + "  ".join(f"{matrix[truth][p]:>8}" for p in CLASSES + ["other"])
        print(row)


def main():
    print("=" * 70)
    print("  RQ2 Router Classification Accuracy Benchmark")
    print("=" * 70)

    queries = load_rq2_queries()
    if not queries:
        print("ERROR: No SQL/TXT/HYB queries found.")
        sys.exit(1)

    print(f"  Loaded {len(queries)} queries: "
          f"{sum(1 for q in queries if q['ground_truth_route']=='sql')} sql  |  "
          f"{sum(1 for q in queries if q['ground_truth_route']=='text')} text  |  "
          f"{sum(1 for q in queries if q['ground_truth_route']=='hybrid')} hybrid")

    results = run_classification(queries)

    # Save per-query results
    results_path = RESULTS_DIR / "rq2_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  ✓ Per-query results saved → {results_path}")

    # Build and save summary
    summary = build_summary(results)
    summary_path = RESULTS_DIR / "rq2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print report
    print(f"\n{'='*70}")
    print(f"  RQ2 SUMMARY")
    print(f"{'='*70}")
    for cls in CLASSES:
        s = summary["per_class_accuracy"][cls]
        bar = "█" * int(s["accuracy"] * 20)
        print(f"  {cls:<8}  {s['correct']:>2}/{s['total']}  {s['accuracy']:>6.1%}  {bar}")
    print(f"\n  Overall: {summary['total_correct']}/{summary['total_queries']}  "
          f"= {summary['overall_accuracy']:.1%}")
    print(f"\n  {summary['verdict']}")

    if summary["misclassified"]:
        print(f"\n  Misclassified ({len(summary['misclassified'])} queries):")
        for e in summary["misclassified"][:10]:
            print(f"    {e['qid']}: truth={e['truth']}  predicted={e['predicted']}")
        if len(summary["misclassified"]) > 10:
            print(f"    … and {len(summary['misclassified'])-10} more (see rq2_results.json)")

    print_confusion_matrix(summary["confusion_matrix"])
    print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
