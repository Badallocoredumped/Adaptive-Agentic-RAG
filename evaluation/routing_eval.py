"""
evaluation/routing_eval.py
===========================
Evaluates system behavior metrics directly from your pipeline internals.
Answers your three research questions:

  RQ1: Does caching reduce latency for recurring queries?
       → Measures Fast Track (cache hit) vs Reasoning Track (cache miss) latency

  RQ2: How accurately does the Router Agent classify queries?
       → Runs the 30-query routing_eval_set.json, compares predicted vs ground truth

  RQ3: Does TableRAG schema pruning reduce SQL error rates?
       → Compares SQL errors with full schema vs pruned schema

Usage:
    python evaluation/routing_eval.py           # run all three RQs
    python evaluation/routing_eval.py --rq 1    # RQ1 only
    python evaluation/routing_eval.py --rq 2    # RQ2 only
    python evaluation/routing_eval.py --rq 3    # RQ3 only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
BENCHMARK_DIR  = PROJECT_ROOT / "benchmark"
RESULTS_DIR    = PROJECT_ROOT / "evaluation" / "results"
ROUTING_SET    = BENCHMARK_DIR / "routing_eval_set.json"
RQ_RESULTS     = RESULTS_DIR / "routing_eval_results.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# RQ1 — CACHE LATENCY
# Measure Fast Track (cache hit) vs Reasoning Track (cache miss) latency.
# Uses your real SQL pipeline's `path` and `latency` fields.
# =============================================================================

def run_rq1(n_queries: int = 20) -> dict:
    """
    RQ1: Cache latency reduction.

    Strategy:
      - Run N novel queries → cache misses → Reasoning Track (populates cache)
      - Re-run the same queries → cache hits → Fast Track
      - Compare average latency between the two passes
    """
    print("\n" + "="*60)
    print("  RQ1: Cache latency (Fast Track vs Reasoning Track)")
    print("="*60)

    from backend import config                          # noqa: PLC0415
    from backend.sql import run_table_rag_pipeline      # noqa: PLC0415

    # Load a few benchmark SQL questions for realistic queries
    with open(BENCHMARK_DIR / "benchmark.json") as f:
        benchmark = json.load(f)

    sql_examples = [ex for ex in benchmark if ex["answer_source"] == "sql"][:n_queries]

    first_pass  = []  # cache misses
    second_pass = []  # cache hits

    for ex in sql_examples:
        # Point config at the right DB
        config.SQLITE_PATH = str(BENCHMARK_DIR / ex["db_path"])

        # Clear schema index for new DB
        for fname in ["schema.faiss", "schema_texts.json"]:
            p = getattr(config, "INDEX_DIR", Path(".")) / fname
            if Path(p).exists():
                Path(p).unlink()

        q = ex["question"]

        # First pass — cold (cache miss → Reasoning Track)
        try:
            result1 = run_table_rag_pipeline(q)
            first_pass.append({
                "question" : q,
                "latency_s": result1.get("latency", 0),
                "path"     : result1.get("path", "unknown"),
                "error"    : result1.get("error"),
            })
        except Exception as e:
            first_pass.append({"question": q, "latency_s": 0, "path": "error", "error": str(e)})

        # Second pass — warm (cache hit → Fast Track)
        try:
            result2 = run_table_rag_pipeline(q)
            second_pass.append({
                "question" : q,
                "latency_s": result2.get("latency", 0),
                "path"     : result2.get("path", "unknown"),
                "error"    : result2.get("error"),
            })
        except Exception as e:
            second_pass.append({"question": q, "latency_s": 0, "path": "error", "error": str(e)})

        print(f"  Q: {q[:55]}")
        print(f"     pass1={first_pass[-1]['latency_s']:.2f}s ({first_pass[-1]['path']}) | "
              f"pass2={second_pass[-1]['latency_s']:.2f}s ({second_pass[-1]['path']})")

    avg_miss = sum(r["latency_s"] for r in first_pass)  / max(len(first_pass), 1)
    avg_hit  = sum(r["latency_s"] for r in second_pass) / max(len(second_pass), 1)
    reduction = ((avg_miss - avg_hit) / avg_miss * 100) if avg_miss > 0 else 0

    # Count actual path labels
    miss_paths = [r["path"] for r in first_pass]
    hit_paths  = [r["path"] for r in second_pass]

    result = {
        "rq"                   : 1,
        "n_queries"            : len(sql_examples),
        "avg_latency_miss_s"   : round(avg_miss, 3),
        "avg_latency_hit_s"    : round(avg_hit, 3),
        "latency_reduction_pct": round(reduction, 1),
        "first_pass_paths"     : {p: miss_paths.count(p) for p in set(miss_paths)},
        "second_pass_paths"    : {p: hit_paths.count(p) for p in set(hit_paths)},
        "per_query"            : {
            "first_pass" : first_pass,
            "second_pass": second_pass,
        },
    }

    print(f"\n  ── RQ1 Summary ──────────────────────────")
    print(f"  Avg latency (cache miss / Reasoning Track): {avg_miss:.3f}s")
    print(f"  Avg latency (cache hit  / Fast Track)     : {avg_hit:.3f}s")
    print(f"  Latency reduction                         : {reduction:.1f}%")
    print(f"  First-pass  paths: {result['first_pass_paths']}")
    print(f"  Second-pass paths: {result['second_pass_paths']}")

    return result


# =============================================================================
# RQ2 — ROUTING ACCURACY
# Run the 30-query routing_eval_set.json through your router and compare
# predicted route vs ground_truth_route.
# =============================================================================

def run_rq2() -> dict:
    """
    RQ2: Router Agent classification accuracy.

    Runs all 30 queries in routing_eval_set.json through your router
    and computes accuracy per category and overall.
    """
    print("\n" + "="*60)
    print("  RQ2: Router Agent classification accuracy")
    print("="*60)

    from backend.router import QueryRouter  # noqa: PLC0415

    with open(ROUTING_SET) as f:
        routing_data = json.load(f)

    router  = QueryRouter()
    queries = routing_data["queries"]
    records = []

    for q in queries:
        qid       = q["id"]
        question  = q["question"]
        expected  = q["ground_truth_route"]
        db_tid    = q.get("table_id", "")

        try:
            # Use your router's zero-shot classification
            predicted = router.route(question)
        except Exception as e:
            predicted = "error"
            print(f"  [{qid}] ERROR: {e}")

        correct = int(predicted == expected)
        records.append({
            "id"       : qid,
            "question" : question,
            "expected" : expected,
            "predicted": predicted,
            "correct"  : correct,
        })
        mark = "✓" if correct else "✗"
        print(f"  {mark} [{qid}] expected={expected:7s} predicted={predicted:7s} | {question[:50]}")

    # Aggregate
    overall_acc = sum(r["correct"] for r in records) / len(records) * 100

    by_category: dict[str, dict] = {}
    for cat in ["sql", "text", "hybrid"]:
        subset = [r for r in records if r["expected"] == cat]
        acc    = sum(r["correct"] for r in subset) / max(len(subset), 1) * 100
        by_category[cat] = {
            "n"       : len(subset),
            "correct" : sum(r["correct"] for r in subset),
            "accuracy": round(acc, 1),
        }

    # Confusion matrix
    cats = ["sql", "text", "hybrid"]
    confusion = {e: {p: 0 for p in cats} for e in cats}
    for r in records:
        e = r["expected"]
        p = r["predicted"] if r["predicted"] in cats else "other"
        if e in confusion:
            confusion[e][p] = confusion[e].get(p, 0) + 1

    result = {
        "rq"             : 2,
        "n_queries"      : len(queries),
        "overall_accuracy": round(overall_acc, 1),
        "by_category"    : by_category,
        "confusion_matrix": confusion,
        "per_query"      : records,
    }

    print(f"\n  ── RQ2 Summary ──────────────────────────")
    print(f"  Overall routing accuracy: {overall_acc:.1f}%")
    for cat, s in by_category.items():
        print(f"  {cat:8s}: {s['correct']}/{s['n']} correct = {s['accuracy']:.1f}%")
    print(f"\n  Confusion matrix (rows=expected, cols=predicted):")
    print(f"  {'':10} {'sql':>8} {'text':>8} {'hybrid':>8}")
    for e_cat in cats:
        row = confusion.get(e_cat, {})
        print(f"  {e_cat:10} {row.get('sql',0):>8} {row.get('text',0):>8} {row.get('hybrid',0):>8}")

    return result


# =============================================================================
# RQ3 — SCHEMA PRUNING SQL ERROR RATE
# Compare SQL error rate with full schema vs TableRAG pruned schema.
# =============================================================================

def run_rq3(n_queries: int = 20) -> dict:
    """
    RQ3: Does TableRAG schema pruning reduce SQL syntax/execution errors?

    Strategy:
      - Run N SQL queries with schema pruning ON (your current pipeline)
      - Run the same queries with schema pruning OFF (full schema injected)
      - Compare error rates
    """
    print("\n" + "="*60)
    print("  RQ3: Schema pruning SQL error rate")
    print("="*60)

    from backend import config                      # noqa: PLC0415
    from backend.sql.sql_agent import (             # noqa: PLC0415
        run_sql_agent,
        _load_schema_dict,
    )
    from backend.sql.table_rag import (             # noqa: PLC0415
        build_schema_index,
        retrieve_relevant_schema,
        get_schema_texts,
    )

    with open(BENCHMARK_DIR / "benchmark.json") as f:
        benchmark = json.load(f)

    sql_examples = [ex for ex in benchmark if ex["answer_source"] == "sql"][:n_queries]

    pruned_results   = []
    unpruned_results = []

    for ex in sql_examples:
        config.SQLITE_PATH = str(BENCHMARK_DIR / ex["db_path"])
        for fname in ["schema.faiss", "schema_texts.json"]:
            p = getattr(config, "INDEX_DIR", Path(".")) / fname
            if Path(p).exists():
                Path(p).unlink()

        q = ex["question"]

        # ── WITH pruning (TableRAG) ────────────────────────────
        try:
            schema_dict = _load_schema_dict()
            build_schema_index(schema_dict)
            pruned_schema = retrieve_relevant_schema(q, top_k=config.SQL_TOP_K)
            r_pruned = run_sql_agent(q, schema_context=pruned_schema)
            pruned_results.append({
                "question"      : q,
                "sql"           : r_pruned.get("sql"),
                "error"         : r_pruned.get("error"),
                "schema_tokens" : len(pruned_schema.split()),
                "has_error"     : bool(r_pruned.get("error")),
            })
        except Exception as e:
            pruned_results.append({"question": q, "sql": None, "error": str(e), "has_error": True, "schema_tokens": 0})

        # ── WITHOUT pruning (full schema) ──────────────────────
        try:
            schema_dict = _load_schema_dict()
            # Inject ALL tables into context (no pruning)
            full_schema = get_schema_texts(schema_dict)
            full_context = "\n".join(full_schema)
            r_full = run_sql_agent(q, schema_context=full_context)
            unpruned_results.append({
                "question"      : q,
                "sql"           : r_full.get("sql"),
                "error"         : r_full.get("error"),
                "schema_tokens" : len(full_context.split()),
                "has_error"     : bool(r_full.get("error")),
            })
        except Exception as e:
            unpruned_results.append({"question": q, "sql": None, "error": str(e), "has_error": True, "schema_tokens": 0})

        p_err = "✗" if pruned_results[-1]["has_error"] else "✓"
        u_err = "✗" if unpruned_results[-1]["has_error"] else "✓"
        p_tok = pruned_results[-1]["schema_tokens"]
        u_tok = unpruned_results[-1]["schema_tokens"]
        print(f"  Q: {q[:55]}")
        print(f"     pruned  {p_err}  ({p_tok} tokens) | full {u_err}  ({u_tok} tokens)")

    n = len(sql_examples)
    pruned_errors   = sum(r["has_error"] for r in pruned_results)
    unpruned_errors = sum(r["has_error"] for r in unpruned_results)
    pruned_err_rate   = pruned_errors   / n * 100
    unpruned_err_rate = unpruned_errors / n * 100
    avg_pruned_tokens   = sum(r["schema_tokens"] for r in pruned_results) / max(n, 1)
    avg_unpruned_tokens = sum(r["schema_tokens"] for r in unpruned_results) / max(n, 1)
    token_reduction = (
        (avg_unpruned_tokens - avg_pruned_tokens) / avg_unpruned_tokens * 100
        if avg_unpruned_tokens > 0 else 0
    )

    result = {
        "rq"                      : 3,
        "n_queries"               : n,
        "pruned_error_count"      : pruned_errors,
        "pruned_error_rate_pct"   : round(pruned_err_rate, 1),
        "unpruned_error_count"    : unpruned_errors,
        "unpruned_error_rate_pct" : round(unpruned_err_rate, 1),
        "avg_pruned_schema_tokens"  : round(avg_pruned_tokens, 1),
        "avg_unpruned_schema_tokens": round(avg_unpruned_tokens, 1),
        "token_reduction_pct"       : round(token_reduction, 1),
        "per_query": {
            "pruned"  : pruned_results,
            "unpruned": unpruned_results,
        },
    }

    print(f"\n  ── RQ3 Summary ──────────────────────────")
    print(f"  SQL error rate  WITH pruning  : {pruned_err_rate:.1f}%  ({pruned_errors}/{n})")
    print(f"  SQL error rate WITHOUT pruning: {unpruned_err_rate:.1f}%  ({unpruned_errors}/{n})")
    print(f"  Avg schema tokens with pruning : {avg_pruned_tokens:.0f}")
    print(f"  Avg schema tokens without      : {avg_unpruned_tokens:.0f}")
    print(f"  Token reduction from pruning   : {token_reduction:.1f}%")

    return result


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate system behavior (RQ1/RQ2/RQ3)")
    parser.add_argument(
        "--rq",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run only one RQ (default: run all three)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of SQL queries to use for RQ1 and RQ3 (default: 20)",
    )
    args = parser.parse_args()

    all_results = {}

    try:
        if args.rq is None or args.rq == 1:
            all_results["rq1"] = run_rq1(n_queries=args.n)
    except Exception as e:
        print(f"\n[ERROR] RQ1 failed: {e}")
        traceback.print_exc()
        all_results["rq1"] = {"error": str(e)}

    try:
        if args.rq is None or args.rq == 2:
            all_results["rq2"] = run_rq2()
    except Exception as e:
        print(f"\n[ERROR] RQ2 failed: {e}")
        traceback.print_exc()
        all_results["rq2"] = {"error": str(e)}

    try:
        if args.rq is None or args.rq == 3:
            all_results["rq3"] = run_rq3(n_queries=args.n)
    except Exception as e:
        print(f"\n[ERROR] RQ3 failed: {e}")
        traceback.print_exc()
        all_results["rq3"] = {"error": str(e)}

    # Save all results
    with open(RQ_RESULTS, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nAll RQ results saved to {RQ_RESULTS}")
    print("Done.")


if __name__ == "__main__":
    main()
