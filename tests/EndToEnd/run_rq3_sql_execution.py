"""
run_rq3_sql_execution.py
─────────────────────────────────────────────────────────────────────────────
RQ3 — SQL Execution Success Rate & TableRAG Schema Pruning

Runs the 60-query SQL execution set:
  • SQL-001  … SQL-030   (30 standard queries)
  • CSQL-001 … CSQL-030  (30 complex multi-join / window-function queries)

For each query the runner:
  1. Calls the system with TableRAG pruning ENABLED  (normal run)
  2. Calls the system with TableRAG pruning DISABLED (baseline comparison)
     — controlled via the 'tablerag_pruning' kwarg passed to run_query()
  3. Compares execution success rates to prove pruning helps (or doesn't hurt)
  4. Checks which tables the pruner kept vs the expected_tables ground truth

Metrics computed
─────────────────
  • execution_success_rate_pruned   (target ≥ execution_success_rate_full)
  • execution_success_rate_full     (baseline — full schema exposed)
  • pruning_precision               (% of kept tables that were actually needed)
  • pruning_recall                  (% of needed tables that were kept)
  • per-complexity breakdown        (simple / medium / complex)

Outputs
────────
  results/rq3_results.json      — per-query results (both modes)
  results/rq3_summary.json      — aggregated metrics
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

# ─────────────────────────────────────────────────────────────────────────────

def load_rq3_queries() -> list[dict]:
    data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    return [q for q in data if q["qid"].split("-")[0] in ("SQL", "CSQL")]


def run_single(q: dict, pruning_enabled: bool) -> dict:
    """Run one query in one pruning mode; return a result record."""
    response = call_system(q["query"], tablerag_pruning=pruning_enabled)

    executed_ok = (
        response["error"] is None
        and response["sql_result"] is not None
    )

    # Tables the system actually used (pruner output) vs what was expected
    expected_tables = set(
        t.strip() for t in q.get("expected_tables", "").split(",") if t.strip()
    )
    # Your system should return the tables it queried in sql_result or a
    # dedicated key; adapt the key name in system_adapter._normalise() if needed.
    kept_tables_raw = response.get("tables_used") or []
    kept_tables     = set(kept_tables_raw) if isinstance(kept_tables_raw, list) else set()

    precision = (
        len(kept_tables & expected_tables) / len(kept_tables)
        if kept_tables else None
    )
    recall = (
        len(kept_tables & expected_tables) / len(expected_tables)
        if expected_tables else None
    )

    return {
        "executed_ok":       executed_ok,
        "sql_executed":      response["sql_executed"],
        "sql_result_rows":   len(response["sql_result"]) if isinstance(response["sql_result"], list) else None,
        "tables_kept":       list(kept_tables),
        "expected_tables":   list(expected_tables),
        "pruning_precision": round(precision, 4) if precision is not None else None,
        "pruning_recall":    round(recall, 4)    if recall    is not None else None,
        "latency_ms":        response["latency_ms"],
        "error":             response["error"],
    }


def run_all(queries: list[dict]) -> list[dict]:
    results = []
    total   = len(queries)

    print(f"\n{'='*70}")
    print(f"  RQ3 SQL Execution — {total} queries  ×  2 modes = {total*2} calls")
    print(f"{'='*70}")

    for idx, q in enumerate(queries, 1):
        qid = q["qid"]
        if VERBOSE:
            print(f"\n[{idx:>2}/{total}] {qid} [{q.get('sql_complexity','?')}]: "
                  f"{q['query'][:65]}…")

        # Mode A — TableRAG pruning ON
        pruned = run_single(q, pruning_enabled=True)

        # Mode B — TableRAG pruning OFF (full schema)
        full   = run_single(q, pruning_enabled=False)

        if VERBOSE:
            ok_p = "✓" if pruned["executed_ok"] else "✗"
            ok_f = "✓" if full["executed_ok"]   else "✗"
            print(f"       Pruned: {ok_p} {pruned['latency_ms']:.0f}ms  |  "
                  f"Full schema: {ok_f} {full['latency_ms']:.0f}ms")
            if pruned["pruning_recall"] is not None or pruned["pruning_precision"] is not None:
                rec_str  = f"{pruned['pruning_recall']:.0%}"    if pruned["pruning_recall"]    is not None else "N/A"
                prec_str = f"{pruned['pruning_precision']:.0%}" if pruned["pruning_precision"] is not None else "N/A"
                print(f"       Pruning recall={rec_str}  precision={prec_str}")
            if pruned["error"]:
                print(f"       ERROR (pruned): {pruned['error']}")
            if full["error"]:
                print(f"       ERROR (full):   {full['error']}")

        record = {
            # ground truth
            "qid":               qid,
            "query":             q["query"],
            "ground_truth_sql":  q.get("ground_truth_sql", ""),
            "expected_tables":   q.get("expected_tables", ""),
            "sql_complexity":    q.get("sql_complexity", ""),
            "rq_target":         q.get("rq_target", ""),
            "source_sheet":      "CSQL" if qid.startswith("CSQL") else "SQL",
            # pruned mode
            "pruned_executed_ok":       pruned["executed_ok"],
            "pruned_sql_executed":      pruned["sql_executed"],
            "pruned_sql_result_rows":   pruned["sql_result_rows"],
            "pruned_tables_kept":       pruned["tables_kept"],
            "pruned_pruning_precision": pruned["pruning_precision"],
            "pruned_pruning_recall":    pruned["pruning_recall"],
            "pruned_latency_ms":        pruned["latency_ms"],
            "pruned_error":             pruned["error"],
            # full-schema mode
            "full_executed_ok":         full["executed_ok"],
            "full_sql_executed":        full["sql_executed"],
            "full_latency_ms":          full["latency_ms"],
            "full_error":               full["error"],
            # timestamp
            "timestamp": datetime.utcnow().isoformat(),
        }

        results.append(record)

        if idx < total:
            time.sleep(INTER_QUERY_DELAY_SEC)

    return results


def build_summary(results: list[dict]) -> dict:
    total = len(results)

    def success_rate(records, key):
        ok = sum(1 for r in records if r[key])
        return {"ok": ok, "total": len(records), "rate": round(ok / len(records), 4) if records else 0}

    # Overall
    overall_pruned = success_rate(results, "pruned_executed_ok")
    overall_full   = success_rate(results, "full_executed_ok")
    improvement    = round(overall_pruned["rate"] - overall_full["rate"], 4)

    # Per-complexity breakdown
    breakdown = {}
    for cx in ("simple", "medium", "complex"):
        subset = [r for r in results if r["sql_complexity"] == cx]
        if subset:
            breakdown[cx] = {
                "pruned": success_rate(subset, "pruned_executed_ok"),
                "full":   success_rate(subset, "full_executed_ok"),
            }

    # Per-sheet breakdown
    sheet_breakdown = {}
    for sheet in ("SQL", "CSQL"):
        subset = [r for r in results if r["source_sheet"] == sheet]
        if subset:
            sheet_breakdown[sheet] = {
                "pruned": success_rate(subset, "pruned_executed_ok"),
                "full":   success_rate(subset, "full_executed_ok"),
            }

    # Pruning quality (only rows where tables_kept was reported)
    prec_vals = [r["pruned_pruning_precision"] for r in results if r["pruned_pruning_precision"] is not None]
    rec_vals  = [r["pruned_pruning_recall"]    for r in results if r["pruned_pruning_recall"]    is not None]
    avg_precision = round(sum(prec_vals) / len(prec_vals), 4) if prec_vals else None
    avg_recall    = round(sum(rec_vals)  / len(rec_vals),  4) if rec_vals  else None

    passes = overall_pruned["rate"] >= overall_full["rate"]

    return {
        "rq":                  "RQ3 — SQL Execution Success Rate",
        "run_timestamp":       datetime.utcnow().isoformat(),
        "total_queries":       total,
        "overall": {
            "pruned_schema":   overall_pruned,
            "full_schema":     overall_full,
            "improvement":     improvement,
        },
        "by_complexity":       breakdown,
        "by_sheet":            sheet_breakdown,
        "tablerag_pruning_quality": {
            "avg_precision": avg_precision,
            "avg_recall":    avg_recall,
            "note": "recall = fraction of needed tables kept; precision = fraction of kept tables that were needed",
        },
        "passes_rq3_threshold": passes,
        "verdict": (
            f"✓ PASS — pruned {overall_pruned['rate']:.1%} ≥ full {overall_full['rate']:.1%}"
            if passes else
            f"✗ FAIL — pruned {overall_pruned['rate']:.1%} < full {overall_full['rate']:.1%}"
        ),
    }


def main():
    print("=" * 70)
    print("  RQ3 SQL Execution Success Rate & TableRAG Pruning Benchmark")
    print("=" * 70)

    queries = load_rq3_queries()
    if not queries:
        print("ERROR: No SQL/CSQL queries found.")
        sys.exit(1)

    sql_count  = sum(1 for q in queries if q["qid"].startswith("SQL"))
    csql_count = sum(1 for q in queries if q["qid"].startswith("CSQL"))
    print(f"  Loaded {len(queries)} queries: {sql_count} SQL  |  {csql_count} CSQL")
    print(f"  Each query will be run twice: once with TableRAG pruning ON, once OFF.")

    results = run_all(queries)

    results_path = RESULTS_DIR / "rq3_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  ✓ Per-query results saved → {results_path}")

    summary = build_summary(results)
    summary_path = RESULTS_DIR / "rq3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"  RQ3 SUMMARY")
    print(f"{'='*70}")
    ov = summary["overall"]
    print(f"  Pruned schema : {ov['pruned_schema']['ok']}/{ov['pruned_schema']['total']}  "
          f"= {ov['pruned_schema']['rate']:.1%}")
    print(f"  Full schema   : {ov['full_schema']['ok']}/{ov['full_schema']['total']}  "
          f"= {ov['full_schema']['rate']:.1%}")
    print(f"  Improvement   : {ov['improvement']:+.1%}")

    print("\n  By complexity:")
    for cx, s in summary["by_complexity"].items():
        print(f"    {cx:<8}  pruned={s['pruned']['rate']:.1%}  full={s['full']['rate']:.1%}")

    tq = summary["tablerag_pruning_quality"]
    if tq["avg_recall"] is not None:
        print(f"\n  TableRAG pruning  avg_recall={tq['avg_recall']:.1%}  "
              f"avg_precision={tq['avg_precision']:.1%}")

    print(f"\n  {summary['verdict']}")
    print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
