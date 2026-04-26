"""
run_all_benchmarks.py
─────────────────────────────────────────────────────────────────────────────
Master runner — executes all four benchmark suites in the correct order and
produces a unified final_report.json suitable for your thesis results chapter.

Execution order
────────────────
  1. RQ2  — Router Classification  (SQL + TXT + HYB, 90 queries)
  2. RQ3  — SQL Execution          (SQL + CSQL, 60 queries × 2 modes)
  3. RQ1  — Latency                (SQL cold → CACHE warm, 30+30 queries)
  4. FI   — Functional Integration (HYB, 30 queries)

RQ1 is run last because:
  • The SQL queries run in step 3 (RQ3) already warm the Golden SQL cache
    with high-confidence executed queries.
  • The CACHE phase therefore measures a more realistic Fast Track speedup.

Flags
──────
  --rq1-only   run only the latency benchmark
  --rq2-only   run only router classification
  --rq3-only   run only SQL execution
  --fi-only    run only functional integration
  --skip-rq1   skip latency (fastest full run during development)
  --dry-run    print queries without calling the system

Outputs
────────
  results/rq1_cold_results.json
  results/rq1_warm_results.json
  results/rq1_summary.json
  results/rq2_results.json
  results/rq2_summary.json
  results/rq3_results.json
  results/rq3_summary.json
  results/fi_results.json
  results/fi_summary.json
  results/final_report.json        ← thesis-ready combined summary
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import RESULTS_DIR


# ── Lazy imports so individual suites can still run standalone ────────────────
def _run_rq2():
    import run_rq2_router as m
    queries = m.load_rq2_queries()
    results = m.run_classification(queries)
    path = RESULTS_DIR / "rq2_results.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = m.build_summary(results)
    (RESULTS_DIR / "rq2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _run_rq3():
    import run_rq3_sql_execution as m
    queries = m.load_rq3_queries()
    results = m.run_all(queries)
    path = RESULTS_DIR / "rq3_results.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = m.build_summary(results)
    (RESULTS_DIR / "rq3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _run_rq1():
    import run_rq1_latency as m
    sql_queries   = m.load_queries("SQL-")
    cache_queries = m.load_queries("CACHE-")
    cold_results  = m.run_phase(sql_queries,   "COLD (Reasoning Track)")
    (RESULTS_DIR / "rq1_cold_results.json").write_text(
        json.dumps(cold_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n  Pausing 3 s before warm phase …")
    time.sleep(3)
    warm_results = m.run_phase(cache_queries, "WARM (Fast Track)")
    (RESULTS_DIR / "rq1_warm_results.json").write_text(
        json.dumps(warm_results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = m.build_summary(cold_results, warm_results)
    (RESULTS_DIR / "rq1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _run_fi():
    import run_fi_hybrid as m
    queries = m.load_hybrid_queries()
    results = m.run_hybrid(queries)
    (RESULTS_DIR / "fi_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = m.build_summary(results)
    (RESULTS_DIR / "fi_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# ── Dry-run mode ─────────────────────────────────────────────────────────────
def dry_run():
    from config import BENCHMARK_FILE
    data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    from collections import Counter
    counts = Counter(q["qid"].split("-")[0] for q in data)
    print("\n  DRY RUN — no system calls will be made\n")
    print(f"  Benchmark file : {BENCHMARK_FILE}")
    print(f"  Results dir    : {RESULTS_DIR}\n")
    for prefix, cnt in sorted(counts.items()):
        route = data[[q["qid"].split("-")[0] for q in data].index(prefix)]["ground_truth_route"]
        print(f"  {prefix:<8}  {cnt:>3} queries   route={route}")
    print(f"\n  Total: {sum(counts.values())} queries")
    print("\n  Execution plan:")
    print("    Step 1: RQ2 (90 queries) -- router classification")
    print("    Step 2: RQ3 (60 q x 2)  -- SQL execution + pruning")
    print("    Step 3: RQ1 (60 queries) -- latency (cold->warm)")
    print("    Step 4: FI  (30 queries) -- hybrid E2E flows")


# ── Final report ──────────────────────────────────────────────────────────────
def build_final_report(summaries: dict) -> dict:
    report = {
        "project":       "Adaptive Agentic RAG Framework",
        "run_timestamp": datetime.utcnow().isoformat(),
        "summaries":     summaries,
        "thesis_table":  {},
    }

    # Build a flat thesis-ready table
    t = report["thesis_table"]

    rq1 = summaries.get("RQ1")
    if rq1:
        t["rq1_speedup_factor"]       = rq1.get("speedup_factor")
        t["rq1_cache_hit_rate"]       = rq1.get("fast_track_hits_only", {}).get("cache_hit_rate")
        t["rq1_reasoning_avg_ms"]     = rq1.get("reasoning_track", {}).get("mean")
        t["rq1_fast_track_avg_ms"]    = rq1.get("fast_track_hits_only", {}).get("mean")
        t["rq1_passes"]               = rq1.get("passes_rq1_threshold")

    rq2 = summaries.get("RQ2")
    if rq2:
        t["rq2_overall_accuracy"]     = rq2.get("overall_accuracy")
        t["rq2_sql_accuracy"]         = rq2.get("per_class_accuracy", {}).get("sql", {}).get("accuracy")
        t["rq2_text_accuracy"]        = rq2.get("per_class_accuracy", {}).get("text", {}).get("accuracy")
        t["rq2_hybrid_accuracy"]      = rq2.get("per_class_accuracy", {}).get("hybrid", {}).get("accuracy")
        t["rq2_passes"]               = rq2.get("passes_threshold")

    rq3 = summaries.get("RQ3")
    if rq3:
        ov = rq3.get("overall", {})
        t["rq3_pruned_success_rate"]  = ov.get("pruned_schema", {}).get("rate")
        t["rq3_full_success_rate"]    = ov.get("full_schema", {}).get("rate")
        t["rq3_improvement"]          = ov.get("improvement")
        tq = rq3.get("tablerag_pruning_quality", {})
        t["rq3_pruning_recall"]       = tq.get("avg_recall")
        t["rq3_pruning_precision"]    = tq.get("avg_precision")
        t["rq3_passes"]               = rq3.get("passes_rq3_threshold")

    fi = summaries.get("FI")
    if fi:
        t["fi_e2e_success_count"]     = fi.get("e2e_success", {}).get("ok")
        t["fi_e2e_success_rate"]      = fi.get("e2e_success", {}).get("rate")
        t["fi_router_accuracy"]       = fi.get("router_accuracy", {}).get("rate")
        t["fi_sql_success_rate"]      = fi.get("sql_success_rate", {}).get("rate")
        t["fi_pdf_retrieval_rate"]    = fi.get("pdf_retrieval_rate", {}).get("rate")
        t["fi_source_accuracy"]       = fi.get("source_accuracy", {}).get("rate")
        t["fi_passes"]                = fi.get("passes_threshold")

    # Overall verdict
    passes = [v for k, v in t.items() if k.endswith("_passes") and v is not None]
    report["all_pass"] = all(passes) if passes else None

    return report


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run all RAG benchmarks")
    parser.add_argument("--rq1-only",  action="store_true")
    parser.add_argument("--rq2-only",  action="store_true")
    parser.add_argument("--rq3-only",  action="store_true")
    parser.add_argument("--fi-only",   action="store_true")
    parser.add_argument("--skip-rq1",  action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    summaries = {}
    start_time = time.perf_counter()

    banner = lambda t: print(f"\n{'#'*70}\n#  {t}\n{'#'*70}")

    if args.rq1_only:
        banner("RQ1 — Latency Differential")
        summaries["RQ1"] = _run_rq1()

    elif args.rq2_only:
        banner("RQ2 — Router Classification")
        summaries["RQ2"] = _run_rq2()

    elif args.rq3_only:
        banner("RQ3 — SQL Execution")
        summaries["RQ3"] = _run_rq3()

    elif args.fi_only:
        banner("Functional Integration — Hybrid Flows")
        summaries["FI"] = _run_fi()

    else:
        # Full run
        banner("RQ2 — Router Classification Accuracy  (Step 1/4)")
        summaries["RQ2"] = _run_rq2()

        banner("RQ3 — SQL Execution Success Rate  (Step 2/4)")
        summaries["RQ3"] = _run_rq3()

        if not args.skip_rq1:
            banner("RQ1 — Latency Differential  (Step 3/4)")
            summaries["RQ1"] = _run_rq1()
        else:
            print("\n  ⚡ Skipping RQ1 (--skip-rq1 flag set)")

        banner("Functional Integration — Hybrid Flows  (Step 4/4)")
        summaries["FI"] = _run_fi()

    # Final report
    elapsed = round((time.perf_counter() - start_time) / 60, 1)
    report = build_final_report(summaries)
    report["total_runtime_minutes"] = elapsed
    report_path = RESULTS_DIR / "final_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"  FINAL REPORT")
    print(f"{'='*70}")
    for k, v in report["thesis_table"].items():
        if v is not None:
            print(f"  {k:<35}  {v}")
    print(f"\n  Total runtime: {elapsed} min")
    print(f"  All metrics pass: {report['all_pass']}")
    print(f"\n  Full report → {report_path}")


if __name__ == "__main__":
    main()
