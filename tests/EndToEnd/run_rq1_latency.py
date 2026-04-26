"""
run_rq1_latency.py
─────────────────────────────────────────────────────────────────────────────
RQ1 — Latency Differential: Fast Track (Golden SQL cache) vs Reasoning Track

Experimental protocol
──────────────────────
Phase 1 — COLD / Reasoning Track
  Submit SQL-001 … SQL-030 with the cache empty.
  Each query must go through TableRAG schema pruning → LangChain ReAct SQL
  agent → execute → cache the result.

Phase 2 — WARM / Fast Track
  Submit CACHE-001 … CACHE-030 (semantic paraphrases of SQL-001 … SQL-030).
  The FAISS Golden-SQL cache should now have matching entries and return a
  cache hit (Fast Track path) for each query, bypassing the full agent.

Metrics computed
─────────────────
  • avg_reasoning_latency_ms   (Phase 1 mean)
  • avg_fast_track_latency_ms  (Phase 2 mean, cache_hit=True only)
  • speedup_factor             = reasoning_avg / fast_track_avg  (target ≥ 1.4)
  • cache_hit_rate             (proportion of Phase 2 queries that hit cache)
  • p50 / p95 for both phases
  • per-query latency CSV for thesis appendix

Outputs
────────
  results/rq1_cold_results.json
  results/rq1_warm_results.json
  results/rq1_summary.json
─────────────────────────────────────────────────────────────────────────────
"""

import json
import sys
import time
import statistics
from pathlib import Path
from datetime import datetime

# ── Path bootstrap (allow running from any working directory) ─────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import BENCHMARK_FILE, RESULTS_DIR, INTER_QUERY_DELAY_SEC, VERBOSE
from system_adapter import call_system

# ─────────────────────────────────────────────────────────────────────────────

def load_queries(prefix: str) -> list[dict]:
    data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    return [q for q in data if q["qid"].startswith(prefix)]


def run_phase(queries: list[dict], phase_label: str) -> list[dict]:
    """Submit each query, record timing and cache status."""
    results = []
    total = len(queries)

    print(f"\n{'='*70}")
    print(f"  Phase: {phase_label}  ({total} queries)")
    print(f"{'='*70}")

    for idx, q in enumerate(queries, 1):
        qid   = q["qid"]
        query = q["query"]

        if VERBOSE:
            print(f"\n[{idx:>2}/{total}] {qid}: {query[:72]}…")

        response = call_system(query)

        record = {
            # --- ground truth ---
            "qid":               qid,
            "query":             query,
            "ground_truth_route": q.get("ground_truth_route", "sql"),
            "ground_truth_sql":  q.get("ground_truth_sql", ""),
            "expected_tables":   q.get("expected_tables", ""),
            "sql_complexity":    q.get("sql_complexity", ""),
            "rq_target":         q.get("rq_target", ""),
            # --- system output ---
            "router_decision":   response["router_decision"],
            "sql_executed":      response["sql_executed"],
            "cache_hit":         response["cache_hit"],
            "latency_ms":        response["latency_ms"],
            "executed_ok":       "Y" if response["error"] is None and response["sql_result"] is not None else "N",
            "answer_snippet":    (response["answer"] or "")[:200],
            "error":             response["error"],
            "phase":             phase_label,
            "timestamp":         datetime.utcnow().isoformat(),
        }

        if VERBOSE:
            hit  = "✓ CACHE HIT"  if record["cache_hit"] else "✗ cache miss"
            ok   = "✓ SQL OK"     if record["executed_ok"] == "Y" else "✗ SQL ERR"
            ms   = f"{record['latency_ms']:.0f} ms" if record["latency_ms"] else "—"
            print(f"       {hit}  |  {ok}  |  {ms}")
            if record["error"]:
                print(f"       ERROR: {record['error']}")

        results.append(record)

        if idx < total:
            time.sleep(INTER_QUERY_DELAY_SEC)

    return results


def compute_latency_stats(records: list[dict], cache_hit_filter: bool | None = None) -> dict:
    """
    Compute latency statistics from a list of result records.
    cache_hit_filter: True = only cache-hit rows, False = only misses, None = all rows.
    """
    latencies = []
    for r in records:
        if cache_hit_filter is not None and bool(r["cache_hit"]) != cache_hit_filter:
            continue
        if r["latency_ms"] is not None:
            latencies.append(r["latency_ms"])

    if not latencies:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}

    latencies_sorted = sorted(latencies)
    p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)

    return {
        "count":  len(latencies),
        "mean":   round(statistics.mean(latencies), 2),
        "median": round(statistics.median(latencies), 2),
        "p95":    round(latencies_sorted[p95_idx], 2),
        "min":    round(min(latencies), 2),
        "max":    round(max(latencies), 2),
    }


def build_summary(cold_results: list[dict], warm_results: list[dict]) -> dict:
    reasoning_stats  = compute_latency_stats(cold_results)
    fast_track_stats = compute_latency_stats(warm_results, cache_hit_filter=True)
    all_warm_stats   = compute_latency_stats(warm_results)

    speedup = None
    passes  = False
    if reasoning_stats["mean"] and fast_track_stats["mean"]:
        speedup = round(reasoning_stats["mean"] / fast_track_stats["mean"], 4)
        passes  = speedup >= 1.4

    cache_hits  = sum(1 for r in warm_results if r["cache_hit"])
    cache_total = len(warm_results)

    cold_errors = sum(1 for r in cold_results if r["executed_ok"] == "N")
    warm_errors = sum(1 for r in warm_results if r["executed_ok"] == "N")

    return {
        "rq":                        "RQ1 — Latency Differential",
        "run_timestamp":             datetime.utcnow().isoformat(),
        "cold_query_count":          len(cold_results),
        "warm_query_count":          len(warm_results),
        # Phase 1 stats
        "reasoning_track": {
            **reasoning_stats,
            "sql_errors": cold_errors,
        },
        # Phase 2 stats (all warm queries)
        "fast_track_all_warm": all_warm_stats,
        # Phase 2 stats (cache-hit subset only — what matters for RQ1)
        "fast_track_hits_only": {
            **fast_track_stats,
            "cache_hit_count": cache_hits,
            "cache_hit_rate":  round(cache_hits / cache_total, 4) if cache_total else 0,
            "sql_errors":      warm_errors,
        },
        # RQ1 headline metric
        "speedup_factor":            speedup,
        "passes_rq1_threshold":      passes,
        "rq1_threshold":             "speedup ≥ 1.4×",
        "verdict": (
            f"✓ PASS — {speedup:.2f}× speedup (threshold ≥ 1.4×)"
            if passes else
            f"✗ FAIL — only {speedup:.2f}× speedup (threshold ≥ 1.4×)"
        ) if speedup else "⚠ Cannot compute — missing latency data",
    }


def main():
    print("=" * 70)
    print("  RQ1 Latency Benchmark — Fast Track vs Reasoning Track")
    print("=" * 70)

    # ── Phase 1: Cold / Reasoning Track ──────────────────────────────────────
    sql_queries = load_queries("SQL-")
    if not sql_queries:
        print("ERROR: No SQL- queries found in benchmark file.")
        sys.exit(1)

    cold_results = run_phase(sql_queries, "COLD (Reasoning Track)")
    cold_path = RESULTS_DIR / "rq1_cold_results.json"
    cold_path.write_text(json.dumps(cold_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  ✓ Cold results saved → {cold_path}")

    # ── Brief pause between phases so the cache has time to persist ──────────
    print("\n  Pausing 3 s before warm phase to allow cache to settle …")
    time.sleep(3)

    # ── Phase 2: Warm / Fast Track ────────────────────────────────────────────
    cache_queries = load_queries("CACHE-")
    if not cache_queries:
        print("ERROR: No CACHE- queries found in benchmark file.")
        sys.exit(1)

    warm_results = run_phase(cache_queries, "WARM (Fast Track)")
    warm_path = RESULTS_DIR / "rq1_warm_results.json"
    warm_path.write_text(json.dumps(warm_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  ✓ Warm results saved → {warm_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = build_summary(cold_results, warm_results)
    summary_path = RESULTS_DIR / "rq1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"  RQ1 SUMMARY")
    print(f"{'='*70}")
    rt = summary["reasoning_track"]
    ft = summary["fast_track_hits_only"]
    print(f"  Reasoning Track  avg: {rt['mean']} ms  |  p95: {rt['p95']} ms  |  errors: {rt['sql_errors']}")
    print(f"  Fast Track       avg: {ft['mean']} ms  |  p95: {ft['p95']} ms  |  cache hits: {ft['cache_hit_count']}/{summary['warm_query_count']}")
    print(f"  Cache hit rate : {ft['cache_hit_rate']:.1%}")
    print(f"  Speedup factor : {summary['speedup_factor']}")
    print(f"\n  {summary['verdict']}")
    print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
