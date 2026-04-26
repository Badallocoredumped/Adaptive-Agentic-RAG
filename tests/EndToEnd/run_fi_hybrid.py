"""
run_fi_hybrid.py
─────────────────────────────────────────────────────────────────────────────
Functional Integration — End-to-End Hybrid Query Flows

Runs HYB-001 … HYB-030.  Each query requires the system to:
  1. Classify the query as 'hybrid'         (Router Agent)
  2. Execute a SQL query on fintech.db      (Structured pipeline)
  3. Retrieve ≥1 relevant PDF chunk        (Unstructured pipeline)
  4. Synthesise both into a coherent answer (Synthesis Agent)

A query is considered a successful E2E flow if ALL of the following hold:
  • router_decision == "hybrid"
  • SQL executed without error AND returned ≥1 row
  • ≥1 PDF chunk was retrieved
  • At least one of expected_pdf_sources appears in retrieved_sources

Metrics computed
─────────────────
  • successful_e2e_flows     (target ≥ 5  out of 30)
  • router_accuracy          (fraction routed to 'hybrid' correctly)
  • sql_success_rate         (fraction where SQL executed OK)
  • pdf_retrieval_rate       (fraction where ≥1 chunk retrieved)
  • source_accuracy          (fraction where correct PDF was retrieved)
  • synthesis_quality        (subjective yes/no — filled manually or via LLM judge)

Outputs
────────
  results/fi_results.json      — per-query results
  results/fi_summary.json      — aggregated metrics
─────────────────────────────────────────────────────────────────────────────
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import BENCHMARK_FILE, RESULTS_DIR, INTER_QUERY_DELAY_SEC, VERBOSE
from system_adapter import call_system

TARGET_E2E_FLOWS = 5   # minimum required by proposal

# ─────────────────────────────────────────────────────────────────────────────

def load_hybrid_queries() -> list[dict]:
    data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    return [q for q in data if q["qid"].startswith("HYB")]


def expected_pdfs(q: dict) -> set[str]:
    raw = q.get("expected_pdf_sources", "")
    return {p.strip() for p in raw.split(",") if p.strip()}


def check_source_match(expected: set[str], retrieved: list | None) -> bool:
    """True if at least one expected PDF appears in the retrieved sources list."""
    if not retrieved or not expected:
        return False
    retrieved_lower = {str(s).lower() for s in retrieved}
    return any(exp.lower() in retrieved_lower for exp in expected)


def run_hybrid(queries: list[dict]) -> list[dict]:
    results = []
    total   = len(queries)

    print(f"\n{'='*70}")
    print(f"  Functional Integration — {total} hybrid queries")
    print(f"{'='*70}")

    for idx, q in enumerate(queries, 1):
        qid      = q["qid"]
        query    = q["query"]
        exp_pdfs = expected_pdfs(q)

        if VERBOSE:
            print(f"\n[{idx:>2}/{total}] {qid}: {query[:65]}…")
            if exp_pdfs:
                print(f"       Expected PDFs: {', '.join(sorted(exp_pdfs))}")

        response = call_system(query)

        # ── Evaluate each sub-component ──────────────────────────────────────
        routed_correctly = (response["router_decision"] or "").lower() == "hybrid"

        sql_ok = (
            response["error"] is None
            and response["sql_result"] is not None
            and (
                not isinstance(response["sql_result"], list)
                or len(response["sql_result"]) > 0
            )
        )

        chunks_retrieved  = bool(response["retrieved_chunks"])
        sources_retrieved = response["retrieved_sources"] or []
        source_match      = check_source_match(exp_pdfs, sources_retrieved)

        # ── Successful E2E = all four checks pass ─────────────────────────────
        e2e_success = routed_correctly and sql_ok and chunks_retrieved and source_match

        if VERBOSE:
            print(f"       Router ={'✓' if routed_correctly else '✗'} ({response['router_decision']})  "
                  f"SQL={'✓' if sql_ok else '✗'}  "
                  f"PDF={'✓' if chunks_retrieved else '✗'}  "
                  f"SrcMatch={'✓' if source_match else '✗'}  "
                  f"E2E={'✓ SUCCESS' if e2e_success else '✗ FAIL'}")
            if response["error"]:
                print(f"       ERROR: {response['error']}")
            if sources_retrieved:
                print(f"       Retrieved: {sources_retrieved[:3]}")

        record = {
            # ground truth
            "qid":                  qid,
            "query":                query,
            "ground_truth_route":   "hybrid",
            "ground_truth_sql":     q.get("ground_truth_sql", ""),
            "expected_tables":      q.get("expected_tables", ""),
            "expected_pdf_sources": q.get("expected_pdf_sources", ""),
            "sql_complexity":       q.get("sql_complexity", ""),
            # system output
            "router_decision":      response["router_decision"],
            "sql_executed":         response["sql_executed"],
            "sql_result_rows":      len(response["sql_result"]) if isinstance(response["sql_result"], list) else None,
            "retrieved_sources":    sources_retrieved,
            "num_chunks_retrieved": len(response["retrieved_chunks"]) if isinstance(response["retrieved_chunks"], list) else 0,
            "answer_snippet":       (response["answer"] or "")[:300],
            "latency_ms":           response["latency_ms"],
            "error":                response["error"],
            # evaluation flags
            "routed_correctly":     routed_correctly,
            "sql_ok":               sql_ok,
            "chunks_retrieved":     chunks_retrieved,
            "source_match":         source_match,
            "e2e_success":          e2e_success,
            # manual quality assessment (fill in after reviewing answers)
            "answer_correct":       None,   # set to True/False after human review
            "notes":                "",
            "timestamp":            datetime.utcnow().isoformat(),
        }

        results.append(record)

        if idx < total:
            time.sleep(INTER_QUERY_DELAY_SEC)

    return results


def build_summary(results: list[dict]) -> dict:
    total = len(results)

    def rate(key):
        ok = sum(1 for r in results if r[key])
        return {"ok": ok, "total": total, "rate": round(ok / total, 4) if total else 0}

    e2e_rate     = rate("e2e_success")
    router_rate  = rate("routed_correctly")
    sql_rate     = rate("sql_ok")
    pdf_rate     = rate("chunks_retrieved")
    source_rate  = rate("source_match")

    passes_threshold = e2e_rate["ok"] >= TARGET_E2E_FLOWS

    # Failed queries — which component failed most?
    failures = [r for r in results if not r["e2e_success"]]
    failure_breakdown = {
        "wrong_route":        sum(1 for r in failures if not r["routed_correctly"]),
        "sql_failed":         sum(1 for r in failures if not r["sql_ok"]),
        "no_pdf_chunks":      sum(1 for r in failures if not r["chunks_retrieved"]),
        "wrong_pdf_source":   sum(1 for r in failures if not r["source_match"]),
    }

    latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else None

    return {
        "rq":                    "Functional Integration — E2E Hybrid Flows",
        "run_timestamp":         datetime.utcnow().isoformat(),
        "total_queries":         total,
        "target_e2e_flows":      TARGET_E2E_FLOWS,
        "e2e_success":           e2e_rate,
        "router_accuracy":       router_rate,
        "sql_success_rate":      sql_rate,
        "pdf_retrieval_rate":    pdf_rate,
        "source_accuracy":       source_rate,
        "failure_breakdown":     failure_breakdown,
        "passes_threshold":      passes_threshold,
        "avg_latency_ms":        avg_latency,
        "verdict": (
            f"✓ PASS — {e2e_rate['ok']}/{total} successful E2E flows "
            f"(target ≥ {TARGET_E2E_FLOWS})"
            if passes_threshold else
            f"✗ FAIL — only {e2e_rate['ok']}/{total} E2E flows succeeded "
            f"(target ≥ {TARGET_E2E_FLOWS})"
        ),
        "note": (
            "answer_correct field is None — run review_hybrid_answers.py "
            "or manually set it in fi_results.json after reviewing responses."
        ),
    }


def main():
    print("=" * 70)
    print("  Functional Integration — Hybrid E2E Flow Benchmark")
    print("=" * 70)

    queries = load_hybrid_queries()
    if not queries:
        print("ERROR: No HYB- queries found.")
        sys.exit(1)

    results = run_hybrid(queries)

    results_path = RESULTS_DIR / "fi_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  ✓ Per-query results saved → {results_path}")

    summary = build_summary(results)
    summary_path = RESULTS_DIR / "fi_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"  FUNCTIONAL INTEGRATION SUMMARY")
    print(f"{'='*70}")
    metrics = [
        ("Router accuracy",     summary["router_accuracy"]),
        ("SQL success rate",    summary["sql_success_rate"]),
        ("PDF retrieval rate",  summary["pdf_retrieval_rate"]),
        ("Source accuracy",     summary["source_accuracy"]),
        ("E2E success",         summary["e2e_success"]),
    ]
    for label, m in metrics:
        bar = "█" * int(m["rate"] * 20)
        print(f"  {label:<22}  {m['ok']:>2}/{m['total']}  {m['rate']:>6.1%}  {bar}")

    fb = summary["failure_breakdown"]
    if sum(fb.values()) > 0:
        print(f"\n  Failure breakdown (non-exclusive):")
        for k, v in fb.items():
            print(f"    {k:<25} {v}")

    print(f"\n  {summary['verdict']}")
    print(f"\n  Summary saved → {summary_path}")
    print(f"\n  ⚠ Manually review fi_results.json and set 'answer_correct' "
          f"for each query,\n    or run review_hybrid_answers.py for LLM-assisted scoring.")


if __name__ == "__main__":
    main()
