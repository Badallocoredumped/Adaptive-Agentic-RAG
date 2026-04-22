"""
ReAct SQL Agent Test Suite
==========================
Tests run_react_sql_agent() DIRECTLY — bypasses the FAISS cache and full
pipeline — so you can see exactly how the agent reasons step by step.

run_react_sql_agent() streams the LangGraph execution internally, printing
every message as it arrives (Tool Call → Observation → Final Message).

Usage:
    python tests/test_react_agent.py                  # run all tests
    python tests/test_react_agent.py --top-k 6        # retrieve more schema
    python tests/test_react_agent.py --test 3         # run only test #3
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

# ── Project root on path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import config
from backend.sql.react_agent import run_react_sql_agent
from backend.sql.sql_agent import _ensure_schema_index_exists
from backend.sql.table_rag import retrieve_relevant_schema

# ── Test cases ────────────────────────────────────────────────────────────
# Each dict has:
#   label      : short name shown in the report header
#   query      : natural-language question sent to the agent
#   expect_sql : True  → we expect a SQL result (pass if rows returned or 0 rows OK)
#                False → we expect an error / graceful failure
#   note       : what this test is checking (printed before the run)

TEST_CASES = [
    # ── 1. Single-table ─────────────────────────────────────────────────────
    {
        "label": "Simple aggregation",
        "query": "What is the total balance across all active accounts?",
        "expect_sql": True,
        "note": "Single table, WHERE + SUM — should be trivial for the agent.",
    },
    # ── 2. Two-table JOIN ────────────────────────────────────────────────────
    {
        "label": "Basic JOIN",
        "query": "Show all frozen accounts with the customer's full name and balance",
        "expect_sql": True,
        "note": "customers JOIN accounts on customer_id.",
    },
    # ── 3. Three-table JOIN ──────────────────────────────────────────────────
    {
        "label": "Three-table JOIN",
        "query": "Show late loan payments with customer name and how many days late",
        "expect_sql": True,
        "note": "loan_payments → loans → customers. Tests multi-hop JOIN.",
    },
    # ── 4. GROUP BY + HAVING ─────────────────────────────────────────────────
    {
        "label": "GROUP BY + HAVING",
        "query": "Which customers have more than one active loan?",
        "expect_sql": True,
        "note": "HAVING COUNT > 1. Tests aggregation filter.",
    },
    # ── 5. Subquery / MAX date ───────────────────────────────────────────────
    {
        "label": "Latest price per asset",
        "query": "What is the latest market price for each asset?",
        "expect_sql": True,
        "note": "Requires correlated subquery or window fn to get MAX(price_date) per asset.",
    },
    # ── 6. Cross-domain NEW FK: relationship_manager_id ─────────────────────
    {
        "label": "Cross-domain: relationship manager",
        "query": "Which relationship managers are assigned the most customers?",
        "expect_sql": True,
        "note": "customers.relationship_manager_id → employees.id."
                " Tests new v2 FK.",
    },
    # ── 7. Cross-domain NEW FK: disbursement_account_id ─────────────────────
    {
        "label": "Cross-domain: loan disbursement account",
        "query": "Show which account each loan was disbursed into, with the customer name",
        "expect_sql": True,
        "note": "loans.disbursement_account_id → accounts.id → customers.id."
                " Three cross-domain hops.",
    },
    # ── 8. Cross-domain NEW FK: portfolio funding account ───────────────────
    {
        "label": "Cross-domain: portfolio funding",
        "query": "Show portfolio name, total value, and the balance of its funding account",
        "expect_sql": True,
        "note": "portfolios.funding_account_id → accounts.id.",
    },
    # ── 9. Compliance cross-domain ───────────────────────────────────────────
    {
        "label": "Compliance cross-domain",
        "query": "Which customers have a pending KYC status and at least one active loan?",
        "expect_sql": True,
        "note": "kyc_records + loans + customers — three-table compliance query.",
    },
    # ── 10. Risk + Lending cross-domain ─────────────────────────────────────
    {
        "label": "Risk + Lending",
        "query": "Show customers with a Very High risk tier who still have an active loan",
        "expect_sql": True,
        "note": "risk_assessments.loan_id → loans.id → customers.",
    },
    # ── 11. Ambiguous: 'order' in fintech context ────────────────────────────
    {
        "label": "Ambiguous date ordering",
        "query": "List all branches ordered by their opening date oldest to newest",
        "expect_sql": True,
        "note": "Word 'ordered' could confuse the agent — should use ORDER BY opened_date.",
    },
    # ── 12. Agent must retry: deliberate schema mismatch hint ────────────────
    {
        "label": "Self-correction: bad column guess",
        "query": "Show total loan amount approved per employee who approved it",
        "expect_sql": False,  # No 'approved_by' column — agent should error gracefully
        "note": "loans has no 'approved_by' column. Agent should attempt, fail,"
                " then surface a clear error rather than fabricating data.",
    },
    # ── 13. 4-table chain ────────────────────────────────────────────────────
    {
        "label": "4-table chain",
        "query": "Show each portfolio holding with asset name, customer name, and portfolio risk profile",
        "expect_sql": True,
        "note": "portfolio_holdings → assets + portfolios → customers. Four-table JOIN.",
    },
    # ── 14. Aggregation over investments ────────────────────────────────────
    {
        "label": "Investment aggregation",
        "query": "What asset class has the highest total unrealized profit across all portfolios?",
        "expect_sql": True,
        "note": "portfolio_holdings.unrealized_pnl + assets.asset_class, GROUP BY.",
    },
    # ── 15. Edge: non-existent concept ──────────────────────────────────────
    {
        "label": "Edge: missing table",
        "query": "Show all mortgage approvals with the approving officer name",
        "expect_sql": False,
        "note": "No 'approvals' table exists. Agent should not invent one.",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════════

def _trunc(text: str, n: int = 200) -> str:
    return text if len(text) <= n else text[:n] + "…"


def run_single_test(
    idx: int,
    case: dict,
    top_k: int,
) -> dict[str, Any]:
    W = 72
    print(f"\n{'═' * W}")
    print(f"  TEST {idx:02d}/{len(TEST_CASES)}  {case['label']}")
    print(f"{'─' * W}")
    print(f"  Query : {case['query']}")
    print(f"  Note  : {case['note']}")
    print(f"{'─' * W}")

    _ensure_schema_index_exists()
    schema_rows = retrieve_relevant_schema(case["query"], top_k=top_k)
    schema_context = "\n".join(schema_rows)

    print(f"  📋 Schema context ({len(schema_rows)} rows retrieved):")
    for row in schema_rows:
        print(f"     {row}")
    print()

    # run_react_sql_agent streams and prints the live trace internally
    start = time.time()
    result = run_react_sql_agent(case["query"], schema_context)
    latency = time.time() - start

    sql      = result.get("sql")
    rows     = result.get("result", [])
    error    = result.get("error")
    expected = case["expect_sql"]

    produced_sql = sql is not None
    passed = (produced_sql == expected)

    print(f"{'─' * W}")
    status_icon = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status_icon}  |  Expected SQL={expected}  Got SQL={produced_sql}  |  {latency:.2f}s")

    if sql:
        print(textwrap.fill(
            _trunc(sql, 200),
            width=68, initial_indent="  SQL    : ", subsequent_indent="           "
        ))
    if rows:
        print(f"  Rows   : {len(rows)} returned")
        for row in rows[:3]:
            print(f"           {row}")
        if len(rows) > 3:
            print(f"           … and {len(rows)-3} more")
    if error:
        print(f"  Error  : {_trunc(error, 150)}")

    return {
        "idx":     idx,
        "label":   case["label"],
        "passed":  passed,
        "latency": latency,
        "sql":     sql,
        "error":   error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the LangGraph ReAct SQL agent directly."
    )
    parser.add_argument(
        "--test", type=int, default=None,
        help="Run only test number N (1-indexed). Default: run all."
    )
    parser.add_argument(
        "--top-k", type=int, default=config.SQL_TOP_K,
        help=f"Number of schema rows to retrieve via TableRAG (default: {config.SQL_TOP_K})."
    )
    args = parser.parse_args()

    print(f"\n{'═' * 72}")
    print("  ReAct SQL Agent — Test Suite")
    print(f"{'─' * 72}")
    print(f"  DB          : {config.SQLITE_PATH or config.DATABASE_URL or 'PostgreSQL (env)'}")
    print(f"  LLM model   : {config.SQL_OPENAI_MODEL}")
    print(f"  Max iter    : {config.SQL_REACT_MAX_ITERATIONS}")
    print(f"  Top-K schema: {args.top_k}")
    print(f"{'═' * 72}")

    cases = (
        [TEST_CASES[args.test - 1]]
        if args.test
        else TEST_CASES
    )
    indices = [args.test] if args.test else list(range(1, len(TEST_CASES) + 1))

    results = []
    for i, case in zip(indices, cases):
        r = run_single_test(i, case, args.top_k)
        results.append(r)

    # ── Final report ──────────────────────────────────────────────────────
    W = 72
    passed  = sum(1 for r in results if r["passed"])
    failed  = len(results) - passed
    avg_lat = sum(r["latency"] for r in results) / len(results) if results else 0

    print(f"\n{'═' * W}")
    print(f"  FINAL RESULTS  {passed}/{len(results)} passed   avg latency {avg_lat:.2f}s")
    print(f"{'─' * W}")
    print(f"  {'#':>3}  {'Label':<35} {'Result':<8} {'Latency':>8}")
    print(f"  {'─'*3}  {'─'*35} {'─'*8} {'─'*8}")
    for r in results:
        icon = "✅" if r["passed"] else "❌"
        print(f"  {r['idx']:>3}  {r['label']:<35} {icon}       {r['latency']:>6.2f}s")
    print(f"{'═' * W}")

    if failed > 0:
        print(f"\n  Failed tests:")
        for r in results:
            if not r["passed"]:
                print(f"    • [{r['idx']:02d}] {r['label']}")
                if r["error"]:
                    print(f"         Error: {r['error'][:120]}")
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
