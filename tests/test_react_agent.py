"""
ReAct SQL Agent Test Suite
==========================
Tests run_react_sql_agent() DIRECTLY — bypasses the FAISS cache and full
pipeline — so you can see exactly how the agent reasons step by step.

Each test streams the LangGraph execution, printing every message as it
arrives so you get a live Thought → Tool Call → Observation trace.

Usage:
    python tests/test_react_agent.py                  # run all tests
    python tests/test_react_agent.py --top-k 6        # retrieve more schema
    python tests/test_react_agent.py --test 3         # run only test #3
    python tests/test_react_agent.py --verbose        # show full SQL in trace
    python tests/test_react_agent.py --no-stream      # suppress live trace
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

# ── Project root on path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import config
from backend.sql.react_agent import build_react_agent, run_react_sql_agent
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
# Streaming trace printer
# ═══════════════════════════════════════════════════════════════════════════

def _truncate(text: str, max_len: int, verbose: bool) -> str:
    if verbose:
        return text
    return text if len(text) <= max_len else text[:max_len] + "…"


def stream_agent_trace(
    query: str,
    schema_context: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Stream the LangGraph ReAct agent and print each message as it arrives.
    Returns the same result dict as run_react_sql_agent().
    """
    max_iter: int = getattr(config, "SQL_REACT_MAX_ITERATIONS", 6)
    graph = build_react_agent(schema_context)

    print(f"\n  💬 Human: {query}")
    print()

    step = 0
    all_messages: list = []

    try:
        stream = graph.stream(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": max_iter * 3},
            stream_mode="values",
        )
        prev_count = 0
        for state in stream:
            msgs = state.get("messages", [])
            new_msgs = msgs[prev_count:]
            prev_count = len(msgs)
            all_messages = msgs

            for msg in new_msgs:
                step += 1
                if isinstance(msg, HumanMessage):
                    continue  # already printed above

                elif isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_name = tc["name"]
                            raw_input = next(iter(tc.get("args", {}).values()), "")
                            print(f"  🔧 [{step}] TOOL CALL → {tool_name}")
                            print(f"       Input: {_truncate(str(raw_input), 120, verbose)}")
                    else:
                        content = msg.content or ""
                        if content.strip():
                            print(f"  🤖 [{step}] AGENT FINAL MESSAGE")
                            wrapped = textwrap.fill(
                                _truncate(content, 300, verbose),
                                width=72,
                                initial_indent="       ",
                                subsequent_indent="       ",
                            )
                            print(wrapped)

                elif isinstance(msg, ToolMessage):
                    obs = msg.content if isinstance(msg.content, str) else str(msg.content)
                    is_success = obs.startswith("SUCCESS")
                    icon = "✅" if is_success else "❌"
                    print(f"  {icon} [{step}] OBSERVATION")
                    print(f"       {_truncate(obs, 200, verbose)}")

                print()

    except Exception as exc:
        print(f"  ❌ Graph execution failed: {exc}\n")
        return {
            "sql": None,
            "result": [],
            "error": str(exc),
            "schema_context": schema_context,
        }

    # ── Parse final result from messages ─────────────────────────────────
    # (same logic as run_react_sql_agent)
    from backend.sql.react_agent import (
        _extract_and_normalise_sql,
        _raw_execute_sql,
    )

    tool_call_map: dict[str, tuple[str, str]] = {}
    for msg in all_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                raw_input = next(iter(tc.get("args", {}).values()), "") if tc.get("args") else ""
                tool_call_map[tc["id"]] = (tc["name"], str(raw_input))

    final_sql: str | None = None
    final_rows: list[dict] = []
    final_error: str | None = None

    for msg in all_messages:
        if not isinstance(msg, ToolMessage):
            continue
        call_id = msg.tool_call_id
        if call_id not in tool_call_map:
            continue
        tool_name, raw_input = tool_call_map[call_id]
        if tool_name != "execute_sql":
            continue

        observation = msg.content if isinstance(msg.content, str) else str(msg.content)
        candidate_sql = _extract_and_normalise_sql(raw_input) or raw_input.strip()

        if observation.startswith("SUCCESS"):
            final_sql = candidate_sql
            final_error = None
            try:
                json_start = observation.index("[")
                json_part = observation[json_start:].split("\n(truncated")[0]
                final_rows = json.loads(json_part)
            except (ValueError, json.JSONDecodeError):
                try:
                    final_rows = _raw_execute_sql(candidate_sql)
                except RuntimeError as exc:
                    final_error = str(exc)
                    final_rows = []
        else:
            if final_sql is None:
                final_error = observation

    if final_sql is None and final_error is None:
        last_ai = next(
            (m for m in reversed(all_messages)
             if isinstance(m, AIMessage) and not m.tool_calls),
            None,
        )
        output_text = last_ai.content if last_ai else ""
        final_error = f"Agent completed without executing SQL. Output: {output_text}"

    return {
        "sql": final_sql,
        "result": final_rows,
        "error": final_error,
        "schema_context": schema_context,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════════

def run_single_test(
    idx: int,
    case: dict,
    top_k: int,
    verbose: bool,
    stream: bool,
) -> dict[str, Any]:
    W = 72
    print(f"\n{'═' * W}")
    print(f"  TEST {idx:02d}/{len(TEST_CASES)}  {case['label']}")
    print(f"{'─' * W}")
    print(f"  Query : {case['query']}")
    print(f"  Note  : {case['note']}")
    print(f"{'─' * W}")

    # Retrieve schema context via TableRAG
    _ensure_schema_index_exists()
    schema_rows = retrieve_relevant_schema(case["query"], top_k=top_k)
    schema_context = "\n".join(schema_rows)

    print(f"  📋 Schema context ({len(schema_rows)} rows retrieved):")
    for row in schema_rows:
        print(f"     {row}")
    print()

    start = time.time()
    if stream:
        result = stream_agent_trace(case["query"], schema_context, verbose=verbose)
    else:
        result = run_react_sql_agent(case["query"], schema_context)
    latency = time.time() - start

    # ── Verdict ─────────────────────────────────────────────────────────
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
        wrapped_sql = textwrap.fill(
            _truncate(sql, 200, verbose),
            width=68, initial_indent="  SQL    : ", subsequent_indent="           "
        )
        print(wrapped_sql)
    if rows:
        print(f"  Rows   : {len(rows)} returned")
        for row in rows[:3]:
            print(f"           {row}")
        if len(rows) > 3:
            print(f"           … and {len(rows)-3} more")
    if error:
        print(f"  Error  : {_truncate(error, 150, verbose)}")

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
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show full SQL / observation text without truncation."
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Run agent silently (no live Thought/Action/Obs trace)."
    )
    args = parser.parse_args()

    stream = not args.no_stream

    print(f"\n{'═' * 72}")
    print("  ReAct SQL Agent — Test Suite")
    print(f"{'─' * 72}")
    print(f"  DB          : {config.SQLITE_PATH or config.DATABASE_URL or 'PostgreSQL (env)'}")
    print(f"  LLM model   : {config.SQL_OPENAI_MODEL}")
    print(f"  Max iter    : {config.SQL_REACT_MAX_ITERATIONS}")
    print(f"  Top-K schema: {args.top_k}")
    print(f"  Streaming   : {'yes' if stream else 'no'}")
    print(f"{'═' * 72}")

    cases = (
        [TEST_CASES[args.test - 1]]
        if args.test
        else TEST_CASES
    )
    indices = [args.test] if args.test else list(range(1, len(TEST_CASES) + 1))

    results = []
    for i, case in zip(indices, cases):
        r = run_single_test(i, case, args.top_k, args.verbose, stream)
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
