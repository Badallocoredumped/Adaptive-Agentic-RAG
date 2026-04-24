"""
LangGraph ReAct agent layer on top of the TableRAG pipeline.

Architecture
------------
Step 1  User question arrives.
Step 2  TableRAG (sql_agent.py) retrieves the relevant schema context
        (table names + column names) via FAISS semantic search.
Step 3  This module receives:
          • query          – original user question
          • schema_context – TableRAG-produced "Table: X | Columns: a,b,c" lines
        and builds a LangGraph ReAct agent (compiled graph) that:
          • has the schema context baked into its system prompt
          • can call schema_lookup to discover extra tables if needed
          • can call execute_sql to run and verify SQL queries
Step 4  The agent reasons via native tool calling and iterates until done.
Step 5  run_react_sql_agent() returns an AgentResult-compatible dict.

Circular-import avoidance
-------------------------
react_agent.py  ←  sql_agent.py  (sql_agent imports run_react_sql_agent)
react_agent.py  →  table_rag.py  (top-level import, no cycle)
react_agent.py  →  sql_agent.py  (LAZY import inside schema_lookup tool body only)
"""

from __future__ import annotations

import json
import logging
import textwrap
import threading
import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from backend import config
from backend.sql.database import execute_query
from backend.sql.table_rag import retrieve_relevant_schema

logger = logging.getLogger(__name__)


def _debug(message: str) -> None:
    if getattr(config, "DEBUG_LOGGING", False):
        print(message)


def _trunc(text: str, n: int = 200) -> str:
    return text if len(text) <= n else text[:n] + "…"


def _print_msg(step: int, msg: Any) -> None:
    """Print a single LangGraph message in the live trace format."""
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            for tc in msg.tool_calls:
                raw = next(iter(tc.get("args", {}).values()), "")
                print(f"  🔧 [{step}] TOOL CALL → {tc['name']}")
                print(f"       Input: {raw}")
        else:
            content = msg.content or ""
            if content.strip():
                print(f"  🤖 [{step}] AGENT FINAL MESSAGE")
                print(textwrap.fill(
                    _trunc(content, 300),
                    width=72,
                    initial_indent="       ",
                    subsequent_indent="       ",
                ))
    elif isinstance(msg, ToolMessage):
        obs = msg.content if isinstance(msg.content, str) else str(msg.content)
        icon = "✅" if obs.startswith("SUCCESS") else "❌"
        print(f"  {icon} [{step}] OBSERVATION")
        print(f"       {_trunc(obs, 200)}")
    print()

# ---------------------------------------------------------------------------
# Regex helpers (kept local to avoid circular imports with sql_agent.py)
# ---------------------------------------------------------------------------

_SQL_READ_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_SQL_EXTRACT_RE = re.compile(
    r"\b(SELECT|WITH)\b[^;]*(?:;|$)",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# ReAct system prompt
#
# LangGraph uses native tool calling (function calling API), so there is no
# need for {{tools}}/{{tool_names}}/{{agent_scratchpad}} placeholders.
# The system message only needs to set context and rules.
# ---------------------------------------------------------------------------

def _build_system_prompt(schema_context: str, max_iter: int) -> str:
    """Return a system message string with schema context and reasoning rules."""
    dialect = "SQLite" if config.SQLITE_PATH else "PostgreSQL"
    syntax_note = (
        "Use SQLite syntax (LIKE not ILIKE, strftime not EXTRACT, etc.)."
        if config.SQLITE_PATH
        else "Use standard PostgreSQL syntax."
    )
    return (
    f"You are an expert SQL analyst working with a {dialect} database.\n\n"
    "TableRAG has already retrieved the following schema context relevant to this query.\n"
    "Use this as your starting point — call schema_lookup if you need additional tables.\n\n"
    f"{schema_context}\n\n"

    "══════════════════════════════════════════════\n"
    "  MULTI-HOP REASONING — proceed hop by hop\n"
    "══════════════════════════════════════════════\n\n"

    "  HOP 1 · UNDERSTAND THE QUESTION\n"
    "    - Identify every table and column the question mentions or implies.\n"
    "    - For each concept (e.g. 'charter funding type', 'school status', 'enrollment'),\n"
    "      confirm WHICH table and WHICH exact column name holds it — do not assume.\n"
    "      If any table or column is absent from the schema context above,\n"
    "      call schema_lookup with a descriptive topic BEFORE continuing.\n"
    "    - If the same concept could plausibly live in multiple tables\n"
    "      (e.g. funding type in both frpm AND schools), probe both:\n"
    "        SELECT * FROM <table> LIMIT 1;\n"
    "      and confirm the right source before using it.\n\n"

    "  HOP 2 · EXPLORE UNCERTAIN VALUES\n"
    "    - For every string filter (county name, category, status code, etc.)\n"
    "      run a small probe query to confirm the exact stored value:\n"
    "        SELECT DISTINCT <column> FROM <table> LIMIT 10;\n"
    "    - If a JOIN key is unclear, probe both sides of the join.\n"
    "    - DIRECTIONAL FILTERS: if the question says 'A is greater than B by N',\n"
    "      that means (A - B) > N, NOT ABS(A - B) > N.\n"
    "      Only use ABS() when the question explicitly says 'difference' without direction.\n"
    "    - Only move to HOP 3 once you know the exact values to use.\n\n"

    "  HOP 3 · BUILD AND VERIFY THE FINAL QUERY\n"
    "    - Write the complete SELECT query using only confirmed tables,\n"
    "      columns, and values discovered in the previous hops.\n"
    "    - Execute it with execute_sql.\n"
    "    - If it errors: read the message, fix the specific problem, retry.\n"
    "    - If it returns 0 rows: revisit your filter values from HOP 2.\n\n"

    f"You have up to {max_iter} tool-calling rounds — spread them across all hops.\n\n"

    "══════════════════════════════════════════════\n"
    "  STRICT RULES\n"
    "══════════════════════════════════════════════\n"
    "  1. Only reference tables/columns confirmed to exist (schema context or schema_lookup).\n"
    "  2. Read-only queries only — never INSERT, UPDATE, or DELETE.\n"
    f"  3. {syntax_note}\n"
    "  4. SELECT ONLY the columns the question asks for — no extra helper columns,\n"
    "     no extra address fields, no extra identifiers beyond what is asked.\n"
    "  5. ALIAS CONSISTENCY: every alias in FROM/JOIN must be used exactly as written.\n"
    "     Example: `FROM satscores ss JOIN frpm f` → use `ss.col` and `f.col`, never `s.col`.\n"
    "  6. COLUMN QUOTING: names with spaces or special characters MUST use double quotes.\n"
    '     Example: `"FRPM Count (K-12)"`, `"Charter School (Y/N)"`.\n'
    "  7. UNKNOWN COLUMN: on 'no such column: X' — call schema_lookup immediately;\n"
    "     do NOT retry the same broken query.\n"
    "  8. NEVER HARDCODE DATA VALUES: never embed a specific row value (a count, an ID,\n"
    "     a CDSCode, a score) that you recall from training or saw in an earlier result\n"
    "     as a filter or lookup target. Always derive the target dynamically:\n"
    "       WRONG: WHERE \"FRPM Count (K-12)\" = 4419\n"
    "       WRONG: WHERE CDSCode = '01611760135244'\n"
    "       RIGHT: WHERE CDSCode = (SELECT CDSCode FROM frpm ORDER BY \"FRPM Count (K-12)\" DESC LIMIT 1)\n"
    "  9. TOP-N AND MAX MUST BE DYNAMIC: when the question asks for the school/district\n"
    "     with the highest or lowest value, always use ORDER BY ... DESC/ASC LIMIT N\n"
    "     or a subquery — never a hardcoded WHERE clause with a constant.\n"
    " 10. AGGREGATION SCOPE: match the aggregation level to the question exactly.\n"
    "     'The school with the highest X' → ORDER BY school-level X, LIMIT 1.\n"
    "     'The district with the highest average X' → GROUP BY district, ORDER BY AVG(X).\n"
    "     Do not substitute one for the other.\n"
    " 11. COMPLETE FILTER SET: every condition in the question must appear in WHERE.\n"
    "     Before finalizing, re-read the question and verify each constraint is present.\n"
    "     Missing even one filter (e.g. Charter Funding Type, StatusType) will fail.\n"
)


# ---------------------------------------------------------------------------
# Internal SQLite helper (no import from sql_agent to avoid circularity)
# ---------------------------------------------------------------------------

def _raw_execute_sql(sql: str) -> list[dict[str, Any]]:
    """Execute a SQL query against the configured DB (SQLite or PostgreSQL)."""
    return execute_query(sql)


def _extract_and_normalise_sql(text: str) -> str | None:
    """Extract the first SELECT/WITH statement from free text, strip fences."""
    cleaned = text.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    match = _SQL_EXTRACT_RE.search(cleaned)
    if not match:
        return None

    sql = match.group(0).strip()
    if not sql.endswith(";"):
        sql += ";"

    # Enforce read-only safety
    if not _SQL_READ_RE.match(sql):
        return None

    return sql


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------

def _make_execute_sql_tool() -> Tool:
    """Build the execute_sql LangChain Tool.

    The agent sends its generated SQL here; the tool returns JSON rows or an
    error string so the agent can iterate.  Results are capped at 50 rows to
    keep the scratchpad manageable.
    """
    _MAX_ROWS = 50

    def execute_sql(sql_input: str) -> str:
        sql = _extract_and_normalise_sql(sql_input) or sql_input.strip()

        if not _SQL_READ_RE.match(sql):
            return (
                "ERROR: Only SELECT or WITH (read-only) queries are permitted. "
                "Rewrite your query."
            )

        try:
            rows = _raw_execute_sql(sql)
        except RuntimeError as exc:
            err = str(exc)
            if "no such column" in err:
                bad_col = err.split("no such column:")[-1].strip().split("\n")[0]
                return (
                    f"ERROR: SQL execution failed: {err}\n"
                    f"ACTION REQUIRED: column '{bad_col}' does not exist. "
                    "You MUST call schema_lookup now to get the exact column names "
                    "before retrying. Do NOT guess or retry the same query."
                )
            if "syntax error" in err:
                return (
                    f"ERROR: SQL execution failed: {err}\n"
                    "HINT: Check for unquoted column names containing spaces or "
                    'special characters — wrap them in double quotes, e.g. "Column Name (unit)".'
                )
            return f"ERROR: SQL execution failed: {err}"

        if not rows:
            return "SUCCESS: 0 row(s) returned.\n[]"

        truncated = rows[:_MAX_ROWS]
        result_json = json.dumps(truncated, ensure_ascii=False, default=str)
        suffix = (
            f"\n(truncated: showing {_MAX_ROWS} of {len(rows)} total rows)"
            if len(rows) > _MAX_ROWS
            else ""
        )
        return f"SUCCESS: {len(rows)} row(s) returned.\n{result_json}{suffix}"

    dialect = "SQLite" if config.SQLITE_PATH else "PostgreSQL"
    return Tool(
        name="execute_sql",
        func=execute_sql,
        description=(
            f"Execute a SQL SELECT or WITH query against the {dialect} database. "
            "Input: a valid SQL query string (may include markdown fences — they are stripped). "
            "Output: JSON rows on success, or an error message to guide your next attempt. "
            "Always call this tool to verify your SQL before reporting the final answer."
        ),
    )


def _make_schema_lookup_tool() -> Tool:
    """Build the schema_lookup LangChain Tool.

    Lets the agent discover additional tables if the initial TableRAG context
    seems insufficient.  Uses a lazy import of _ensure_schema_index_exists
    from sql_agent to avoid a circular module-level import.
    """

    def schema_lookup(topic: str) -> str:
        try:
            # Lazy import: sql_agent is guaranteed to be loaded by the time
            # this tool function is invoked (it is the caller's module).
            from backend.sql.sql_agent import _ensure_schema_index_exists  # noqa: PLC0415

            _ensure_schema_index_exists()
            results = retrieve_relevant_schema(topic, top_k=config.SQL_TOP_K)
            if not results:
                return (
                    "No additional schema found for this topic. "
                    "Only use the tables already provided in your context."
                )
            return "\n".join(results)
        except Exception as exc:  # pragma: no cover
            return f"Schema lookup failed: {exc}"

    return Tool(
        name="schema_lookup",
        func=schema_lookup,
        description=(
            "Look up database tables and their columns relevant to a topic. "
            "Input: a natural-language description of the data domain you need "
            "(e.g. 'employee salaries', 'order amounts by city'). "
            "Output: matching table/column lines from the schema index. "
            "Use this ONLY when the initial schema context seems incomplete."
        ),
    )


# ---------------------------------------------------------------------------
# LLM factory (reuses project config — gpt-4o-mini, temperature 0)
# ---------------------------------------------------------------------------

_react_llm_lock = threading.Lock()
_react_llm_instance: ChatOpenAI | None = None


def _get_react_llm() -> ChatOpenAI:
    """Return a cached, deterministic ChatOpenAI instance (thread-safe)."""
    global _react_llm_instance
    with _react_llm_lock:
        if _react_llm_instance is None:
            api_key = os.environ.get("OPENAI_API_KEY", config.OPENAI_API_KEY)
            _react_llm_instance = ChatOpenAI(
                model=config.SQL_OPENAI_MODEL,
                temperature=0.0,
                api_key=api_key,
            )
    return _react_llm_instance


# ---------------------------------------------------------------------------
# Agent factory (public, exposed for testing / extension)
# ---------------------------------------------------------------------------

def build_react_agent(schema_context: str) -> CompiledStateGraph:
    """Construct a LangGraph ReAct agent (compiled graph).

    The TableRAG schema_context is baked into the system prompt at build time
    so the agent always reasons within the discovered schema boundary.

    Args:
        schema_context: Multi-line string produced by retrieve_relevant_schema(),
                        e.g. "Table: orders | Columns: id, amount, status\\n..."

    Returns:
        A compiled LangGraph state graph invoked with
        .invoke({"messages": [("human", query)]}).
    """
    max_iter: int = getattr(config, "SQL_REACT_MAX_ITERATIONS", 10)

    tools = [
        _make_schema_lookup_tool(),
        _make_execute_sql_tool(),
    ]

    system_prompt = _build_system_prompt(schema_context=schema_context, max_iter=max_iter)
    llm = _get_react_llm()

    return create_react_agent(model=llm, tools=tools, prompt=system_prompt)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _get_full_schema_context() -> str:
    """Return every schema text stored in the FAISS index (or live DB schema)."""
    from backend.sql.table_rag import _SCHEMA_TEXTS_PATH  # noqa: PLC0415

    if _SCHEMA_TEXTS_PATH.exists():
        with _SCHEMA_TEXTS_PATH.open("r", encoding="utf-8") as _f:
            texts: list[str] = json.load(_f)
        return "\n".join(texts)

    # Fallback when index has not been built yet
    from backend.sql.database import get_live_schema  # noqa: PLC0415

    return "\n".join(get_live_schema().to_embedding_texts())


def run_react_sql_agent(
    query: str,
    schema_context: str,
    *,
    _retry: bool = False,
) -> dict[str, Any]:
    """Run the ReAct agent to answer *query* using the TableRAG *schema_context*.

    This is the primary entry point called by run_table_rag_pipeline() after
    schema retrieval, replacing the simple single-pass _generate_sql() call
    with an iterative reasoning loop.

    Args:
        query:          Original user question (natural language).
        schema_context: TableRAG-produced schema lines.  The agent is
                        instructed to stay within these tables/columns.

    Returns:
        Dict with keys matching AgentResult:
          "sql"            – last SQL query successfully executed, or None
          "result"         – rows from that execution as list[dict]
          "error"          – error message if the agent could not produce SQL
          "schema_context" – the schema context passed to the agent
    """
    _debug(f"\n[ReAct Agent] Starting for query: {query!r}")
    logger.info("[ReAct Agent] query=%r", query)

    # ── Print TableRAG schema context ────────────────────────────────────────
    schema_lines = [l for l in schema_context.splitlines() if l.strip()]
    from backend.sql.sql_agent import _extract_table_names  # noqa: PLC0415
    table_names = _extract_table_names(schema_context)
    label = "(full schema fallback)" if _retry else "(TableRAG)"
    print(f"\n  📋 Schema {label} — tables: {table_names}")
    for line in schema_lines:
        if not line.startswith("---"):
            print(f"       {line}")

    max_iter: int = getattr(config, "SQL_REACT_MAX_ITERATIONS", 6)
    graph = build_react_agent(schema_context)

    print(f"\n  💬 Human: {query}\n")

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
                if isinstance(msg, HumanMessage):
                    continue
                step += 1
                _print_msg(step, msg)
    except Exception as exc:
        logger.error("[ReAct Agent] graph.stream() failed: %s", exc)
        return {
            "sql": None,
            "result": [],
            "error": f"ReAct agent graph failed: {exc}",
            "schema_context": schema_context,
        }

    # ── Extract final SQL from collected messages ────────────────────────────
    # Priority order:
    #   1. SQL in the agent's FINAL AIMessage (its declared conclusion) —
    #      avoids picking up an intermediate/exploratory execute_sql call.
    #   2. Last successful execute_sql input (fallback when the agent does
    #      not restate the SQL in its conclusion text).

    # Map tool_call_id → (tool_name, raw_input_str)
    tool_call_map: dict[str, tuple[str, str]] = {}
    for msg in all_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                raw_input = next(iter(tc.get("args", {}).values()), "") if tc.get("args") else ""
                tool_call_map[tc["id"]] = (tc["name"], str(raw_input))

    final_sql: str | None = None
    final_rows: list[dict] = []
    final_error: str | None = None

    # Priority 1: SQL in the agent's final non-tool-calling AIMessage
    last_ai_msg = next(
        (m for m in reversed(all_messages) if isinstance(m, AIMessage) and not m.tool_calls),
        None,
    )
    if last_ai_msg:
        content = last_ai_msg.content if isinstance(last_ai_msg.content, str) else ""
        final_sql = _extract_and_normalise_sql(content)

    # Priority 2: last successful execute_sql call (kept as fallback)
    fallback_sql: str | None = None
    fallback_rows: list[dict] = []

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
            fallback_sql = candidate_sql
            final_error = None  # Agent recovered from earlier errors
            if "[" not in observation:
                fallback_rows = []
            else:
                try:
                    json_start = observation.index("[")
                    json_part = observation[json_start:].split("\n(truncated")[0]
                    fallback_rows = json.loads(json_part)
                except (ValueError, json.JSONDecodeError):
                    try:
                        fallback_rows = _raw_execute_sql(candidate_sql)
                    except RuntimeError:
                        fallback_rows = []
        else:
            if fallback_sql is None and final_sql is None:
                final_error = observation

    if final_sql is None:
        if fallback_sql:
            final_sql = fallback_sql
            final_rows = fallback_rows
        elif final_error is None:
            output_text = last_ai_msg.content if last_ai_msg else ""
            final_error = (
                f"ReAct agent completed without executing any SQL. "
                f"Agent output: {output_text}"
            )
    else:
        # Final AIMessage SQL found — execute it to get rows
        try:
            final_rows = _raw_execute_sql(final_sql)
        except RuntimeError:
            # Execution failed — fall back to the last verified execute_sql result
            if fallback_sql:
                final_sql = fallback_sql
                final_rows = fallback_rows

    print(f"  📌 Final SQL : {final_sql or 'None'}")
    print(f"  📊 Rows      : {len(final_rows)}")
    if final_error:
        print(f"  ⚠️  Error     : {_trunc(final_error, 120)}")
    print()

    # ── Error-based retry with full schema ──────────────────────────────────
    # The agent always starts with TableRAG schema.  If execute_sql returned
    # ERROR on 4+ calls the partial schema is likely missing relevant tables;
    # fetch the complete DB schema and re-run once so the agent has full context.
    if not _retry:
        error_count = sum(
            1
            for msg in all_messages
            if isinstance(msg, ToolMessage)
            and tool_call_map.get(msg.tool_call_id, ("", ""))[0] == "execute_sql"
            and (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            ).startswith("ERROR")
        )
        if error_count >= 4:
            logger.info(
                "[ReAct Agent] %d execute_sql errors — fetching full schema and retrying",
                error_count,
            )
            print(
                f"\n  🔄 [{error_count}x SQL errors] Fetching full DB schema and re-running...\n"
            )
            try:
                full_schema = _get_full_schema_context()
                augmented_schema = (
                    f"{schema_context}\n\n"
                    "--- FULL DATABASE SCHEMA (added after repeated SQL errors; use this to find the correct tables/columns) ---\n"
                    f"{full_schema}"
                )
                retry_result = run_react_sql_agent(
                    query, augmented_schema, _retry=True
                )
                if retry_result["result"] or retry_result["sql"]:
                    return retry_result
            except Exception as exc:
                logger.warning("[ReAct Agent] Full-schema retry failed: %s", exc)

    return {
        "sql": final_sql,
        "result": final_rows,
        "error": final_error,
        "schema_context": schema_context,
    }
