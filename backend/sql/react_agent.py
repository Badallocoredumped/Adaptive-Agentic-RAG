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
        "This is the AUTHORITATIVE list of tables and columns you may use:\n\n"
        f"{schema_context}\n\n"
        "STRICT RULES — follow these without exception:\n"
        "  1. Only reference tables and columns that appear in the schema context above.\n"
        "  2. Do NOT invent, assume, or guess tables or columns that are not listed.\n"
        "  3. Write only SELECT or WITH (read-only) queries — never INSERT, UPDATE, DELETE.\n"
        f"  4. {syntax_note}\n"
        "  5. If you think more tables might be needed, call schema_lookup FIRST to verify.\n"
        "  6. WRITE COMPLETE QUERIES. Use JOINs to connect tables. Do NOT write exploratory step-by-step queries just to look up intermediate IDs.\n"
        "  7. Always call execute_sql to test your SQL before reporting the final answer.\n"
        "  8. If execute_sql returns an error, read the error message, fix the SQL, and retry.\n"
        f"     You have up to {max_iter} tool-calling rounds — use them wisely.\n"
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
            return f"ERROR: {exc}"

        if not rows:
            return "Query executed successfully. Result set is empty (0 rows)."

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
    max_iter: int = getattr(config, "SQL_REACT_MAX_ITERATIONS", 6)

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

def run_react_sql_agent(
    query: str,
    schema_context: str,
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

    max_iter: int = getattr(config, "SQL_REACT_MAX_ITERATIONS", 6)
    graph = build_react_agent(schema_context)

    try:
        # LangGraph uses a messages-based interface.
        # recursion_limit caps the number of graph steps (each tool call = 2 steps).
        agent_output = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": max_iter * 3},
        )
    except Exception as exc:
        logger.error("[ReAct Agent] graph.invoke() failed: %s", exc)
        _debug(f"[ReAct Agent] Graph error: {exc}")
        return {
            "sql": None,
            "result": [],
            "error": f"ReAct agent graph failed: {exc}",
            "schema_context": schema_context,
        }

    # ── Walk messages to find the last *successful* execute_sql call ──
    # Message sequence:  HumanMessage → AIMessage(tool_calls) → ToolMessage → …
    # Build a map from tool_call_id → (tool_name, first_arg_value) using AIMessages,
    # then correlate each ToolMessage observation back to its call.
    messages: list = agent_output.get("messages", [])

    # Map tool_call_id → (tool_name, raw_input_str)
    tool_call_map: dict[str, tuple[str, str]] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                # args is a dict; for single-arg tools grab the first value
                raw_input = next(iter(tc.get("args", {}).values()), "") if tc.get("args") else ""
                tool_call_map[tc["id"]] = (tc["name"], str(raw_input))

    final_sql: str | None = None
    final_rows: list[dict] = []
    final_error: str | None = None

    for msg in messages:
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
            # Parse JSON rows embedded in the observation string
            if "[" not in observation:
                final_rows = []
            else:
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

    # If the agent never executed SQL at all, surface the final AI message text
    if final_sql is None and final_error is None:
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls),
            None,
        )
        output_text = last_ai.content if last_ai else ""
        final_error = (
            f"ReAct agent completed without executing any SQL. "
            f"Agent output: {output_text}"
        )

    _debug(f"[ReAct Agent] Final SQL : {final_sql}")
    _debug(f"[ReAct Agent] Rows      : {len(final_rows)}")
    if final_error:
        _debug(f"[ReAct Agent] Error     : {final_error}")

    return {
        "sql": final_sql,
        "result": final_rows,
        "error": final_error,
        "schema_context": schema_context,
    }
