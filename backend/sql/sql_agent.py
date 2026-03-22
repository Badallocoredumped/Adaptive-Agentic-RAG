"""LangChain-based SQL agent module (isolated from existing rule-based SQL pipeline)."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from backend import config

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class AgentResult(TypedDict):
    sql: str | None
    result: list[dict[str, Any]]
    error: str | None


# ---------------------------------------------------------------------------
# SQL tracing
# ---------------------------------------------------------------------------

_SQL_TOOL_NAMES = {"sql_db_query", "query_sql_db"}

class SQLTraceCallbackHandler(BaseCallbackHandler):
    """Captures the most recent SQL statement executed by LangChain SQL tools."""

    def __init__(self) -> None:
        self.last_sql: str | None = None

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        tool_name = str((serialized or {}).get("name", ""))
        if tool_name in _SQL_TOOL_NAMES or any(t in tool_name for t in _SQL_TOOL_NAMES):
            sql = input_str.strip()
            if sql:
                self.last_sql = sql


# ---------------------------------------------------------------------------
# SQL extraction helpers
# ---------------------------------------------------------------------------

# Matches SELECT or WITH at word boundary, then everything up to a semicolon
# or end-of-string.  The non-greedy inner match + possessive-style boundary
# avoids catastrophic backtracking on large agent outputs.
_SQL_PATTERN = re.compile(
    r"\b(SELECT|WITH)\b[^;]*(?:;|$)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_sql_from_text(text: str) -> str | None:
    """Return the first SELECT/WITH statement found in *text*, or None."""
    if not text:
        return None
    match = _SQL_PATTERN.search(text)
    return match.group(0).strip() if match else None


def _extract_sql_from_steps(response: dict[str, Any]) -> str | None:
    """Walk intermediate steps and return the first sql_db_query input found."""
    for step in response.get("intermediate_steps") or []:
        if not isinstance(step, (tuple, list)) or not step:
            continue

        action = step[0]
        if not _is_sql_tool(getattr(action, "tool", "")):
            continue

        sql = _sql_from_tool_input(getattr(action, "tool_input", None))
        if sql:
            return sql

        sql = _extract_sql_from_text(str(getattr(action, "log", "")))
        if sql:
            return sql

    return None


def _is_sql_tool(tool_name: str) -> bool:
    return tool_name in _SQL_TOOL_NAMES or any(t in tool_name for t in _SQL_TOOL_NAMES)


def _sql_from_tool_input(tool_input: Any) -> str | None:
    if isinstance(tool_input, str):
        return tool_input.strip() or None
    if isinstance(tool_input, dict):
        for key in ("query", "sql", "input"):
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_sql(trace: SQLTraceCallbackHandler, response: dict[str, Any]) -> str | None:
    """Return the best available SQL in priority order: callback > steps > output text."""
    return (
        trace.last_sql
        or _extract_sql_from_steps(response)
        or _extract_sql_from_text(str(response.get("output", "")))
    )


# ---------------------------------------------------------------------------
# DB / agent construction
# ---------------------------------------------------------------------------

_AGENT_PREFIX = (
    "You are an expert SQL assistant working against a SQLite database. "
    "You must use the available SQL tools to answer questions — never guess. "
    "Required sequence: list tables → inspect relevant schema → "
    "execute sql_db_query for final computation → answer."
)

_INITIAL_SUFFIX = (
    "\n\nImportant: Use sql_db_query for the final computation and base "
    "the final answer on that query result."
)

_RETRY_SUFFIX = (
    "\n\nMandatory tool sequence before answering: "
    "1) sql_db_list_tables  2) sql_db_schema for relevant tables  "
    "3) sql_db_query for the final result."
)


def _sqlite_uri(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve().as_posix()}"


def _build_agent(llm: ChatOpenAI, db: SQLDatabase, callbacks: list[Any]):
    return create_sql_agent(
        llm=llm,
        db=db,
        prefix=_AGENT_PREFIX,
        verbose=True,
        agent_executor_kwargs={
            "callbacks": callbacks,
            "return_intermediate_steps": True,
            "handle_parsing_errors": True,
        },
    )


# ---------------------------------------------------------------------------
# SQL execution
# ---------------------------------------------------------------------------

def _execute_sql(sql: str) -> list[dict[str, Any]]:
    """Run *sql* against the configured SQLite DB and return rows as dicts."""
    try:
        with sqlite3.connect(str(config.SQLITE_DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql)
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        raise RuntimeError(f"SQL execution failed: {exc}\nQuery: {sql}") from exc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_sql_agent(query: str) -> AgentResult:
    """Run a LangChain SQL agent against *query* and return SQL + row results.

    Returns an :class:`AgentResult` with:
    - ``sql``    — the SQL that was (or would have been) executed, or None.
    - ``result`` — rows returned by that SQL, or an empty list.
    - ``error``  — a human-readable error string if something went wrong, else None.
    """
    trace = SQLTraceCallbackHandler()
    llm = ChatOpenAI(
        model=config.ROUTER_MODEL,
        temperature=0,
        base_url=f"{config.ROUTER_BASE_URL.rstrip('/')}/v1",
        api_key=config.ROUTER_API_KEY,
        timeout=config.ROUTER_TIMEOUT_SECONDS,
    )
    db = SQLDatabase.from_uri(_sqlite_uri(Path(config.SQLITE_DB_PATH)))
    agent = _build_agent(llm, db, callbacks=[trace])

    response = agent.invoke({"input": query + _INITIAL_SUFFIX})
    sql = _extract_sql(trace, response)

    if not sql:
        response = agent.invoke({"input": query + _RETRY_SUFFIX})
        sql = _extract_sql(trace, response)

    if not sql:
        return AgentResult(sql=None, result=[], error="Agent did not produce a SQL query.")

    try:
        rows = _execute_sql(sql)
    except RuntimeError as exc:
        return AgentResult(sql=sql, result=[], error=str(exc))

    return AgentResult(sql=sql, result=rows, error=None)


if __name__ == "__main__":
    demo_query = "What is the total revenue from all orders?"
    print(run_sql_agent(demo_query))