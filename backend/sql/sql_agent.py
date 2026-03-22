"""TableRAG-infused SQL generation + execution module."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI

from .. import config
from .table_rag import (
    build_schema_index,
    get_schema_texts,
    retrieve_relevant_schema,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class AgentResult(TypedDict):
    sql: str | None
    result: list[dict[str, Any]]
    error: str | None
    schema_context: str


_SQL_PATTERN = re.compile(
    r"\b(SELECT|WITH)\b[^;]*(?:;|$)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_sql_from_text(text: str) -> str | None:
    """Return the first SELECT/WITH statement found in text, or None."""
    if not text:
        return None
    match = _SQL_PATTERN.search(text)
    return match.group(0).strip() if match else None


def _normalize_sql(raw_text: str) -> str | None:
    """Extract and normalize SQL from LLM text output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    sql = _extract_sql_from_text(cleaned)
    if not sql:
        return None

    sql = sql.strip()
    if not sql.endswith(";"):
        sql = f"{sql};"

    # Safety boundary: this module only executes read queries.
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
        return None

    return sql


def _extract_table_names(schema_context: str) -> list[str]:
    """Extract table names from schema context lines."""
    names = re.findall(r"Table:\s*([A-Za-z_][A-Za-z0-9_]*)", schema_context)
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def _load_sqlite_schema_dict() -> dict[str, list[str]]:
    """Read SQLite schema into {table: [columns...]} for TableRAG indexing."""
    schema: dict[str, list[str]] = {}
    with sqlite3.connect(str(config.SQLITE_DB_PATH)) as conn:
        cursor = conn.cursor()
        table_rows = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        ).fetchall()

        for (table_name,) in table_rows:
            col_rows = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
            schema[str(table_name)] = [str(row[1]) for row in col_rows]

    return schema


def _ensure_schema_index_exists() -> None:
    """Build TableRAG schema index if it does not already exist."""
    index_path = config.INDEX_DIR / "schema.faiss"
    texts_path = config.INDEX_DIR / "schema_texts.json"
    if index_path.exists() and texts_path.exists():
        return

    schema_dict = _load_sqlite_schema_dict()
    if schema_dict:
        build_schema_index(schema_dict)


def _resolve_schema_context(query: str, schema_context: str | None, top_k: int) -> str:
    """Resolve schema context from caller input or TableRAG retrieval."""
    if schema_context and schema_context.strip():
        return schema_context.strip()

    _ensure_schema_index_exists()
    retrieved = retrieve_relevant_schema(query, top_k=top_k)
    return "\n".join(retrieved)


def _generate_sql(query: str, schema_context: str) -> str | None:
    """Generate SQL in a single pass using only pruned schema context."""
    llm = ChatOpenAI(
        model=config.ROUTER_MODEL,
        temperature=0,
        base_url=f"{config.ROUTER_BASE_URL.rstrip('/')}/v1",
        api_key=config.ROUTER_API_KEY,
        timeout=config.ROUTER_TIMEOUT_SECONDS,
    )

    prompt = (
        "You are a SQL expert.\n\n"
        "You ONLY have access to the following tables:\n\n"
        f"{schema_context}\n\n"
        "Task: Write one correct SQLite SELECT query for the user request.\n"
        "Rules:\n"
        "- Use only the tables/columns shown above.\n"
        "- Return SQL only, no explanation, no markdown.\n"
        "- Must be a SELECT/WITH read query.\n\n"
        f"User request: {query}"
    )

    response = llm.invoke(prompt)
    return _normalize_sql(str(getattr(response, "content", "")))


def _log_schema_tables(schema_context: str) -> None:
    tables = _extract_table_names(schema_context)
    print(f"[SQL AGENT] schema tables passed: {tables if tables else 'none parsed'}")
    return None


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

def run_sql_agent(
    query: str,
    schema_context: str | None = None,
    top_k: int = 3,
) -> AgentResult:
    """Generate SQL from pruned schema context, execute it, and return rows.

    This is a single-pass flow without retry/fallback chains.
    """
    resolved_schema_context = _resolve_schema_context(query, schema_context, top_k)
    _log_schema_tables(resolved_schema_context)

    if not resolved_schema_context.strip():
        return AgentResult(
            sql=None,
            result=[],
            error="No schema context available. Build table schema index and try again.",
            schema_context="",
        )

    sql = _generate_sql(query, resolved_schema_context)
    if not sql:
        return AgentResult(
            sql=None,
            result=[],
            error="LLM did not return a valid SELECT/WITH SQL query.",
            schema_context=resolved_schema_context,
        )

    try:
        rows = _execute_sql(sql)
    except RuntimeError as exc:
        return AgentResult(
            sql=sql,
            result=[],
            error=str(exc),
            schema_context=resolved_schema_context,
        )

    return AgentResult(
        sql=sql,
        result=rows,
        error=None,
        schema_context=resolved_schema_context,
    )


# ---------------------------------------------------------------------------
# TableRAG pipeline entry point
# ---------------------------------------------------------------------------

def run_table_rag_pipeline(
    query: str,
    top_k: int = 3,
) -> dict:
    """End-to-end pipeline: TableRAG schema retrieval → SQL generation → execution.

    Steps:
        1. Retrieve relevant schema via TableRAG.
        2. Format schema into a single context string.
        3. Pass the pruned schema to run_sql_agent for SQL generation + execution.
        4. Return a dict with schema_used, sql, and result.
    """
    # Step 1 — retrieve relevant schema rows from the FAISS index
    _ensure_schema_index_exists()
    schema_rows: list[str] = retrieve_relevant_schema(query, top_k=top_k)

    # Logging: which tables were selected
    table_names = _extract_table_names("\n".join(schema_rows))
    print(f"[TableRAG Pipeline] Query: {query!r}")
    print(f"[TableRAG Pipeline] Retrieved schema rows ({len(schema_rows)}):")
    for row in schema_rows:
        print(f"  → {row}")
    print(f"[TableRAG Pipeline] Selected tables: {table_names}")

    # Step 2 — format into a single context string
    schema_context = "\n".join(schema_rows)

    # Step 3 — generate + execute SQL via the single-pass agent
    agent_result = run_sql_agent(query, schema_context=schema_context, top_k=top_k)

    # Logging: generated SQL
    print(f"[TableRAG Pipeline] Generated SQL: {agent_result['sql']}")
    if agent_result["error"]:
        print(f"[TableRAG Pipeline] Error: {agent_result['error']}")
    else:
        print(f"[TableRAG Pipeline] Rows returned: {len(agent_result['result'])}")

    # Step 4 — return unified result dict
    return {
        "schema_used": schema_rows,
        "sql": agent_result["sql"],
        "result": agent_result["result"],
        "error": agent_result["error"],
    }


if __name__ == "__main__":
    demo_query = "What is the total revenue from all orders?"
    print("=" * 60)
    print("run_sql_agent demo")
    print("=" * 60)
    print(run_sql_agent(demo_query))

    print()
    print("=" * 60)
    print("run_table_rag_pipeline demo")
    print("=" * 60)
    print(run_table_rag_pipeline(demo_query))