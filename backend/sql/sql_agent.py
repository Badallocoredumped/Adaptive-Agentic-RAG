"""TableRAG-infused SQL generation + execution module."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI

from backend import config
from backend.sql.table_rag import (
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


if __name__ == "__main__":
    demo_query = "What is the total revenue from all orders?"
    print(run_sql_agent(demo_query))