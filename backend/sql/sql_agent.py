"""TableRAG-infused SQL generation + execution module."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, TypedDict

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


# ---------------------------------------------------------------------------
# Few-shot examples used to guide the cache-hit SQL refiner
# ---------------------------------------------------------------------------

_REFINE_FEW_SHOT_EXAMPLES = [
    {
        "original_question": "What is the total revenue from all orders?",
        "cached_sql": "SELECT SUM(amount) AS total_revenue FROM orders;",
        "new_question": "What is the total revenue from completed orders?",
        "refined_sql": "SELECT SUM(amount) AS total_revenue FROM orders WHERE status = 'completed';",
    },
    {
        "original_question": "Show me the top 3 cities by number of customers.",
        "cached_sql": "SELECT city, COUNT(id) AS customer_count FROM customers GROUP BY city ORDER BY customer_count DESC LIMIT 3;",
        "new_question": "Show me the top 5 cities by number of customers.",
        "refined_sql": "SELECT city, COUNT(id) AS customer_count FROM customers GROUP BY city ORDER BY customer_count DESC LIMIT 5;",
    },
    {
        "original_question": "What is the average salary per department?",
        "cached_sql": "SELECT d.name, AVG(e.salary) AS avg_salary FROM employees e JOIN departments d ON e.department_id = d.id GROUP BY d.name;",
        "new_question": "What is the maximum salary per department?",
        "refined_sql": "SELECT d.name, MAX(e.salary) AS max_salary FROM employees e JOIN departments d ON e.department_id = d.id GROUP BY d.name;",
    },
]


def _build_refine_few_shot_block() -> str:
    """Format the few-shot examples into a prompt block."""
    lines = []
    for ex in _REFINE_FEW_SHOT_EXAMPLES:
        lines.append(f"Original question: {ex['original_question']}")
        lines.append(f"Cached SQL:        {ex['cached_sql']}")
        lines.append(f"New question:      {ex['new_question']}")
        lines.append(f"Refined SQL:       {ex['refined_sql']}")
        lines.append("")
    return "\n".join(lines).strip()


def _refine_sql_from_cache(
    new_question: str,
    original_question: str,
    cached_sql: str,
    schema_context: str,
) -> str | None:
    """Use OpenAI with few-shot examples to make small adjustments to a cached SQL.

    Called on a cache HIT when the new question is semantically similar but not
    identical to the cached question.  The model only adjusts what is necessary
    (e.g., a filter value, a LIMIT, or an aggregate function); it should NOT
    rewrite the query from scratch.
    """
    import os
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", config.OPENAI_API_KEY)
    client = OpenAI(api_key=api_key)

    few_shot_block = _build_refine_few_shot_block()

    system_prompt = (
        "You are a SQL refinement assistant.\n"
        "You are given a cached SQL query that was written for a similar (but not identical) question.\n"
        "Your job is to make the SMALLEST possible edits to the cached SQL so that it correctly answers "
        "the new question.\n"
        "Rules:\n"
        "  - Only change what is necessary (filter values, column names, LIMIT, aggregate function, etc.).\n"
        "  - Do NOT rewrite the query from scratch unless the structure must change.\n"
        "  - Return only the final SQL — no explanation, no markdown fences.\n"
        "  - Must remain a SELECT/WITH read-only query.\n"
        "  - Use only tables/columns available in the schema context provided.\n\n"
        "Schema context (available tables and columns):\n"
        f"{schema_context}"
    )

    user_prompt = (
        f"Here are examples of how to refine cached SQL:\n\n"
        f"{few_shot_block}\n\n"
        "---\n\n"
        f"Now refine the following:\n"
        f"Original question: {original_question}\n"
        f"Cached SQL:        {cached_sql}\n"
        f"New question:      {new_question}\n"
        f"Refined SQL:"
    )

    try:
        response = client.chat.completions.create(
            model=config.SQL_OPENAI_MODEL,
            temperature=config.SQL_REFINE_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        return _normalize_sql(raw)
    except Exception as exc:
        print(f"[SQL AGENT] Cache-hit refiner failed: {exc}. Falling back to cached SQL.")
        return None


def _generate_sql(query: str, schema_context: str) -> str | None:
    """Generate SQL in a single pass using OpenAI and pruned schema context."""
    import os
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", config.OPENAI_API_KEY)
    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a SQL expert.\n"
        "You ONLY have access to the tables listed in the schema context.\n"
        "Rules:\n"
        "  - Use only the tables/columns shown in the schema.\n"
        "  - Return SQL only, no explanation, no markdown fences.\n"
        "  - Must be a SELECT/WITH read query.\n"
    )

    user_prompt = (
        f"Schema context (available tables and columns):\n{schema_context}\n\n"
        f"Task: Write one correct SQLite SELECT query for the following request.\n"
        f"User request: {query}\n"
        f"SQL:"
    )

    try:
        response = client.chat.completions.create(
            model=config.SQL_OPENAI_MODEL,
            temperature=config.SQL_GENERATE_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        return _normalize_sql(raw)
    except Exception as exc:
        print(f"[SQL AGENT] OpenAI SQL generation failed: {exc}")
        return None


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
    top_k: int = config.SQL_TOP_K,
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


import time

# ---------------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------------

def _get_sql_cache() -> "SQLCache":
    """Lazy initialization of the FAISS SQL cache."""
    if not hasattr(_get_sql_cache, "instance"):
        from .sql_cache import SQLCache
        cache = SQLCache()
        cache.load_cache()
        _get_sql_cache.instance = cache
    return _get_sql_cache.instance

# ---------------------------------------------------------------------------
# TableRAG pipeline entry point
# ---------------------------------------------------------------------------

def run_table_rag_pipeline(
    query: str,
    top_k: int = config.SQL_TOP_K,
) -> dict:
    """End-to-end pipeline: SQL Cache Check -> TableRAG schema retrieval → SQL generation → execution."""
    start_time = time.time()
    
    # 1. Check Semantic Cache
    cache = _get_sql_cache()
    print(f"\n[TableRAG Pipeline] Analyzing query: {query!r}")
    cache_result = cache.check_cache_hit(query, threshold=0.85)

    if cache_result["hit"]:
        cached_sql = cache_result["sql"]
        original_question = cache_result.get("question", query)
        # Use the schema that was stored in FAISS metadata when this query was cached
        cached_schema = cache_result.get("schema", "")
        print(f"[TableRAG Pipeline] ⚡ CACHE HIT: Cached SQL → {cached_sql}")

        # --- Few-shot refinement on cache hit ---
        # Uses the schema already stored in the cache — no extra retrieval needed
        print("[TableRAG Pipeline] 🔧 Running few-shot SQL refiner for cache hit...")
        refined_sql = _refine_sql_from_cache(
            new_question=query,
            original_question=original_question,
            cached_sql=cached_sql,
            schema_context=cached_schema,
        )

        if refined_sql and refined_sql.strip().upper() != cached_sql.strip().upper():
            print(f"[TableRAG Pipeline] ✏️  Refiner adjusted SQL → {refined_sql}")
            sql_to_execute = refined_sql
        else:
            print("[TableRAG Pipeline] ✅ Refiner kept cached SQL unchanged.")
            sql_to_execute = cached_sql

        try:
            rows = _execute_sql(sql_to_execute)
            error = None
        except Exception as e:
            rows = []
            error = str(e)
            print(f"[TableRAG Pipeline] ❌ SQL execution failed: {error}")

        latency = time.time() - start_time
        print(f"[TableRAG Pipeline] ⏱️  Latency: {latency:.2f}s")
        return {
            "schema_used": ["<from semantic cache>"],
            "sql": sql_to_execute,
            "result": rows,
            "error": error,
            "path": "fast",
            "latency": latency
        }

    # 2. RUN FULL PIPELINE (If Cache MISS)
    print(f"[TableRAG Pipeline] 🤖 AGENT PATH: Routing to TableRAG + LLM")
    
    _ensure_schema_index_exists()
    schema_rows: list[str] = retrieve_relevant_schema(query, top_k=top_k)

    # Logging: which tables were selected
    table_names = _extract_table_names("\n".join(schema_rows))
    print(f"[TableRAG Pipeline] Retrieved schema rows ({len(schema_rows)}):")
    for row in schema_rows:
        print(f"  → {row}")
    print(f"[TableRAG Pipeline] Selected tables: {table_names}")

    # Format into a single context string
    schema_context = "\n".join(schema_rows)

    # Generate + execute SQL via the single-pass agent
    agent_result = run_sql_agent(query, schema_context=schema_context, top_k=top_k)

    # Logging: generated SQL
    print(f"[TableRAG Pipeline] Generated SQL: {agent_result['sql']}")
    if agent_result["error"]:
        print(f"[TableRAG Pipeline] Error: {agent_result['error']}")
    else:
        print(f"[TableRAG Pipeline] Rows returned: {len(agent_result['result'])}")
        
        # 3. Add successful run to Cache
        if agent_result["sql"] and not agent_result["error"]:
            print(f"[TableRAG Pipeline] 💾 Saving successful query to cache...")
            cache.add_to_cache(query, agent_result["sql"], schema_context)
            cache.save_cache()

    latency = time.time() - start_time
    print(f"[TableRAG Pipeline] ⏱️  Latency: {latency:.2f}s")

    # 4. Return unified result dict
    return {
        "schema_used": schema_rows,
        "sql": agent_result["sql"],
        "result": agent_result["result"],
        "error": agent_result["error"],
        "path": "agent",
        "latency": latency
    }


    """     # 1. Simple aggregation
    "What is the total revenue from all orders?",
    # 2. Filter by status
    "Show all pending orders with customer names",
    # 3. GROUP BY
    "How many orders does each customer have?",
    # 4. Multi-table JOIN + aggregation
    "What is the total revenue per city?",
    # 5. HAVING clause
    "Which customers have placed more than 2 orders?",
    # 6. Date range filter
    "Show all orders placed in March 2026",
    # 7. Product + order_items JOIN
    "What is the most ordered product by total quantity?",
    # 8. Complex multi-JOIN
    "Show total spending per customer on subscription products", """

if __name__ == "__main__":
    from .database import SQLiteDatabase

    # Re-create DB with all 9 tables + seed data
    import os
    db_path = str(config.SQLITE_DB_PATH)
    if os.path.exists(db_path):
        os.remove(db_path)
    # Also clear cached schema index
    for f in ["schema.faiss", "schema_texts.json"]:
        p = config.INDEX_DIR / f
        if p.exists():
            p.unlink()

    db = SQLiteDatabase(db_path)
    db.initialize_schema()
    print(f"[SETUP] Database created with 9 tables at {db_path}\n")

    # ── Stress-test queries (15 total) ───────────────────────────
    test_queries = [
        # --- DOMAIN ISOLATION: will it ignore unrelated tables? ---
        # 1. Pure HR query — should use employees + departments only
        "What is the average salary per department?",
        # 2. Pure inventory — should ignore orders/HR/blog
        "Which warehouse has the most total stock?",
        # 3. Pure support — should use support_tickets + customers
        "Show all open high-priority support tickets with customer names",
        # 4. Pure blog — should pick only blog_posts
        "What is the most viewed blog post?",

        # --- CROSS-DOMAIN JOINS: tables from different domains ---
        # 5. Sales + support — customer appears in both
        "Which customers have both completed orders and open support tickets?",
        # 6. HR + blog — employees are blog authors
        "Show all blog posts written by Engineering department employees",
        # 7. Inventory + sales — products bridge both domains
        "Which products have low stock but high order volume?",

        # --- AMBIGUOUS / TRICKY QUERIES ---
        # 8. Word "order" could mean SQL ORDER BY or the orders table
        "List departments in order of budget from highest to lowest",
        # 9. "Amount" exists in orders — but this is an HR question
        "What is the total salary amount for employees hired in 2025?",
        # 10. Needs to distinguish product.name from customer.name
        "Show product names and their total revenue from completed orders",

        # --- COMPLEX MULTI-TABLE QUERIES ---
        # 11. 3 tables: customers + orders + support_tickets
        "For each city, show total order revenue and number of support tickets",
        # 12. 4 tables: order_items + products + inventory + orders
        "Show each product with its total units sold and current stock across all warehouses",
        # 13. Subquery / CTE: top customer by revenue and their tickets
        "Who is the highest-spending customer and how many support tickets do they have?",

        # --- EDGE CASES ---
        # 14. Query about table that doesn't exist
        "Show all shipping addresses for customers",
        # 15. Vague query that could go many ways
        "Give me a summary of everything from March 2026",
    ]

    passed = 0
    failed = 0

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"  TEST {i}/{len(test_queries)}: {query}")
        print(f"{'='*70}")

        result = run_table_rag_pipeline(query)

        print(f"\n  SCHEMA → {result.get('schema_used', [])}")
        print(f"  SQL    → {result['sql']}")
        if result["error"]:
            print(f"  ❌ ERR  → {result['error']}")
            failed += 1
        else:
            print(f"  ✅ ROWS → {len(result['result'])} returned")
            for row in result["result"][:5]:
                print(f"           {row}")
            if len(result["result"]) > 5:
                print(f"           ... and {len(result['result']) - 5} more")
            passed += 1
        print()

    print(f"\n{'='*70}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {len(test_queries)} total")
    print(f"{'='*70}")