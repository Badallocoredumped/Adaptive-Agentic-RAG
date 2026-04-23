"""TableRAG-infused SQL generation + execution module."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from .. import config
from .database import execute_query, get_db_connection, get_live_schema
from .table_rag import (
    build_schema_index,
    get_schema_texts,
    retrieve_relevant_schema,
)

if TYPE_CHECKING:
    from .sql_cache import SQLCache

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


def _debug(message: str) -> None:
    if getattr(config, "DEBUG_LOGGING", False):
        print(message)


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
    names = re.findall(r"Table\s+([A-Za-z_][A-Za-z0-9_]*)\s+with\s+columns", schema_context)
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped







def _ensure_schema_index_exists() -> None:
    """Build TableRAG schema index if it does not already exist or is stale."""
    index_path = config.INDEX_DIR / "schema.faiss"
    texts_path = config.INDEX_DIR / "schema_texts.json"
    if index_path.exists() and texts_path.exists():
        import json
        with open(texts_path, "r", encoding="utf-8") as f:
            stored = json.load(f)
        live_schema = get_live_schema()
        if set(stored) == set(live_schema.to_embedding_texts()):
            return  # schema unchanged (tables and columns match)

    schema_info = get_live_schema()
    if schema_info.tables:
        build_schema_index(schema_info)


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

def _build_refine_few_shot_block(similar_queries: list[dict[str, Any]] | None) -> str:
    """Format dynamic few-shot examples into a prompt block."""
    if not similar_queries:
        return "No similar queries available."
        
    lines = []
    for ex in similar_queries:
        q = ex.get("question", "")
        s = ex.get("sql", "")
        if q and s:
            lines.append(f"Question: {q}")
            lines.append(f"SQL:      {s}")
            lines.append("")
    return "\n".join(lines).strip()


_openai_client_lock = threading.Lock()
_openai_client: Any = None


def _get_openai_client() -> Any:
    global _openai_client
    with _openai_client_lock:
        if _openai_client is None:
            import os
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY", config.OPENAI_API_KEY)
            _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _refine_sql_from_cache(
    new_question: str,
    original_question: str,
    cached_sql: str,
    schema_context: str,
    similar_queries: list[dict[str, Any]] | None = None,
) -> str | None:
    """Use OpenAI with few-shot examples to make small adjustments to a cached SQL.

    Called on a cache HIT when the new question is semantically similar but not
    identical to the cached question.  The model only adjusts what is necessary
    (e.g., a filter value, a LIMIT, or an aggregate function); it should NOT
    rewrite the query from scratch.
    """
    client = _get_openai_client()

    few_shot_block = _build_refine_few_shot_block(similar_queries)

    dialect = "SQLite" if config.SQLITE_PATH else "PostgreSQL"
    system_prompt = (
        f"You are a {dialect} SQL refinement assistant.\n"
        "You are given a cached SQL query that was written for a similar (but not identical) question.\n"
        "Your job is to make the SMALLEST possible edits to the cached SQL so that it correctly answers "
        "the new question.\n"
        "Rules:\n"
        "  - Only change what is necessary (filter values, column names, LIMIT, aggregate function, etc.).\n"
        "  - Do NOT rewrite the query from scratch unless the structure must change.\n"
        "  - Return only the final SQL — no explanation, no markdown fences.\n"
        "  - Must remain a SELECT/WITH read-only query.\n"
        "  - Use only tables/columns available in the schema context provided.\n"
        f"  - Write standard {dialect} syntax.\n\n"
        "Schema context (available tables and columns):\n"
        f"{schema_context}"
    )

    user_prompt = (
        f"Here are some other valid queries from our system to help you understand the database patterns:\n\n"
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
        _debug(f"[SQL AGENT] Cache-hit refiner failed: {exc}. Falling back to cached SQL.")
        return None


def _generate_sql(query: str, schema_context: str) -> str | None:
    """Generate SQL in a single pass using OpenAI and pruned schema context."""
    client = _get_openai_client()

    dialect = "SQLite" if config.SQLITE_PATH else "PostgreSQL"
    syntax_note = (
        "Use SQLite syntax (e.g. LIKE, strftime, CAST, COALESCE). "
        "Do NOT use ILIKE or PostgreSQL-specific functions."
        if config.SQLITE_PATH
        else "Write standard PostgreSQL syntax (e.g. ILIKE, EXTRACT, COALESCE)."
    )
    system_prompt = (
        f"You are a {dialect} expert.\n"
        "You ONLY have access to the tables listed in the schema context.\n"
        "Rules:\n"
        "  - Use only the tables/columns shown in the schema.\n"
        "  - Return SQL only, no explanation, no markdown fences.\n"
        "  - Must be a SELECT/WITH read query.\n"
        f"  - {syntax_note}\n"
    )

    user_prompt = (
        f"Schema context (available tables and columns):\n{schema_context}\n\n"
        f"Task: Write one correct {dialect} SELECT query for the following request.\n"
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
        _debug(f"[SQL AGENT] OpenAI SQL generation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# SQL execution
# ---------------------------------------------------------------------------

def _execute_sql(sql: str) -> list[dict[str, Any]]:
    """Run *sql* against the configured DB and return rows as dicts."""
    return execute_query(sql)


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
    tables = _extract_table_names(resolved_schema_context)
    _debug(f"[SQL AGENT] schema tables passed: {tables if tables else 'none parsed'}")

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
# ReAct agent import (lazy alias to keep module load fast and avoid issues
# when SQL_REACT_ENABLED=False)
# ---------------------------------------------------------------------------

def _get_react_sql_agent():
    """Return run_react_sql_agent; imported lazily so the module loads cleanly."""
    from .react_agent import run_react_sql_agent  # noqa: PLC0415
    return run_react_sql_agent


# ---------------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------------

_sql_cache_lock = threading.Lock()


def _get_sql_cache() -> "SQLCache":
    """Lazy initialization of the FAISS SQL cache (thread-safe)."""
    with _sql_cache_lock:
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
    _debug(f"\n[TableRAG Pipeline] Analyzing query: {query!r}")
    
    _t0 = time.time()
    cache_result = cache.check_cache_hit(query, threshold=0.85)
    _t_cache = time.time() - _t0
    _debug(f"[Timer] Cache Check took {_t_cache:.3f}s")

    if cache_result["hit"]:
        cached_sql = cache_result["sql"]
        original_question = cache_result.get("question", query)
        cached_schema = cache_result.get("schema") or ""
        _debug(f"[TableRAG Pipeline] ⚡ CACHE HIT: Cached SQL -> {cached_sql}")

        # Ensure we always have schema context for the refiner.
        # If the cache entry has no schema, retrieve it fresh now.
        if not cached_schema.strip():
            _debug("[TableRAG Pipeline] No schema in cache entry - retrieving fresh schema for refiner...")
            _ensure_schema_index_exists()
            _t0 = time.time()
            cached_schema = "\n".join(retrieve_relevant_schema(query, top_k=top_k))
            _debug(f"[Timer] TableRAG Schema Retrieval took {(time.time() - _t0):.3f}s")

        # --- LLM refinement on cache hit ---
        # Always send to the refiner so the LLM can make minimal adjustments
        # even when the cached question is nearly identical to the new one.
        _debug("[TableRAG Pipeline] Running LLM SQL refiner for cache hit...")
        _t0 = time.time()
        refined_sql = _refine_sql_from_cache(
            new_question=query,
            original_question=original_question,
            cached_sql=cached_sql,
            schema_context=cached_schema,
            similar_queries=cache_result.get("similar_queries", []),
        )
        _debug(f"[Timer] LLM Refiner took {(time.time() - _t0):.3f}s")

        if refined_sql and refined_sql.strip().upper() != cached_sql.strip().upper():
            _debug(f"[TableRAG Pipeline] Refiner adjusted SQL -> {refined_sql}")
            sql_to_execute = refined_sql
        elif refined_sql:
            _debug("[TableRAG Pipeline] Refiner kept cached SQL unchanged.")
            sql_to_execute = cached_sql
        else:
            # Refiner failed — generate fresh SQL instead of blindly using
            # the cached SQL which may be wrong for this new query.
            _debug("[TableRAG Pipeline] Refiner failed - generating fresh SQL from schema...")
            fresh_sql = _generate_sql(query, cached_schema)
            if fresh_sql:
                _debug(f"[TableRAG Pipeline] Fresh SQL generated -> {fresh_sql}")
                sql_to_execute = fresh_sql
            else:
                _debug("[TableRAG Pipeline] Fresh generation also failed - using cached SQL as-is.")
                sql_to_execute = cached_sql

        try:
            rows = _execute_sql(sql_to_execute)
            error = None
        except Exception as e:
            rows = []
            error = str(e)
            _debug(f"[TableRAG Pipeline] SQL execution failed: {error}")

            # ── ReAct fallback: only reached when execution fails ──────────
            # LLM refinement already ran above; this is a true last resort.
            if getattr(config, "SQL_REACT_ENABLED", True):
                _debug("[TableRAG Pipeline] Execution failed - escalating to ReAct agent as last resort...")
                react_fn = _get_react_sql_agent()
                react_result = react_fn(query, cached_schema)
                if react_result["sql"] or react_result["result"]:
                    sql_to_execute = react_result["sql"] or sql_to_execute
                    rows = react_result["result"]
                    error = react_result["error"]
                    _debug(f"[TableRAG Pipeline] ReAct agent recovered: sql={sql_to_execute!r}")

        latency = time.time() - start_time
        _debug(f"[TableRAG Pipeline] Latency: {latency:.2f}s")
        return {
            "schema_used": ["<from semantic cache>"],
            "sql": sql_to_execute,
            "result": rows,
            "error": error,
            "path": "fast",
            "latency": latency
        }

    # 2. RUN FULL PIPELINE (If Cache MISS)
    _debug("[TableRAG Pipeline] AGENT PATH: Routing to TableRAG + LLM")
    
    _ensure_schema_index_exists()
    _t0 = time.time()
    schema_rows: list[str] = retrieve_relevant_schema(query, top_k=top_k)
    _debug(f"[Timer] TableRAG Schema Retrieval took {(time.time() - _t0):.3f}s")

    # Logging: which tables were selected
    table_names = _extract_table_names("\n".join(schema_rows))
    _debug(f"[TableRAG Pipeline] Retrieved schema rows ({len(schema_rows)}):")
    for row in schema_rows:
        _debug(f"  -> {row}")
    _debug(f"[TableRAG Pipeline] Selected tables: {table_names}")

    # Format into a single context string
    schema_context = "\n".join(schema_rows)

    # ── Step 3: Run ReAct agent (primary) ─────────────────────────────────
    # The ReAct agent receives the original question + the TableRAG schema
    # context and reasons iteratively (Thought → Action → Observation) to
    # generate and verify its SQL.  The single-pass run_sql_agent() is kept
    # as a fallback in case the ReAct layer produces nothing at all.
    used_react = False
    if getattr(config, "SQL_REACT_ENABLED", True):
        _debug("[TableRAG Pipeline] Running ReAct agent with schema context...")
        react_fn = _get_react_sql_agent()
        _t0 = time.time()
        agent_result = react_fn(query, schema_context)
        _debug(f"[Timer] ReAct Agent took {(time.time() - _t0):.3f}s")

        # Fallback: ReAct returned neither SQL nor rows → use single-pass agent
        if not agent_result["sql"] and not agent_result["result"]:
            _debug(
                "[TableRAG Pipeline] ⚠️  ReAct agent produced no output — "
                "falling back to single-pass agent..."
            )
            agent_result = run_sql_agent(query, schema_context=schema_context, top_k=top_k)
        else:
            used_react = True
    else:
        # ReAct disabled — use the original single-pass SQL agent
        _t0 = time.time()
        agent_result = run_sql_agent(query, schema_context=schema_context, top_k=top_k)
        _debug(f"[Timer] Single-pass SQL Agent took {(time.time() - _t0):.3f}s")

    # Logging: generated SQL
    _debug(f"[TableRAG Pipeline] Generated SQL: {agent_result['sql']}")
    if agent_result["error"]:
        _debug(f"[TableRAG Pipeline] Error: {agent_result['error']}")
    else:
        _debug(f"[TableRAG Pipeline] Rows returned: {len(agent_result['result'])}")

        # 3. Add successful run to Cache
        if agent_result["sql"] and not agent_result["error"]:
            _debug("[TableRAG Pipeline] Saving successful query to cache...")
            cache.add_to_cache(query, agent_result["sql"], schema_context)
            cache.save_cache()

    latency = time.time() - start_time
    _debug(f"[TableRAG Pipeline] Latency: {latency:.2f}s")

    # path label: "react" only when the ReAct agent itself produced the result
    path_label = "react" if used_react else "agent"

    # 4. Return unified result dict
    return {
        "schema_used": schema_rows,
        "sql": agent_result["sql"],
        "result": agent_result["result"],
        "error": agent_result["error"],
        "path": path_label,
        "latency": latency
    }

if __name__ == "__main__":
    # Clear the FAISS schema index so it gets rebuilt from the live PG schema.
    for f in ["schema.faiss", "schema_texts.json"]:
        p = config.INDEX_DIR / f
        if p.exists():
            p.unlink()

    print(f"[SETUP] Starting testing queries on {config.PG_HOST}/{config.PG_DB}\n")

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