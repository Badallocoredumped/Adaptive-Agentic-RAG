"""
Tests for the TableRAG + ReAct pipeline.

Three layers:
  Unit       – no PostgreSQL, no OpenAI required (everything mocked)
  Integration – PostgreSQL required (set PG_* vars in .env)
  E2E         – PostgreSQL + OpenAI required

Run only unit tests (default / CI):
    pytest tests/test_sql_pipeline.py -m "not integration and not e2e"

Run with a real database:
    pytest tests/test_sql_pipeline.py -m "unit or integration"

Run everything:
    pytest tests/test_sql_pipeline.py
"""

import sys
from pathlib import Path

import pytest

# Make sure 'backend' is importable regardless of where pytest is invoked from
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.sql.sql_agent import run_table_rag_pipeline
from backend.sql.sql_cache import SQLCache

# ── Constants ────────────────────────────────────────────────────────────────

# A query that will never collide with production cache entries
TEST_QUERY = "pytest unique completely new query xyz"
MOCK_SQL   = "SELECT 1 AS test_col;"
MOCK_ROWS  = [{"test_col": 1}]


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_sql_cache(tmp_path, monkeypatch):
    """Replace the production FAISS cache with an empty throwaway instance."""
    test_cache = SQLCache(
        index_path=tmp_path / "test_cache.faiss",
        metadata_path=tmp_path / "test_cache_texts.json",
    )
    test_cache.initialize_cache()
    monkeypatch.setattr("backend.sql.sql_agent._get_sql_cache", lambda: test_cache)
    return test_cache


@pytest.fixture
def mock_react_agent(monkeypatch):
    """Stub run_react_sql_agent so no OpenAI call is made on the agent path.

    Returns a successful result whose SQL can be executed by _execute_sql.
    We also stub _execute_sql so no PostgreSQL connection is needed.
    """
    def fake_react(query, schema_context):
        return {
            "sql": MOCK_SQL,
            "result": MOCK_ROWS,
            "error": None,
            "schema_context": schema_context,
        }

    # The pipeline calls run_react_sql_agent via the _get_react_sql_agent() factory.
    # Patch the function in the react_agent module so the factory picks it up.
    monkeypatch.setattr("backend.sql.react_agent.run_react_sql_agent", fake_react)
    # Also patch the import target inside sql_agent so the lazy import resolves to the stub.
    import backend.sql.react_agent as _ra
    monkeypatch.setattr(_ra, "run_react_sql_agent", fake_react)

    # Stub _execute_sql (used on the cache-HIT path) so tests don't need a live DB.
    monkeypatch.setattr(
        "backend.sql.sql_agent._execute_sql",
        lambda sql: MOCK_ROWS,
    )

    # Stub few-shot refiner (uses OpenAI) so cache-HIT path stays offline.
    monkeypatch.setattr(
        "backend.sql.sql_agent._refine_sql_from_cache",
        lambda **kw: None,   # "no change" → pipeline uses cached SQL as-is
    )

    # Stub schema retrieval so there's no FAISS schema index dependency.
    monkeypatch.setattr(
        "backend.sql.sql_agent.retrieve_relevant_schema",
        lambda query, top_k=4: ["Table: orders | Columns: id, amount, status"],
    )
    monkeypatch.setattr(
        "backend.sql.sql_agent._ensure_schema_index_exists",
        lambda: None,
    )


# ── Unit tests (no DB, no API) ────────────────────────────────────────────────

@pytest.mark.unit
def test_cache_miss_triggers_react_agent_and_saves(mock_react_agent, isolated_sql_cache):
    """Cache MISS → ReAct agent runs → result cached under 'react' path."""
    res = run_table_rag_pipeline(TEST_QUERY)

    assert res["path"] == "react", f"Expected 'react' path, got {res['path']!r}"
    assert res["sql"] == MOCK_SQL
    assert res["error"] is None
    assert res["result"] == MOCK_ROWS

    # Confirm the result was saved to the FAISS cache
    assert isolated_sql_cache.index.ntotal == 1, "One query should be cached."
    assert isolated_sql_cache.metadata[0]["sql"] == MOCK_SQL
    assert isolated_sql_cache.metadata[0]["question"] == TEST_QUERY


@pytest.mark.unit
def test_cache_hit_bypasses_react_agent(mock_react_agent, isolated_sql_cache):
    """Cache HIT → fast path → ReAct agent is NOT called."""
    isolated_sql_cache.add_to_cache(TEST_QUERY, MOCK_SQL, "schema")
    isolated_sql_cache.save_cache()

    res = run_table_rag_pipeline(TEST_QUERY)

    assert res["path"] == "fast", f"Expected 'fast' path, got {res['path']!r}"
    assert res["sql"] == MOCK_SQL
    assert res["error"] is None
    assert res["result"] == MOCK_ROWS


@pytest.mark.unit
def test_different_query_misses_cache(mock_react_agent, isolated_sql_cache):
    """A semantically unrelated query must not get a false-positive cache hit."""
    isolated_sql_cache.add_to_cache(TEST_QUERY, MOCK_SQL, "schema")
    isolated_sql_cache.save_cache()

    unrelated = "What is the capital of France?"
    res = run_table_rag_pipeline(unrelated)

    assert res["path"] in {"react", "agent"}, "Unrelated query must take the agent path."
    assert isolated_sql_cache.index.ntotal == 2, "Miss should add a second cache entry."


@pytest.mark.unit
def test_react_agent_no_output_falls_back_to_single_pass(monkeypatch, isolated_sql_cache):
    """If ReAct returns nothing, the pipeline must fall back to run_sql_agent."""
    # ReAct returns empty
    import backend.sql.react_agent as _ra
    monkeypatch.setattr(_ra, "run_react_sql_agent", lambda q, s: {"sql": None, "result": [], "error": "fail", "schema_context": s})

    # Fallback single-pass agent returns something
    monkeypatch.setattr(
        "backend.sql.sql_agent.run_sql_agent",
        lambda q, schema_context, top_k: {"sql": MOCK_SQL, "result": MOCK_ROWS, "error": None, "schema_context": schema_context},
    )
    monkeypatch.setattr("backend.sql.sql_agent.retrieve_relevant_schema", lambda q, top_k=4: ["Table: orders | Columns: id"])
    monkeypatch.setattr("backend.sql.sql_agent._ensure_schema_index_exists", lambda: None)

    res = run_table_rag_pipeline(TEST_QUERY)
    assert res["sql"] == MOCK_SQL, "Fallback agent should have produced SQL."


# ── SQLCache unit tests ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestSQLCache:
    def test_initialize_creates_empty_index(self, isolated_sql_cache):
        assert isolated_sql_cache.index is not None
        assert isolated_sql_cache.index.ntotal == 0

    def test_add_and_hit(self, isolated_sql_cache):
        isolated_sql_cache.add_to_cache("total revenue", "SELECT SUM(amount) FROM orders;", "schema")
        result = isolated_sql_cache.check_cache_hit("total revenue", threshold=0.5)
        assert result["hit"] is True
        assert "SELECT" in result["sql"]

    def test_miss_on_unrelated_query(self, isolated_sql_cache):
        isolated_sql_cache.add_to_cache("total revenue", "SELECT SUM(amount) FROM orders;", "schema")
        result = isolated_sql_cache.check_cache_hit("blog post views", threshold=0.85)
        assert result["hit"] is False

    def test_save_and_reload(self, isolated_sql_cache, tmp_path):
        """Cache written to disk can be reloaded into a fresh instance."""
        isolated_sql_cache.add_to_cache("revenue query", MOCK_SQL, "schema")
        isolated_sql_cache.save_cache()

        fresh = SQLCache(
            index_path=isolated_sql_cache.index_path,
            metadata_path=isolated_sql_cache.metadata_path,
        )
        assert fresh.load_cache() is True
        assert fresh.index.ntotal == 1
        assert fresh.metadata[0]["sql"] == MOCK_SQL


# ── Integration tests (real PostgreSQL required) ──────────────────────────────

@pytest.mark.integration
def test_db_connection():
    """Smoke-test: can we connect to PostgreSQL?"""
    from backend.sql.database import get_db_connection

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 AS ok;")
        row = cur.fetchone()
        assert row is not None
    finally:
        conn.close()


@pytest.mark.integration
def test_execute_sql_returns_rows():
    """execute_query must execute a basic SELECT against PostgreSQL."""
    from backend.sql.database import execute_query

    rows = execute_query("SELECT id, name FROM customers ORDER BY id LIMIT 3;")
    assert isinstance(rows, list)
    assert len(rows) > 0
    assert "id" in rows[0]
    assert "name" in rows[0]


@pytest.mark.integration
def test_get_live_schema():
    """get_live_schema must return a SchemaInfo with fintech tables."""
    from backend.sql.database import get_live_schema

    schema_info = get_live_schema()
    assert "customers" in schema_info.tables
    assert "orders" in schema_info.tables
    assert any(col.name == "id" for col in schema_info.tables["customers"].columns)


# ── E2E tests (real PostgreSQL + real OpenAI API required) ────────────────────

@pytest.mark.e2e
def test_full_pipeline_simple_query():
    """End-to-end: run_table_rag_pipeline answers a basic aggregation question."""
    result = run_table_rag_pipeline("What is the total revenue from all orders?")

    assert result["sql"] is not None, "Pipeline must produce a SQL query."
    assert result["error"] is None,   f"Pipeline error: {result['error']}"
    assert isinstance(result["result"], list)
    assert len(result["result"]) > 0
    assert result["path"] in {"react", "fast"}


@pytest.mark.e2e
def test_full_pipeline_caches_result():
    """Running the same query twice should give a cache hit on the second run."""
    q = "How many customers are there in each city? e2e_unique_8472"
    r1 = run_table_rag_pipeline(q)
    r2 = run_table_rag_pipeline(q)

    assert r1["sql"] is not None
    assert r2["path"] == "fast", "Second run must hit the semantic cache."
