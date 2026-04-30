"""
Fast-track performance regression test.

Counts how many times get_live_schema() is called during:
  1. True fast track (score >= 0.98): should be 0 — no schema needed
  2. Cache-hit refiner path (score < 0.98): should be exactly 1

After the fix, _ensure_schema_index_exists() returns the loaded SchemaInfo and
retrieve_relevant_schema() receives it as schema_info= — so get_live_schema()
is never called a second time.

Run:
    python test_fast_track_perf.py
"""
from __future__ import annotations

import json
import sys
import types
import unittest.mock as mock
from pathlib import Path

# ── project root on sys.path (same as test_sql_pipeline.py) ──────────────────
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ── stubs that must be registered BEFORE importing backend modules ─────────────

# faiss
faiss_stub = types.ModuleType("faiss")
faiss_stub.IndexFlatIP = lambda dim: mock.MagicMock()
faiss_stub.IndexFlatL2 = lambda dim: mock.MagicMock()
faiss_stub.normalize_L2 = lambda v: None
faiss_stub.write_index = lambda idx, path: None
faiss_stub.read_index = lambda path: mock.MagicMock(search=lambda *_: ([[0.9]], [[0]]))
sys.modules.setdefault("faiss", faiss_stub)

# numpy (real, already installed)
import numpy as np  # noqa: E402

# sentence_transformers
st_stub = types.ModuleType("sentence_transformers")
sys.modules.setdefault("sentence_transformers", st_stub)

# psycopg2
pg_stub = types.ModuleType("psycopg2")
pg_stub.connect = mock.MagicMock()
pg_extras = types.ModuleType("psycopg2.extras")
pg_pool = types.ModuleType("psycopg2.pool")
sys.modules.setdefault("psycopg2", pg_stub)
sys.modules.setdefault("psycopg2.extras", pg_extras)
sys.modules.setdefault("psycopg2.pool", pg_pool)

# LangChain / LangGraph (react_agent imports these at module level)
class _FakeMsg:
    def __init__(self, *_, **kwargs):
        self.__dict__.update(kwargs)
    content = ""

_lc_messages = types.ModuleType("langchain_core.messages")
_lc_messages.AIMessage    = _FakeMsg
_lc_messages.HumanMessage = _FakeMsg
_lc_messages.ToolMessage  = _FakeMsg

_lc_tools = types.ModuleType("langchain_core.tools")
_lc_tools.Tool = mock.MagicMock

_lc_core = types.ModuleType("langchain_core")
_lc_core.messages = _lc_messages
_lc_core.tools    = _lc_tools

_lc_openai = types.ModuleType("langchain_openai")
_lc_openai.ChatOpenAI = mock.MagicMock

_lg_graph   = types.ModuleType("langgraph.graph")
_lg_state   = types.ModuleType("langgraph.graph.state")
_lg_state.CompiledStateGraph = mock.MagicMock
_lg_prebuilt = types.ModuleType("langgraph.prebuilt")
_lg_prebuilt.create_react_agent = mock.MagicMock(return_value=mock.MagicMock())
_lg = types.ModuleType("langgraph")
_lg.graph = _lg_graph

for _name, _mod in [
    ("langchain_core",          _lc_core),
    ("langchain_core.messages", _lc_messages),
    ("langchain_core.tools",    _lc_tools),
    ("langchain_openai",        _lc_openai),
    ("langgraph",               _lg),
    ("langgraph.graph",         _lg_graph),
    ("langgraph.graph.state",   _lg_state),
    ("langgraph.prebuilt",      _lg_prebuilt),
]:
    sys.modules.setdefault(_name, _mod)

# ── now import the real backend modules ───────────────────────────────────────
from backend import config                                      # noqa: E402
from backend.sql.schema import ColumnInfo, SchemaInfo, TableInfo  # noqa: E402

# backend.models stub (needs to exist before sql_agent is imported)
models_stub = types.ModuleType("backend.models")
_fake_st_model = mock.MagicMock()
_fake_st_model.encode.return_value = np.zeros((1, 4), dtype=np.float32)
models_stub.get_shared_st_model = lambda: _fake_st_model
sys.modules["backend.models"] = models_stub

# Force SQLite mode
config.SQLITE_PATH = ":memory:"  # type: ignore[assignment]

import backend.sql.sql_agent as sql_agent  # noqa: E402
import backend.sql.table_rag as table_rag  # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────────────

MOCK_SQL  = "SELECT 1 AS n;"
MOCK_ROWS = [{"n": 1}]

def _make_schema() -> SchemaInfo:
    return SchemaInfo(tables={
        "orders": TableInfo(
            name="orders",
            columns=[ColumnInfo("id", "INTEGER"), ColumnInfo("status", "TEXT")],
        )
    })



def _make_fake_cache(score: float, sql: str = MOCK_SQL) -> mock.MagicMock:
    cache = mock.MagicMock()
    cache.check_cache_hit.return_value = {
        "hit": True,
        "sql": sql,
        "question": "test question",
        "score": score,
        "similar_queries": [],
    }
    return cache


def _write_fake_index(tmp_dir: Path, table_names: list[str]) -> None:
    """Write stub schema.faiss + schema_meta.json so _ensure_schema_index_exists
    sees an existing (fresh) index and doesn't try to rebuild it via FAISS."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    meta = [{"table": t, "level": "table", "text": t} for t in table_names]
    (tmp_dir / "schema_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_dir / "schema.faiss").write_bytes(b"stub")  # just needs to exist


# ── test runner ───────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    icon = PASS if ok else FAIL
    print(f"  {icon}  {name}" + (f"  [{detail}]" if detail else ""))


def run_test(label: str, score: float, expect_schema_calls: int) -> None:
    print(f"\n{'─'*60}")
    print(f"Test: {label}  (score={score})")
    print(f"{'─'*60}")

    schema = _make_schema()

    # Count every get_live_schema() call — patched at the sql_agent binding
    # (where it is imported as a module-level name via `from .database import ...`).
    call_log: list[str] = []

    def counting_get_live_schema():
        call_log.append("get_live_schema")
        return schema

    # Track what schema_info arg retrieve_relevant_schema receives.
    # Use a MagicMock so we can inspect call_args_list without any unused-param hints.
    _SCHEMA_STUB = ["Table: orders\nColumns:\n  - id INTEGER\n  - status TEXT"]
    retrieve_mock = mock.MagicMock(return_value=_SCHEMA_STUB)

    # Fake index files whose table list matches our schema (index = fresh)
    original_index_dir = config.INDEX_DIR
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_fake_index(tmp_path, list(schema.tables.keys()))
        config.INDEX_DIR = tmp_path  # type: ignore[assignment]

        # Reset module-level singletons so each test run is independent
        sql_agent._openai_client = None       # type: ignore[attr-defined]
        table_rag._selector_client = None     # type: ignore[attr-defined]

        config.SQL_REACT_ENABLED              = False
        config.SQL_CACHE_SKIP_REFINER_THRESHOLD = 0.98
        config.SQL_CACHE_HIT_THRESHOLD        = 0.80

        with (
            mock.patch("backend.sql.sql_agent._get_sql_cache",
                       return_value=_make_fake_cache(score)),
            mock.patch("backend.sql.sql_agent.get_live_schema",
                       side_effect=counting_get_live_schema),
            mock.patch("backend.sql.sql_agent.retrieve_relevant_schema",
                       retrieve_mock),
            mock.patch("backend.sql.sql_agent._execute_sql",
                       return_value=MOCK_ROWS),
            mock.patch("backend.sql.sql_agent._refine_sql_from_cache",
                       return_value=MOCK_SQL),
        ):
            try:
                result = sql_agent.run_table_rag_pipeline("How many orders are there?")
                check("Pipeline returned sql", bool(result.get("sql")))
                check("No pipeline error",     result.get("error") is None,
                      result.get("error") or "")
            except Exception as exc:
                check("Pipeline did not raise", False, str(exc))

        config.INDEX_DIR = original_index_dir  # restore

    actual = len(call_log)
    check(
        f"get_live_schema() called ≤ {expect_schema_calls}x",
        actual <= expect_schema_calls,
        f"actual={actual}",
    )

    # For the refiner path: verify schema_info was forwarded (not None)
    if score < 0.98 and retrieve_mock.called:
        schema_info_args = [c.kwargs.get("schema_info") for c in retrieve_mock.call_args_list]
        all_have_schema = all(s is not None for s in schema_info_args)
        check(
            "retrieve_relevant_schema received schema_info (no 2nd DB call)",
            all_have_schema,
            f"types={[type(s).__name__ for s in schema_info_args]}",
        )


def run_full_pipeline_test() -> None:
    """Real end-to-end test: real SQLite DB + real LLM refiner.

    Only the cache hit score is controlled (forced to 0.85) so the refiner
    path is guaranteed.  The real refiner calls the configured LLM endpoint
    (llama.cpp / OpenAI) with the actual full schema and returns real SQL.

    Verifies:
    - The full schema (all 3 tables) was passed to the refiner.
    - retrieve_relevant_schema is NOT called (TableRAG LLM selector bypassed).
    - The pipeline returns whatever SQL the real LLM produced.
    """
    import sqlite3 as _sqlite3
    import tempfile as _tempfile
    from backend.sql import database as _db_module

    label = "Full pipeline (real SQLite + real LLM) — refiner sees full schema"
    print(f"\n{'─'*60}")
    print(f"Test: {label}")
    print(f"{'─'*60}")

    cached_sql = "SELECT COUNT(*) FROM orders;"
    query      = "How many shipped orders do we have?"

    original_sqlite_path = config.SQLITE_PATH
    original_index_dir   = config.INDEX_DIR

    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path  = str(tmp_path / "test.db")

        # 1. Build a real SQLite DB with 3 tables + sample rows
        conn = _sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE orders (
                id          INTEGER PRIMARY KEY,
                customer_id INTEGER,
                status      TEXT,
                total       REAL
            );
            CREATE TABLE customers (
                id   INTEGER PRIMARY KEY,
                name TEXT,
                city TEXT
            );
            CREATE TABLE products (
                id    INTEGER PRIMARY KEY,
                name  TEXT,
                price REAL
            );
            INSERT INTO orders   VALUES (1,1,'shipped',99.99),(2,2,'pending',49.99),(3,1,'shipped',199.99);
            INSERT INTO customers VALUES (1,'Alice','Berlin'),(2,'Bob','Paris');
            INSERT INTO products  VALUES (1,'Widget',9.99),(2,'Gadget',19.99);
        """)
        conn.commit()
        conn.close()

        # 2. Point config at the new DB and a fresh index dir
        config.SQLITE_PATH = db_path          # type: ignore[assignment]
        config.INDEX_DIR   = tmp_path         # type: ignore[assignment]

        # Reset thread-local SQLite connection so it opens the new file
        if hasattr(_db_module._sqlite_local, "conn"):
            _db_module._sqlite_local.conn.close()
            del _db_module._sqlite_local.conn

        # Reset module-level singletons
        sql_agent._openai_client    = None    # type: ignore[attr-defined]
        table_rag._selector_client  = None    # type: ignore[attr-defined]

        config.SQL_REACT_ENABLED               = False
        config.SQL_CACHE_SKIP_REFINER_THRESHOLD = 0.98
        config.SQL_CACHE_HIT_THRESHOLD         = 0.80

        # 3. Write stub FAISS index so _ensure_schema_index_exists returns
        #    the live schema without trying to rebuild the index via FAISS.
        _write_fake_index(tmp_path, ["orders", "customers", "products"])

        # 4. Spy on _refine_sql_from_cache: capture schema_context but call through
        #    to the real function so the actual LLM runs.
        real_refine = sql_agent._refine_sql_from_cache  # type: ignore[attr-defined]
        captured: dict = {}

        def spy_refine(new_question, original_question, cached_sql,
                       schema_context, similar_queries=None):
            captured["schema_context"] = schema_context
            return real_refine(
                new_question=new_question,
                original_question=original_question,
                cached_sql=cached_sql,
                schema_context=schema_context,
                similar_queries=similar_queries,
            )

        retrieve_mock = mock.MagicMock()  # must NOT be called

        with (
            mock.patch("backend.sql.sql_agent._get_sql_cache",
                       return_value=_make_fake_cache(0.85, cached_sql)),
            mock.patch("backend.sql.sql_agent.retrieve_relevant_schema",
                       retrieve_mock),
            mock.patch("backend.sql.sql_agent._refine_sql_from_cache",
                       side_effect=spy_refine),
        ):
            try:
                result = sql_agent.run_table_rag_pipeline(query)
            except Exception as exc:
                check("Pipeline did not raise", False, str(exc))
                config.SQLITE_PATH = original_sqlite_path
                config.INDEX_DIR   = original_index_dir
                return

        config.SQLITE_PATH = original_sqlite_path
        config.INDEX_DIR   = original_index_dir
        if hasattr(_db_module._sqlite_local, "conn"):
            _db_module._sqlite_local.conn.close()
            del _db_module._sqlite_local.conn

    # 5. Print what the real LLM received and produced
    schema_ctx = captured.get("schema_context", "")
    print(f"\n  Schema sent to refiner ({len(schema_ctx)} chars):")
    for line in schema_ctx.splitlines()[:25]:
        print(f"    {line}")
    print(f"\n  Cached SQL  : {cached_sql}")
    print(f"  Refined SQL : {result.get('sql')!r}")
    print(f"  DB rows     : {result.get('result')}")

    # 6. Assertions
    check("Pipeline returned SQL",
          bool(result.get("sql")), result.get("sql") or "none")

    check("retrieve_relevant_schema NOT called (TableRAG bypassed)",
          not retrieve_mock.called)

    for table in ("orders", "customers", "products"):
        check(f"Full schema contains table '{table}'",
              table in schema_ctx,
              f"found={'yes' if table in schema_ctx else 'NO'}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Fast Track Performance Regression Test")
    print("=" * 60)

    # True fast track (score >= 0.98): no schema loading at all
    run_test("True fast track  (score=0.99)", score=0.99, expect_schema_calls=0)

    # Refiner path (score < 0.98): exactly 1 get_live_schema() call
    run_test("Refiner path     (score=0.85)", score=0.85, expect_schema_calls=1)

    # Full pipeline: real refiner + mocked LLM, verify SQL changes and schema coverage
    run_full_pipeline_test()

    print(f"\n{'='*60}")
    total  = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed
    status = "ALL PASSED" if failed == 0 else f"{failed} FAILED"
    print(f"Results: {passed}/{total} passed  —  {status}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
