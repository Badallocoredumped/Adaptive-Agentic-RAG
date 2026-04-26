"""
rag_system.py
─────────────────────────────────────────────────────────────────────────────
Benchmark adapter for the Adaptive Agentic RAG system.

Placed in tests/EndToEnd/ so importlib.import_module("rag_system") finds it
when the runner scripts add this directory to sys.path.

Exposes:
    run_query(query: str, **kwargs) -> dict

    kwargs:
        tablerag_pruning (bool, default True)
            Pass False to disable TableRAG schema pruning (RQ3 baseline).

Returns a dict with the canonical benchmark schema:
    router_decision    "sql" | "text" | "hybrid" | None
    answer             str | None
    sql_executed       str | None
    sql_result         list | None
    retrieved_chunks   list | None
    retrieved_sources  list | None
    cache_hit          bool
    latency_ms         float
    error              str | None
    tables_used        list[str]
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

# ── Project root on sys.path ──────────────────────────────────────────────────
# tests/EndToEnd/ → tests/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend import config as _cfg
from backend.main import AdaptiveAgenticRAGSystem
from backend.sql import run_table_rag_pipeline

# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_table_names(schema_used: list) -> list[str]:
    """Pull table names out of 'Table: X | Columns: ...' schema lines."""
    names: list[str] = []
    for entry in schema_used:
        m = re.match(r"Table:\s+([A-Za-z_][A-Za-z0-9_]*)", str(entry))
        if m:
            names.append(m.group(1))
    return list(dict.fromkeys(names))  # deduplicate, preserve order


# ── Instrumented system ───────────────────────────────────────────────────────

class _InstrumentedSystem(AdaptiveAgenticRAGSystem):
    """
    Subclass that captures per-query structured state while delegating all
    real work to the parent.  Three interception points:

    1. _run_sql_pipeline   (instance override of parent's @staticmethod)
       → captures sql_executed, sql_result, cache_hit, tables_used

    2. self.retriever.retrieve   (wrapped in __init__)
       → captures retrieved_chunks, retrieved_sources

    3. router route methods   (wrapped in __init__)
       → captures router_decision (route)
    """

    def __init__(self) -> None:
        super().__init__()
        self._call_state: dict[str, Any] = {}

        # ── Wrap retriever.retrieve ───────────────────────────────────────
        _orig_retrieve = self.retriever.retrieve

        def _capturing_retrieve(*args: Any, **kwargs: Any):
            chunks = _orig_retrieve(*args, **kwargs)
            if isinstance(chunks, list):
                self._call_state.setdefault("retrieved_chunks", []).extend(chunks)
                sources: set[str] = self._call_state.setdefault("_sources_set", set())
                for c in chunks:
                    name = Path(c.get("source", "")).name
                    if name:
                        sources.add(name)
            return chunks

        self.retriever.retrieve = _capturing_retrieve

        # ── Wrap router methods to capture the decided route ──────────────
        def _make_route_wrapper(fn: Any) -> Any:
            def _wrapper(*args: Any, **kwargs: Any):
                result = fn(*args, **kwargs)
                self._call_state["route"] = result
                return result
            return _wrapper

        for _mname in (
            "route_from_subtasks",
            "route_with_llm",
            "route",
            "route_with_semantic",
        ):
            _orig = getattr(self.router, _mname, None)
            if _orig is not None:
                setattr(self.router, _mname, _make_route_wrapper(_orig))

    # Override the parent's @staticmethod as an instance method ───────────────
    def _run_sql_pipeline(self, query: str) -> dict:  # type: ignore[override]
        pipeline_result = run_table_rag_pipeline(query, top_k=_cfg.SQL_TOP_K)

        st = self._call_state
        if pipeline_result.get("sql"):
            st["sql_executed"] = pipeline_result["sql"]
        rows = pipeline_result.get("result") or []
        if rows:
            st["sql_result"] = rows
        if pipeline_result.get("path") == "fast":
            st["cache_hit"] = True
        st.setdefault("tables_used", set()).update(
            _extract_table_names(pipeline_result.get("schema_used", []))
        )

        return {
            "ok":         not bool(pipeline_result.get("error")),
            "query":      pipeline_result.get("sql") or "n/a",
            "error":      pipeline_result.get("error"),
            "rows":       rows,
            "row_count":  len(rows),
            "schema_used": pipeline_result.get("schema_used", []),
            "path":       pipeline_result.get("path", "unknown"),
            "latency":    pipeline_result.get("latency", 0.0),
        }

    def run_query_detailed(
        self,
        user_query: str,
        tablerag_pruning: bool = True,
    ) -> dict:
        """Run the full pipeline and return the canonical benchmark dict."""
        self._call_state = {
            "cache_hit":       False,
            "sql_executed":    None,
            "sql_result":      None,
            "tables_used":     set(),
            "retrieved_chunks": [],
            "_sources_set":    set(),
            "route":           None,
        }

        # Temporarily widen schema retrieval to simulate full-schema baseline
        orig_top_k = _cfg.SQL_TOP_K
        if not tablerag_pruning:
            _cfg.SQL_TOP_K = 200  # returns all tables when DB has fewer than 200

        t0 = time.perf_counter()
        error: str | None = None
        answer: str | None = None

        try:
            answer = super().run_query(user_query)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        finally:
            _cfg.SQL_TOP_K = orig_top_k

        latency_ms = (time.perf_counter() - t0) * 1000
        st = self._call_state

        chunks = st["retrieved_chunks"] or None
        sources = list(st["_sources_set"]) or None

        return {
            "router_decision":   st.get("route"),
            "answer":            answer,
            "sql_executed":      st.get("sql_executed"),
            "sql_result":        st.get("sql_result"),
            "retrieved_chunks":  chunks,
            "retrieved_sources": sources,
            "cache_hit":         st.get("cache_hit", False),
            "latency_ms":        latency_ms,
            "error":             error,
            "tables_used":       list(st.get("tables_used", set())),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_system: _InstrumentedSystem | None = None


def _get_system() -> _InstrumentedSystem:
    global _system
    if _system is None:
        print("[rag_system] Initialising AdaptiveAgenticRAGSystem (first call only)…")
        _system = _InstrumentedSystem()
        print("[rag_system] System ready.")
    return _system


# ── Public entry point ────────────────────────────────────────────────────────

def run_query(query: str, **kwargs: Any) -> dict:
    """
    Canonical benchmark entry point.

    kwargs:
        tablerag_pruning (bool): True (default) = TableRAG schema pruning ON.
                                 False          = full schema exposed (RQ3 baseline).
    """
    tablerag_pruning: bool = bool(kwargs.get("tablerag_pruning", True))
    return _get_system().run_query_detailed(query, tablerag_pruning=tablerag_pruning)
