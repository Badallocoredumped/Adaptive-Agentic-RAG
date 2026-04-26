"""
system_adapter.py
─────────────────────────────────────────────────────────────────────────────
Thin adapter between the benchmark runners and your RAG system.

If your system's API changes (different function name, keyword args, return
format), you only need to edit THIS file — all five runner scripts stay the
same.
─────────────────────────────────────────────────────────────────────────────
"""

import importlib
import time
from typing import Any

from config import SYSTEM_MODULE, SYSTEM_FN, QUERY_TIMEOUT_SEC, MAX_RETRIES, RETRY_DELAY_SEC

# ── Lazy-load the RAG system once ────────────────────────────────────────────
_system_fn = None


def _load_system():
    global _system_fn
    if _system_fn is None:
        try:
            mod = importlib.import_module(SYSTEM_MODULE)
            _system_fn = getattr(mod, SYSTEM_FN)
            print(f"[adapter] Loaded {SYSTEM_MODULE}.{SYSTEM_FN} ✓")
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                f"Cannot import '{SYSTEM_FN}' from '{SYSTEM_MODULE}'.\n"
                f"  → Make sure your RAG system module is importable and "
                f"SYSTEM_MODULE/SYSTEM_FN in config.py are correct.\n"
                f"  Original error: {e}"
            )
    return _system_fn


# ── Canonical response schema ─────────────────────────────────────────────────
EMPTY_RESPONSE = {
    "router_decision":   None,
    "answer":            None,
    "sql_executed":      None,
    "sql_result":        None,
    "retrieved_chunks":  None,
    "retrieved_sources": None,
    "cache_hit":         False,
    "latency_ms":        None,
    "error":             None,
    "tables_used":       [],   # list[str] — tables kept by TableRAG (used by RQ3)
}


def call_system(query: str, **kwargs) -> dict[str, Any]:
    """
    Call the RAG system for a single query.

    Handles:
      - lazy loading of the system module
      - wall-clock timing
      - retries on transient errors
      - timeout enforcement (best-effort via elapsed check after call)
      - response normalisation to the canonical schema

    Returns a dict conforming to EMPTY_RESPONSE keys.
    """
    fn = _load_system()

    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):   # +2 = initial attempt + retries
        t0 = time.perf_counter()
        try:
            raw: dict = fn(query, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if elapsed_ms / 1000 > QUERY_TIMEOUT_SEC:
                print(f"  ⚠ query exceeded timeout ({elapsed_ms:.0f} ms > "
                      f"{QUERY_TIMEOUT_SEC * 1000:.0f} ms)")

            return _normalise(raw, elapsed_ms)

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            last_error = str(exc)
            if attempt <= MAX_RETRIES:
                print(f"  ↻ attempt {attempt} failed ({exc}), "
                      f"retrying in {RETRY_DELAY_SEC}s …")
                time.sleep(RETRY_DELAY_SEC)
            else:
                break

    # All attempts exhausted
    return {**EMPTY_RESPONSE, "error": last_error, "latency_ms": 0.0}


def _normalise(raw: dict, elapsed_ms: float) -> dict[str, Any]:
    """
    Map whatever your system returns onto the canonical schema.
    Extend this function if your system uses different key names.
    """
    result = dict(EMPTY_RESPONSE)   # start from defaults

    # Standard keys — copied directly if present
    for key in EMPTY_RESPONSE:
        if key in raw:
            result[key] = raw[key]

    # Always use measured elapsed time (overrides any system-reported latency)
    result["latency_ms"] = elapsed_ms

    # Common alternative key names your system might use
    aliases = {
        "router_decision":   ["route", "routing_decision", "pipeline"],
        "answer":            ["response", "final_answer", "output", "text"],
        "sql_executed":      ["generated_sql", "sql_query", "executed_sql"],
        "sql_result":        ["db_result", "query_result", "rows"],
        "retrieved_chunks":  ["chunks", "context_chunks", "documents"],
        "retrieved_sources": ["sources", "source_files", "pdf_sources"],
        "cache_hit":         ["is_cache_hit", "from_cache", "fast_track"],
        "error":             ["exception", "err", "error_message"],
        "tables_used":       ["schema_tables", "kept_tables", "pruned_tables"],
    }
    for canonical, alts in aliases.items():
        if result[canonical] is None:
            for alt in alts:
                if alt in raw and raw[alt] is not None:
                    result[canonical] = raw[alt]
                    break

    return result
