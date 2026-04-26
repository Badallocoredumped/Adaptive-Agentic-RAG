"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for all benchmark runners.
Edit the paths/settings in this file to match your project layout.
─────────────────────────────────────────────────────────────────────────────
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

# Directory that contains THIS file and all runner scripts
RUNNERS_DIR = Path(__file__).parent

# Root of your project (tests/EndToEnd/ → tests/ → project root)
PROJECT_ROOT = RUNNERS_DIR.parent.parent

# Benchmark query file (the flat JSON you provided)
BENCHMARK_FILE = RUNNERS_DIR / "all_benchmark_queries.json"

# SQLite database (matches backend/config.py DATA_DIR layout)
DB_PATH = PROJECT_ROOT / "data" / "fintech.db"

# Directory where PDF files live (used by text/hybrid retrievers)
PDF_DIR = PROJECT_ROOT / "data"

# Output directory – all result JSON files are written here
RESULTS_DIR = RUNNERS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── System entrypoint ────────────────────────────────────────────────────────
# Your RAG system must expose a function or callable that matches this import.
# Runner scripts call:
#   from your_rag_system import run_query
#
# run_query(query: str) must return a dict with AT MINIMUM these keys:
#
#   {
#     "router_decision":    "sql" | "text" | "hybrid",
#     "answer":             str,          # final synthesised answer
#     "sql_executed":       str | None,   # SQL that was actually run (if any)
#     "sql_result":         list | None,  # rows returned (if any)
#     "retrieved_chunks":   list | None,  # PDF chunks retrieved (if any)
#     "retrieved_sources":  list | None,  # source filenames (if any)
#     "cache_hit":          bool,         # True = Fast Track, False = Reasoning Track
#     "latency_ms":         float,        # wall-clock time in milliseconds
#     "error":              str | None,   # exception message if something failed
#   }
#
# If your system uses a different interface, update SYSTEM_MODULE / SYSTEM_FN
# or wrap it in a thin adapter in system_adapter.py.

SYSTEM_MODULE = "rag_system"        # importable module name
SYSTEM_FN     = "run_query"         # callable inside that module

# ── Benchmark run settings ───────────────────────────────────────────────────

# Seconds to wait between queries (avoids rate-limit bursts on LLM APIs)
INTER_QUERY_DELAY_SEC = 1.0

# Maximum seconds to wait for a single query before marking it timed out
QUERY_TIMEOUT_SEC = 120

# Number of retries on transient errors (network, API 429)
MAX_RETRIES = 2
RETRY_DELAY_SEC = 5.0

# Verbosity: True = print every query result to stdout
VERBOSE = True
