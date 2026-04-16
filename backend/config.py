"""Central configuration for the Adaptive Agentic RAG MVP."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def win_short_path(path: Path) -> str:
    """Return Windows 8.3 short path so FAISS C++ fopen handles non-ASCII dirs.

    GetShortPathNameW only works on paths that already exist.  When the target
    file does not exist yet (e.g. we are about to create it), we shorten the
    parent directory instead and re-attach the filename — the parent must exist
    (call mkdir before this function in write paths).

    On non-Windows platforms returns str(path) unchanged.
    """
    if sys.platform != "win32":
        return str(path)
    try:
        import ctypes
        path = Path(path)
        buf = ctypes.create_unicode_buffer(32768)
        # Fast path: file already exists — shorten the whole path.
        if ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf)):
            return buf.value
        # File does not exist yet: shorten the parent (must already exist).
        if ctypes.windll.kernel32.GetShortPathNameW(str(path.parent), buf, len(buf)):
            return buf.value + "\\" + path.name
    except Exception:
        pass
    return str(path)

load_dotenv(override=True)  # loads .env, overriding any pre-existing system env vars

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "chunk_metadata.json"

# ── Database backend ──────────────────────────────────────────────────────────
# Set SQLITE_PATH to a .db file path to use SQLite instead of PostgreSQL.
# Example: SQLITE_PATH=data/mydata.db
SQLITE_PATH: str = os.getenv("SQLITE_PATH", "")
DB_BACKEND: str = "sqlite" if SQLITE_PATH else "postgres"

# ── PostgreSQL connection settings (ignored when SQLITE_PATH is set) ──────────
# Set DATABASE_URL to override all individual vars (preferred in production).
# Format: postgresql://user:password@host:port/dbname
DATABASE_URL: str = os.getenv("DATABASE_URL", "")
PG_HOST: str      = os.getenv("PG_HOST", "localhost")
PG_PORT: int      = int(os.getenv("PG_PORT", "5432"))
PG_DB: str        = os.getenv("PG_DB", "adaptive_rag")
PG_USER: str      = os.getenv("PG_USER", "postgres")
PG_PASSWORD: str  = os.getenv("PG_PASSWORD", "")

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
E5_PREFIX_ENABLED = True
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

RAG_TOP_K = 6        # number of text chunks to retrieve
SQL_TOP_K = 4        # number of schema tables to retrieve


RAG_FETCH_MULTIPLIER = 6
RAG_ENABLE_SEMANTIC_RERANK = True
RAG_PREVIEW_CHARS = 400

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHUNKER_MODE = "recursive"
VECTOR_DISTANCE_METRIC = "cosine"
VECTOR_NORMALIZE_L2 = True

RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

DEFAULT_ROUTE = "text"

# decompose -> LLM decomposes query into routed sub-tasks (project default)
# llm       -> local LLM classifies the whole query into a single route
# keyword   -> rule-based keyword matching into a single route (fallback)
ROUTER_MODE = "decompose"

ROUTER_LLM_TEMPERATURE = 0.0
ROUTER_MODEL = "qwen2.5-coder-7b-instruct"
ROUTER_BASE_URL = "http://localhost:8080"
ROUTER_API_KEY = "local"
SYNTHESIS_MODEL = ROUTER_MODEL
SYNTHESIS_TEMPERATURE = 0.0

# OpenAI settings for SQL generation & refinement
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # set in .env
SQL_OPENAI_MODEL = "gpt-4o-mini"
SQL_GENERATE_TEMPERATURE = 0.0
SQL_REFINE_TEMPERATURE = 0.0

# Global runtime debug logging toggle for Router/RAG/TableRAG/ReAct internals.
# Set False to keep terminal output concise (final answers/errors only).
DEBUG_LOGGING = False

# Legacy alias kept for compatibility with existing references.
ROUTER_DEBUG = DEBUG_LOGGING

# maximum number of subtasks for initial query
ROUTER_DECOMPOSE_MAX_SUBTASKS = 4

# ReAct agent settings
# Set SQL_REACT_ENABLED=False to bypass the ReAct layer and use the single-pass agent directly.
SQL_REACT_ENABLED = True
SQL_REACT_MAX_ITERATIONS = 6  # max Thought/Action/Observation cycles per query

ROUTER_CHAT_ENDPOINT = "/v1/chat/completions"
ROUTER_COMPLETION_ENDPOINT = "/completion"
ROUTER_TIMEOUT_SECONDS = 30

SQL_KEYWORDS = {
    "count",
    "sum",
    "average",
    "avg",
    "total",
    "order",
    "orders",
    "customer",
    "customers",
    "table",
    "sql",
    "database",
    "revenue",
}

TEXT_KEYWORDS = {
    "document",
    "pdf",
    "text",
    "explain",
    "summary",
    "summarize",
    "file",
    "context",
    "policy",
    "report",
}

DOMAIN_KEYWORDS = {
    "sales": {
        "sale",
        "sales",
        "order",
        "orders",
        "revenue",
        "customer",
        "customers",
        "invoice",
        "amount",
    },
    "legal": {
        "law",
        "legal",
        "contract",
        "compliance",
        "policy",
        "regulation",
        "gdpr",
        "terms",
    },
}

FILENAME_DOMAIN_HINTS = {
    "sales": "sales",
    "order": "sales",
    "invoice": "sales",
    "finance": "sales",
    "law": "legal",
    "legal": "legal",
    "contract": "legal",
    "policy": "legal",
    "compliance": "legal",
}
