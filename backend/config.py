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

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
INGEST_DIR = DATA_DIR / "ingest"
INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "chunk_metadata.json"

# ============================================================================
# Database
# ============================================================================
# Set SQLITE_PATH to a .db file path to use SQLite instead of PostgreSQL.
# Example: SQLITE_PATH=data/mydata.db
SQLITE_PATH: str = os.getenv("SQLITE_PATH", "")
DB_BACKEND: str = "sqlite" if SQLITE_PATH else "postgres"

# PostgreSQL connection settings (ignored when SQLITE_PATH is set).
# Set DATABASE_URL to override all individual vars (preferred in production).
# Format: postgresql://user:password@host:port/dbname
DATABASE_URL: str = os.getenv("DATABASE_URL", "")
PG_HOST: str      = os.getenv("PG_HOST", "localhost")
PG_PORT: int      = int(os.getenv("PG_PORT", "5432"))
PG_DB: str        = os.getenv("PG_DB", "fintech")
PG_USER: str      = os.getenv("PG_USER", "postgres")
PG_PASSWORD: str  = os.getenv("PG_PASSWORD", "admin")

# ============================================================================
# Embeddings
# ============================================================================
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
E5_PREFIX_ENABLED = True
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# RAG_TOP_K (number of retrieved text chunks passed into synthesis)
RAG_TOP_K = 15  # matches RAG_K15_BGE_BASE_C1000_POOL25 — best ablation config
# SQL_TOP_K (max schema tables retrieved by TableRAG)
SQL_TOP_K = 10
# SQL_SCHEMA_THRESHOLD (min cosine similarity score to include a schema table).
# Tables below this score are dropped. The top-1 match is always kept as a
# fallback so the schema context is never empty.

SQL_SCHEMA_THRESHOLD: float = None


# ============================================================================
# Reranker
# ============================================================================
RAG_RERANKER_MODEL = "BAAI/bge-reranker-base"

# ============================================================================
# Retrieval
# ============================================================================
# Retrieval mode:
#   "hybrid" -> BM25 + FAISS fused via Reciprocal Rank Fusion (higher recall)
#   "faiss"  -> dense-only FAISS cosine search (faster, lower memory)
# This setting controls retrieval strategy only.
RAG_RETRIEVAL_MODE = "faiss"
RAG_RRF_K = 60             # RRF constant -- only used when RAG_RETRIEVAL_MODE="hybrid"

RAG_FETCH_K = 0               # 0 = use RAG_FETCH_MULTIPLIER × RAG_TOP_K
RAG_FETCH_MULTIPLIER = 10
# Enables an extra semantic reranking pass over retrieved candidates.
# In hybrid mode, reranking is applied after BM25+FAISS fusion and can increase latency.
RAG_ENABLE_SEMANTIC_RERANK = True
RAG_RERANK_POOL = 25       # matches RAG_K15_BGE_BASE_C1000_POOL25 — top-25 candidates fed to cross-encoder
RAG_RERANK_DEBUG = True    # Enable verbose reranker logging
# FAISS-only thresholding hint: hybrid mode uses rank fusion, so this should be unset there.
RAG_SCORE_THRESHOLD = 0.5
RAG_MAX_CHUNKS = 10
RAG_PREVIEW_CHARS = 100

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNKER_MODE = "recursive"
VECTOR_DISTANCE_METRIC = "cosine"
VECTOR_NORMALIZE_L2 = True

RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def validate_retrieval_config() -> None:
    """Optional warning-only checks for retrieval-related settings."""
    if RAG_RETRIEVAL_MODE == "hybrid" and RAG_SCORE_THRESHOLD is not None:
        print("WARNING: RAG_SCORE_THRESHOLD should not be used in hybrid mode.")
    if RAG_RETRIEVAL_MODE not in {"hybrid", "faiss"}:
        print(
            f"WARNING: Unknown RAG_RETRIEVAL_MODE='{RAG_RETRIEVAL_MODE}'. "
            "Expected 'hybrid' or 'faiss'."
        )
    if RAG_TOP_K <= 0:
        print("WARNING: RAG_TOP_K should be > 0 to return retrieval context.")

# ============================================================================
# Router
# ============================================================================
DEFAULT_ROUTE = "text"

# decompose -> LLM decomposes query into routed sub-tasks (project default)
# llm       -> local LLM classifies the whole query into a single route
# keyword   -> rule-based keyword matching into a single route (fallback)
# semantic  -> cosine similarity to seed query embeddings (no LLM required)
ROUTER_MODE = "decompose"

# Seed queries for the semantic router — representative user intents per route.
# Embedded once at first use (query: prefix applied, same model as RAG retrieval).
SEMANTIC_ROUTER_SQL_SEEDS: list[str] = [
    # --- Original Clean Sentences ---
    "How many orders were placed last month?",
    "What is the total revenue by customer?",
    "List all customers who placed more than 5 orders.",
    "Show me the average order value per region.",
    "Count the number of products in each category.",
    "Which sales rep has the highest revenue this quarter?",
    "Give me a breakdown of orders by status.",
    "What is the sum of all invoices for 2023?",
    "How many records are in the database?",
    "Show the top 10 customers by spending.",
    
    # --- Additional Clean / Complex Aggregations ---
    "What were the total sales figures for Q3?",
    "Which region had the highest customer churn rate?",
    "Show the average delivery time for international shipments.",
    "Calculate the year-over-year growth in subscription revenue.",
    "Identify the top-selling products in the electronics category.",
    "What is the median salary for software engineers?",
    "Group the active users by their subscription tier.",
    "Find the maximum discount ever applied to a single cart.",
    
    # --- Terse / Keyword-Heavy Queries ---
    "2023 Q4 sales numbers",
    "revenue by quarter",
    "active users count",
    "inventory levels laptops",
    "daily active users last 30 days",
    "highest margin products",
    
    # --- Implicit Intents (Conversational DB lookups) ---
    "Who bought the most expensive item?",
    "Are there any pending orders from yesterday?",
    "I need to know how much we spent on marketing in January.",
    "Did sales drop after the pricing change?",
    "Which warehouse has the least stock right now?",
    "Can you pull the transaction history for client ID 4092?"
]

SEMANTIC_ROUTER_TEXT_SEEDS: list[str] = [
    # --- Original Clean Sentences ---
    "Explain the return policy in the document.",
    "Summarize the contract terms.",
    "What does the document say about data privacy?",
    "Describe the onboarding process from the manual.",
    "What are the compliance requirements mentioned in the report?",
    "Find information about GDPR in the uploaded files.",
    "What are the key terms in the agreement?",
    "Explain the company policy on remote work.",
    "What does the PDF say about refunds?",
    "Give me a summary of the uploaded report.",
    
    # --- Additional Clean / Extraction Queries ---
    "What is the procedure for filing a travel expense claim?",
    "Detail the hardware requirements outlined in the installation manual.",
    "How does the company handle intellectual property disputes?",
    "According to the employee handbook, what are the core values?",
    "Extract the main findings from the quarterly market analysis.",
    "What are the prerequisites mentioned in the syllabus?",
    "Identify the risk factors listed in the investment prospectus.",
    
    # --- Terse / Keyword-Heavy Queries ---
    "vacation days accrual",
    "onboarding checklist",
    "SLA response times document",
    "code of conduct guidelines",
    "troubleshooting guide error 404",
    "maternity leave benefits",
    
    # --- Implicit Intents / Help-Seeking ---
    "I need help setting up my corporate email.",
    "Where can I find instructions for resetting the staging server?",
    "My company laptop broke, what do I do?",
    "How am I supposed to configure the local firewall?",
    
    # --- Edge Cases (Sound quantitative, but are document-based) ---
    "List all the uploaded files in the directory.",
    "Count the paragraphs in section 4.",
    "What are the section headers in the proposal design?"
]

# If the similarity margin between the top-ranked and second-ranked route is
# below this value, both intents are considered present and "hybrid" is returned.
SEMANTIC_ROUTER_HYBRID_MARGIN: float = 0.015

# OpenAI settings for SQL generation & refinement
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # set in .env

# ============================================================================
# LLM Provider
# ============================================================================
# Set LLM_PROVIDER=ollama in .env (or environment) to use the Ollama endpoint.
# Set LLM_PROVIDER=openai (default) to use the OpenAI API.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")

# Ollama server address — WITHOUT trailing /v1 (it is appended automatically).
OLLAMA_SERVER: str = os.getenv("OLLAMA_SERVER", "http://81.214.35.48:11434")
OLLAMA_MODEL: str  = os.getenv("OLLAMA_MODEL",  "qwen2.5-coder:14b-instruct-q4_K_M")
OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "ollama")

# Effective LLM settings used by all agents (SQL, ReAct, Router, Synthesizer).
# Override any individual variable via environment if needed.
LLM_MODEL: str   = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_BASE_URL: str = f"{OLLAMA_SERVER.rstrip('/')}/v1" if LLM_PROVIDER == "ollama" else ""
LLM_API_KEY: str  = OLLAMA_API_KEY if LLM_PROVIDER == "ollama" else OPENAI_API_KEY

ROUTER_LLM_TEMPERATURE = 0.0
ROUTER_MODEL   = os.getenv("ROUTER_MODEL",   LLM_MODEL)
# ROUTER_BASE_URL is used WITHOUT /v1 — the router appends /v1 itself.
ROUTER_BASE_URL = os.getenv("ROUTER_BASE_URL", OLLAMA_SERVER if LLM_PROVIDER == "ollama" else "")
ROUTER_API_KEY  = os.getenv("ROUTER_API_KEY",  LLM_API_KEY)

SYNTHESIS_MODEL = ROUTER_MODEL
SYNTHESIS_TEMPERATURE = 0.0

# ============================================================================
# SQL
# ============================================================================
SQL_OPENAI_MODEL = LLM_MODEL
SQL_GENERATE_TEMPERATURE = 0.0
SQL_REFINE_TEMPERATURE = 0.0

# Mode for retrieving fresh schema on a semantic cache hit:
#   "always"    -> Always retrieve fresh schema for the new query.
#   "threshold" -> Use cached schema if score >= threshold, otherwise retrieve.
#   "never"     -> Never retrieve fresh schema, strictly use cached schema.
SQL_CACHE_REFRESH_MODE = "threshold"

# If SQL_CACHE_REFRESH_MODE is "threshold", this defines the minimum score
# required to skip fresh schema retrieval and strictly use the cached schema.
SQL_CACHE_REFRESH_THRESHOLD = 0.95

# ============================================================================
# Debug
# ============================================================================
# Global runtime debug logging toggle for Router/RAG/TableRAG/ReAct internals.
# Set False to keep terminal output concise (final answers/errors only).
DEBUG_LOGGING = True

# Legacy alias kept for compatibility with existing references.
ROUTER_DEBUG = DEBUG_LOGGING

# maximum number of subtasks for initial query
ROUTER_DECOMPOSE_MAX_SUBTASKS = 4

# ReAct agent settings
# Set SQL_REACT_ENABLED=False to bypass the ReAct layer and use the single-pass agent directly.
SQL_REACT_ENABLED = True
SQL_REACT_MAX_ITERATIONS = 3  # max Thought/Action/Observation cycles per query
SQL_CACHE_HIT_THRESHOLD: float = 0.85  # cosine similarity threshold for cache hit (RQ1 variable)
SQL_CACHE_SKIP_REFINER_THRESHOLD: float = 0.98  # Score above which the LLM refiner is skipped (Fast Track)

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
    "min",
    "max",
    "maximum",
    "minimum",
    "mean",
    "percent",
    "percentage",
    "ratio",
    "rank",
    "top",
    "bottom",
    "highest",
    "lowest",
    "per",
    "group",
    "by",
    "compare",
    "versus",
    "vs",
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
    "how",
    "why",
    "meaning",
    "definition",
    "describe",
    "description",
    "purpose",
    "background",
    "history",
    "narrative",
    "overview",
    "manual",
    "guide",
    "mentions",
    "about",
    "related",
    "concept",
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
