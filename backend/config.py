"""Central configuration for the Adaptive Agentic RAG MVP."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "chunk_metadata.json"
SQLITE_DB_PATH = DATA_DIR / "app.db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHUNKER_MODE = "recursive"

RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

DEFAULT_ROUTE = "text"
ROUTER_MODE = "llm"
ROUTER_LLM_TEMPERATURE = 0.0
ROUTER_MODEL = "Qwen"
ROUTER_BASE_URL = "http://localhost:8080"
ROUTER_CHAT_ENDPOINT = "/v1/chat/completions"
ROUTER_COMPLETION_ENDPOINT = "/completion"
ROUTER_TIMEOUT_SECONDS = 10

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
