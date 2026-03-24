"""Central configuration for the Adaptive Agentic RAG MVP."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads .env from the project root (or current working directory)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "chunk_metadata.json"
SQLITE_DB_PATH = DATA_DIR / "app.db"

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

# decompose -> divide into subtask before routing (zeroshot) (project's goal)
# zeroshot -> zeroshot routing without dividing into subtasks
# llm -> uses llm for routing process (local)
# by default it uses predefined keyword check for routing
ROUTER_MODE = "decompose" 

ROUTER_LLM_TEMPERATURE = 0.0
ROUTER_MODEL = "qwen2.5-coder-7b-instruct"
ROUTER_BASE_URL = "http://localhost:8080"
ROUTER_API_KEY = "local"

# OpenAI settings for SQL generation & refinement
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # set in .env
SQL_OPENAI_MODEL = "gpt-4o-mini"
SQL_GENERATE_TEMPERATURE = 0.0
SQL_REFINE_TEMPERATURE = 0.0
ROUTER_DEBUG = True

# maximum number of subtasks for initial query
ROUTER_DECOMPOSE_MAX_SUBTASKS = 4


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
