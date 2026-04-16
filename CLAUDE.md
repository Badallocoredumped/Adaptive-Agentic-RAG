# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest documents into FAISS vector store
python ingest_data.py [--data-dir data/]

# Run the demo / ad-hoc query
python -m backend.main

# Run unit tests only (no external services required)
pytest tests/test_sql_pipeline.py -m "not integration and not e2e"

# Run unit + integration tests (requires PostgreSQL)
pytest tests/test_sql_pipeline.py -m "unit or integration"

# Run all tests (requires PostgreSQL + OpenAI-compatible LLM)
pytest tests/test_sql_pipeline.py
```

## Architecture

This is an **Adaptive Agentic RAG** system — a Python backend that routes natural-language queries to the right retrieval pipeline and synthesizes answers with an LLM.

### Query flow

```
User Query
    ↓
QueryRouter  (backend/router/router.py)
    ↓ classifies as: sql | text | hybrid
    ├── SQL Pipeline  → TableRAG schema retrieval → ReAct NL-to-SQL agent → DB execution
    ├── RAG Pipeline  → FAISS vector search → optional reranking → top-k chunks
    └── Hybrid        → both pipelines in parallel
    ↓
ResponseSynthesizer  (backend/synthesis/synthesizer.py)
    ↓
Final Answer
```

### Key modules

- **`backend/main.py`** — `AdaptiveAgenticRAGSystem`: top-level orchestrator; entry point for all queries.
- **`backend/router/router.py`** — Routes queries; three modes controlled by `ROUTER_MODE`:
  - `decompose` (default): LLM decomposes query into subtasks, each routed independently.
  - `llm`: single-route classification via a local llama.cpp endpoint.
  - `keyword`: rule-based fallback (no LLM needed).
- **`backend/sql/`** — Structured retrieval stack:
  - `table_rag.py` — semantic schema retrieval (finds relevant tables/columns via embeddings).
  - `sql_agent.py` — NL-to-SQL generation wrapping `table_rag.py`.
  - `react_agent.py` — ReAct loop (max 6 iterations) around the SQL agent.
  - `database.py` — DB abstraction over SQLite (default) and PostgreSQL.
  - `sql_cache.py` — query result caching.
- **`backend/rag/`** — Unstructured retrieval stack:
  - `embedder.py` — SentenceTransformers (`intfloat/multilingual-e5-base`) with E5 `query:`/`passage:` prefixes.
  - `vector_store.py` — FAISS with cosine distance (normalized L2). Includes Windows `ctypes` workaround.
  - `retriever.py` + `reranker.py` — retrieval and optional semantic reranking.
  - `loader.py` + `chunker.py` — PDF/text ingestion, recursive or fixed-size chunking.
- **`backend/config.py`** — All tunable knobs in one place (see below).

### Configuration (`backend/config.py`)

| Variable | Default | Options |
|---|---|---|
| `ROUTER_MODE` | `decompose` | `decompose`, `llm`, `keyword` |
| `CHUNKER_MODE` | `recursive` | `recursive`, `fixed` |
| `DB_BACKEND` | `sqlite` | `sqlite`, `postgres` |
| `EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-base` | any SentenceTransformers model |
| `E5_PREFIX_ENABLED` | `True` | adds `query:` / `passage:` prefixes |
| `RAG_TOP_K` | `6` | chunks returned per text query |
| `SQL_TOP_K` | `4` | schema tables retrieved by TableRAG |

Database is selected by environment variables: set `SQLITE_PATH` for SQLite, or `DATABASE_URL` / `PG_*` vars for PostgreSQL.

LLM endpoints (router + synthesizer) expect an OpenAI-compatible API; a local llama.cpp server is the intended default. If the LLM is unavailable, the router falls back to keyword mode automatically.

### Test markers

Tests use `pytest` markers: `unit`, `integration`, `e2e`. Unit tests mock the DB and LLM. Integration tests require a live PostgreSQL instance. E2E tests require PostgreSQL + a live LLM endpoint.
