# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install eval-only dependencies (separate file)
pip install -r requirements-eval.txt

# Ingest documents into FAISS vector store
python ingest_data.py [--data-dir data/]

# Seed SQL cache with few-shot examples
python seed_sql_cache.py [--clear] [--build-schema] [--shots fintech|generic]

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
- **`backend/models.py`** — Thread-safe singleton factories: `get_shared_hf_embeddings()`, `get_shared_st_model()`, `get_shared_cross_encoder()`. Import from here rather than instantiating models directly.
- **`backend/router/router.py`** — Routes queries; three modes controlled by `ROUTER_MODE`:
  - `decompose` (default): LLM decomposes query into subtasks, each routed independently.
  - `llm`: single-route classification via a local llama.cpp endpoint.
  - `keyword`: rule-based fallback (no LLM needed).
- **`backend/sql/`** — Structured retrieval stack:
  - `table_rag.py` — semantic schema retrieval (finds relevant tables/columns via embeddings).
  - `sql_agent.py` — NL-to-SQL generation wrapping `table_rag.py`.
  - `react_agent.py` — ReAct loop (max **10** iterations) around the SQL agent.
  - `database.py` — DB abstraction over SQLite (default) and PostgreSQL.
  - `sql_cache.py` — query result caching (cosine similarity, hit threshold `SQL_CACHE_HIT_THRESHOLD=0.85`).
- **`backend/rag/`** — Unstructured retrieval stack:
  - `embedder.py` — SentenceTransformers (`intfloat/multilingual-e5-base`) with E5 `query:`/`passage:` prefixes.
  - `vector_store.py` — FAISS with cosine distance (normalized L2). Includes Windows `ctypes` short-path workaround (`config.win_short_path()`).
  - `retriever.py` + `reranker.py` — retrieval and optional cross-encoder reranking.
  - `loader.py` + `chunker.py` — PDF/text ingestion, recursive or fixed-size chunking.
  - `bm25_index.py` — BM25Okapi index for hybrid mode (requires `rank_bm25`; built lazily on first hybrid retrieve call).
- **`backend/config.py`** — All tunable knobs in one place. **Must be imported and overridden before any pipeline module** (it calls `load_dotenv(override=True)` at module level).

### Configuration (`backend/config.py`)

| Variable | Default | Notes |
|---|---|---|
| `ROUTER_MODE` | `decompose` | `decompose`, `llm`, `keyword`, `semantic` |
| `CHUNKER_MODE` | `recursive` | `recursive`, `fixed` |
| `DB_BACKEND` | `sqlite` | `sqlite`, `postgres` |
| `EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-base` | any SentenceTransformers model |
| `E5_PREFIX_ENABLED` | `True` | adds `query:` / `passage:` prefixes |
| `RAG_TOP_K` | `5` | final chunks returned per text query |
| `RAG_RETRIEVAL_MODE` | `faiss` | `faiss` (dense only) or `hybrid` (BM25 + FAISS via RRF) |
| `RAG_ENABLE_SEMANTIC_RERANK` | `True` | enables cross-encoder reranking pass |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-base` | cross-encoder model; max_length=512 |
| `RAG_RERANK_POOL` | `100` | candidate pool size fed to reranker before selecting top-k |
| `RAG_FETCH_K` | `0` | explicit BM25 fetch count (0 = use `RAG_FETCH_MULTIPLIER × RAG_TOP_K`) |
| `CHUNK_SIZE` | `1000` | chars per chunk |
| `CHUNK_OVERLAP` | `200` | char overlap between chunks |
| `SQL_TOP_K` | `30` | schema tables retrieved by TableRAG |
| `SQL_REACT_MAX_ITERATIONS` | `10` | max ReAct cycles per SQL query |
| `SQL_OPENAI_MODEL` | `gpt-4o-mini` | model used for SQL generation and RAGAS judging |
| `ROUTER_DECOMPOSE_MAX_SUBTASKS` | `4` | max subtasks from decomposition router |
| `DEBUG_LOGGING` | `True` | verbose logging for Router/RAG/TableRAG/ReAct |

Database is selected by environment variables: set `SQLITE_PATH` for SQLite, or `DATABASE_URL` / `PG_*` vars for PostgreSQL.

LLM endpoints (router + synthesizer) expect an OpenAI-compatible API; set `OPENAI_API_KEY` in `.env`. If the LLM is unavailable, the router falls back to keyword mode automatically.

### Test markers

Tests use `pytest` markers: `unit`, `integration`, `e2e`. Unit tests mock the DB and LLM. Integration tests require a live PostgreSQL instance. E2E tests require PostgreSQL + a live LLM endpoint.

## Evaluation (Unstructured RAG)

### Data preparation (run once per dataset)

```bash
python evaluation/Unstructured/prepare_ragbench_dataset.py  # RAGBench corpus + eval set
python evaluation/Unstructured/prepare_squad_dataset.py     # SQuAD corpus + eval set
python evaluation/Unstructured/prepare_nq_dataset.py        # Natural Questions corpus + eval set
```

Each eval script auto-ingests the corpus into a chunk-size-specific FAISS index (e.g., `ragbench_index_500/`, `ragbench_index_1000/`) on first run — subsequent runs reuse the cached index.

### Running evaluations

```bash
# RAGBench — main ablation ladder
python evaluation/Unstructured/run_ragbench_eval.py <CONFIG_ID>
# Ablation configs: LLM_ONLY | RAG_K10_FAISS | RAG_K10_BGE_BASE | RAG_K10_BGE_BASE_C1000 | RAG_K15_BGE_BASE_C1000_POOL25

# SQuAD configs (C1–C12, T2A–T5B)
python evaluation/Unstructured/run_config_eval.py <CONFIG_ID>

# Natural Questions
python evaluation/Unstructured/run_nq_eval.py <CONFIG_ID>

# Batch runners
python evaluation/Unstructured/run_ragbench_series.py   # all ragbench ablation configs
python evaluation/Unstructured/run_t_series.py          # all T-series configs

# Post-hoc scoring (if raw results already exist)
python evaluation/Unstructured/score_squad_ragas.py

# Smoke-test: temporarily set eval_set = eval_set[:3] in the script, then revert to [:50]
```

### Evaluation output

Results saved to `evaluation/Unstructured/results/`:
- `ragbench_<CONFIG>_rag_results.json` — retrieved contexts, generated answers, gold hit flags
- `ragbench_<CONFIG>_ragas_scores.json` — overall + per-question RAGAS metrics

### Evaluation gotchas

- **RAGAS fork**: installed package is `vibrantlabsai/ragas` 0.4.3, not the official `explodinggradients/ragas` — APIs differ.
- **Async client required**: `llm_factory` must receive `AsyncOpenAI(...)`, not `OpenAI(...)`. Sync client blocks asyncio event loop, making RAGAS evaluation fully sequential despite `RunConfig(max_workers=N)`.
- **Parallelism**: `RunConfig(max_workers=8, max_wait=180, max_retries=3)` passed to `evaluate()` in `run_ragbench_eval.py` — reduces RAGAS phase from ~45 min to ~8–15 min per config.
- **Separate FAISS indexes**: eval scripts write to their own index dirs (e.g., `ragbench_index_1000/`) and override `config.INDEX_DIR` at import time — they never touch `data/index/`.
- **Hybrid mode dependency**: `rank_bm25` must be installed for `RAG_RETRIEVAL_MODE="hybrid"` (may not be in `requirements.txt` explicitly).

## Platform notes (Windows)

- `config.win_short_path()` converts paths to 8.3 short names via `ctypes.windll.kernel32.GetShortPathNameW` — used before FAISS save/load to avoid C++ fopen failures on non-ASCII paths. Call `mkdir` on the parent before using it on a not-yet-existing file.
- Shell commands in docs use Unix syntax; run under Git Bash or WSL on Windows.
