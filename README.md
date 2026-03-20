# Adaptive-Agentic-RAG

MVP scaffold for an Adaptive Agentic RAG-style backend that supports:

- Unstructured retrieval from PDF/text using LangChain embeddings + LangChain FAISS.
- Structured retrieval from SQLite using an NL-to-SQL stub.
- A router layer that selects `sql`, `text`, or `hybrid`.
- A synthesis layer that merges outputs into one final response.

## Project Structure

```text
backend/
├── router/
│   └── router.py
├── rag/
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── retriever.py
├── sql/
│   ├── database.py
│   ├── query_generator.py
│   └── executor.py
├── synthesis/
│   └── synthesizer.py
├── main.py
└── config.py
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run your local llama.cpp server for the WP1 LLM router (OpenAI-compatible endpoint):

```bash
python -m llama_cpp.server --host 0.0.0.0 --port 8080 --model /path/to/model.gguf
```

## Function-Call Usage

The project uses function-call orchestration (no interactive CLI required).

```python
from backend.main import build_system, run_query

# Option A: one-shot convenience
answer = run_query("What is the total number of orders?")
print(answer)

# Option B: reuse one system instance
system = build_system()

# Optional: ingest your files before text/hybrid retrieval
indexed = system.ingest_documents([
	"path/to/file1.txt",
	"path/to/file2.pdf",
])
print(f"Indexed chunks: {indexed}")

answer = system.run_query("Summarize policy details and show order totals")
print(answer)
```

## Run Demo

```bash
python -m backend.main
```

## WP1 Flow (Router + Unstructured Retrieval)

1. Ingest PDF/text documents into FAISS:

```bash
python -c "from backend.main import build_system; s=build_system(); print('indexed=', s.ingest_documents(['data/Project_Proposal.pdf']))"
```

2. Run a document-centric query:

```bash
python -c "from backend.main import run_query; print(run_query('summarize the document'))"
```

Notes:

- Router mode is controlled in `backend/config.py` via `ROUTER_MODE` (`zeroshot`, `llm`, or `rule`).
- `zeroshot` mode uses LangChain with an OpenAI-compatible chat endpoint for strict route classification (`sql|text|hybrid`).
- LLM routing calls your local llama.cpp REST server (`/v1/chat/completions` primary, `/completion` fallback) with automatic fallback to rule-based routing if the server is unavailable.
- Chunking mode is controlled via `CHUNKER_MODE` (`recursive` recommended for WP1, `fixed` also available).

This runs a single hardcoded demo query through:

1. Router
2. SQL and/or RAG path
3. Synthesis layer

## Notes

- LLM calls are intentionally stubbed to keep this MVP simple and local.
- SQLite tables are initialized automatically (without seed data).
- If no documents are indexed yet, text retrieval returns no chunks gracefully.
- The RAG internals use LangChain wrappers while keeping the same project module structure.