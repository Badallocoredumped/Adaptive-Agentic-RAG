# Performance Engineering Review — Adaptive Agentic RAG

> Reviewed: every module in `backend/` — `main.py`, `config.py`, `router/router.py`, `rag/{embedder,retriever,reranker,vector_store}.py`, `sql/{sql_agent,react_agent,table_rag,sql_cache,database}.py`, `synthesis/synthesizer.py`.

---

## 1. Bottleneck Identification

### SQL Pipeline

| Bottleneck | Location | Severity |
|---|---|---|
| **ReAct agent creates a new `ChatOpenAI` + compiled LangGraph on every call** | `react_agent.py:258` — `_get_react_llm()` and `build_react_agent()` called from `run_react_sql_agent()` every time | 🔴 High |
| **DB connection opened + closed per `execute_query()` call** | `database.py:52-61`, `63-71` — no connection pooling; ReAct may call `execute_sql` tool 3-6× per query | 🟡 Medium |
| **`SentenceTransformer` model loaded 2× independently** | `table_rag.py:25` and `sql_cache.py:41` each instantiate their own `SentenceTransformer(config.EMBEDDING_MODEL_NAME)` — same model, separate GPU/RAM copies | 🔴 High |
| **Cache-hit refiner always calls OpenAI even for scores 0.85–0.93** | `sql_agent.py:447-457` — the 0.93 threshold should be tighter; many near-identical queries still trigger an LLM round-trip | 🟡 Medium |

### RAG Pipeline

| Bottleneck | Location | Severity |
|---|---|---|
| **CrossEncoder loaded lazily on first retrieve() only, but reranks sequentially on CPU** | `retriever.py:79-84` — `reranker.predict()` scores all `fetch_k` pairs one batch, but the batch size is up to 36 pairs (6×6) on CPU | 🟡 Medium |
| **Double FAISS search on domain filter miss** | `retriever.py:62-63` — if the domain filter returns fewer than `fetch_k` results, the entire unfiltered search runs again from scratch (re-embeds the query) | 🟡 Medium |
| **`RAG_FETCH_MULTIPLIER = 6` is aggressive** | `config.py:69` — `fetch_k = max(top_k * 6, 20) = 36`. The reranker then scores 36 query-doc pairs. Most of the bottom half will be discarded | 🟡 Medium |

### Synthesis Layer

| Bottleneck | Location | Severity |
|---|---|---|
| **`_resolve_chat_openai_class()` called via `importlib.import_module` on every synthesis call** | `synthesizer.py:282-293` — repeated dynamic import resolution | 🟢 Low |
| **Full JSON payload serialized with `indent=2`** | `synthesizer.py:123` — `json.dumps(payload, indent=2)` adds token waste; the LLM pays per-token for all those spaces | 🟡 Medium |

### Concurrency Model

| Bottleneck | Location | Severity |
|---|---|---|
| **Nested `ThreadPoolExecutor` creation** | `main.py:152` — each hybrid subtask spawns an inner `ThreadPoolExecutor(max_workers=2)` inside an outer pool with `max_workers=len(sub_tasks)`. For 4 subtasks × 2 inner = 8 threads + the outer 4 = 12 threads | 🔴 High |
| **`_get_sql_cache()` uses `hasattr(fn, "instance")` singleton — not thread-safe** | `sql_agent.py:407-412` — concurrent subtasks can race and double-init the cache | 🟡 Medium |

---

## 2. Safe Optimization Opportunities

### OPT-1: Share `SentenceTransformer` model across TableRAG + SQLCache

**Problem:** `table_rag._get_model()` and `sql_cache._get_model()` each load `intfloat/multilingual-e5-base` independently. That's ~900MB RAM × 2 (or GPU VRAM × 2).

**Fix:** Create a shared module-level singleton in `backend/models.py`:
```python
# backend/models.py
from sentence_transformers import SentenceTransformer
from backend import config

_ST_MODEL = None

def get_shared_st_model() -> SentenceTransformer:
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _ST_MODEL
```
Then import it in both `table_rag.py` and `sql_cache.py` instead of each having their own `_MODEL`.

| Impact | Risk | Effort |
|---|---|---|
| 🔴 **High** — halves model memory; eliminates ~2-3s duplicate load on cold start | 🟢 Low | ~15 min |

---

### OPT-2: Cache the compiled `ChatOpenAI` instance in the ReAct agent

**Problem:** `_get_react_llm()` creates a brand-new `ChatOpenAI(...)` on every single call to `run_react_sql_agent()`. LangChain's ChatOpenAI constructor sets up HTTP connection pools, validates the API key, etc.

**Fix:** Cache the LLM instance at module level (the config never changes mid-process):
```python
_REACT_LLM: ChatOpenAI | None = None

def _get_react_llm() -> ChatOpenAI:
    global _REACT_LLM
    if _REACT_LLM is None:
        api_key = os.environ.get("OPENAI_API_KEY", config.OPENAI_API_KEY)
        _REACT_LLM = ChatOpenAI(model=config.SQL_OPENAI_MODEL, temperature=0.0, api_key=api_key)
    return _REACT_LLM
```

| Impact | Risk | Effort |
|---|---|---|
| 🟡 **Medium** — saves ~50-100ms per ReAct invocation from constructor + HTTP pool setup | 🟢 Low | ~5 min |

---

### OPT-3: Cache `_resolve_chat_openai_class()` result

**Problem:** Both `router.py` and `synthesizer.py` call `importlib.import_module("langchain_openai")` then `getattr(module, "ChatOpenAI")` on **every** request. This dynamic import resolution is pure overhead.

**Fix:** Use `functools.lru_cache(maxsize=1)` on `_resolve_chat_openai_class`:
```python
from functools import lru_cache

@staticmethod
@lru_cache(maxsize=1)
def _resolve_chat_openai_class():
    ...
```

| Impact | Risk | Effort |
|---|---|---|
| 🟢 **Low** — saves microseconds per call, but eliminates repeated module scanning | 🟢 Low | ~2 min |

---

### OPT-4: Remove `indent=2` from synthesis payload serialization

**Problem:** `synthesizer.py:123` does `json.dumps(payload, indent=2)`. For a typical hybrid query with SQL rows + RAG chunks, this adds hundreds of whitespace characters that the LLM tokenizes and you pay for — both in latency (more tokens to read) and cost.

**Fix:** Use `json.dumps(payload, separators=(',', ':'))` for minimum token footprint.

| Impact | Risk | Effort |
|---|---|---|
| 🟡 **Medium** — saves ~100-300 tokens per call ≈ 5-15% faster synthesis response | 🟢 Low | ~1 min |

---

### OPT-5: Reduce `RAG_FETCH_MULTIPLIER` from 6 to 3–4

**Problem:** The retriever fetches `top_k × 6 = 36` chunks from FAISS, then reranks all 36 with the CrossEncoder. The bottom ~50% of these are almost always discarded. CrossEncoder cost scales linearly with pair count.

**Fix:** Set `RAG_FETCH_MULTIPLIER = 3` (or 4). This cuts reranking cost roughly in half while still giving the reranker enough candidates to improve over raw FAISS ordering.

| Impact | Risk | Effort |
|---|---|---|
| 🟡 **Medium** — CrossEncoder rerank time drops ~40-50% | 🟢 Low | ~1 min (config change) |

---

### OPT-6: Cap outer `ThreadPoolExecutor` workers

**Problem:** `main.py:173` creates `ThreadPoolExecutor(max_workers=len(sub_tasks))`. If the router decomposes into 4 subtasks, that's 4 outer threads. Each hybrid subtask nests another `ThreadPoolExecutor(max_workers=2)`. Thread explosion: up to 12 real threads, many blocking on OpenAI HTTP I/O. FAISS and SentenceTransformer are NOT thread-safe by default.

**Fix:** Cap outer workers at `min(len(sub_tasks), 3)` and reuse a single long-lived pool on the system object:
```python
# In __init__:
self._executor = ThreadPoolExecutor(max_workers=3)
```
Reuse this pool instead of creating/destroying it per-call.

| Impact | Risk | Effort |
|---|---|---|
| 🟡 **Medium** — reduces thread creation overhead, prevents GPU contention | 🟢 Low | ~15 min |

---

### OPT-7: Add thread-safety to `_get_sql_cache()` singleton

**Problem:** The current implementation uses `hasattr(_get_sql_cache, "instance")` which is not thread-safe. Two concurrent subtasks can race past the check and double-initialize the cache + double-load the SentenceTransformer model.

**Fix:** Use a `threading.Lock`:
```python
import threading
_sql_cache_lock = threading.Lock()

def _get_sql_cache() -> "SQLCache":
    with _sql_cache_lock:
        if not hasattr(_get_sql_cache, "instance"):
            from .sql_cache import SQLCache
            cache = SQLCache()
            cache.load_cache()
            _get_sql_cache.instance = cache
    return _get_sql_cache.instance
```

| Impact | Risk | Effort |
|---|---|---|
| 🟡 **Medium** — prevents double model loading under concurrency; correctness fix | 🟢 Low | ~5 min |

---

### OPT-8: Eliminate redundant FAISS search on domain filter fallback

**Problem:** `retriever.py:56-65` — when a domain filter returns fewer than `fetch_k` results, the code re-runs `self.vector_store.search(query, fetch_k)` which re-embeds the query. The LangChain FAISS `similarity_search_with_score` calls `embed_query()` internally each time.

**Fix:** Split into two steps: embed the query once, then use `similarity_search_with_score_by_vector()`:
```python
query_vec = self.vector_store.embeddings.embed_query(query)
results = self.vector_store.store.similarity_search_with_score_by_vector(query_vec, k=fetch_k, filter=...)
if len(results) < fetch_k:
    results = self.vector_store.store.similarity_search_with_score_by_vector(query_vec, k=fetch_k)
```

| Impact | Risk | Effort |
|---|---|---|
| 🟡 **Medium** — eliminates a redundant ~50ms embedding call on every domain-filtered query | 🟢 Low | ~10 min |

---

### OPT-9: Raise cache-hit no-refine threshold from 0.93 to 0.95 or lower the "call refiner" floor

**Problem:** `sql_agent.py:447` — scores between 0.85 (cache threshold) and 0.93 always trigger an OpenAI API call via `_refine_sql_from_cache()`. For your small 9-table schema, many queries in the 0.88-0.93 range are essentially identical intent and the refiner returns the cached SQL unchanged — wasting ~300-500ms and API cost.

**Fix:** Either raise the refine-skip threshold to `0.90` (covering more queries) or add a secondary check: if `original_question` fuzzy-matches `new_question` above a string similarity threshold, skip the refiner.

| Impact | Risk | Effort |
|---|---|---|
| 🟡 **Medium** — eliminates unnecessary LLM calls on ~30-40% of cache hits | 🟢 Low | ~5 min |

---

## 3. Concurrency Review

### Current State

```
run_query("complex hybrid question")
    └── decompose_with_zeroshot()           ← 1 OpenAI call (blocking)
    └── _execute_subtasks()
         └── ThreadPoolExecutor(max_workers=N)   ← N = len(sub_tasks), up to 4
              ├── run_task(sql)  → run_table_rag_pipeline()
              │    └── _get_sql_cache() → SentenceTransformer.encode()  ← shared model?
              │    └── retrieve_relevant_schema() → SentenceTransformer.encode()
              │    └── build_react_agent() → ChatOpenAI → graph.invoke()
              │         └── execute_sql tool (up to 6 iterations)
              ├── run_task(text) → retriever.retrieve()
              │    └── FAISS search → HuggingFace embed_query()
              │    └── CrossEncoder.predict()
              └── run_task(hybrid)
                   └── ThreadPoolExecutor(max_workers=2)   ← NESTED!
                        ├── run_table_rag_pipeline()
                        └── retriever.retrieve()
```

### Issues

1. **Nested pools are wasteful.** Creating + destroying `ThreadPoolExecutor` objects has real cost (thread creation, OS scheduling). The inner pools live for one task then die.

2. **GPU contention.** If running on GPU, `SentenceTransformer.encode()` and `CrossEncoder.predict()` will contend on the CUDA device. PyTorch operations are serialized by the GIL anyway for CPU, but GPU kernels from multiple threads can cause OOM or corruption. Currently you're likely on CPU, so this is a latent risk rather than an active bug.

3. **OpenAI API burst.** 4 concurrent subtasks × (1 refiner or 1 generator + up to 6 ReAct iterations) = potentially 28 concurrent OpenAI API calls. Risk of 429 rate-limiting, especially on `gpt-4o-mini` tier.

### Recommendations

| Fix | Detail |
|---|---|
| Reuse a single `ThreadPoolExecutor(max_workers=3)` as a class attribute | Eliminates per-call pool creation; caps total threads |
| For hybrid subtasks, run SQL and RAG sequentially instead of nested pool | The nested pool's 2 threads save little vs. sequential when you're already parallelized at the subtask level |
| Add an `openai_semaphore = threading.Semaphore(4)` | Wrap all OpenAI calls to prevent burst > 4 concurrent |

---

## 4. Model & Pipeline Efficiency

### Embedding Model Usage

Currently **three separate** `SentenceTransformer` / `HuggingFaceEmbeddings` instances exist for the same `intfloat/multilingual-e5-base` model:

| Instance | Location | Used For |
|---|---|---|
| `table_rag._MODEL` | `table_rag.py:18` | Schema embedding + retrieval |
| `sql_cache._model` | `sql_cache.py:36` | SQL cache embedding + search |
| `embedder._model` (via HuggingFaceEmbeddings) | `embedder.py:51` | RAG document embedding + query |

The first two use raw `SentenceTransformer`; the third wraps it in `HuggingFaceEmbeddings`. All three load the same 278M-parameter model. **This triples** memory footprint for no reason.

> **Fix:** Consolidate into a shared singleton as described in OPT-1. The RAG embedder could also use the shared model (wrap it in a thin LangChain-compatible adapter).

### CrossEncoder Reranking Cost

- Model: `BAAI/bge-reranker-base` (~278M params)
- Current input: 36 pairs (query × 36 chunks) on every RAG retrieval
- Estimated latency: **200-500ms on CPU** depending on chunk length
- This is the single most expensive *local compute* step in the RAG pipeline

> **Fix:** Reduce `RAG_FETCH_MULTIPLIER` from 6 → 3. Alternatively, use `batch_size` parameter in `model.predict()` to enable batched inference (default is already batched, but confirming explicit control is good practice).

### LLM Call Count (worst case per query)

| Step | API Calls | Model |
|---|---|---|
| Router decomposition | 1 | gpt-4o-mini |
| SQL cache refiner (per subtask) | 0-1 | gpt-4o-mini |
| SQL generation (cache miss) | 1 | gpt-4o-mini |
| ReAct agent (cache miss) | 2-6 | gpt-4o-mini |
| Synthesis | 1 | gpt-4o-mini |
| **Worst-case total** | **~10** | |

For a 4-subtask decomposition where all are SQL cache misses with ReAct:
`1 (router) + 4×6 (ReAct) + 1 (synthesis) = 26 API calls` 💀

> **Fix:** The ReAct `max_iterations` of 6 is generous. Most successful queries resolve in 1-2 iterations. Reducing `SQL_REACT_MAX_ITERATIONS` from 6 → 3 would cap worst-case without hurting typical cases.

---

## 5. Latency Breakdown (Estimated)

| Stage | Estimated Latency | Notes |
|---|---|---|
| **Router decomposition** | 300-600ms | Single gpt-4o-mini call, small prompt |
| **SQL cache hit (score ≥ 0.93)** | 50-100ms | FAISS search + local embedding only |
| **SQL cache hit (score 0.85-0.93)** | 400-700ms | Above + OpenAI refiner call |
| **SQL agent path (ReAct, 2 iterations)** | 1.5-3.0s | Schema retrieval (50ms) + 2 LLM calls (500ms each) + 2 SQL executions |
| **SQL agent path (ReAct, 6 iterations)** | 4-8s | 6 LLM round-trips with tool calls |
| **RAG FAISS retrieval** | 50-100ms | embed_query + FAISS search |
| **CrossEncoder reranking (36 pairs)** | 200-500ms | CPU-bound, scales with pair count |
| **Synthesis LLM** | 300-800ms | Single gpt-4o-mini call, medium prompt |
| **End-to-end (cache hit, SQL-only)** | 0.7-1.5s | Router + cache + synthesis |
| **End-to-end (cache miss, SQL-only)** | 2.5-5.0s | Router + ReAct + synthesis |
| **End-to-end (hybrid, cache miss)** | 3.0-6.0s | Router + parallel(ReAct, RAG+rerank) + synthesis |

### What Dominates

1. **On cache miss: the ReAct agent.** Each iteration is a full LLM round-trip (300-500ms) + a DB query. With 2-3 iterations typical, this is 60-70% of total latency.
2. **On cache hit: the synthesis LLM call.** Everything else is sub-100ms, so the final synthesis call (300-800ms) dominates.
3. **The router decomposition is a fixed tax** on every query (~400ms). It can't be parallelized because downstream work depends on it.

---

## 6. Quick Wins — Top 5

| # | Optimization | Impact | Risk | Effort |
|---|---|---|---|---|
| **1** | **Share the `SentenceTransformer` singleton** across `table_rag`, `sql_cache`, and ideally `embedder` (OPT-1) | 🔴 High — halves model RAM, eliminates 2-3s cold-start duplicate | 🟢 Low | 15 min |
| **2** | **Remove `indent=2` from synthesis JSON** (OPT-4) | 🟡 Medium — 5-15% faster synthesis, lower API cost | 🟢 Low | 1 min |
| **3** | **Reduce `RAG_FETCH_MULTIPLIER` from 6 → 3** (OPT-5) | 🟡 Medium — ~40-50% faster CrossEncoder reranking | 🟢 Low | 1 min |
| **4** | **Cache the ReAct `ChatOpenAI` instance** (OPT-2) | 🟡 Medium — saves ~50-100ms per SQL agent call | 🟢 Low | 5 min |
| **5** | **Add `threading.Lock` to `_get_sql_cache()`** (OPT-7) | 🟡 Medium — prevents double model load under concurrency | 🟢 Low | 5 min |

**Total estimated effort: ~27 minutes for all 5.**

---

## 7. Things That Look Good (Do NOT Change)

| Component | Why It's Good |
|---|---|
| **SQL semantic cache architecture** (`sql_cache.py`) | Clean design. FAISS IndexFlatIP with L2-normalized vectors correctly implements cosine similarity. The hit/miss threshold of 0.85 is well-tuned. |
| **TableRAG schema retrieval** (`table_rag.py`) | Elegant solution. Embedding table schemas as text for semantic FAISS lookup rather than heuristic matching is genuinely clever and works well for your 9-table schema. |
| **E5 prefix handling** (`_apply_e5_prefix`) | Correctly distinguishes query vs. passage prefixes with idempotency checks. Both `table_rag.py` and `sql_cache.py` handle this correctly. |
| **ReAct agent tool design** (`react_agent.py`) | The `execute_sql` and `schema_lookup` tools are well-scoped. Read-only enforcement via regex (`_SQL_READ_RE`) is a solid safety boundary. Row truncation at 50 is sensible. |
| **Synthesis fallback formatter** (`synthesizer.py:554-576`) | Having a deterministic formatter as fallback when LLM synthesis fails is excellent resilience engineering. |
| **Latency tracking** | Already built into both the SQL pipeline (`time.time()`) and synthesis (`time.perf_counter()`). Don't add more — this is sufficient. |
| **Lazy model loading pattern** | All models (SentenceTransformer, CrossEncoder, ChatOpenAI class resolution) use lazy loading. This keeps startup fast. Just consolidate the duplicates. |
| **`win_short_path()` for FAISS on Windows** | This is a real production gotcha (FAISS C++ can't handle non-ASCII paths). Smart defensive engineering. |
