# Bug Report — Adaptive Agentic RAG

> Generated: 2026-04-19  
> Reviewer: Claude (deep static analysis)  
> Branch: `main`

---

## Table of Contents

- [Critical Bugs](#critical-bugs)
- [High Severity](#high-severity)
- [Medium / Cosmetic](#medium--cosmetic)
- [Priority Summary](#priority-summary)

---

## Critical Bugs

These bugs cause crashes, silent data loss, or fundamentally wrong behavior.

---

### BUG-01 — Hybrid retrieval silently broken when reranking is OFF

**File:** `backend/rag/retriever.py` lines 141–147  
**Severity:** Critical — feature completely non-functional in default config

When semantic reranking is disabled, the code iterates RRF-fused results (FAISS + BM25) and scores each chunk via `text_to_score.get(text, 0.0)`. BM25-only hits have no FAISS score, so they receive `0.0`. The downstream threshold filter (`RAG_SCORE_THRESHOLD = 0.5`) then discards every BM25-only result. Hybrid retrieval silently degrades to pure FAISS-only.

**Fix:** Assign BM25-only hits a non-zero default score (e.g. their normalized BM25 score), or skip the threshold filter when a hit originates exclusively from BM25.

---

### BUG-02 — Synthesizer uses wrong model config key

**File:** `backend/synthesis/synthesizer.py` line 51  
**Severity:** Critical — `SYNTHESIS_MODEL` config is dead

```python
# Current (wrong)
self._llm = chat_openai_cls(model=config.SQL_OPENAI_MODEL, ...)

# Should be
self._llm = chat_openai_cls(model=config.SYNTHESIS_MODEL, ...)
```

`config.SYNTHESIS_MODEL` is defined in `backend/config.py` line 106 but is never referenced anywhere. Changing it has zero effect; the synthesizer always uses `SQL_OPENAI_MODEL` (`"gpt-4o-mini"`).

---

### BUG-03 — Thread-unsafe cache writes can segfault

**File:** `backend/sql/sql_cache.py` line 88, `backend/sql/sql_agent.py` lines 569–573  
**Severity:** Critical — concurrent requests can corrupt FAISS index

The `AdaptiveAgenticRAGSystem` runs sub-tasks in a `ThreadPoolExecutor`. When two requests both produce successful SQL, they call `cache.add_to_cache()` → `index.add()` → `cache.save_cache()` concurrently. FAISS index mutation is not thread-safe and will likely segfault under concurrent load.

**Fix:** Add a lock (e.g. `threading.Lock`) around `add_to_cache` and `save_cache` in `SQLCache`.

---

### BUG-04 — Single sub-task exception kills the entire query

**File:** `backend/main.py` lines 170–171  
**Severity:** Critical — no per-task isolation

```python
outputs = [f.result() for f in futures]  # re-raises any sub-task exception
```

`Future.result()` re-raises exceptions. A RAG retrieval failure or DB connection error in one sub-task propagates up and aborts the entire `run_query()` call. No fallback, no partial result, no user-facing error message.

**Fix:** Wrap each `f.result()` in a `try/except` and return a structured error dict for failed sub-tasks instead of propagating.

---

### BUG-05 — `load_cache()` leaves corrupted state on partial failure

**File:** `backend/sql/sql_cache.py` lines 74–86  
**Severity:** Critical — causes `IndexError` on subsequent searches

If the FAISS index file loads successfully but the JSON metadata file is corrupt, `self.index` is set to a valid object while `self.metadata` remains `[]`. Any subsequent `search_cache()` call will attempt to index into `self.metadata` using FAISS result indices, causing an `IndexError`.

**Fix:** Reset `self.index = None` in the `except` block of `load_cache()` to guarantee a clean state on any failure.

---

### BUG-06 — Empty SQL result set crashes JSON parse in ReAct agent

**File:** `backend/sql/react_agent.py` lines 358–362  
**Severity:** Critical — raises `ValueError`, triggers wasteful re-execution

```python
json_start = observation.index("[")   # ValueError if no "[" in string
json_part = observation[json_start:].split("\n(truncated")[0]
final_rows = json.loads(json_part)
```

When a query returns 0 rows, the observation string is something like `"Query executed successfully. Result set is empty (0 rows)."` — it contains no `[`. `.index("[")` raises `ValueError`, which is caught and triggers a redundant re-execution of the SQL query.

**Fix:** Check for the empty-result string before attempting JSON parsing:
```python
if "0 rows" in observation or "[" not in observation:
    final_rows = []
else:
    json_start = observation.index("[")
    ...
```

---

### BUG-07 — `ROUTER_BASE_URL` blindly appends `/v1`, causing 404

**File:** `backend/router/router.py` line 125  
**Severity:** Critical — router fails with 404 when user supplies `/v1` in the URL

```python
llm_kwargs["base_url"] = f"{config.ROUTER_BASE_URL.rstrip('/')}/v1"
```

If the user sets `ROUTER_BASE_URL=http://localhost:8080/v1`, this produces `http://localhost:8080/v1/v1` → HTTP 404. The router then silently falls back to keyword mode with no warning.

**Fix:** Strip any trailing `/v1` from the user-supplied URL before appending:
```python
base = config.ROUTER_BASE_URL.rstrip("/")
if not base.endswith("/v1"):
    base = f"{base}/v1"
llm_kwargs["base_url"] = base
```

---

### BUG-08 — Clarification JSON silently dropped if LLM adds extra keys

**File:** `backend/synthesis/synthesizer.py` lines 518–520  
**Severity:** Critical — clarification feature non-functional with most LLMs

```python
required_keys = {"needs_clarification", "reason", "question"}
if set(payload.keys()) != required_keys:   # strict equality
    return None
```

LLMs commonly add extra fields (`"confidence"`, `"examples"`, etc.) to JSON responses. Strict `==` comparison rejects any response with extra keys, silently discarding valid clarification requests and treating them as normal answers.

**Fix:** Use `required_keys.issubset(payload.keys())` instead of `==`.

---

### BUG-09 — SQLite thread-local connection has no rollback on error

**File:** `backend/sql/database.py` lines 96–105  
**Severity:** Critical — thread remains in broken transaction state

```python
cur = conn.cursor()
try:
    cur.execute(sql)
    return [dict(row) for row in cur.fetchall()]
except sqlite3.Error as e:
    raise  # no rollback
```

On any `sqlite3.Error`, the thread-local connection stays in an aborted transaction state. Subsequent queries on the same worker thread will fail with `"cannot execute query"` until the connection is closed/reopened.

**Fix:** Add `conn.rollback()` in the `except` block.

---

### BUG-10 — SQL refiner comparison uppercases string literals

**File:** `backend/sql/sql_agent.py` lines 476–480  
**Severity:** Critical — valid refiner edits to string literals are silently reverted

```python
if refined_sql and refined_sql.strip().upper() != cached_sql.strip().upper():
    sql_to_execute = refined_sql
elif refined_sql:
    sql_to_execute = cached_sql   # uses cached, not refined
```

`.upper()` is applied to the entire SQL string including string literals. If the refiner correctly changes `WHERE city = 'CAIRO'` to `WHERE city = 'Cairo'` (case-sensitive DB), the `.upper()` comparison sees them as identical and reverts to the cached (incorrect) SQL.

**Fix:** Compare the raw strings without `.upper()`:
```python
if refined_sql and refined_sql.strip() != cached_sql.strip():
```

---

## High Severity

These bugs produce incorrect behavior but do not crash immediately.

---

### BUG-11 — `"ok"` flag treats empty-string error as success

**File:** `backend/main.py` line 72  
**Severity:** High — pipeline reports success on certain error conditions

```python
"ok": not bool(pipeline_result.get("error"))  # "" → ok=True
```

If any upstream function sets `error=""` (empty string), the empty string is falsy and the pipeline reports success. Should compare to `None` explicitly.

**Fix:** `"ok": pipeline_result.get("error") is None`

---

### BUG-12 — Domain filter fallback discards valid filtered results

**File:** `backend/rag/retriever.py` lines 87–94  
**Severity:** High — domain-aware search degrades silently

```python
if len(faiss_results) < fetch_k:
    faiss_results = self.vector_store.search_by_vector(query_vector, fetch_k)
```

When the domain filter returns fewer than `fetch_k` results, the code throws them away and re-fetches without any filter. The filtered results (which are highly relevant) are lost entirely. Should merge filtered and unfiltered results, not replace.

---

### BUG-13 — Schema-staleness check fails open on DB error

**File:** `backend/sql/sql_agent.py` lines 156–170  
**Severity:** High — stale schema index used silently when DB is unreachable

If `_load_schema_dict()` raises an exception (e.g. DB connection failure), the exception propagates uncaught out of `_ensure_schema_index_exists`. The calling code does not handle it, so the old (stale) FAISS index is used for schema retrieval without any warning.

**Fix:** Wrap the staleness check in a `try/except` and log a warning if the live schema cannot be fetched.

---

### BUG-14 — Silent fallback masks real OpenAI errors in router

**File:** `backend/router/router.py` lines 276–293  
**Severity:** High — auth errors and rate limits silently produce wrong routes

```python
except Exception:
    pass  # intentional fallback to completion endpoint
```

This swallows authentication errors, rate limit responses, and network failures from the chat endpoint. The router falls back to keyword mode with no log entry, so the user receives a wrong route with no indication of what went wrong.

**Fix:** Log the exception before passing, and distinguish retriable errors from hard failures.

---

### BUG-15 — `SQL_REACT_MAX_ITERATIONS` contradicts documentation

**File:** `backend/config.py` line 126 vs `backend/sql/react_agent.py` line 309  
**Severity:** High — documentation/code mismatch causes confusion

`SQL_REACT_MAX_ITERATIONS = 3` but both `CLAUDE.md` and the `react_agent.py` module docstring state "max 6 iterations". The LangGraph `recursion_limit = max_iter * 3 = 9` steps total (each LLM call + tool call = 2 steps), yielding roughly 3–4 full ReAct cycles — not 6.

**Fix:** Align the config default with the documentation, or update the docs to reflect 3 iterations.

---

### BUG-16 — `get_shared_st_model()` accesses private LangChain attribute

**File:** `backend/models.py` line 43  
**Severity:** High — will break on LangChain upgrades

```python
return get_shared_hf_embeddings()._client
```

`._client` is a private attribute of `HuggingFaceEmbeddings`. LangChain has a history of renaming private attributes between minor versions (e.g., `_client` → `client`). This will throw `AttributeError` silently on upgrade.

**Fix:** Use the public API, or pin the LangChain version and document the dependency.

---

## Medium / Cosmetic

These are non-critical but worth addressing for correctness and maintainability.

---

### BUG-17 — Non-deterministic document ordering in loader

**File:** `backend/rag/loader.py` lines 57–62`  
`as_completed()` returns futures in order of completion, not submission. The resulting `documents` list ordering varies across runs, meaning chunk IDs (assigned by document order in `chunker.py`) differ between ingestion runs. External references to chunk IDs become stale.

---

### BUG-18 — E5 prefix detection too broad

**File:** `backend/rag/embedder.py` line 24`  
`"e5" in self.model_name` matches any model name containing `"e5"` (e.g., a hypothetical `"base5"` or `"e5x-large"`). Should use a more precise match (exact list or regex word boundary).

---

### BUG-19 — `__main__` block in `sql_agent.py` deletes schema index unconditionally

**File:** `backend/sql/sql_agent.py` lines 601–604`  
Running `python -m backend.sql.sql_agent` deletes the FAISS schema index before any test runs. Accidental execution in production destroys the deployment cache.

---

### BUG-20 — Dead code branches in `print_result`

**File:** `backend/main.py` lines 215–220`  
The `dict`/object branches in `print_result` are unreachable because `synthesize()` always returns a `str`. The dead code creates misleading impressions about possible return types.

---

### BUG-21 — PostgreSQL cursor commits twice when `commit=True`

**File:** `backend/sql/database.py` lines 139–144`  
`with conn:` already commits on clean exit. The explicit `conn.commit()` inside the block is redundant and runs twice when `commit=True`. Harmless for correctness but wasteful.

---

### BUG-22 — `OPENAI_API_KEY` defaults to empty string, produces misleading error

**File:** `backend/config.py` line 99`  
Missing API key causes `_generate_sql` to return `None`, which surfaces as `"LLM did not return a valid SELECT/WITH SQL query"` — misleading. The actual error (missing credentials) is hidden in logs.

---

## Priority Summary

| ID | File | Issue | Severity |
|----|------|--------|----------|
| BUG-01 | `rag/retriever.py:141` | BM25 hits filtered out — hybrid retrieval broken | Critical |
| BUG-02 | `synthesis/synthesizer.py:51` | Wrong model config key used | Critical |
| BUG-03 | `sql/sql_cache.py:88` | Thread-unsafe FAISS writes | Critical |
| BUG-04 | `main.py:171` | Sub-task exception kills whole query | Critical |
| BUG-05 | `sql/sql_cache.py:79` | Corrupted cache state on partial load failure | Critical |
| BUG-06 | `sql/react_agent.py:360` | Empty result set crashes JSON parse | Critical |
| BUG-07 | `router/router.py:125` | Double `/v1` in base URL → 404 | Critical |
| BUG-08 | `synthesis/synthesizer.py:519` | Strict key equality drops clarifications | Critical |
| BUG-09 | `sql/database.py:96` | No SQLite rollback on error | Critical |
| BUG-10 | `sql/sql_agent.py:476` | `.upper()` comparison reverts refiner edits | Critical |
| BUG-11 | `main.py:72` | Empty-string error treated as success | High |
| BUG-12 | `rag/retriever.py:87` | Domain filter fallback discards results | High |
| BUG-13 | `sql/sql_agent.py:156` | Stale schema used when DB unreachable | High |
| BUG-14 | `router/router.py:292` | Silent fallback masks OpenAI errors | High |
| BUG-15 | `config.py:126` | `MAX_ITERATIONS` contradicts docs | High |
| BUG-16 | `models.py:43` | Private LangChain attribute access | High |
| BUG-17 | `rag/loader.py:57` | Non-deterministic chunk IDs | Medium |
| BUG-18 | `rag/embedder.py:24` | E5 prefix detection too broad | Medium |
| BUG-19 | `sql/sql_agent.py:601` | `__main__` deletes schema index unconditionally | Medium |
| BUG-20 | `main.py:215` | Dead code branches in `print_result` | Medium |
| BUG-21 | `sql/database.py:139` | Double commit on PostgreSQL | Medium |
| BUG-22 | `config.py:99` | Misleading error when API key missing | Medium |
