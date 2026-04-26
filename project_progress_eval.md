 Codebase Audit: Graduation Project Proposal Alignment

  ---
  Step 1 — Codebase Map

  ┌─────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────┐
  │        Group        │                                           Files                                           │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Routing             │ backend/router/router.py — QueryRouter (4 modes: keyword, llm, semantic, decompose)       │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Unstructured        │ backend/rag/embedder.py, vector_store.py, retriever.py, reranker.py, chunker.py,          │
  │ pipeline            │ loader.py, bm25_index.py                                                                  │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Structured pipeline │ backend/sql/sql_cache.py (FAISS semantic cache), sql_agent.py:_refine_sql_from_cache()    │
  │  — Fast Track       │                                                                                           │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Structured pipeline │ backend/sql/sql_agent.py:run_table_rag_pipeline(), table_rag.py, react_agent.py           │
  │  — Reasoning Track  │                                                                                           │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Synthesis           │ backend/synthesis/synthesizer.py — ResponseSynthesizer                                    │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Orchestrator        │ backend/main.py — AdaptiveAgenticRAGSystem.run_query()                                    │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Config              │ backend/config.py, backend/models.py                                                      │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ UI                  │ frontend/src/App.tsx (placeholder only)                                                   │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Router evaluation   │ evaluation/Router/eval_router.py, test_cases.json (240 queries), decomp_cases.json,       │
  │                     │ results/                                                                                  │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Structured          │ evaluation/Structured/bird/bird_eval.py, spider_eval.py, evaluation.py, results/,         │
  │ evaluation          │ schema_cache/                                                                             │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Unstructured        │ evaluation/Unstructured/run_ragbench_eval.py, run_nq_eval.py, run_squad_rag_eval.py,      │
  │ evaluation          │ run_config_eval.py, run_t_series.py, run_ragbench_series.py, results/ (100+ result files) │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Tests               │ tests/test_sql_pipeline.py, tests/test_react_agent.py                                     │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Utilities           │ ingest_data.py, seed_sql_cache.py, sql_data/fintech_db_setup_pg.py,                       │
  │                     │ scratch/test_dynamic_cache.py                                                             │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Data                │ data/, benchmark/, evaluation/Unstructured/evaluation/ (corpus + eval sets for 3          │
  │                     │ benchmarks)                                                                               │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ Orphaned / internal │ backend/evaluation/retrieval_comparison.py — not connected to any pipeline                │
  └─────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Step 2 — Component Implementation Status

  1. Router / Decomposition Agent

  Status: IMPLEMENTED — exceeds proposal scope

  The proposal described a "Zero-Shot LLM classifier" with three routes (sql/text/hybrid). The implementation has four
  distinct router modes, controlled by config.ROUTER_MODE:

  - keyword — router.py:42-59 — rule-based token matching against SQL_KEYWORDS and TEXT_KEYWORDS sets; returns "sql",
  "text", or "hybrid".
  - llm — router.py:61-83 — single-classification LLM call to local llama.cpp or OpenAI API. The prompt at line 63–70:
  "Classify the following user query into exactly one label: sql, text, or hybrid. Return only the label."
  - semantic — router.py:85-131 — cosine similarity of the query embedding against per-route seed clusters (30 SQL
  seeds, 27 text seeds defined in config.py:146-226). Falls back to "hybrid" when the margin between top two routes is
  below SEMANTIC_ROUTER_HYBRID_MARGIN = 0.015.
  - decompose (default) — router.py:172-240 — LangChain ChatOpenAI call with a structured system prompt (lines 183-202)
  that asks the model to decompose the query into multiple sub-tasks, each labeled "sql" or "text". Returns a list of
  SubTask objects. The overall route is derived by route_from_subtasks() (lines 242-255).

  Wiring: main.py:82-141 dispatches all four modes. The decompose path calls _execute_subtasks() (lines 143-187) via a
  ThreadPoolExecutor(max_workers=3) to run sub-tasks concurrently.

  Key difference from proposal: decompose mode never returns "hybrid" as a route — it splits the query into separate
  "sql" and "text" sub-tasks and runs them independently. True hybrid (both pipelines on the same sub-query) only occurs
   in the non-decompose modes.

  ---
  2. Unstructured Pipeline

  Status: FULLY IMPLEMENTED — exceeds proposal scope

  ┌──────────────────┬───────────────────────────────────────┬─────────────────────────────────────────────────────┐
  │    Component     │               Location                │                       Details                       │
  ├──────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Chunk embedding  │ backend/rag/embedder.py               │ intfloat/multilingual-e5-base, E5 query:/passage:   │
  │                  │                                       │ prefixes                                            │
  ├──────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ FAISS vector     │ backend/rag/vector_store.py           │ Cosine distance via normalized L2, Windows ctypes   │
  │ store            │                                       │ short-path workaround                               │
  ├──────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Top-k retrieval  │ backend/rag/retriever.py:77-191       │ RagRetriever.retrieve()                             │
  ├──────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Cross-encoder    │ backend/rag/reranker.py               │ BAAI/bge-reranker-base, RAG_RERANK_POOL=100         │
  │ reranking        │                                       │ candidates fed before selecting top-k               │
  ├──────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Hybrid           │ backend/rag/bm25_index.py +           │ Reciprocal Rank Fusion (_rrf(), RAG_RRF_K=60).      │
  │ BM25+FAISS       │ retriever.py:114-130                  │ Activated when RAG_RETRIEVAL_MODE="hybrid"          │
  └──────────────────┴───────────────────────────────────────┴─────────────────────────────────────────────────────┘

  Not in proposal: BM25 hybrid mode was added beyond the original design.

  ---
  3. Structured Pipeline — Fast Track (SQL Cache)

  Status: IMPLEMENTED — mechanism differs from proposal

  Proposal described: "FAISS SQL cache, few-shot retrieval, template filling"

  Actual implementation (backend/sql/sql_cache.py + sql_agent.py:365-485):

  - SQLCache class stores (question, sql, schema) triples in a FAISS IndexFlatIP index.
  - check_cache_hit() at sql_cache.py:136 computes cosine similarity; threshold is config.SQL_CACHE_HIT_THRESHOLD = 0.85
   (config.py line 296).
  - On a cache hit: run_table_rag_pipeline() in sql_agent.py:377-485 calls _refine_sql_from_cache() (lines 165-224),
  which passes the cached SQL, the original question, similar queries (dynamic few-shot), and the new question to the
  LLM to make "the smallest possible edits."
  - The result is labeled "path": "fast" (line 484).

  Key difference: Instead of static template filling, the Fast Track uses LLM refinement with dynamic few-shot examples
  pulled from the cache's top-k results. This is semantically equivalent to few-shot retrieval + adaptation, not
  template filling. The spirit is preserved, the mechanism is more robust.

  ---
  4. Structured Pipeline — Reasoning Track (TableRAG + ReAct)

  Status: FULLY IMPLEMENTED

  ┌──────────────────┬────────────────────────────────────┬─────────────────────────────────────────────────────────┐
  │    Component     │              Location              │                         Details                         │
  ├──────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ TableRAG schema  │                                    │ retrieve_relevant_schema() — builds multi-granularity   │
  │ pruning          │ backend/sql/table_rag.py:252-341   │ chunks (table, column, value), FAISS IndexFlatIP,       │
  │                  │                                    │ max-pool scoring, FK expansion with 3 gates             │
  ├──────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ LangGraph ReAct  │                                    │ run_react_sql_agent() — create_react_agent with         │
  │ agent            │ backend/sql/react_agent.py:394-595 │ execute_sql and schema_lookup tools, recursion_limit =  │
  │                  │                                    │ max_iter * 3, up to 10 iterations                       │
  ├──────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │                  │                                    │ If execute_sql returns ERROR on ≥4 calls, the agent     │
  │ Self-correction  │ react_agent.py:557-588             │ retries with the full DB schema appended (_retry=True   │
  │                  │                                    │ guard prevents infinite recursion)                      │
  ├──────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Single-pass      │ sql_agent.py:504-525               │ If ReAct produces no output, run_sql_agent() is used as │
  │ fallback         │                                    │  fallback                                               │
  └──────────────────┴────────────────────────────────────┴─────────────────────────────────────────────────────────┘

  Path labels: "react" when ReAct agent succeeded, "agent" when single-pass was used, "fast" when cache hit.

  ReAct system prompt (react_agent.py:102-180): 3-hop reasoning structure (UNDERSTAND → EXPLORE VALUES → BUILD &
  VERIFY), 11 strict rules including read-only enforcement, alias consistency, no hardcoded values, dynamic TOP-N.

  ---
  5. Synthesis Agent

  Status: FULLY IMPLEMENTED

  backend/synthesis/synthesizer.py — ResponseSynthesizer:

  - synthesize() (line 61) accepts sql_result, rag_result, and subtask_results from any pipeline combination.
  - _synthesize_with_llm() (line 99) builds a source-mode-tailored system prompt (sql_only, text_only, hybrid,
  decompose, empty) and invokes ChatOpenAI.
  - Clarification JSON parsing at line 501 — if the LLM cannot answer, it returns {"needs_clarification": true,
  "reason": "...", "question": "..."}.
  - Inline source citation extraction ([source: <name>]) at line 452.
  - Deterministic formatter fallback at line 563 — renders a structured text table when LLM synthesis fails entirely.
  - SynthesisResult.latency field tracks synthesis time (line 132).

  ---
  6. User Interface

  Status: NOT IMPLEMENTED

  frontend/src/App.tsx contains exactly this:

  function App() {
    return (
      <main style={{ fontFamily: 'sans-serif', padding: '2rem' }}>
        <h1>Adaptive Agentic RAG</h1>
        <p>Frontend coming soon.</p>
      </main>
    );
  }

  The React/Vite/TypeScript scaffold is configured (package.json, vite.config.ts, tsconfig.json, services/api.ts type
  stubs), but no QueryPage, IngestPage, or any functional component exists. The TODO comment in the file explicitly
  lists planned pages that were never built. The proposal mentioned Streamlit or Chainlit; neither was attempted — the
  team started a custom React frontend and stopped.

  ---
  7. End-to-End Hybrid Flow

  Status: IMPLEMENTED (with a decompose-mode caveat)

  - In llm, keyword, semantic modes: main.py:122-126 — when route == "hybrid", both _run_sql_pipeline() and
  self.retriever.retrieve() are called sequentially, then passed together to synthesizer.synthesize().
  - In decompose mode (default): The LLM never returns "hybrid" as a sub-task route — it only produces "sql" or "text"
  sub-tasks. Hybrid behavior is achieved by having separate sql and text sub-tasks that are merged at synthesis. This
  produces the same output but is structurally different from a true hybrid call.

  ---
  Step 3 — Research Questions

  RQ1: Latency measurement — Fast Track vs Reasoning Track

  Status: PARTIALLY IMPLEMENTED — path tracking exists, dedicated comparison experiment missing

  What is implemented:
  - run_table_rag_pipeline() in sql_agent.py returns "path" ("fast" / "react" / "agent") and "latency" (wall-clock
  seconds via time.time()) for every query (lines 476 and 540).
  - _run_sql_pipeline() in main.py:67-80 extracts this latency into the result dict.
  - The __main__ block in main.py:236-280 prints End-to-End Latency for 6 test queries.
  - SynthesisResult.latency tracks synthesis time separately.

  What is missing:
  There is no evaluation script that systematically runs a labeled set of queries through both the Fast Track (cache
  hit) and the Reasoning Track (cache miss) and produces a head-to-head latency comparison table. The bird_eval.py has
  three ablation modes (full_single_pass, tablerag_single_pass, tablerag_react) but these measure SQL accuracy, not
  cache-hit vs. ReAct latency. No script in evaluation/Structured/ or anywhere else deliberately seeds the cache, then
  runs queries to force hits, then measures and compares latencies.

  ---
  RQ2: Router classification accuracy — test set of 30 mixed queries

  Status: FULLY IMPLEMENTED — test set is 240 queries (exceeds proposal)

  evaluation/Router/eval_router.py + evaluation/Router/test_cases.json:

  - Test set: 240 queries — 80 sql, 80 text, 80 hybrid. Each has "difficulty" (easy/medium/hard) and "description"
  annotating which keywords fire and where the keyword router fails.
  - Three routers compared: keyword (rule-based), semantic (embedding cosine), zeroshot (decompose_with_zeroshot +
  route_from_subtasks).
  - Metrics computed (_compute_metrics(), lines 147-201): accuracy, macro precision/recall/F1, weighted F1, Cohen's κ,
  MCC, per-class F1/precision/recall, confusion matrix, accuracy by difficulty.
  - Results already exist in evaluation/Router/results/: classification_keyword.json, classification_semantic.json,
  classification_zeroshot.json, classification_comparison.json, decomposition.json.
  - Decomposition quality evaluated separately on decomp_cases.json with metrics: route_validity, atomicity, coverage,
  no_raw_sql, quality_score.

  Proposal said 30 queries; implementation has 240. This is a strength.

  ---
  RQ3: SQL accuracy — with vs without schema pruning

  Status: IMPLEMENTED via BIRD benchmark ablation

  evaluation/Structured/bird/bird_eval.py provides three ablation modes:

  - full_single_pass — Full schema passed to LLM, no TableRAG pruning. This is the baseline without schema pruning.
  - tablerag_single_pass — TableRAG prunes the schema, then single-pass SQL agent generates. Direct comparison to full
  schema.
  - tablerag_react — TableRAG pruning + ReAct agent. Tests pruning + iterative reasoning.

  The BIRD evaluation script calls the Spider evaluation_ex.py harness to compute Execution Accuracy across
  easy/medium/hard/extra difficulty levels. Results are written to evaluation/Structured/results_bird_*.txt.

  Gap: The BIRD dev databases (dev_databases/) are not present in the repository (they would be too large). The
  evaluation script references BIRD_DIR / "dev_databases" / "dev_databases" which does not exist in the repo as checked
  by the file map. This means RQ3 cannot be reproduced without downloading the BIRD dataset separately. Existing result
  files (results_bird_*.txt) are present, suggesting the evaluation was run externally.

  ---
  Step 4 — Gaps and Risks

  Missing from the codebase entirely

  ┌───────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────┐
  │       Item        │                                          Severity                                           │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Functional UI     │ Critical — App.tsx is a 15-line placeholder. No query interface, no file ingest page,       │
  │                   │ nothing usable.                                                                             │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ RQ1 latency       │ High — there is no script that systematically forces cache hits vs. misses on the same      │
  │ comparison script │ query set and measures latency differences. The data needed (path label + latency per       │
  │                   │ query) is logged but never aggregated.                                                      │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ BIRD dev          │ Medium — required for RQ3 reproduction but not committed to the repo. Pre-existing result   │
  │ databases         │ files suggest prior runs, but the experiment cannot be re-run without the dataset.          │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ End-to-end hybrid │ Medium — no evaluation that tests a mixed query set (requiring both SQL and text retrieval) │
  │  evaluation       │  end-to-end through the full pipeline including synthesis.                                  │
  └───────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────┘

  Implemented but not connected (orphaned code)

  ┌────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┐
  │                    File                    │                                Issue                                │
  ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ backend/evaluation/retrieval_comparison.py │ Not imported by any pipeline module, not mentioned in CLAUDE.md,    │
  │                                            │ not called from any evaluation script. Likely an early experiment.  │
  ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ scratch/test_dynamic_cache.py              │ Scratch file not connected to anything.                             │
  └────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────┘

  Implemented differently than proposed in a way that affects research questions

  Component: Fast Track mechanism
  Difference: Proposal: template filling. Actual: LLM refinement. Both are valid; however the "fast" label in path only
    applies to cache hits, not to a distinct architectural path that can be cleanly compared against the Reasoning Track

    on a per-query basis.
  RQ Impact: RQ1: comparing "fast" vs "reasoning" latency requires knowing which path each query took — the data is
  there
    but no aggregation script exists.
  ────────────────────────────────────────
  Component: Decompose mode hybrid
  Difference: In decompose mode (the default), queries with both sql and text intent produce separate sub-tasks rather
    than a single hybrid route. The synthesis receives subtask_results instead of combined sql_result + rag_result.
  RQ Impact: Affects any end-to-end hybrid evaluation — the "hybrid flow" as described in the proposal only exists in
    non-decompose modes.

  Present in the codebase but not evaluated/tested

  ┌────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────┐
  │            Item            │                                        Gap                                         │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ RAG_RETRIEVAL_MODE =       │ The unstructured evaluation scripts appear to use FAISS-only mode in most configs. │
  │ "hybrid" (BM25+FAISS)      │  The BM25 hybrid mode is implemented and configurable but no systematic ablation   │
  │                            │ comparing BM25 vs FAISS-only appears in the evaluation results.                    │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ route_with_semantic()      │ Implemented, eval results exist in classification_semantic.json, so this is        │
  │ (semantic router)          │ evaluated. ✓                                                                       │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ SQL_REACT_ENABLED = False  │ Tested manually in the __main__ block of sql_agent.py but not in the pytest suite. │
  │ path                       │                                                                                    │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ backend/models.py          │ No unit tests cover the thread-safety guarantees.                                  │
  │ singleton factories        │                                                                                    │
  └────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Step 5 — Completion Estimate per Work Package

  WP I — Routing + Unstructured Retrieval: 88% complete

  ┌──────────────────────────────────────────────────────────┬────────────┐
  │                      Sub-component                       │   Status   │
  ├──────────────────────────────────────────────────────────┼────────────┤
  │ Keyword router                                           │ ✅ Done    │
  ├──────────────────────────────────────────────────────────┼────────────┤
  │ LLM router (llm mode)                                    │ ✅ Done    │
  ├──────────────────────────────────────────────────────────┼────────────┤
  │ Semantic router                                          │ ✅ Done    │
  ├──────────────────────────────────────────────────────────┼────────────┤
  │ Decompose router (default)                               │ ✅ Done    │
  ├──────────────────────────────────────────────────────────┼────────────┤
  │ Router evaluation (240 queries, 3 routers, full metrics) │ ✅ Done    │
  ├──────────────────────────────────────────────────────────┼────────────┤
  │ FAISS vector store + embedder                            │ ✅ Done    │
  ├──────────────────────────────────────────────────────────┼────────────┤
  │ Recursive/fixed chunker                                  │ ✅ Done    │
  ├──────────────────────────────────────────────────────────┼────────────┤
  │ Cross-encoder reranking                                  │ ✅ Done    │
  ├──────────────────────────────────────────────────────────┼────────────┤
  │ BM25 hybrid retrieval (RRF)                              │ ✅ Done    │
  ├──────────────────────────────────────────────────────────┼────────────┤
  │ Unstructured eval (RAGBench, SQuAD, NQ, 100+ results)    │ ✅ Done    │
  ├──────────────────────────────────────────────────────────┼────────────┤
  │ End-to-end hybrid eval (SQL + RAG together)              │ ❌ Missing │
  └──────────────────────────────────────────────────────────┴────────────┘

  ---
  WP II — Structured Retrieval + Dual Path: 78% complete

  ┌──────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────┐
  │                        Sub-component                         │                      Status                      │
  ├──────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ TableRAG schema pruning (multi-granularity, FK expansion)    │ ✅ Done                                          │
  ├──────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ Single-pass SQL agent                                        │ ✅ Done                                          │
  ├──────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ LangGraph ReAct agent (10 iterations, tools,                 │ ✅ Done                                          │
  │ self-correction)                                             │                                                  │
  ├──────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ SQL Cache (FAISS semantic cache,                             │ ✅ Done                                          │
  │ SQL_CACHE_HIT_THRESHOLD=0.85)                                │                                                  │
  ├──────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ LLM cache refiner (few-shot from cache)                      │ ✅ Done                                          │
  ├──────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ BIRD evaluation (3 ablation modes, accuracy metrics)         │ ✅ Done                                          │
  ├──────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ Spider evaluation                                            │ ✅ Done                                          │
  ├──────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ RQ1: Dedicated Fast vs Reasoning Track latency experiment    │ ❌ Missing                                       │
  ├──────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ BIRD dev databases in repo                                   │ ❌ Missing (external dataset, results            │
  │                                                              │ pre-computed)                                    │
  └──────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────┘

  ---
  WP III — Synthesis + UI: 52% complete

  ┌───────────────────────────────────────────────┬───────────────────────────────────────┐
  │                 Sub-component                 │                Status                 │
  ├───────────────────────────────────────────────┼───────────────────────────────────────┤
  │ LLM synthesis (source-mode-specific prompts)  │ ✅ Done                               │
  ├───────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Clarification request handling                │ ✅ Done                               │
  ├───────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Inline source citation                        │ ✅ Done                               │
  ├───────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Synthesis latency tracking                    │ ✅ Done                               │
  ├───────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Deterministic fallback formatter              │ ✅ Done                               │
  ├───────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Frontend UI (QueryPage, IngestPage)           │ ❌ Not implemented — placeholder only │
  ├───────────────────────────────────────────────┼───────────────────────────────────────┤
  │ React/TypeScript scaffold + API service stubs │ ⚠️ Skeleton exists, no pages          │
  └───────────────────────────────────────────────┴───────────────────────────────────────┘

  ---
  WP IV — Testing + Validation: 72% complete

  ┌──────────────────────────────────────────────────────────────┬────────────────────────────────────┐
  │                        Sub-component                         │               Status               │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ Router eval (240 queries, accuracy/F1/κ/MCC, all 3 routers)  │ ✅ Done                            │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ Decomposition quality eval (atomicity, coverage, no_raw_sql) │ ✅ Done                            │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ Structured eval: BIRD + Spider ablations                     │ ✅ Done                            │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ Unstructured eval: RAGBench + SQuAD + NQ                     │ ✅ Done                            │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ Unit tests (SQL pipeline, ReAct agent)                       │ ✅ Done                            │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ RQ1: Fast vs Reasoning Track latency comparison experiment   │ ❌ Missing                         │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ End-to-end hybrid query evaluation                           │ ❌ Missing                         │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ Integration tests (live DB + LLM)                            │ ⚠️ Markers exist, coverage unclear │
  └──────────────────────────────────────────────────────────────┴────────────────────────────────────┘

  ---
  Summary

  The core backend is in very good shape — the Router (4 modes), Unstructured Pipeline (FAISS+BM25+reranker), Structured
   Pipeline (TableRAG + FAISS cache + LangGraph ReAct), and Synthesizer are all fully implemented and wired together.
  Evaluation coverage is strong for routing (240 queries) and both structured (BIRD/Spider ablations) and unstructured
  (RAGBench/SQuAD/NQ) pipelines.

  The two most significant gaps relative to the proposal are: (1) the UI is a non-functional placeholder, and (2) there
  is no dedicated RQ1 latency comparison experiment — the per-query path and latency data is being produced by the
  pipeline but never aggregated into a comparative table. Both are completable quickly; the UI is the larger effort.

✻ Brewed for 4m 2s

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
> 
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ? for shortcuts
  ├───────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Inline source citation                        │ ✅ Done                               │
  ├───────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Synthesis latency tracking                    │ ✅ Done                               │
  ├───────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Deterministic fallback formatter              │ ✅ Done                               │
  ├───────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Frontend UI (QueryPage, IngestPage)           │ ❌ Not implemented — placeholder only │
  ├───────────────────────────────────────────────┼───────────────────────────────────────┤
  │ React/TypeScript scaffold + API service stubs │ ⚠️ Skeleton exists, no pages          │
  └───────────────────────────────────────────────┴───────────────────────────────────────┘

  ---
  WP IV — Testing + Validation: 72% complete

  ┌──────────────────────────────────────────────────────────────┬────────────────────────────────────┐
  │                        Sub-component                         │               Status               │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ Router eval (240 queries, accuracy/F1/κ/MCC, all 3 routers)  │ ✅ Done                            │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ Decomposition quality eval (atomicity, coverage, no_raw_sql) │ ✅ Done                            │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ Structured eval: BIRD + Spider ablations                     │ ✅ Done                            │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ Unstructured eval: RAGBench + SQuAD + NQ                     │ ✅ Done                            │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ Unit tests (SQL pipeline, ReAct agent)                       │ ✅ Done                            │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ RQ1: Fast vs Reasoning Track latency comparison experiment   │ ❌ Missing                         │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ End-to-end hybrid query evaluation                           │ ❌ Missing                         │
  ├──────────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ Integration tests (live DB + LLM)                            │ ⚠️ Markers exist, coverage unclear │
  └──────────────────────────────────────────────────────────────┴────────────────────────────────────┘

  ---
  Summary

  The core backend is in very good shape — the Router (4 modes), Unstructured Pipeline (FAISS+BM25+reranker), Structured Pipeline (TableRAG + FAISS cache + LangGraph ReAct), and Synthesizer are all fully
  implemented and wired together. Evaluation coverage is strong for routing (240 queries) and both structured (BIRD/Spider ablations) and unstructured (RAGBench/SQuAD/NQ) pipelines.

  The two most significant gaps relative to the proposal are: (1) the UI is a non-functional placeholder, and (2) there is no dedicated RQ1 latency comparison experiment — the per-query path and latency data
  is being produced by the pipeline but never aggregated into a comparative table. Both are completable quickly; the UI is the larger effort.

✻ Brewed for 4m 2s

※ recap: Audited the codebase against the graduation project proposal. Everything is implemented except the UI (placeholder only) and a Fast Track vs Reasoning Track latency comparison script for RQ1. Build
  those two next. (disable recaps in /config)

> Context for evaluation:
  I am a 4th year computer engineering undergraduate at Kadir Has University (Istanbul, Turkey) working on my graduation project. I need an honest, calibrated assessment of whether this project is strong,
  average, or weak for a bachelor's-level computer engineering thesis — not encouragement, just an accurate benchmark.
  The project:
  We built an Adaptive Agentic RAG system that answers queries requiring both structured data (SQL databases) and unstructured data (PDF documents). The core idea is a routing agent that classifies user
  queries as SQL-only, text-only, or hybrid, then dispatches to the appropriate pipeline.
  Key components implemented:

  Zero-shot LLM router that classifies queries into structured / unstructured / hybrid
  Unstructured pipeline: FAISS vector search + BGE cross-encoder reranker + GPT-4o-mini generation
  Structured pipeline dual-path: FAISS SQL cache (fast track) + TableRAG schema pruning + LangChain SQL Agent (reasoning track)
  Synthesis agent that combines outputs from both pipelines
  Streamlit/Chainlit web UI

  Evaluation work done:

  14+ controlled ablation experiments on the unstructured pipeline varying chunk size, top-k, reranker model, reranker pool size — each changing exactly one variable
  Benchmarked on RAGBench TechQA subset (50 questions, ~150 IBM technical documents)
  Evaluated with RAGAS (answer relevancy, faithfulness, context precision, context recall)
  Conducted a methodology audit that identified evaluation bias in RAGBench+RAGAS — specifically that NO_GOLD questions (unanswerable by design) were being penalised identically to wrong answers, and that
  gold_hit=True is not a reliable signal of context quality
  Identified and fixed an over-conservative LLM abstention prompt through systematic false-abstention analysis
  Best config: K=15, chunk_size=1000, rerank_pool=25, achieving 40/50 questions answered, 20% not-found rate, 0.685 overall faithfulness
  Router evaluated on 30 mixed queries (structured / unstructured / hybrid)
  SQL pipeline evaluated on execution accuracy with vs without TableRAG schema pruning
  Latency comparison between fast track (cache hit) and reasoning track (full agent)

  Technologies used:

  Python, FAISS, LangChain, OpenAI API (gpt-4o-mini), RAGAS, HuggingFace (BGE-M3 embeddings, bge-reranker-base), TableRAG, Streamlit

  Team: 4 people, ~700 total hours across 5 months
  Please evaluate:

  Scope — Is this too much, too little, or appropriately scoped for a 4-person undergraduate thesis over 5 months?
  Technical depth — Is the ablation study methodology (single-variable controlled experiments, not-found audit, RAGAS methodology critique) above average, average, or below average for an undergraduate
  project?
  Novelty — Is there anything genuinely novel here, or is this purely an integration project assembling existing tools?
  Weaknesses an examiner would likely challenge — What are the 3 most likely questions or criticisms from a technical examiner?
  Honest grade range — If presented well, what grade range does this project deserve at a Turkish engineering faculty (AA / BA / BB / CB / CC scale), and what would it need to reach AA?

  Be direct. If this is average, say it's average. If the ablation work elevates it, say why. If the sample size of 50 questions is a weakness, say so.

● Honest Calibration: Bachelor Thesis Assessment

  I'll evaluate this against what I know from both what you described and what the codebase audit actually shows. Where the two diverge, I'll say so directly, because an examiner will find the same gaps.

  ---
  First: A Discrepancy You Need to Address

  You listed "Streamlit/Chainlit web UI" as a delivered component. The codebase has a React scaffold with App.tsx containing exactly:

  <h1>Adaptive Agentic RAG</h1>
  <p>Frontend coming soon.</p>

  There is no UI. This is not a minor gap — if you present this to an examiner as delivered, you have a credibility problem the moment they ask for a demo. The router eval test set also has 240 queries in the
  file, not 30. These numbers matter if you write them in a report. Fix the presentation before you fix anything else.

  ---
  1. Scope

  Assessment: Appropriately scoped, possibly slightly over-broad for the depth achieved.

  Four people, five months, a dual-pipeline RAG system with ablations and benchmark evaluation is a reasonable undergraduate scope. The mistake most teams make at this scope is building wide and shallow — many
   components, none of them rigorously understood. You partially avoided this on the unstructured side (genuine ablation discipline) but fell into it on the UI and the RQ1 latency experiment, both of which are
   listed as goals but not completed.

  The structured pipeline (TableRAG + LangGraph ReAct) is genuinely complex to wire together. The fact that it works end-to-end with self-correction and schema FK expansion is a meaningful engineering
  accomplishment. But because you are using LangGraph, LangChain, FAISS, RAGAS, and the OpenAI API as black boxes, the "scope" is partially infrastructure wrangling, not research. That is normal for
  undergraduate work — just be honest with yourself about which parts of the system you deeply understand versus assembled.

  ---
  2. Technical Depth

  Assessment: Above average for an undergraduate thesis, specifically because of the evaluation methodology — not because of the system design.

  The ablation study approach is the strongest part of this project. Changing exactly one variable per experiment, maintaining a consistent evaluation set, and then catching a systematic bias in your own
  evaluation methodology (the NO_GOLD problem) is the kind of critical thinking that distinguishes a good thesis from an average one. Most undergraduate projects run one configuration, get a number, and
  present it. You ran 14+, noticed the results were misleading, and investigated why. That is scientifically disciplined.

  The false-abstention analysis (identifying that the LLM was over-refusing and systematically auditing it) is also genuinely mature. That is not something most 4th-year students think to do.

  However, three things limit how much credit the methodology earns:

  - N=50 is too small. At 50 questions, a difference of 2–3 questions is 4–6 percentage points. Your confidence intervals are wide enough that many of your ablation conclusions could reverse with a different
  50-question sample. A reviewer who knows statistics will say this immediately.
  - No external baseline. You compare configurations against each other (best chunk size, best K, best reranker pool). You never compare your best system against naive RAG (FAISS top-k, no reranker, no cache,
  direct GPT-4o-mini), or against a BM25-only retriever, or against GPT-4o-mini with the full document context. Without a baseline, "K=15 with reranker achieves 0.685 faithfulness" means nothing on its own.
  What does a simple system get? 0.60? 0.70? If a simpler system matches you, your architecture is not justified.
  - RAGAS is itself contested. Your critique of RAGAS is valid, but an examiner could argue that your fix (adjusting abstention behavior) games the metric rather than solving the evaluation problem. You noted
  the bias but then continued using RAGAS as your primary metric. A stronger position would be to supplement RAGAS with human evaluation on a sample — even 20 manually judged questions — or with a
  reference-based metric that is not LLM-judged.

  ---
  3. Novelty

  Assessment: Integration project with one above-average design decision and one genuine methodological contribution. No algorithmic novelty.

  There is no new algorithm here. FAISS, BGE embeddings, cross-encoder reranking, LangGraph ReAct agents, TableRAG schema pruning — every component is an existing library or published technique. This is
  standard for undergraduate work and not a criticism by itself. A thesis can be valuable without novelty if the engineering is solid and the evaluation is rigorous.

  The one design decision that goes slightly beyond off-the-shelf assembly is the dual-path structured pipeline: a FAISS semantic cache (fast track) with LLM refinement on cache hits, falling back to TableRAG
  + ReAct on cache misses. This is a practical architecture that addresses latency in a non-trivial way and has a clear research question attached to it (RQ1). The problem is that the latency experiment was
  not actually completed — you have the path labels and latency logged per query, but no aggregated comparison table.

  The evaluation methodology critique (identifying NO_GOLD bias in RAGAS) is the closest thing to a genuine novel contribution. It is a methodological observation that could be useful to others evaluating RAG
  systems on similar benchmarks. If you write this up clearly, it is the most citable and defensible part of the thesis.

  What is not novel and should not be overclaimed: the router architecture (LLM classification of query intent is described in dozens of RAG papers from 2023–2024), the hybrid retrieval (BM25+FAISS via RRF is
  standard since DPR-hybrid papers), and the synthesis agent (prompt engineering on top of GPT-4o-mini).

  ---
  4. Weaknesses an Examiner Will Challenge

  Challenge 1: "Your evaluation set has 50 questions. How do you know your conclusions generalize?"

  This is the first and most damaging question. You do not have a good answer with 50 questions. The standard response in the NLP literature for a dataset this small is to run multiple random samples and
  report variance, or to calculate confidence intervals explicitly. If you used the same 50 questions for all 14 ablations, you also have a multiple comparisons problem — some of your "best configs" are
  overfitted to that specific 50.

  What to prepare: Calculate standard deviations across your RAGAS metrics. Acknowledge the limitation explicitly in the thesis and discuss what a larger study would require. Do not try to defend 50 questions
  as sufficient — acknowledge it and explain why it was the practical limit.

  Challenge 2: "Where is the system comparison against a baseline? How do we know the complexity is justified?"

  Your ablations compare configurations of your system against each other. No examiner who has read RAG literature will accept "our best config is better than our worst config" as a conclusion. They will ask:
  compared to what? A retrieval-augmented system that uses BM25 only, no reranker, and a direct prompt to GPT-4o-mini is a 30-line Python script. If it scores 0.65 faithfulness and your elaborate system scores
   0.685, you have built a lot of infrastructure for 2 percentage points.

  What to prepare: Run at least one ablation that removes your reranker and semantic cache entirely (pure FAISS top-k → GPT-4o-mini) and report the result honestly. If your system is better, this strengthens
  the thesis significantly. If it is not much better, that is also a valid finding — it means the marginal contribution of each component is small, which is a legitimate research result.

  Challenge 3: "You describe hybrid queries as a core use case, but in your default mode (decompose), you never route to 'hybrid' — you run separate sub-tasks. Is this actually hybrid retrieval, or just
  sequential retrieval with merging?"

  This is a precise technical criticism. In decompose mode, the LangChain LLM only emits "sql" or "text" sub-tasks. Hybrid routing (both pipelines on the same query) only occurs in the keyword, llm, or
  semantic modes, none of which are the default. Your synthesis agent handles this correctly, but the framing of "hybrid query routing" overstates what is happening in the default pipeline.

  What to prepare: Either switch the default mode to llm or semantic for demos involving true hybrid queries, or be precise in the thesis that decompose mode handles mixed-intent via sub-task splitting rather
  than dual-pipeline execution.

  ---
  5. Honest Grade Range

  Current state, presented honestly: BB to BA

  Why not higher:
  - No working UI (you claimed one)
  - N=50 without confidence intervals
  - No baseline comparison
  - RQ1 latency experiment not completed
  - No novelty beyond integration and methodology critique

  Why not lower:
  - The ablation discipline is genuinely above average
  - The RAGAS methodology critique shows real critical thinking
  - The dual-path structured pipeline is correctly implemented end-to-end
  - The codebase is actually clean and works (not always the case at this level)
  - The evaluation guide and experiment infrastructure show systematic thinking

  To reach AA, specifically:

  1. Do the baseline comparison. One run of naive FAISS-top-k-GPT-4o-mini on the same 50 questions. This single addition would significantly strengthen the ablation section because it gives your numbers
  context.
  2. Complete or remove the UI. Either spend 2 days wiring a Streamlit interface to backend/main.py (the API layer exists, it is a thin frontend problem), or remove the claim entirely from the presentation. Do
   not present something that does not exist.
  3. Complete RQ1. The path label and latency are already being logged per query. You need a script that seeds the SQL cache with a test set, then runs the same queries to force cache hits, records latency for
   both conditions, and produces a comparison table. This is probably 2–4 hours of work given the infrastructure is there.
  4. Add confidence intervals to your ablation table. Even bootstrapped CIs from 50 questions would show you are thinking statistically. This directly addresses the N=50 criticism without requiring more data.
  5. Sharpen the RAGAS critique into a standalone finding. Write a dedicated section: "We identified a systematic evaluation bias in RAGAS when applied to benchmarks with unanswerable questions, and we propose
   the following correction." This is the closest thing to a novel contribution you have. Make it visible and precise.

  If points 1–3 are delivered, and the presentation is technically confident, BA to AA is a reasonable range at most Turkish engineering faculties. If the UI is absent and the baseline is absent, examiners who
   ask the right questions will knock you to BB.

✻ Worked for 1m 31s

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
> 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ? for shortcuts                                                                                                                                                        new task? /clear to save 147.3k tokens

  ---
  1. Scope

  Assessment: Appropriately scoped, possibly slightly over-broad for the depth achieved.

  Four people, five months, a dual-pipeline RAG system with ablations and benchmark evaluation is a reasonable
  undergraduate scope. The mistake most teams make at this scope is building wide and shallow — many components, none of
   them rigorously understood. You partially avoided this on the unstructured side (genuine ablation discipline) but
  fell into it on the UI and the RQ1 latency experiment, both of which are listed as goals but not completed.

  The structured pipeline (TableRAG + LangGraph ReAct) is genuinely complex to wire together. The fact that it works
  end-to-end with self-correction and schema FK expansion is a meaningful engineering accomplishment. But because you
  are using LangGraph, LangChain, FAISS, RAGAS, and the OpenAI API as black boxes, the "scope" is partially
  infrastructure wrangling, not research. That is normal for undergraduate work — just be honest with yourself about
  which parts of the system you deeply understand versus assembled.

  ---
  2. Technical Depth

  Assessment: Above average for an undergraduate thesis, specifically because of the evaluation methodology — not
  because of the system design.

  The ablation study approach is the strongest part of this project. Changing exactly one variable per experiment,
  maintaining a consistent evaluation set, and then catching a systematic bias in your own evaluation methodology (the
  NO_GOLD problem) is the kind of critical thinking that distinguishes a good thesis from an average one. Most
  undergraduate projects run one configuration, get a number, and present it. You ran 14+, noticed the results were
  misleading, and investigated why. That is scientifically disciplined.

  The false-abstention analysis (identifying that the LLM was over-refusing and systematically auditing it) is also
  genuinely mature. That is not something most 4th-year students think to do.

  However, three things limit how much credit the methodology earns:

  - N=50 is too small. At 50 questions, a difference of 2–3 questions is 4–6 percentage points. Your confidence
  intervals are wide enough that many of your ablation conclusions could reverse with a different 50-question sample. A
  reviewer who knows statistics will say this immediately.
  - No external baseline. You compare configurations against each other (best chunk size, best K, best reranker pool).
  You never compare your best system against naive RAG (FAISS top-k, no reranker, no cache, direct GPT-4o-mini), or
  against a BM25-only retriever, or against GPT-4o-mini with the full document context. Without a baseline, "K=15 with
  reranker achieves 0.685 faithfulness" means nothing on its own. What does a simple system get? 0.60? 0.70? If a
  simpler system matches you, your architecture is not justified.
  - RAGAS is itself contested. Your critique of RAGAS is valid, but an examiner could argue that your fix (adjusting
  abstention behavior) games the metric rather than solving the evaluation problem. You noted the bias but then
  continued using RAGAS as your primary metric. A stronger position would be to supplement RAGAS with human evaluation
  on a sample — even 20 manually judged questions — or with a reference-based metric that is not LLM-judged.

  ---
  3. Novelty

  Assessment: Integration project with one above-average design decision and one genuine methodological contribution. No
   algorithmic novelty.

  There is no new algorithm here. FAISS, BGE embeddings, cross-encoder reranking, LangGraph ReAct agents, TableRAG
  schema pruning — every component is an existing library or published technique. This is standard for undergraduate
  work and not a criticism by itself. A thesis can be valuable without novelty if the engineering is solid and the
  evaluation is rigorous.

  The one design decision that goes slightly beyond off-the-shelf assembly is the dual-path structured pipeline: a FAISS
   semantic cache (fast track) with LLM refinement on cache hits, falling back to TableRAG + ReAct on cache misses. This
   is a practical architecture that addresses latency in a non-trivial way and has a clear research question attached to
   it (RQ1). The problem is that the latency experiment was not actually completed — you have the path labels and
  latency logged per query, but no aggregated comparison table.

  The evaluation methodology critique (identifying NO_GOLD bias in RAGAS) is the closest thing to a genuine novel
  contribution. It is a methodological observation that could be useful to others evaluating RAG systems on similar
  benchmarks. If you write this up clearly, it is the most citable and defensible part of the thesis.

  What is not novel and should not be overclaimed: the router architecture (LLM classification of query intent is
  described in dozens of RAG papers from 2023–2024), the hybrid retrieval (BM25+FAISS via RRF is standard since
  DPR-hybrid papers), and the synthesis agent (prompt engineering on top of GPT-4o-mini).

  ---
  4. Weaknesses an Examiner Will Challenge

  Challenge 1: "Your evaluation set has 50 questions. How do you know your conclusions generalize?"

  This is the first and most damaging question. You do not have a good answer with 50 questions. The standard response
  in the NLP literature for a dataset this small is to run multiple random samples and report variance, or to calculate
  confidence intervals explicitly. If you used the same 50 questions for all 14 ablations, you also have a multiple
  comparisons problem — some of your "best configs" are overfitted to that specific 50.

  What to prepare: Calculate standard deviations across your RAGAS metrics. Acknowledge the limitation explicitly in the
   thesis and discuss what a larger study would require. Do not try to defend 50 questions as sufficient — acknowledge
  it and explain why it was the practical limit.

  Challenge 2: "Where is the system comparison against a baseline? How do we know the complexity is justified?"

  Your ablations compare configurations of your system against each other. No examiner who has read RAG literature will
  accept "our best config is better than our worst config" as a conclusion. They will ask: compared to what? A
  retrieval-augmented system that uses BM25 only, no reranker, and a direct prompt to GPT-4o-mini is a 30-line Python
  script. If it scores 0.65 faithfulness and your elaborate system scores 0.685, you have built a lot of infrastructure
  for 2 percentage points.

  What to prepare: Run at least one ablation that removes your reranker and semantic cache entirely (pure FAISS top-k →
  GPT-4o-mini) and report the result honestly. If your system is better, this strengthens the thesis significantly. If
  it is not much better, that is also a valid finding — it means the marginal contribution of each component is small,
  which is a legitimate research result.

  Challenge 3: "You describe hybrid queries as a core use case, but in your default mode (decompose), you never route to
   'hybrid' — you run separate sub-tasks. Is this actually hybrid retrieval, or just sequential retrieval with merging?"

  This is a precise technical criticism. In decompose mode, the LangChain LLM only emits "sql" or "text" sub-tasks.
  Hybrid routing (both pipelines on the same query) only occurs in the keyword, llm, or semantic modes, none of which
  are the default. Your synthesis agent handles this correctly, but the framing of "hybrid query routing" overstates
  what is happening in the default pipeline.

  What to prepare: Either switch the default mode to llm or semantic for demos involving true hybrid queries, or be
  precise in the thesis that decompose mode handles mixed-intent via sub-task splitting rather than dual-pipeline
  execution.

  ---
  5. Honest Grade Range

  Current state, presented honestly: BB to BA

  Why not higher:
  - No working UI (you claimed one)
  - N=50 without confidence intervals
  - No baseline comparison
  - RQ1 latency experiment not completed
  - No novelty beyond integration and methodology critique

  Why not lower:
  - The ablation discipline is genuinely above average
  - The RAGAS methodology critique shows real critical thinking
  - The dual-path structured pipeline is correctly implemented end-to-end
  - The codebase is actually clean and works (not always the case at this level)
  - The evaluation guide and experiment infrastructure show systematic thinking

  To reach AA, specifically:

  1. Do the baseline comparison. One run of naive FAISS-top-k-GPT-4o-mini on the same 50 questions. This single addition
   would significantly strengthen the ablation section because it gives your numbers context.
  2. Complete or remove the UI. Either spend 2 days wiring a Streamlit interface to backend/main.py (the API layer
  exists, it is a thin frontend problem), or remove the claim entirely from the presentation. Do not present something
  that does not exist.
  3. Complete RQ1. The path label and latency are already being logged per query. You need a script that seeds the SQL
  cache with a test set, then runs the same queries to force cache hits, records latency for both conditions, and
  produces a comparison table. This is probably 2–4 hours of work given the infrastructure is there.
  4. Add confidence intervals to your ablation table. Even bootstrapped CIs from 50 questions would show you are
  thinking statistically. This directly addresses the N=50 criticism without requiring more data.
  5. Sharpen the RAGAS critique into a standalone finding. Write a dedicated section: "We identified a systematic
  evaluation bias in RAGAS when applied to benchmarks with unanswerable questions, and we propose the following
  correction." This is the closest thing to a novel contribution you have. Make it visible and precise.

  If points 1–3 are delivered, and the presentation is technically confident, BA to AA is a reasonable range at most
  Turkish engineering faculties. If the UI is absent and the baseline is absent, examiners who ask the right questions
  will knock you to BB.

✻ Worked for 1m 31s

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
> 
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ? for shortcuts                                                               new task? /clear to save 147.3k tokens
  3. Novelty

  Assessment: Integration project with one above-average design decision and one genuine methodological contribution. No algorithmic novelty.

  There is no new algorithm here. FAISS, BGE embeddings, cross-encoder reranking, LangGraph ReAct agents, TableRAG schema pruning — every component is an existing library or published technique. This is
  standard for undergraduate work and not a criticism by itself. A thesis can be valuable without novelty if the engineering is solid and the evaluation is rigorous.

  The one design decision that goes slightly beyond off-the-shelf assembly is the dual-path structured pipeline: a FAISS semantic cache (fast track) with LLM refinement on cache hits, falling back to TableRAG
  + ReAct on cache misses. This is a practical architecture that addresses latency in a non-trivial way and has a clear research question attached to it (RQ1). The problem is that the latency experiment was
  not actually completed — you have the path labels and latency logged per query, but no aggregated comparison table.

  The evaluation methodology critique (identifying NO_GOLD bias in RAGAS) is the closest thing to a genuine novel contribution. It is a methodological observation that could be useful to others evaluating RAG
  systems on similar benchmarks. If you write this up clearly, it is the most citable and defensible part of the thesis.

  What is not novel and should not be overclaimed: the router architecture (LLM classification of query intent is described in dozens of RAG papers from 2023–2024), the hybrid retrieval (BM25+FAISS via RRF is
  standard since DPR-hybrid papers), and the synthesis agent (prompt engineering on top of GPT-4o-mini).

  ---
  4. Weaknesses an Examiner Will Challenge

  Challenge 1: "Your evaluation set has 50 questions. How do you know your conclusions generalize?"

  This is the first and most damaging question. You do not have a good answer with 50 questions. The standard response in the NLP literature for a dataset this small is to run multiple random samples and
  report variance, or to calculate confidence intervals explicitly. If you used the same 50 questions for all 14 ablations, you also have a multiple comparisons problem — some of your "best configs" are
  overfitted to that specific 50.

  What to prepare: Calculate standard deviations across your RAGAS metrics. Acknowledge the limitation explicitly in the thesis and discuss what a larger study would require. Do not try to defend 50 questions
  as sufficient — acknowledge it and explain why it was the practical limit.

  Challenge 2: "Where is the system comparison against a baseline? How do we know the complexity is justified?"

  Your ablations compare configurations of your system against each other. No examiner who has read RAG literature will accept "our best config is better than our worst config" as a conclusion. They will ask:
  compared to what? A retrieval-augmented system that uses BM25 only, no reranker, and a direct prompt to GPT-4o-mini is a 30-line Python script. If it scores 0.65 faithfulness and your elaborate system scores
   0.685, you have built a lot of infrastructure for 2 percentage points.

  What to prepare: Run at least one ablation that removes your reranker and semantic cache entirely (pure FAISS top-k → GPT-4o-mini) and report the result honestly. If your system is better, this strengthens
  the thesis significantly. If it is not much better, that is also a valid finding — it means the marginal contribution of each component is small, which is a legitimate research result.

  Challenge 3: "You describe hybrid queries as a core use case, but in your default mode (decompose), you never route to 'hybrid' — you run separate sub-tasks. Is this actually hybrid retrieval, or just
  sequential retrieval with merging?"

  This is a precise technical criticism. In decompose mode, the LangChain LLM only emits "sql" or "text" sub-tasks. Hybrid routing (both pipelines on the same query) only occurs in the keyword, llm, or
  semantic modes, none of which are the default. Your synthesis agent handles this correctly, but the framing of "hybrid query routing" overstates what is happening in the default pipeline.

  What to prepare: Either switch the default mode to llm or semantic for demos involving true hybrid queries, or be precise in the thesis that decompose mode handles mixed-intent via sub-task splitting rather
  than dual-pipeline execution.

  ---
  5. Honest Grade Range

  Current state, presented honestly: BB to BA

  Why not higher:
  - No working UI (you claimed one)
  - N=50 without confidence intervals
  - No baseline comparison
  - RQ1 latency experiment not completed
  - No novelty beyond integration and methodology critique

  Why not lower:
  - The ablation discipline is genuinely above average
  - The RAGAS methodology critique shows real critical thinking
  - The dual-path structured pipeline is correctly implemented end-to-end
  - The codebase is actually clean and works (not always the case at this level)
  - The evaluation guide and experiment infrastructure show systematic thinking

  To reach AA, specifically:

  1. Do the baseline comparison. One run of naive FAISS-top-k-GPT-4o-mini on the same 50 questions. This single addition would significantly strengthen the ablation section because it gives your numbers
  context.
  2. Complete or remove the UI. Either spend 2 days wiring a Streamlit interface to backend/main.py (the API layer exists, it is a thin frontend problem), or remove the claim entirely from the presentation. Do
   not present something that does not exist.
  3. Complete RQ1. The path label and latency are already being logged per query. You need a script that seeds the SQL cache with a test set, then runs the same queries to force cache hits, records latency for
   both conditions, and produces a comparison table. This is probably 2–4 hours of work given the infrastructure is there.
  4. Add confidence intervals to your ablation table. Even bootstrapped CIs from 50 questions would show you are thinking statistically. This directly addresses the N=50 criticism without requiring more data.
  5. Sharpen the RAGAS critique into a standalone finding. Write a dedicated section: "We identified a systematic evaluation bias in RAGAS when applied to benchmarks with unanswerable questions, and we propose
   the following correction." This is the closest thing to a novel contribution you have. Make it visible and precise.

  If points 1–3 are delivered, and the presentation is technically confident, BA to AA is a reasonable range at most Turkish engineering faculties. If the UI is absent and the baseline is absent, examiners who
   ask the right questions will knock you to BB.

✻ Worked for 1m 31s