# Evaluation Guide

This document explains how to run every benchmark in the system. All commands are run from the **project root** (`Adaptive-Agentic-RAG/`).

---

## Prerequisites

```bash
pip install -r requirements.txt
pip install ragas datasets langchain-huggingface
```

Set your OpenAI key in `.env` or the shell:

```bash
export OPENAI_API_KEY=sk-...
```

---

## 1. Unstructured Pipeline — SQuAD

Evaluates the RAG pipeline (FAISS / hybrid retrieval + optional reranking) on 50 SQuAD v1.1 questions. Scores: `answer_relevancy`, `faithfulness`, `context_precision`, `context_recall` via RAGAS.

### Step 1 — Prepare data (once)

```bash
python evaluation/Unstructured/prepare_squad_dataset.py
```

Produces `evaluation/Unstructured/squad_corpus.json` and `squad_eval_set.json`.

### Step 2 — Run a single config

```bash
python evaluation/Unstructured/run_config_eval.py C1
```

### Step 3 — Run all T-series configs

```bash
python evaluation/Unstructured/run_t_series.py
```

### Available configs

**C-series** — retrieval mode × reranker × chunk size

| Config | Mode   | Reranker | top_k | chunk_size |
|--------|--------|----------|-------|------------|
| C1     | faiss  | No       | 5     | 500        |
| C2     | faiss  | Yes      | 5     | 500        |
| C3     | hybrid | No       | 5     | 500        |
| C4     | hybrid | Yes      | 5     | 500        |
| C5     | hybrid | Yes      | 3     | 500        |
| C6     | hybrid | Yes      | 9     | 500        |
| C7     | faiss  | No       | 5     | 250        |
| C8     | faiss  | Yes      | 5     | 250        |
| C9     | hybrid | No       | 5     | 250        |
| C10    | hybrid | Yes      | 5     | 250        |
| C11    | hybrid | Yes      | 3     | 250        |
| C12    | hybrid | Yes      | 9     | 250        |

**T-series** — fixed fetch_k=100, varying rerank_pool and top_k

| Config | Mode   | Reranker | top_k | fetch_k | rerank_pool |
|--------|--------|----------|-------|---------|-------------|
| T2A    | hybrid | Yes      | 5     | 100     | 60          |
| T2B    | hybrid | Yes      | 10    | 100     | 60          |
| T3A    | hybrid | Yes      | 5     | 100     | 20          |
| T3B    | hybrid | Yes      | 10    | 100     | 20          |
| T4A    | hybrid | Yes      | 5     | 60      | 60          |
| T4B    | hybrid | Yes      | 10    | 60      | 60          |
| T5A    | hybrid | No       | 5     | 100     | —           |
| T5B    | hybrid | No       | 10    | 100     | —           |

> **Note:** `rerank_pool=0` (or omitted) means the reranker sees all `fetch_k` candidates. When not set in config, `fetch_k` defaults to `RAG_TOP_K × RAG_FETCH_MULTIPLIER` (5 × 20 = 100).

### Output files

```
evaluation/Unstructured/results/<cfg>_rag_results.json    # retrieved contexts + answers
evaluation/Unstructured/results/<cfg>_ragas_scores.json   # RAGAS metrics
```

---

## 2. Unstructured Pipeline — RAGBench

Same pipeline as SQuAD but on the denser RAGBench corpus (techqa, emanual, covidqa, expertqa, msmarco subsets). Adds a `gold_hit_rate` metric — whether the gold passage was retrieved.

### Step 1 — Prepare data (once)

```bash
python evaluation/Unstructured/prepare_ragbench_dataset.py
```

Optional flags:

```bash
# specific subsets only
python evaluation/Unstructured/prepare_ragbench_dataset.py --subsets techqa emanual covidqa

# limit questions per subset
python evaluation/Unstructured/prepare_ragbench_dataset.py --subsets techqa --n 100
```

Produces `evaluation/Unstructured/evaluation/ragbench_corpus.json` and `ragbench_eval_set.json`.

### Step 2 — Run a single config

```bash
python evaluation/Unstructured/run_ragbench_eval.py C1
```

### Step 3 — Run multiple configs with a comparison table

```bash
# Run specific configs
python evaluation/Unstructured/run_ragbench_series.py --configs C1 C2 C3 C4

# Run all T-series
python evaluation/Unstructured/run_ragbench_series.py

# Skip configs that already have result files
python evaluation/Unstructured/run_ragbench_series.py --skip-existing
```

The same config table (C1–C12, T2A–T5B) applies here as in the SQuAD eval.

### Output files

```
evaluation/Unstructured/results/ragbench_<cfg>_rag_results.json
evaluation/Unstructured/results/ragbench_<cfg>_ragas_scores.json
```

---

## 3. Structured Pipeline — Spider Benchmark

Evaluates the SQL pipeline (TableRAG + ReAct NL-to-SQL agent) on the Spider text-to-SQL dataset. Uses stratified sampling: 20 questions × 4 difficulty tiers = 80 questions total.

### Required files (download separately)

Place these in `evaluation/Structured/`:

```
evaluation/Structured/dev.json
evaluation/Structured/tables.json
evaluation/Structured/database/<db_id>/<db_id>.sqlite   # already included
```

Spider dataset: https://yale-lily.github.io/spider

### Run

```bash
# Full run — all 3 configs, 80 questions each
python evaluation/Structured/benchmark_spider.py

# Fewer questions per difficulty tier
python evaluation/Structured/benchmark_spider.py --per_tier 10

# Specific configs only
python evaluation/Structured/benchmark_spider.py --configs S1 S3

# Force S3 without ReAct loop
python evaluation/Structured/benchmark_spider.py --no_react
```

### Configs

| Config | Description                                      |
|--------|--------------------------------------------------|
| S1     | Zero-shot: full schema, no TableRAG, no ReAct    |
| S2     | TableRAG-only: pruned schema, no ReAct           |
| S3     | Full pipeline: TableRAG + ReAct + SQL cache      |

### Metrics reported

- **Execution Accuracy (EX)** — primary metric
- Valid SQL rate, error rate, avg latency
- EX breakdown by difficulty (easy / medium / hard / extra)
- EX breakdown by database
- TableRAG pruning ratio (S2, S3)
- Cache hit rate and EX cache-hit vs ReAct (S3 only)

### Output files

```
evaluation/Structured/results/
```

---

## 4. Router Evaluation

Evaluates the query router (`decompose_with_zeroshot`) on classification accuracy and decomposition quality.

### Required files

```
evaluation/Router/test_cases.json     # 60 labeled queries (sql / text / hybrid)
evaluation/Router/decomp_cases.json   # 10 complex multi-intent queries
```

### Run

```bash
# Both modes (default)
python evaluation/Router/eval_router.py

# Classification only
python evaluation/Router/eval_router.py --only-classification

# Decomposition quality only
python evaluation/Router/eval_router.py --only-decomp

# Custom output path
python evaluation/Router/eval_router.py --output path/to/out.json
```

### Metrics reported

**Classification:** per-class Precision / Recall / F1 + confusion matrix

**Decomposition quality:**
- Route validity — all sub-tasks route to `sql` or `text`
- Atomicity — each sub-task is single-intent
- Coverage — key concepts from original query appear in sub-tasks
- No raw SQL — sub-query fields are natural language only

### Output files

```
evaluation/results/router_eval.json   # default output
```

---

## Quick Reference

| Benchmark          | Script                                              | Dataset  | Questions |
|--------------------|-----------------------------------------------------|----------|-----------|
| RAG — SQuAD        | `evaluation/Unstructured/run_config_eval.py <cfg>`  | SQuAD    | 50        |
| RAG — RAGBench     | `evaluation/Unstructured/run_ragbench_eval.py <cfg>`| RAGBench | 50        |
| RAG — T-series     | `evaluation/Unstructured/run_t_series.py`           | SQuAD    | 50 × 8   |
| RAG — RAGBench all | `evaluation/Unstructured/run_ragbench_series.py`    | RAGBench | 50 × 8   |
| SQL — Spider       | `evaluation/Structured/benchmark_spider.py`         | Spider   | 80        |
| Router             | `evaluation/Router/eval_router.py`                  | custom   | 70        |
