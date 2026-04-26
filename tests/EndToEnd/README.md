# Benchmark Runners — Adaptive Agentic RAG Framework

Five Python files that wire the 150-query benchmark to your live RAG system.

## File layout

```
benchmark_runners/
├── config.py                  ← paths, timeouts, system module name
├── system_adapter.py          ← thin shim between runners and your RAG system
├── run_rq1_latency.py         ← RQ1: Fast Track vs Reasoning Track latency
├── run_rq2_router.py          ← RQ2: Router classification accuracy (90 queries)
├── run_rq3_sql_execution.py   ← RQ3: SQL execution success + TableRAG pruning
├── run_fi_hybrid.py           ← Functional Integration: hybrid E2E flows
├── run_all_benchmarks.py      ← Master runner (runs all 4 in correct order)
├── all_benchmark_queries.json ← 150-query benchmark (your file)
└── results/                   ← auto-created; all JSON outputs land here
```

## Setup — 3 steps

### 1. Configure paths
Edit `config.py`:
```python
DB_PATH        = PROJECT_ROOT / "fintech.db"
PDF_DIR        = PROJECT_ROOT / "pdf_files"
SYSTEM_MODULE  = "rag_system"   # importable name of your RAG module
SYSTEM_FN      = "run_query"    # function to call inside that module
```

### 2. Expose `run_query` from your system
Your RAG system must expose a callable with this signature:

```python
def run_query(query: str, **kwargs) -> dict:
    """
    kwargs may include:
        tablerag_pruning: bool  (used by RQ3 runner)

    Must return a dict with at least:
        router_decision:    "sql" | "text" | "hybrid"
        answer:             str
        sql_executed:       str | None
        sql_result:         list | None
        retrieved_chunks:   list | None
        retrieved_sources:  list | None   # PDF filenames
        cache_hit:          bool
        latency_ms:         float
        error:              str | None
    """
```

If your system uses different key names, map them in `system_adapter.py`
(the `aliases` dict in `_normalise()`).

### 3. Copy benchmark file
Make sure `all_benchmark_queries.json` sits in the same directory as these scripts.

---

## Running

```bash
# Run all benchmarks (recommended order):
python run_all_benchmarks.py

# Run individually:
python run_rq1_latency.py
python run_rq2_router.py
python run_rq3_sql_execution.py
python run_fi_hybrid.py

# Selective flags:
python run_all_benchmarks.py --skip-rq1     # skip latency (faster dev run)
python run_all_benchmarks.py --rq2-only
python run_all_benchmarks.py --dry-run      # print plan, no system calls
```

---

## Output files

| File | Contents |
|------|----------|
| `results/rq1_cold_results.json` | Per-query latency — Reasoning Track (30 SQL) |
| `results/rq1_warm_results.json` | Per-query latency — Fast Track (30 CACHE) |
| `results/rq1_summary.json` | Speedup factor, cache hit rate, p50/p95 |
| `results/rq2_results.json` | Per-query router decisions (90 queries) |
| `results/rq2_summary.json` | Accuracy per class + confusion matrix |
| `results/rq3_results.json` | Per-query SQL execution (60 × 2 modes) |
| `results/rq3_summary.json` | Success rates + TableRAG pruning recall/precision |
| `results/fi_results.json` | Per-query hybrid E2E flags (30 queries) |
| `results/fi_summary.json` | E2E flow count, component success rates |
| `results/final_report.json` | **Thesis-ready unified summary of all metrics** |

---

## Pass thresholds (from proposal)

| Metric | Target |
|--------|--------|
| RQ1 speedup factor | ≥ 1.4× |
| RQ1 cache hit rate | ≥ 80% |
| RQ2 overall accuracy | ≥ 80% (72/90) |
| RQ3 pruned success ≥ full schema | positive improvement |
| FI successful E2E flows | ≥ 5 out of 30 |
