#!/usr/bin/env python3
"""
Spider Benchmark Runner — Adaptive Agentic RAG Structured Pipeline.
 
Stratified sampling: 20 questions × 4 difficulty tiers = 80 questions total.
 
Three pipeline configurations (mirrors the unstructured C1/C3/T5A pattern):
  S1  Zero-shot        : GPT + full schema, no TableRAG, no ReAct, no cache
  S2  TableRAG-only    : GPT + pruned schema via TableRAG, no ReAct, no cache
  S3  Full pipeline    : TableRAG + ReAct + SQL cache  (proposed system)
 
Metrics reported:
  - Execution Accuracy (EX)       primary metric
  - Valid SQL Rate                 % questions where any SQL was produced
  - Error Rate                     % questions that raised an exception
  - Avg latency (s)                per question, per config
  - EX by difficulty               easy / medium / hard / extra breakdown
  - EX by database                 per-db breakdown
  - TableRAG pruning ratio         avg tables selected / total tables in schema (S2, S3)
  - Cache hit rate                 S3 only
  - EX cache-hit vs react          S3 only — key thesis finding
  - Latency cache-hit vs react     S3 only
 
Usage:
    python benchmark_spider.py
    python benchmark_spider.py --per_tier 10   # 10 per difficulty = 40 total
    python benchmark_spider.py --configs S1 S3  # run only specific configs
    python benchmark_spider.py --no_react       # force S3 without ReAct loop
 
Place this file at:  evaluation/Structured/benchmark_spider.py
Place Spider files at: evaluation/Structured/dev.json
                       evaluation/Structured/tables.json
                       evaluation/Structured/database/<db_id>/<db_id>.sqlite
"""
 
from __future__ import annotations
 
import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
 
# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
 
EVAL_DIR         = Path(__file__).resolve().parent
DB_DIR           = EVAL_DIR / "database"
DEV_JSON         = EVAL_DIR / "dev.json"
TABLES_JSON      = EVAL_DIR / "tables.json"
SCHEMA_CACHE_DIR = EVAL_DIR / "schema_cache"
RESULTS_DIR      = EVAL_DIR / "results"
 
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCHEMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
 
# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--per_tier", type=int,   default=20,
                    help="Questions per difficulty tier (default 20 → 80 total)")
parser.add_argument("--configs",  nargs="+",  default=["S1", "S2", "S3"],
                    choices=["S1", "S2", "S3"],
                    help="Which pipeline configs to run")
parser.add_argument("--no_react", action="store_true",
                    help="Disable ReAct in S3 (single-pass agent only)")
parser.add_argument("--seed",     type=int,   default=42)
parser.add_argument("--delay",    type=float, default=0.4,
                    help="Sleep between LLM calls (seconds)")
args = parser.parse_args()
 
random.seed(args.seed)
 
DIFFICULTIES = ["easy", "medium", "hard", "extra"]
 
# ── Config definitions ────────────────────────────────────────────────────────
#
# Mirrors the unstructured benchmark table:
#   S1 = C1  (baseline — no TableRAG, no ReAct, no cache)
#   S2 = C3  (TableRAG pruning only, no ReAct, no cache)
#   S3 = T5A (full proposed system — TableRAG + ReAct + cache)
#
CONFIGS: dict[str, dict] = {
    "S1": dict(
        label       = "Zero-shot (full schema)",
        use_tablerag = False,
        use_react    = False,
        use_cache    = False,
        sql_top_k    = None,     # irrelevant — full schema passed
    ),
    "S2": dict(
        label       = "TableRAG only (no ReAct)",
        use_tablerag = True,
        use_react    = False,
        use_cache    = False,
        sql_top_k    = 3,
    ),
    "S3": dict(
        label       = "Full pipeline (TableRAG + ReAct + cache)",
        use_tablerag = True,
        use_react    = not args.no_react,
        use_cache    = True,
        sql_top_k    = 3,
    ),
}
 
# ── Load Spider data ──────────────────────────────────────────────────────────
for p in [DEV_JSON, TABLES_JSON, DB_DIR]:
    if not p.exists():
        print(f"[ERROR] Not found: {p}")
        print("        Place Spider files inside evaluation/Structured/")
        sys.exit(1)
 
with open(DEV_JSON, encoding="utf-8") as f:
    dev_data: list[dict] = json.load(f)
 
with open(TABLES_JSON, encoding="utf-8") as f:
    tables_raw: list[dict] = json.load(f)
 
# ── SQL hardness classifier (dev.json has no 'hardness' field) ────────────────
def compute_hardness(item: dict) -> str:
    """Classify a Spider question by SQL complexity.

    Uses the pre-parsed 'sql' dict from dev.json to count structural
    complexity features, then maps to easy / medium / hard / extra.
    Falls back to inspecting the raw query string if the sql dict is absent.
    """
    sql = item.get("sql", {})
    q   = item.get("query", "").upper()

    # Count complexity signals from the parsed SQL dict
    score = 0

    where = sql.get("where", [])
    if isinstance(where, list) and where:
        score += 1
        if len(where) > 2:          # multiple conditions
            score += 1

    if sql.get("groupBy"):          score += 1
    if sql.get("having"):           score += 2
    if sql.get("orderBy"):          score += 1
    if sql.get("limit") is not None: score += 1

    for set_op in ("except", "union", "intersect"):
        if sql.get(set_op) is not None:
            score += 2

    # Nested subqueries: any table_unit of type 'sql' in from clause
    from_clause = sql.get("from", {})
    if isinstance(from_clause, dict):
        for unit in from_clause.get("table_units", []):
            if isinstance(unit, (list, tuple)) and unit and unit[0] == "sql":
                score += 2

    # Fallback: count raw SQL keywords when parsed dict is empty
    if score == 0:
        if "GROUP BY" in q:   score += 1
        if "HAVING"  in q:    score += 2
        if "UNION"   in q:    score += 2
        if "EXCEPT"  in q:    score += 2
        if "INTERSECT" in q:  score += 2
        if q.count("SELECT") > 1: score += 2  # subquery

    if   score == 0: return "easy"
    elif score <= 2: return "medium"
    elif score <= 4: return "hard"
    else:            return "extra"


# Build lookup: db_id → {table: [columns]}
def spider_schema_dict(db_id: str) -> dict[str, list[str]]:
    for db in tables_raw:
        if db["db_id"] != db_id:
            continue
        schema: dict[str, list[str]] = {}
        for tbl_idx, col_name in db["column_names_original"]:
            if tbl_idx < 0:
                continue
            tbl = db["table_names_original"][tbl_idx]
            schema.setdefault(tbl, []).append(col_name)
        return schema
    raise ValueError(f"db_id '{db_id}' not found in tables.json")
 
def full_schema_text(db_id: str) -> str:
    schema = spider_schema_dict(db_id)
    return "\n".join(
        f"Table: {tbl} | Columns: {', '.join(cols)}"
        for tbl, cols in schema.items()
    )
 
def total_table_count(db_id: str) -> int:
    return len(spider_schema_dict(db_id))
 
# ── Stratified sampling ───────────────────────────────────────────────────────
# dev.json has no 'hardness' field — compute it from SQL structure.
for item in dev_data:
    item["hardness"] = compute_hardness(item)

by_difficulty: dict[str, list[dict]] = {d: [] for d in DIFFICULTIES}
for item in dev_data:
    h = item["hardness"]
    if h in by_difficulty:
        by_difficulty[h].append(item)
 
sampled: list[dict] = []
for diff in DIFFICULTIES:
    pool = by_difficulty[diff]
    take = min(args.per_tier, len(pool))
    sampled.extend(random.sample(pool, take))
 
random.shuffle(sampled)
 
tier_counts = {d: sum(1 for q in sampled if q.get("hardness") == d) for d in DIFFICULTIES}
print(f"\n{'='*65}")
print(f"  Spider Benchmark — stratified sample")
print(f"  Total questions : {len(sampled)}  ({args.per_tier} per tier)")
print(f"  Tier breakdown  : {tier_counts}")
print(f"  Configs         : {args.configs}")
print(f"{'='*65}\n")
 
# ── Database switching ────────────────────────────────────────────────────────
_current_db: str | None = None
 
def switch_db(db_id: str) -> None:
    global _current_db
    if db_id == _current_db:
        return
 
    from backend import config
    import backend.sql.table_rag as table_rag
    from backend.sql.database import _sqlite_local
 
    db_path   = DB_DIR / db_id / f"{db_id}.sqlite"
    cache_dir = SCHEMA_CACHE_DIR / db_id
    cache_dir.mkdir(parents=True, exist_ok=True)
 
    config.SQLITE_PATH = str(db_path)
    config.DB_BACKEND  = "sqlite"
    config.INDEX_DIR   = cache_dir
 
    table_rag._SCHEMA_INDEX_PATH = cache_dir / "schema.faiss"
    table_rag._SCHEMA_TEXTS_PATH = cache_dir / "schema_texts.json"
 
    # Drop thread-local connection
    if hasattr(_sqlite_local, "conn") and _sqlite_local.conn is not None:
        try:
            _sqlite_local.conn.close()
        except Exception:
            pass
        _sqlite_local.conn = None
 
    # Reset the lazy SQLCache instance so it re-initialises for this DB's
    # INDEX_DIR on the next call — prevents cross-database cache contamination.
    from backend.sql import sql_agent
    if hasattr(sql_agent._get_sql_cache, "instance"):
        del sql_agent._get_sql_cache.instance

    _current_db = db_id
    print(f"  [DB switch] → {db_id}")
 
def ensure_schema_index(db_id: str) -> None:
    from backend.sql.sql_agent import _ensure_schema_index_exists
    _ensure_schema_index_exists()
 
# ── Execution accuracy ────────────────────────────────────────────────────────
def exec_sql_direct(sql: str, db_id: str) -> list[tuple] | None:
    """Execute SQL directly on the Spider SQLite, bypassing pipeline."""
    try:
        conn = sqlite3.connect(str(DB_DIR / db_id / f"{db_id}.sqlite"))
        cur  = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return None
 
def normalise_rows(rows: list[tuple]) -> set[tuple]:
    return set(
        tuple(str(v).strip().lower() if v is not None else "" for v in row)
        for row in rows
    )
 
def execution_accuracy(pred_sql: str | None, gold_sql: str, db_id: str) -> bool:
    if not pred_sql:
        return False
    gold = exec_sql_direct(gold_sql, db_id)
    pred = exec_sql_direct(pred_sql, db_id)
    if gold is None or pred is None:
        return False
    return normalise_rows(gold) == normalise_rows(pred)
 
# ── SQL extraction helper ─────────────────────────────────────────────────────
_SQL_RE = re.compile(r"\b(SELECT|WITH)\b[^;]*(?:;|$)", re.IGNORECASE | re.DOTALL)
 
def extract_sql(text: str | None) -> str | None:
    if not text:
        return None
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = _SQL_RE.search(text)
    if not m:
        return None
    sql = m.group(0).strip()
    return sql if re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE) else None
 
# ── S1: Zero-shot baseline ────────────────────────────────────────────────────
from openai import OpenAI
import backend.config as config
 
_oai = OpenAI(api_key=config.OPENAI_API_KEY)
 
S1_PROMPT = """\
You are an expert SQLite SQL writer. Given the schema below, write a single \
SQLite SELECT query that answers the question. Return ONLY the SQL, no markdown, \
no explanation.
 
Schema:
{schema}
 
Question: {question}
 
SQL:"""
 
def run_s1(question: str, db_id: str) -> dict:
    t0 = time.perf_counter()
    try:
        resp = _oai.chat.completions.create(
            model=config.SQL_OPENAI_MODEL,
            messages=[{"role": "user", "content": S1_PROMPT.format(
                schema=full_schema_text(db_id), question=question
            )}],
            temperature=0.0, max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        sql = extract_sql(raw)
    except Exception as e:
        sql, raw = None, str(e)
    return {
        "sql"             : sql,
        "latency"         : round(time.perf_counter() - t0, 3),
        "error"           : None if sql else raw,
        "tables_selected" : None,   # N/A for zero-shot
        "total_tables"    : total_table_count(db_id),
        "path"            : "zeroshot",
    }
 
# ── S2: TableRAG-only ────────────────────────────────────────────────────────
def run_s2(question: str, db_id: str) -> dict:
    from backend.sql.sql_agent import run_sql_agent, _ensure_schema_index_exists
    from backend.sql.table_rag import retrieve_relevant_schema
 
    t0 = time.perf_counter()
    try:
        _ensure_schema_index_exists()
        schema_rows    = retrieve_relevant_schema(question, top_k=config.SQL_TOP_K)
        schema_context = "\n".join(schema_rows)
        result         = run_sql_agent(question, schema_context=schema_context)
        sql            = result.get("sql")
        error          = result.get("error")
    except Exception as e:
        sql, error, schema_rows = None, str(e), []
 
    return {
        "sql"             : sql,
        "latency"         : round(time.perf_counter() - t0, 3),
        "error"           : error,
        "tables_selected" : len(schema_rows),
        "total_tables"    : total_table_count(db_id),
        "path"            : "tablerag_only",
    }
 
# ── S3: Full pipeline ─────────────────────────────────────────────────────────
def run_s3(question: str, db_id: str, use_react: bool) -> dict:
    from backend.sql.sql_agent import (
        _ensure_schema_index_exists, run_sql_agent,
        _get_sql_cache,
    )
    from backend.sql.table_rag import retrieve_relevant_schema
 
    t0 = time.perf_counter()
    try:
        _ensure_schema_index_exists()
 
        # Check cache first
        sql_cache  = _get_sql_cache()
        cache_hit  = False
        cache_score = 0.0
 
        if sql_cache.index is not None and sql_cache.index.ntotal > 0:
            hits = sql_cache.search_cache(question, top_k=1)
            if hits:
                hit_entry  = hits[0]                    # dict: {score, sql, question, schema}
                score      = hit_entry["score"]
                cached_sql = hit_entry["sql"]
                if score >= getattr(config, "SQL_CACHE_THRESHOLD", 0.85):
                    sql        = cached_sql
                    cache_hit  = True
                    cache_score = score
                    error      = None
                    path       = "cache_hit"
                    schema_rows = []
 
        if not cache_hit:
            schema_rows    = retrieve_relevant_schema(question, top_k=config.SQL_TOP_K)
            schema_context = "\n".join(schema_rows)
 
            if use_react and getattr(config, "SQL_REACT_ENABLED", True):
                from backend.sql.react_agent import run_react_sql_agent
                result = run_react_sql_agent(question, schema_context)
                path   = "react"
            else:
                result = run_sql_agent(question, schema_context=schema_context)
                path   = "single_pass"
 
            sql   = result.get("sql")
            error = result.get("error")
 
            # Add successful SQL to cache
            if sql and not error:
                schema_context_for_cache = "\n".join(schema_rows)
                sql_cache.add_to_cache(question, sql, schema_context_for_cache)
 
    except Exception as e:
        sql, error, schema_rows, path = None, str(e), [], "error"
        cache_hit, cache_score = False, 0.0
 
    return {
        "sql"             : sql,
        "latency"         : round(time.perf_counter() - t0, 3),
        "error"           : error,
        "tables_selected" : len(schema_rows) if not cache_hit else None,
        "total_tables"    : total_table_count(db_id),
        "path"            : path,
        "cache_hit"       : cache_hit,
        "cache_score"     : round(cache_score, 4),
    }
 
# ── Main loop ─────────────────────────────────────────────────────────────────
all_results: list[dict] = []
 
for i, item in enumerate(sampled):
    question = item["question"]
    gold_sql = item["query"]
    db_id    = item["db_id"]
    hardness = item.get("hardness", "unknown")
 
    print(f"\n[{i+1:>3}/{len(sampled)}] [{hardness:>10}] {db_id:<25} {question[:50]}...")
 
    switch_db(db_id)
    ensure_schema_index(db_id)
 
    record: dict[str, Any] = {
        "idx"      : i,
        "question" : question,
        "gold_sql" : gold_sql,
        "db_id"    : db_id,
        "hardness" : hardness,
    }
 
    runner_map = {
        "S1": lambda q, d: run_s1(q, d),
        "S2": lambda q, d: run_s2(q, d),
        "S3": lambda q, d: run_s3(q, d, use_react=CONFIGS["S3"]["use_react"]),
    }
 
    for cfg_id in args.configs:
        time.sleep(args.delay)
        out  = runner_map[cfg_id](question, db_id)
        ex   = execution_accuracy(out["sql"], gold_sql, db_id)
        out["ex"] = ex
 
        status = "✓" if ex else ("ERR" if out["error"] and not out["sql"] else "✗")
        lat    = out["latency"]
        tables = out.get("tables_selected")
        total  = out.get("total_tables", "?")
        prune  = f"{tables}/{total}" if tables is not None else "full"
        print(f"  {cfg_id} [{CONFIGS[cfg_id]['label'][:28]:<28}] "
              f"{status}  {lat:.2f}s  schema={prune}")
 
        record[cfg_id] = out
 
    all_results.append(record)
 
# ── Compute metrics ───────────────────────────────────────────────────────────
def metrics_for(results: list[dict], cfg_id: str) -> dict:
    rows = [r for r in results if cfg_id in r]
    if not rows:
        return {}
 
    n           = len(rows)
    ex_list     = [r[cfg_id]["ex"]      for r in rows]
    lat_list    = [r[cfg_id]["latency"] for r in rows]
    valid_list  = [1 if r[cfg_id].get("sql") else 0 for r in rows]
    error_list  = [1 if r[cfg_id].get("error") and not r[cfg_id].get("sql") else 0 for r in rows]
 
    # Per-difficulty EX
    by_diff: dict[str, list[bool]] = {}
    for r in rows:
        by_diff.setdefault(r["hardness"], []).append(r[cfg_id]["ex"])
    diff_ex = {d: round(sum(v)/len(v), 4) for d, v in sorted(by_diff.items())}
 
    # Per-db EX
    by_db: dict[str, list[bool]] = {}
    for r in rows:
        by_db.setdefault(r["db_id"], []).append(r[cfg_id]["ex"])
    db_ex = {db: round(sum(v)/len(v), 4) for db, v in sorted(by_db.items())}
 
    # TableRAG pruning ratio (S2, S3 non-cache-hit rows)
    prune_rows = [r for r in rows
                  if r[cfg_id].get("tables_selected") is not None
                  and r[cfg_id].get("total_tables", 0) > 0]
    pruning_ratio = (
        round(sum(r[cfg_id]["tables_selected"] / r[cfg_id]["total_tables"]
                  for r in prune_rows) / len(prune_rows), 4)
        if prune_rows else None
    )
 
    out: dict = {
        "n"                  : n,
        "execution_accuracy" : round(sum(ex_list)    / n, 4),
        "valid_sql_rate"     : round(sum(valid_list)  / n, 4),
        "error_rate"         : round(sum(error_list)  / n, 4),
        "avg_latency_s"      : round(sum(lat_list)    / n, 3),
        "by_difficulty"      : diff_ex,
        "by_database"        : db_ex,
        "tablerag_prune_ratio": pruning_ratio,  # avg tables_selected/total
    }
 
    # S3-only cache metrics
    if cfg_id == "S3":
        s3_rows   = rows
        hits      = [r for r in s3_rows if r["S3"].get("cache_hit")]
        reacts    = [r for r in s3_rows if r["S3"].get("path") == "react"]
        singles   = [r for r in s3_rows if r["S3"].get("path") == "single_pass"]
 
        out["cache_breakdown"] = {
            "cache_hits"       : len(hits),
            "react_calls"      : len(reacts),
            "single_pass_calls": len(singles),
            "cache_hit_rate"   : round(len(hits) / n, 4),
            "cache_hit_ex"     : round(sum(r["S3"]["ex"] for r in hits)   / max(len(hits), 1), 4),
            "react_ex"         : round(sum(r["S3"]["ex"] for r in reacts) / max(len(reacts), 1), 4),
            "cache_avg_lat_s"  : round(sum(r["S3"]["latency"] for r in hits)   / max(len(hits), 1), 3),
            "react_avg_lat_s"  : round(sum(r["S3"]["latency"] for r in reacts) / max(len(reacts), 1), 3),
        }
 
    return out
 
report = {
    "dataset"     : "Spider dev (stratified sample)",
    "n_total"     : len(all_results),
    "per_tier"    : args.per_tier,
    "tier_counts" : tier_counts,
    "configs"     : {
        cfg_id: {
            "label"   : CONFIGS[cfg_id]["label"],
            "settings": CONFIGS[cfg_id],
            "metrics" : metrics_for(all_results, cfg_id),
        }
        for cfg_id in args.configs
    },
    "results": all_results,
}
 
# ── Save outputs ──────────────────────────────────────────────────────────────
raw_path    = RESULTS_DIR / "spider_raw.json"
report_path = RESULTS_DIR / "spider_report.json"
 
with open(raw_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
 
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
 
# ── Print final report ────────────────────────────────────────────────────────
def bar(v: float, w: int = 20) -> str:
    return "#" * int(v * w) + "-" * (w - int(v * w))
 
print(f"\n{'='*65}")
print(f"  SPIDER BENCHMARK REPORT  —  n={len(all_results)}")
print(f"{'='*65}")
 
for cfg_id in args.configs:
    m = report["configs"][cfg_id]["metrics"]
    ex = m.get("execution_accuracy", 0)
    print(f"\n  {cfg_id}  {CONFIGS[cfg_id]['label']}")
    print(f"    Execution Accuracy  : {ex:.4f}  {bar(ex)}")
    print(f"    Valid SQL Rate      : {m.get('valid_sql_rate', 0):.4f}")
    print(f"    Error Rate          : {m.get('error_rate', 0):.4f}")
    print(f"    Avg latency         : {m.get('avg_latency_s', 0):.3f}s")
    pr = m.get("tablerag_prune_ratio")
    if pr is not None:
        print(f"    Pruning ratio      : {pr:.4f}  (avg tables_selected / total)")
    print(f"    By difficulty:")
    for d, v in m.get("by_difficulty", {}).items():
        print(f"      {d:<12} {v:.4f}  {bar(v, 12)}")
 
    if cfg_id == "S3" and "cache_breakdown" in m:
        cb = m["cache_breakdown"]
        print(f"    Cache breakdown:")
        print(f"      hits={cb['cache_hits']}  react={cb['react_calls']}  "
              f"hit_rate={cb['cache_hit_rate']:.1%}")
        print(f"      cache EX={cb['cache_hit_ex']:.4f}  lat={cb['cache_avg_lat_s']}s")
        print(f"      react EX={cb['react_ex']:.4f}  lat={cb['react_avg_lat_s']}s")
 
print(f"\n  Raw     → {raw_path}")
print(f"  Report  → {report_path}")
print(f"{'='*65}\n")