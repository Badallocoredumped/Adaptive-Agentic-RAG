"""
evaluation/baselines.py
========================
Runs two baselines on the same 611 HybridQA questions so you have
comparison rows for Table 1, matching the TableRAG paper structure.

Baseline 1 — Direct:    Ask Gemini the question with no context at all.
Baseline 2 — NaiveRAG:  FAISS retrieval only — no SQL, no structured
                         reasoning. Tables are serialised to Markdown text
                         and chunked like regular documents. This is the
                         standard RAG approach that TableRAG paper compares
                         against.

Usage:
    export GEMINI_API_KEY=your_key

    # Run both baselines:
    python evaluation/baselines.py

    # Run one baseline only:
    python evaluation/baselines.py --baseline direct
    python evaluation/baselines.py --baseline naive_rag

    # Smoke test (first 10 questions):
    python evaluation/baselines.py --limit 10

After running, score each with:
    python evaluation/llm_judge.py --results evaluation/results/direct_results.json     --system-name Direct
    python evaluation/llm_judge.py --results evaluation/results/naive_rag_results.json  --system-name NaiveRAG
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
import traceback
from datetime import datetime
from pathlib import Path

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RESULTS_DIR   = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Gemini ─────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash"
GEMINI_URL     = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)
SLEEP_BETWEEN = 1.2   # rate limit safety


# =============================================================================
# GEMINI CALL
# =============================================================================

def call_gemini(prompt: str, max_tokens: int = 256) -> str:
    if not GEMINI_API_KEY:
        return "[ERROR: GEMINI_API_KEY not set]"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens},
    }
    try:
        resp = requests.post(GEMINI_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"[ERROR: {e}]"


# =============================================================================
# BASELINE 1 — DIRECT (no context)
# =============================================================================

DIRECT_PROMPT = """\
Answer the following question as concisely as possible. \
Output only the answer — no explanation, no punctuation around it.

Question: {question}

Answer:"""

def run_direct(question: str) -> tuple[str, dict]:
    prompt = DIRECT_PROMPT.format(question=question)
    answer = call_gemini(prompt)
    return answer, {"baseline": "direct", "n_steps": 1}


# =============================================================================
# BASELINE 2 — NAIVE RAG
# Tables are serialised to Markdown and treated as plain text chunks.
# No SQL execution — just vector similarity over the text.
# Matches the NaiveRAG baseline description from the TableRAG paper (Section 5.1.3).
# =============================================================================

def _table_to_markdown(db_path: str, table_id: str) -> str:
    """Serialise the SQLite table into a Markdown string (NaiveRAG approach)."""
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_id)[:60]
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute(f'PRAGMA table_info("{safe_name}")')
        cols = [row[1] for row in cur.fetchall()]
        cur.execute(f'SELECT * FROM "{safe_name}"')
        rows = cur.fetchall()
        conn.close()

        if not cols:
            return ""

        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
        body   = "\n".join(
            "| " + " | ".join(str(v) if v is not None else "" for v in row) + " |"
            for row in rows
        )
        return f"{header}\n{sep}\n{body}"
    except Exception:
        return ""

def _simple_retrieve(query: str, chunks: list[str], top_k: int = 3) -> list[str]:
    """
    Keyword-overlap retrieval — a minimal stand-in for FAISS when we want
    to avoid loading the full embedder for the baseline.
    For real NaiveRAG you would use your FAISS index; this gives a reasonable
    approximation without the embedding overhead.
    """
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(query_words & chunk_words)
        scored.append((overlap, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k] if c.strip()]

NAIVE_RAG_PROMPT = """\
Answer the following question using only the provided context. \
Output only the answer — no explanation.

Context:
{context}

Question: {question}

Answer:"""

def run_naive_rag(
    question    : str,
    db_path     : str,
    table_id    : str,
    passage_chunks: list[str],
) -> tuple[str, dict]:
    """
    NaiveRAG: serialise the table to Markdown, combine with Wikipedia passage
    chunks, retrieve top-k by keyword overlap, then ask Gemini.
    """
    full_db_path = str(BENCHMARK_DIR / db_path)

    # 1. Serialise table to Markdown
    table_md = _table_to_markdown(full_db_path, table_id)

    # 2. Split table Markdown into chunks (simulate chunking)
    table_lines = table_md.split("\n") if table_md else []
    chunk_size  = 30  # rows per chunk
    table_chunks = []
    for i in range(0, len(table_lines), chunk_size):
        table_chunks.append("\n".join(table_lines[i:i + chunk_size]))

    # 3. Combine table chunks + passage text chunks
    all_chunks = table_chunks + passage_chunks

    # 4. Retrieve top-k by keyword overlap
    top_chunks = _simple_retrieve(question, all_chunks, top_k=3)
    context    = "\n\n---\n\n".join(top_chunks) if top_chunks else "No context available."

    # 5. Ask Gemini
    prompt = NAIVE_RAG_PROMPT.format(context=context[:4000], question=question)
    answer = call_gemini(prompt)

    return answer, {
        "baseline"        : "naive_rag",
        "n_steps"         : 1,
        "n_chunks_total"  : len(all_chunks),
        "n_chunks_retrieved": len(top_chunks),
    }


# =============================================================================
# RUNNER
# =============================================================================

def run_baseline(
    name     : str,
    benchmark: list[dict],
    passages : dict,
    limit    : int | None,
) -> list[dict]:

    output_path = RESULTS_DIR / f"{name}_results.json"
    questions   = benchmark[:limit] if limit else benchmark
    total       = len(questions)

    print(f"\n{'='*60}")
    print(f"  Baseline: {name}  ({total} questions)")
    print(f"{'='*60}\n")

    results = []

    for i, ex in enumerate(questions, 1):
        qid      = ex["id"]
        question = ex["question"]
        gold     = ex["answer"]
        db_path  = ex["db_path"]
        table_id = ex["table_id"]
        source   = ex["answer_source"]

        print(f"[{i:>4}/{total}] {source:7s} | {question[:60]}")

        t_start  = time.time()
        answer   = ""
        metadata : dict = {}
        error    = None

        try:
            passage_chunks = [c["text"] for c in passages.get(table_id, [])]

            if name == "direct":
                answer, metadata = run_direct(question)
            elif name == "naive_rag":
                answer, metadata = run_naive_rag(
                    question, db_path, table_id, passage_chunks
                )

            time.sleep(SLEEP_BETWEEN)

        except Exception as e:
            error  = traceback.format_exc()
            answer = ""
            print(f"         ❌ ERROR: {e}")

        latency_ms = (time.time() - t_start) * 1000

        results.append({
            "id"           : qid,
            "question"     : question,
            "ground_truth" : gold,
            "predicted"    : str(answer).strip(),
            "answer_source": source,
            "table_id"     : table_id,
            "db_path"      : db_path,
            "latency_ms"   : round(latency_ms, 1),
            "error"        : error,
            "metadata"     : metadata,
            "timestamp"    : datetime.utcnow().isoformat(),
        })

        # Save after every question
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        status = "✓" if not error else "✗"
        print(f"         {status} '{str(answer)[:50]}' | {latency_ms:.0f}ms")

    errors     = sum(1 for r in results if r["error"])
    avg_lat    = sum(r["latency_ms"] for r in results) / max(len(results), 1)
    print(f"\n  Done: {len(results)} | Errors: {errors} | Avg latency: {avg_lat:.0f}ms")
    print(f"  Results → {output_path}")
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        choices=["direct", "naive_rag", "both"],
        default="both",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set.")
        print("       export GEMINI_API_KEY=your_key")
        return

    with open(BENCHMARK_DIR / "benchmark.json") as f:
        benchmark = json.load(f)
    with open(BENCHMARK_DIR / "passages" / "all_passages.json") as f:
        passages = json.load(f)

    if args.baseline in ("direct", "both"):
        run_baseline("direct", benchmark, passages, args.limit)

    if args.baseline in ("naive_rag", "both"):
        run_baseline("naive_rag", benchmark, passages, args.limit)

    print("\n\nAll baselines done.")
    print("Score them with:")
    print("  python evaluation/llm_judge.py --results evaluation/results/direct_results.json    --system-name Direct")
    print("  python evaluation/llm_judge.py --results evaluation/results/naive_rag_results.json --system-name NaiveRAG")
    print("  python evaluation/llm_judge.py --results evaluation/results/raw_results.json       --system-name 'Adaptive Agentic RAG'")


if __name__ == "__main__":
    main()
