"""
evaluation/run_benchmark.py
============================
Runs all 611 benchmark questions through the Adaptive Agentic RAG system
and records raw results for scoring.

Usage:
    # From your project root:
    python evaluation/run_benchmark.py

    # Run a small smoke-test (first 10 questions only):
    python evaluation/run_benchmark.py --limit 10

    # Resume an interrupted run (skips already-completed IDs):
    python evaluation/run_benchmark.py --resume

HOW TO PLUG IN YOUR REAL SYSTEM
---------------------------------
Find the PipelineAdapter class below and replace the body of `query()`.
That is the only change needed — everything else (logging, timing, resuming,
per-question DB injection) is handled automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_RESULTS_PATH = RESULTS_DIR / "raw_results.json"
PROGRESS_PATH    = RESULTS_DIR / "_progress.json"

# ── Add project root to sys.path so backend imports work ─────────────────────
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PIPELINE ADAPTER — edit only this class when your system is ready
# =============================================================================

class PipelineAdapter:
    """
    Wraps your AdaptiveAgenticRAGSystem for benchmark evaluation.

    The adapter does two important things beyond a plain run_query() call:
      1. Swaps config.SQLITE_PATH to the per-question SQLite DB before each call.
      2. Injects the per-question passage chunks into the FAISS index before each call.

    This is necessary because your system currently uses a single fixed database.
    The adapter temporarily overrides that for each benchmark question.
    """

    def __init__(self, benchmark_dir: Path) -> None:
        self.benchmark_dir = benchmark_dir
        self._system = None  # lazy init — built once on first query()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system(self):
        """Import and build your system. Called once on first use."""
        from backend.main import build_system  # noqa: PLC0415
        return build_system()

    def _swap_db(self, db_path: str) -> None:
        """Point config.SQLITE_PATH at the question's own SQLite DB."""
        from backend import config  # noqa: PLC0415
        full_path = str(self.benchmark_dir / db_path)
        config.SQLITE_PATH = full_path

        # Also clear the schema index cache so it rebuilds for the new DB
        schema_faiss = getattr(config, "INDEX_DIR", Path(".")) / "schema.faiss"
        schema_texts = getattr(config, "INDEX_DIR", Path(".")) / "schema_texts.json"
        for p in [schema_faiss, schema_texts]:
            if Path(p).exists():
                Path(p).unlink()

    def _inject_passages(self, table_id: str, passages_index: dict) -> int:
        """
        Load the Wikipedia passage chunks for this table into the FAISS
        text index so the RAG pipeline has the right context.
        Returns number of chunks injected.
        """
        # Use your backend's own Chunk dataclass — NOT LangChain Document
        from backend.rag.chunker import Chunk  # noqa: PLC0415

        raw_chunks = passages_index.get(table_id, [])
        if not raw_chunks:
            return 0

        chunks = [
            Chunk(
                chunk_id=i,
                text=c["text"],
                source=c.get("wiki_url") or table_id,
                metadata={"chunk_id": c["chunk_id"]},
            )
            for i, c in enumerate(raw_chunks)
        ]

        # Clear existing FAISS index, then re-index with this table's passages
        system = self._system
        system.vector_store.store = None  # reset in-memory store
        system.retriever.index_chunks(chunks)
        return len(chunks)

    # ------------------------------------------------------------------
    # Public interface — called once per benchmark question
    # ------------------------------------------------------------------

    def setup(self, passages_index: dict) -> None:
        """
        Build the system and pre-load passage index into memory.
        Called once before the evaluation loop starts.
        """
        print("  Building system...")
        self._system = self._build_system()
        self._passages_index = passages_index
        print("  System ready.")

    def query(
        self,
        question: str,
        db_path: str,
        table_id: str,
        table_schema: dict,
    ) -> tuple[str, dict]:
        """
        Run one benchmark question through your pipeline.

        Parameters
        ----------
        question     : natural language question
        db_path      : relative path to this question's SQLite DB
                       e.g. "sqlite_dbs/2004_United_States_Grand_Prix_0.db"
        table_id     : Wikipedia table ID (used to look up passage chunks)
        table_schema : {"table_name": "...", "columns": [...]} — useful for
                       schema pruning context

        Returns
        -------
        answer   : str — the system's predicted answer
        metadata : dict — anything useful to log (route, latency, sql_error…)
        """

        # ── Step 1: point the SQL pipeline at this question's DB ──────────
        self._swap_db(db_path)

        # ── Step 2: load this table's Wikipedia passages into FAISS ───────
        n_chunks = self._inject_passages(table_id, self._passages_index)

        # ── Step 3: run your pipeline ─────────────────────────────────────
        #
        # YOUR SYSTEM IS CALLED HERE.
        # When your pipeline is fully connected, this should just work.
        # The system now has:
        #   - config.SQLITE_PATH pointing at the right SQLite DB
        #   - FAISS index loaded with this table's Wikipedia passages
        #
        answer = self._system.run_query(question)

        metadata = {
            "n_passage_chunks_injected": n_chunks,
            "db_path": db_path,
        }
        return answer, metadata


# =============================================================================
# STUB ADAPTER — used when system is not yet connected
# =============================================================================

class StubAdapter:
    """
    Returns placeholder answers so you can test the harness end-to-end
    before your pipeline is ready. Safe to delete once PipelineAdapter works.
    """

    def setup(self, passages_index: dict) -> None:
        print("  [STUB] Stub adapter ready — no real system loaded.")

    def query(
        self,
        question: str,
        db_path: str,
        table_id: str,
        table_schema: dict,
    ) -> tuple[str, dict]:
        time.sleep(0.05)  # simulate latency
        return "[STUB] Answer not yet implemented", {
            "db_path": db_path,
            "stub": True,
        }


# =============================================================================
# HARNESS
# =============================================================================

def load_benchmark(benchmark_dir: Path) -> list[dict]:
    path = benchmark_dir / "benchmark.json"
    with open(path) as f:
        return json.load(f)


def load_passages(benchmark_dir: Path) -> dict:
    path = benchmark_dir / "passages" / "all_passages.json"
    with open(path) as f:
        return json.load(f)


def load_progress() -> set[str]:
    """Return set of already-completed question IDs."""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return set(json.load(f))
    return set()


def save_progress(done_ids: set[str]) -> None:
    with open(PROGRESS_PATH, "w") as f:
        json.dump(list(done_ids), f)


def load_existing_results() -> list[dict]:
    if RAW_RESULTS_PATH.exists():
        with open(RAW_RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(results: list[dict]) -> None:
    with open(RAW_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def run_evaluation(
    adapter,
    benchmark: list[dict],
    passages_index: dict,
    resume: bool = False,
    limit: int | None = None,
) -> list[dict]:

    done_ids = load_progress() if resume else set()
    results  = load_existing_results() if resume else []

    # Filter questions
    questions = [q for q in benchmark if q["id"] not in done_ids]
    if limit:
        questions = questions[:limit]

    total = len(questions)
    print(f"\n{'='*60}")
    print(f"  Benchmark evaluation")
    print(f"  Questions to run : {total}")
    print(f"  Resumed          : {resume} ({len(done_ids)} already done)")
    print(f"  Results path     : {RAW_RESULTS_PATH}")
    print(f"{'='*60}\n")

    # Setup adapter once
    print("Setting up pipeline adapter...")
    adapter.setup(passages_index)
    print()

    for i, ex in enumerate(questions, 1):
        qid          = ex["id"]
        question     = ex["question"]
        ground_truth = ex["answer"]
        db_path      = ex["db_path"]
        table_id     = ex["table_id"]
        table_schema = ex["table_schema"]
        answer_source = ex["answer_source"]

        print(f"[{i:>4}/{total}] {qid[:12]}… | {answer_source:7s} | {question[:65]}")

        t_start = time.time()
        error   = None
        answer  = ""
        metadata: dict[str, Any] = {}

        try:
            answer, metadata = adapter.query(
                question=question,
                db_path=db_path,
                table_id=table_id,
                table_schema=table_schema,
            )
        except Exception as e:
            error  = traceback.format_exc()
            answer = ""
            print(f"         ❌ ERROR: {e}")

        latency_ms = (time.time() - t_start) * 1000

        result = {
            "id"            : qid,
            "question"      : question,
            "ground_truth"  : ground_truth,
            "predicted"     : str(answer).strip(),
            "answer_source" : answer_source,
            "table_id"      : table_id,
            "db_path"       : db_path,
            "latency_ms"    : round(latency_ms, 1),
            "error"         : error,
            "metadata"      : metadata,
            "timestamp"     : datetime.utcnow().isoformat(),
        }

        results.append(result)
        done_ids.add(qid)

        # Persist after every question so interruptions don't lose work
        save_results(results)
        save_progress(done_ids)

        status = "✓" if not error else "✗"
        print(f"         {status} predicted='{str(answer)[:60]}' | {latency_ms:.0f}ms")

    # Summary
    total_done  = len(results)
    error_count = sum(1 for r in results if r["error"])
    avg_latency = sum(r["latency_ms"] for r in results) / max(total_done, 1)

    print(f"\n{'='*60}")
    print(f"  Done: {total_done} questions")
    print(f"  Errors: {error_count}")
    print(f"  Avg latency: {avg_latency:.0f} ms")
    print(f"  Results saved to: {RAW_RESULTS_PATH}")
    print(f"{'='*60}")
    print("\nNext step: run  python evaluation/llm_judge.py  to score results.")

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument(
        "--stub",
        action="store_true",
        default=False,
        help="Use stub adapter (no real system needed — for testing the harness)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions (useful for smoke-testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip already-completed questions and append to existing results",
    )
    args = parser.parse_args()

    # Choose adapter
    if args.stub:
        print("Using STUB adapter (pass --stub=False when your system is ready)")
        adapter = StubAdapter()
    else:
        print("Using REAL PipelineAdapter")
        adapter = PipelineAdapter(BENCHMARK_DIR)

    benchmark      = load_benchmark(BENCHMARK_DIR)
    passages_index = load_passages(BENCHMARK_DIR)

    run_evaluation(
        adapter=adapter,
        benchmark=benchmark,
        passages_index=passages_index,
        resume=args.resume,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
