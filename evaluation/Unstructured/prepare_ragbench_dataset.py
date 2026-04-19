"""
evaluation/Unstructured/prepare_ragbench_dataset.py
=====================================================
Downloads RAGBench subsets from HuggingFace and builds two files
that slot directly into your existing run_config_eval.py pipeline:

  ragbench_corpus.json    — all unique passages to ingest into FAISS
  ragbench_eval_set.json  — questions with ground truth + gold passage IDs

Usage:
    pip install datasets
    python evaluation/Unstructured/prepare_ragbench_dataset.py
    python evaluation/Unstructured/prepare_ragbench_dataset.py --subsets techqa emanual covidqa
    python evaluation/Unstructured/prepare_ragbench_dataset.py --subsets techqa --n 100

Subsets recommended for RAG evaluation (semantic queries, enterprise-like):
    techqa   — IBM tech support questions        (1.8k)
    emanual  — user manual queries               (1.3k)
    covidqa  — COVID biomedical questions        (1.8k)
    expertqa — multi-domain expert questions     (2.0k)
    msmarco  — open-domain web search queries    (2.7k)

Avoid for RAG eval:
    finqa / tatqa — require numerical table reasoning (SQL pipeline better)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Default output directory — same as your existing eval files
OUT_DIR = SCRIPT_DIR / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Subset metadata ───────────────────────────────────────────────────────────
SUBSET_INFO = {
    "techqa":   "Tech support (IBM) — how-to and troubleshooting queries",
    "emanual":  "User manuals — product operation queries",
    "covidqa":  "COVID biomedical — scientific evidence queries",
    "expertqa": "Multi-domain expert questions — complex reasoning",
    "msmarco":  "Open-domain web search queries",
    "hotpotqa": "Multi-hop Wikipedia reasoning",
    "hagrid":   "Hallucination-grounded multi-hop QA",
    "pubmedqa": "Medical research yes/no + evidence",
}


def make_passage_id(text: str, subset: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:10]
    return f"{subset}_{h}"


def load_ragbench_subset(
    subset: str,
    split: str = "test",
    n: int | None = None,
) -> list[dict]:
    """Load a RAGBench subset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"  Loading rungalileo/ragbench [{subset}] split={split}...")
    ds = load_dataset("rungalileo/ragbench", subset, split=split, trust_remote_code=True)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    print(f"  Loaded {len(ds)} rows.")
    return list(ds)


def build_corpus_and_eval(
    rows: list[dict],
    subset: str,
) -> tuple[list[dict], list[dict]]:
    """
    Convert RAGBench rows into corpus + eval_set format.

    RAGBench schema (relevant fields):
      question          — natural language question
      documents         — list of 4 pre-retrieved passage strings
                          format: "Title: ...\nPassage: ..."
      response          — GPT-3.5 generated answer (our ground_truth)
      all_relevant_sentence_keys  — e.g. ["0b","0c","1d"] 
                          keys of sentences actually relevant to answer
      relevance_score   — float 0-1 (context relevance label)
      adherence_score   — bool (response grounded in context)
      ragas_faithfulness — float
      ragas_context_relevance — float

    Our corpus format (same as squad_corpus.json):
      { context_id, text, title, subset, source_doc_idx }

    Our eval format (same as squad_eval_set.json):
      { question, ground_truth, all_answers, reference_context,
        context_id, title, gold_passage_ids, relevance_score,
        adherence_score, subset }
    """
    corpus_map: dict[str, dict] = {}   # passage_id -> corpus entry
    eval_set: list[dict] = []

    for row in rows:
        question   = row["question"]
        documents  = row["documents"]        # list of 4 passage strings
        response   = row["response"]         # GPT-3.5 answer = our ground_truth
        rel_keys   = row.get("all_relevant_sentence_keys") or []
        rel_score  = row.get("relevance_score") or 0.0
        adh_score  = row.get("adherence_score") or False
        faith      = row.get("ragas_faithfulness") or 0.0
        ctx_rel    = row.get("ragas_context_relevance") or 0.0

        # Parse document index of relevant sentences: "0b" -> doc 0 is relevant
        relevant_doc_indices = set()
        for key in rel_keys:
            if key and key[0].isdigit():
                relevant_doc_indices.add(int(key[0]))

        gold_passage_ids = []
        reference_context = ""

        for doc_idx, doc_text in enumerate(documents):
            # Parse title and passage from "Title: ...\nPassage: ..."
            title = subset
            body  = doc_text
            if "Title:" in doc_text and "Passage:" in doc_text:
                parts = doc_text.split("\nPassage:", 1)
                title = parts[0].replace("Title:", "").strip()
                body  = parts[1].strip() if len(parts) > 1 else doc_text

            passage_id = make_passage_id(body, subset)

            if passage_id not in corpus_map:
                corpus_map[passage_id] = {
                    "context_id": passage_id,
                    "text":       body,
                    "title":      title,
                    "subset":     subset,
                }

            if doc_idx in relevant_doc_indices:
                gold_passage_ids.append(passage_id)
                if not reference_context:
                    reference_context = body

        # Use first document as fallback reference if no gold label
        if not reference_context and documents:
            parts = documents[0].split("\nPassage:", 1)
            reference_context = parts[1].strip() if len(parts) > 1 else documents[0]

        eval_set.append({
            "question":           question,
            "ground_truth":       response,          # paragraph answer
            "all_answers":        [response],
            "reference_context":  reference_context,
            "context_id":         gold_passage_ids[0] if gold_passage_ids else "",
            "gold_passage_ids":   gold_passage_ids,  # ALL relevant passage IDs
            "title":              subset,
            "relevance_score":    float(rel_score),
            "adherence_score":    bool(adh_score),
            "ragas_faithfulness": float(faith),
            "ragas_context_relevance": float(ctx_rel),
            "subset":             subset,
            # How many of the 4 docs are relevant — useful for analysis
            "n_relevant_docs":    len(relevant_doc_indices),
        })

    corpus = list(corpus_map.values())
    return corpus, eval_set


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare RAGBench corpus and eval set for your RAG pipeline"
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["techqa", "emanual", "covidqa"],
        choices=list(SUBSET_INFO.keys()),
        help="RAGBench subsets to include (default: techqa emanual covidqa)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Max rows per subset (default: all). Use 100-200 for quick testing.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    all_corpus: list[dict] = []
    all_eval:   list[dict] = []
    seen_ids: set[str] = set()

    print(f"\nPreparing RAGBench — subsets: {args.subsets}")
    print(f"Split: {args.split}  |  N per subset: {args.n or 'all'}")
    print("="*60)

    for subset in args.subsets:
        print(f"\n[{subset}] {SUBSET_INFO.get(subset, '')}")
        rows = load_ragbench_subset(subset, split=args.split, n=args.n)

        corpus, eval_set = build_corpus_and_eval(rows, subset)

        # Deduplicate corpus across subsets
        new_passages = 0
        for entry in corpus:
            if entry["context_id"] not in seen_ids:
                seen_ids.add(entry["context_id"])
                all_corpus.append(entry)
                new_passages += 1

        all_eval.extend(eval_set)

        # Stats
        n_with_gold = sum(1 for e in eval_set if e["gold_passage_ids"])
        avg_rel = sum(e["relevance_score"] for e in eval_set) / len(eval_set)
        avg_faith = sum(e["ragas_faithfulness"] for e in eval_set) / len(eval_set)
        print(f"  Questions     : {len(eval_set)}")
        print(f"  Unique passages: {new_passages} new (total corpus: {len(all_corpus)})")
        print(f"  Has gold labels: {n_with_gold}/{len(eval_set)}")
        print(f"  Avg relevance : {avg_rel:.3f}")
        print(f"  Avg faithfulness: {avg_faith:.3f}")

    # Save
    corpus_path = args.out / "ragbench_corpus.json"
    eval_path   = args.out / "ragbench_eval_set.json"

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(all_corpus, f, indent=2, ensure_ascii=False)

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(all_eval, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Corpus   : {len(all_corpus)} unique passages → {corpus_path}")
    print(f"  Eval set : {len(all_eval)} questions → {eval_path}")
    print(f"\nNext step:")
    print(f"  python evaluation/Unstructured/run_ragbench_eval.py C1")


if __name__ == "__main__":
    main()
