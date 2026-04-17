"""
evaluation/NaturalQuestions/prepare_nq_dataset.py

Builds two files in the same format your existing run_config_eval.py expects:
  nq_corpus.json    — passage contexts to ingest into FAISS
  nq_eval_set.json  — 50 questions with ground truth answers + reference contexts

Source: HuggingFace datasets — 'natural_questions' (google/natural_questions, trust_remote_code=True)
  OR the lighter 'nq_open' which is already cleaned and much faster to download.

We use 'nq_open' by default — it has:
  - Clean short answers (no HTML, no document markup)
  - Already filtered to questions with unambiguous short answers
  - Much smaller download (~50MB vs ~40GB for full NQ)

Install once:
  pip install datasets

Usage:
  python prepare_nq_dataset.py
  python prepare_nq_dataset.py --split validation --n 50 --out ./
"""

import argparse
import json
import hashlib
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Remove excess whitespace and common Wikipedia artifacts."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[\d+\]', '', text)   # remove citation markers like [1]
    text = re.sub(r'^\s*\.\s*', '', text)  # leading dot artifacts
    return text.strip()


def make_context_id(text: str) -> str:
    return "nq_ctx_" + hashlib.md5(text.encode()).hexdigest()[:8]


def build_nq_datasets(split: str, n: int, out_dir: Path) -> None:
    from datasets import load_dataset

    print(f"[NQ] Loading nq_open ({split} split)...")
    # nq_open has 'question' and 'answer' (list of strings) fields
    # No passage context — we need to source passages separately.
    # Strategy: use 'nq_open' for Q&A pairs, then fetch Wikipedia
    # passages via the 'wikipedia' dataset as our retrieval corpus.

    # Actually use the cleaner approach: load from BeIR's NQ version
    # which already pairs questions with their gold passage.
    # BeIR NQ: https://huggingface.co/datasets/BeIR/nq
    print("[NQ] Loading BeIR/nq (contains questions + gold passages)...")

    try:
        dataset = load_dataset("BeIR/nq", "corpus", trust_remote_code=True)
        queries = load_dataset("BeIR/nq", "queries", trust_remote_code=True)
        qrels   = load_dataset("BeIR/nq-qrels", trust_remote_code=True)
    except Exception as e:
        print(f"[NQ] BeIR/nq load failed: {e}")
        print("[NQ] Falling back to nq_open + wikipedia passages...")
        _build_from_nq_open(split, n, out_dir)
        return

    print(f"[NQ] BeIR corpus size : {len(dataset['corpus'])}")
    print(f"[NQ] BeIR queries size: {len(queries['queries'])}")

    # Build corpus lookup: doc_id -> {text, title}
    corpus_lookup = {
        row["_id"]: {"text": clean_text(row["text"]), "title": row.get("title", "")}
        for row in dataset["corpus"]
    }

    # Build query lookup: query_id -> question text
    query_lookup = {
        row["_id"]: row["text"]
        for row in queries["queries"]
    }

    # Get relevance pairs from qrels (test split)
    qrel_split = "test" if "test" in qrels else list(qrels.keys())[0]
    print(f"[NQ] Using qrels split: {qrel_split}")

    # Group relevant doc_ids per query
    from collections import defaultdict
    relevant_docs: dict[str, list[str]] = defaultdict(list)
    for row in qrels[qrel_split]:
        if int(row.get("score", 0)) > 0:
            relevant_docs[row["query-id"]].append(row["corpus-id"])

    # Filter to queries that have exactly one gold passage (cleaner eval)
    clean_pairs = [
        (qid, relevant_docs[qid][0])
        for qid, docs in relevant_docs.items()
        if len(docs) == 1 and qid in query_lookup and relevant_docs[qid][0] in corpus_lookup
    ]

    print(f"[NQ] Clean Q-passage pairs: {len(clean_pairs)}")

    # Sample n pairs
    import random
    random.seed(42)
    sampled = random.sample(clean_pairs, min(n, len(clean_pairs)))

    # ── Build corpus ──────────────────────────────────────────────────────────
    # Include gold passages + extra distractors for a realistic retrieval task
    gold_doc_ids = {doc_id for _, doc_id in sampled}

    # Add ~5x distractors from corpus (random sample)
    all_doc_ids = list(corpus_lookup.keys())
    random.shuffle(all_doc_ids)
    distractor_ids = [d for d in all_doc_ids if d not in gold_doc_ids][:n * 5]
    corpus_ids = list(gold_doc_ids) + distractor_ids

    corpus_out = []
    for doc_id in corpus_ids:
        entry = corpus_lookup[doc_id]
        ctx_id = f"nq_ctx_{doc_id}"
        corpus_out.append({
            "context_id": ctx_id,
            "text":       entry["text"],
            "title":      entry["title"],
        })

    corpus_path = out_dir / "nq_corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus_out, f, indent=2, ensure_ascii=False)
    print(f"[NQ] Saved corpus: {len(corpus_out)} passages → {corpus_path}")

    # ── Build eval set ────────────────────────────────────────────────────────
    # For NQ we don't have short answer strings in BeIR format directly.
    # We derive ground_truth as the first sentence of the gold passage
    # that contains a named entity (heuristic). Better: use nq_open answers.
    # Load nq_open to get actual short answers matched by question text.
    print("[NQ] Loading nq_open for short answers...")
    try:
        nq_open = load_dataset("nq_open", split="validation")
        nq_open_lookup = {row["question"].lower().strip(): row["answer"] for row in nq_open}
    except Exception:
        nq_open_lookup = {}
        print("[NQ] nq_open unavailable — ground truth will be passage-derived")

    eval_out = []
    for qid, doc_id in sampled:
        question     = query_lookup[qid]
        ctx_id       = f"nq_ctx_{doc_id}"
        passage_text = corpus_lookup[doc_id]["text"]
        title        = corpus_lookup[doc_id]["title"]

        # Try to get short answer from nq_open
        answers = nq_open_lookup.get(question.lower().strip(), [])
        ground_truth = answers[0] if answers else ""

        # Fallback: use passage title as ground truth hint (weak but better than empty)
        if not ground_truth:
            ground_truth = title

        eval_out.append({
            "question":         question,
            "ground_truth":     ground_truth,
            "all_answers":      answers if answers else [ground_truth],
            "reference_context": passage_text,
            "context_id":       ctx_id,
            "title":            title,
        })

    eval_path = out_dir / "nq_eval_set.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2, ensure_ascii=False)
    print(f"[NQ] Saved eval set: {len(eval_out)} questions → {eval_path}")
    print("\n[NQ] Done. Next step:")
    print("     Update run_config_eval.py to point at nq_corpus.json / nq_eval_set.json")
    print("     and set INDEX_DIR to a separate nq_index_<chunk_size>/ folder.\n")


def _build_from_nq_open(split: str, n: int, out_dir: Path) -> None:
    """
    Fallback: build from nq_open + wikipedia passages.
    nq_open gives us clean Q&A pairs. We fetch Wikipedia intro
    paragraphs as both the gold context and corpus.
    Requires: pip install wikipedia-api  (or just use requests)
    """
    from datasets import load_dataset
    import requests

    print("[NQ fallback] Loading nq_open...")
    ds = load_dataset("nq_open", split=split)

    import random
    random.seed(42)
    indices = random.sample(range(len(ds)), min(n * 3, len(ds)))
    sampled_rows = [ds[i] for i in indices]

    def fetch_wikipedia_intro(title: str) -> str:
        """Fetch intro paragraph from Wikipedia API."""
        try:
            resp = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_"),
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json().get("extract", "")
        except Exception:
            pass
        return ""

    corpus_out = []
    eval_out   = []
    seen_titles: set[str] = set()

    for row in sampled_rows:
        if len(eval_out) >= n:
            break
        question    = row["question"]
        answers     = row["answer"]
        ground_truth = answers[0] if answers else ""

        # Use first answer as Wikipedia lookup hint
        passage = fetch_wikipedia_intro(ground_truth) or fetch_wikipedia_intro(question.split()[-1])
        if not passage or len(passage) < 50:
            continue

        ctx_id = make_context_id(passage)
        title  = ground_truth

        if ctx_id not in seen_titles:
            seen_titles.add(ctx_id)
            corpus_out.append({"context_id": ctx_id, "text": clean_text(passage), "title": title})

        eval_out.append({
            "question":          question,
            "ground_truth":      ground_truth,
            "all_answers":       answers,
            "reference_context": clean_text(passage),
            "context_id":        ctx_id,
            "title":             title,
        })

    # Pad corpus with extra Wikipedia intros for distractor passages
    print(f"[NQ fallback] Collected {len(eval_out)} Q&A pairs with Wikipedia passages")

    corpus_path = out_dir / "nq_corpus.json"
    eval_path   = out_dir / "nq_eval_set.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus_out, f, indent=2, ensure_ascii=False)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2, ensure_ascii=False)
    print(f"[NQ fallback] corpus → {corpus_path}  ({len(corpus_out)} passages)")
    print(f"[NQ fallback] eval   → {eval_path}  ({len(eval_out)} questions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", help="Dataset split to use")
    parser.add_argument("--n",     type=int, default=50,  help="Number of Q&A pairs")
    parser.add_argument("--out",   default=str(Path(__file__).resolve().parent / "evaluation"),
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_nq_datasets(split=args.split, n=args.n, out_dir=out_dir)
