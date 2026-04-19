"""
evaluation/Unstructured/run_ragbench_eval.py
=============================================
Runs one benchmark configuration against the RAGBench corpus.
Drop-in replacement for run_config_eval.py — same config table, same output format.

Key differences from the SQuAD eval:
  1. Paragraph-length answers  → RAGAS answer_relevancy works correctly
  2. Larger, denser corpus      → reranker adds genuine value
  3. Gold passage IDs available → extra metric: gold_hit_rate (did we
     retrieve the RAGBench-labeled relevant passage?)
  4. Generation prompt updated  → "2-3 sentences" not "one short phrase"

Usage:
    # First prepare the data (once):
    python evaluation/Unstructured/prepare_ragbench_dataset.py

    # Then run configs:
    python evaluation/Unstructured/run_ragbench_eval.py C1
    python evaluation/Unstructured/run_ragbench_eval.py T5A
    python evaluation/Unstructured/run_ragbench_eval.py T3A

    # Run all T-series:
    for cfg in T2A T2B T3A T3B T4A T4B T5A T5B; do
        python evaluation/Unstructured/run_ragbench_eval.py $cfg
    done

Output (evaluation/Unstructured/results/):
    ragbench_<cfg>_rag_results.json   — retrieved contexts + generated answers
    ragbench_<cfg>_ragas_scores.json  — RAGAS metrics + per-question breakdown
"""

import os
import sys
import json
import time
from pathlib import Path

# ── Config table (identical to run_config_eval.py) ────────────────────────────
CONFIGS = {
    "C1":  dict(retrieval_mode="faiss",  reranker=False, top_k=5,  chunk_size=500),
    "C2":  dict(retrieval_mode="faiss",  reranker=True,  top_k=5,  chunk_size=500),
    "C3":  dict(retrieval_mode="hybrid", reranker=False, top_k=5,  chunk_size=500),
    "C4":  dict(retrieval_mode="hybrid", reranker=True,  top_k=5,  chunk_size=500),
    "C5":  dict(retrieval_mode="hybrid", reranker=True,  top_k=3,  chunk_size=500),
    "C6":  dict(retrieval_mode="hybrid", reranker=True,  top_k=9,  chunk_size=500),
    "C7":  dict(retrieval_mode="faiss",  reranker=False, top_k=5,  chunk_size=250),
    "C8":  dict(retrieval_mode="faiss",  reranker=True,  top_k=5,  chunk_size=250),
    "C9":  dict(retrieval_mode="hybrid", reranker=False, top_k=5,  chunk_size=250),
    "C10": dict(retrieval_mode="hybrid", reranker=True,  top_k=5,  chunk_size=250),
    "C11": dict(retrieval_mode="hybrid", reranker=True,  top_k=3,  chunk_size=250),
    "C12": dict(retrieval_mode="hybrid", reranker=True,  top_k=9,  chunk_size=250),
    "T2A": dict(retrieval_mode="hybrid", reranker=True,  top_k=5,  chunk_size=500, fetch_k=100, rerank_pool=60),
    "T2B": dict(retrieval_mode="hybrid", reranker=True,  top_k=10, chunk_size=500, fetch_k=100, rerank_pool=60),
    "T3A": dict(retrieval_mode="hybrid", reranker=True,  top_k=5,  chunk_size=500, fetch_k=100, rerank_pool=20),
    "T3B": dict(retrieval_mode="hybrid", reranker=True,  top_k=10, chunk_size=500, fetch_k=100, rerank_pool=20),
    "T4A": dict(retrieval_mode="hybrid", reranker=True,  top_k=5,  chunk_size=500, fetch_k=60,  rerank_pool=60),
    "T4B": dict(retrieval_mode="hybrid", reranker=True,  top_k=10, chunk_size=500, fetch_k=60,  rerank_pool=60),
    "T5A": dict(retrieval_mode="hybrid", reranker=False, top_k=5,  chunk_size=500, fetch_k=100, rerank_pool=0),
    "T5B": dict(retrieval_mode="hybrid", reranker=False, top_k=10, chunk_size=500, fetch_k=100, rerank_pool=0),
}

if len(sys.argv) < 2 or sys.argv[1].upper() not in CONFIGS:
    print("Usage: python run_ragbench_eval.py <config_id>")
    print(f"Valid configs: {', '.join(CONFIGS)}")
    sys.exit(1)

cfg_name = sys.argv[1].upper()
cfg = CONFIGS[cfg_name]

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Apply config overrides BEFORE importing pipeline modules ──────────────────
import backend.config as config

config.CHUNK_SIZE    = cfg["chunk_size"]
config.CHUNK_OVERLAP = 50 if cfg["chunk_size"] == 250 else 100
config.RAG_RETRIEVAL_MODE         = cfg["retrieval_mode"]
config.RAG_ENABLE_SEMANTIC_RERANK = cfg["reranker"]
config.RAG_TOP_K                  = cfg["top_k"]
config.RAG_FETCH_K                = cfg.get("fetch_k", 0)
config.RAG_RERANK_POOL            = cfg.get("rerank_pool", 0)

# Separate RAGBench index per chunk size
INDEX_DIR = SCRIPT_DIR / f"ragbench_index_{cfg['chunk_size']}"
config.INDEX_DIR = INDEX_DIR
INDEX_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUT   = RESULTS_DIR / f"ragbench_{cfg_name}_rag_results.json"
RAGAS_OUT = RESULTS_DIR / f"ragbench_{cfg_name}_ragas_scores.json"

# ── Data paths ────────────────────────────────────────────────────────────────
CORPUS_PATH = SCRIPT_DIR / "evaluation" / "ragbench_corpus.json"
EVAL_PATH   = SCRIPT_DIR / "evaluation" / "ragbench_eval_set.json"

if not CORPUS_PATH.exists() or not EVAL_PATH.exists():
    print(f"ERROR: RAGBench data not found.")
    print(f"  Expected: {CORPUS_PATH}")
    print(f"  Run first: python evaluation/Unstructured/prepare_ragbench_dataset.py")
    sys.exit(1)

print(f"\n{'='*65}")
print(f"  Config         : ragbench_{cfg_name}")
print(f"  retrieval_mode : {cfg['retrieval_mode']}")
print(f"  reranker       : {cfg['reranker']}")
print(f"  top_k          : {cfg['top_k']}")
print(f"  chunk_size     : {cfg['chunk_size']}  (overlap={config.CHUNK_OVERLAP})")
if cfg.get("fetch_k"):
    print(f"  fetch_k        : {cfg['fetch_k']}")
if cfg.get("rerank_pool"):
    print(f"  rerank_pool    : {cfg['rerank_pool']}")
print(f"  index_dir      : {INDEX_DIR.name}/")
print(f"{'='*65}\n")

# ── Import pipeline ───────────────────────────────────────────────────────────
from backend.rag import TextChunker, SentenceTransformerEmbedder, FAISSVectorStore, RagRetriever
from backend.rag.loader import Document
from openai import OpenAI

embedder      = SentenceTransformerEmbedder(config.EMBEDDING_MODEL_NAME)
lc_embeddings = embedder.get_langchain_embeddings()
vector_store  = FAISSVectorStore(
    index_path    = str(INDEX_DIR / "documents"),
    metadata_path = str(INDEX_DIR / "documents_meta.json"),
    embeddings    = lc_embeddings,
)

# ── STEP 1: Ingest (once per chunk size) ─────────────────────────────────────
index_sentinel = INDEX_DIR / "documents.faiss"
if not index_sentinel.exists():
    print(f"[INGEST] Building RAGBench index (chunk_size={cfg['chunk_size']})...")

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"[INGEST] {len(corpus)} passages loaded.")

    chunker = TextChunker(
        chunk_size    = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP,
    )
    ingest_retriever = RagRetriever(embedder=embedder, vector_store=vector_store)

    all_chunks = []
    for item in corpus:
        doc = Document(
            text     = item["text"],
            source   = item["context_id"],
            metadata = {
                "title":      item.get("title", ""),
                "context_id": item["context_id"],
                "subset":     item.get("subset", ""),
            },
        )
        all_chunks.extend(chunker.chunk_documents([doc]))

    print(f"[INGEST] {len(all_chunks)} chunks after splitting.")
    BATCH = 500
    for i in range(0, len(all_chunks), BATCH):
        ingest_retriever.index_chunks(all_chunks[i: i + BATCH])
        print(f"[INGEST]   {min(i + BATCH, len(all_chunks))}/{len(all_chunks)} indexed...")
    print(f"[INGEST] Done.\n")
else:
    print(f"[INGEST] Reusing existing index ({INDEX_DIR.name}/)\n")

# ── STEP 2: Retrieval + generation ───────────────────────────────────────────
vector_store.load()
retriever = RagRetriever(embedder=embedder, vector_store=vector_store)
_client   = OpenAI(api_key=config.OPENAI_API_KEY)

# KEY CHANGE from SQuAD: ask for 2-3 sentences, not "one short phrase"
# This fixes:
#   1. RAGAS answer_relevancy (needs paragraph-length answer)
#   2. False "Not found" responses (model had enough context but gave up)
ANSWER_PROMPT = """\
Answer the following question in 2-3 sentences using ONLY the context below.
Be specific and ground every claim in the provided context.
If the answer cannot be found in the context, reply exactly: "Not found."

Context:
{context}

Question: {question}

Answer:"""

with open(EVAL_PATH, encoding="utf-8") as f:
    eval_set = json.load(f)

# Limit to 50 questions per config
eval_set = eval_set[:50]

print(f"[EVAL] Running {len(eval_set)} questions through ragbench_{cfg_name}...\n")

results = []
for i, item in enumerate(eval_set):
    print(f"[{i+1:>4}/{len(eval_set)}] [{item.get('subset','')}] {item['question'][:65]}...")

    # Retrieve
    chunks             = retriever.retrieve(item["question"], top_k=cfg["top_k"])
    retrieved_contexts = [c["text"]           for c in chunks]
    retrieved_sources  = [c.get("source", "") for c in chunks]

    # Gold hit: did we retrieve any of the RAGBench-labeled relevant passages?
    gold_ids = set(item.get("gold_passage_ids", []))
    gold_hit = any(src in gold_ids for src in retrieved_sources) if gold_ids else None

    # Also check substring match against ground truth (paragraph-level)
    gt_lower = item["ground_truth"].lower()
    answer_in_retrieved = any(
        # check 50+ char overlap — paragraph answers won't be exact substrings
        any(sent.lower() in ctx.lower() for sent in gt_lower.split(". ")[:2] if len(sent) > 30)
        for ctx in retrieved_contexts
    )

    context_str = "\n\n---\n\n".join(retrieved_contexts[:cfg["top_k"]])
    try:
        resp = _client.chat.completions.create(
            model       = config.SQL_OPENAI_MODEL,
            messages    = [{"role": "user", "content": ANSWER_PROMPT.format(
                context  = context_str[:4000],   # larger window for RAGBench
                question = item["question"],
            )}],
            temperature = 0.0,
            max_tokens  = 300,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"[ERROR: {e}]"

    results.append({
        "question":             item["question"],
        "answer":               answer,
        "ground_truth":         item["ground_truth"],
        "all_answers":          item["all_answers"],
        "retrieved_contexts":   retrieved_contexts,
        "reference_contexts":   [item["reference_context"]],
        "context_id":           item["context_id"],
        "gold_passage_ids":     item.get("gold_passage_ids", []),
        "retrieved_sources":    retrieved_sources,
        "gold_hit":             gold_hit,            # NEW: did we find the gold passage?
        "answer_in_retrieved":  answer_in_retrieved,
        "subset":               item.get("subset", ""),
        "title":                item.get("title", ""),
        # RAGBench ground truth labels for comparison
        "ragbench_relevance_score":    item.get("relevance_score"),
        "ragbench_adherence_score":    item.get("adherence_score"),
        "ragbench_faithfulness":       item.get("ragas_faithfulness"),
        "ragbench_context_relevance":  item.get("ragas_context_relevance"),
    })
    time.sleep(0.5)

with open(RAW_OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Summary stats
valid = [r for r in results if not r["answer"].startswith("[ERROR")]
gold_results = [r for r in valid if r["gold_hit"] is not None]
gold_hit_rate = sum(r["gold_hit"] for r in gold_results) / len(gold_results) if gold_results else 0
not_found = sum(1 for r in valid if "not found" in r["answer"].lower())

# Per-subset breakdown
from collections import defaultdict
by_subset = defaultdict(list)
for r in valid:
    by_subset[r["subset"]].append(r)

print(f"\n[EVAL] Saved {len(results)} results → {RAW_OUT.name}")
print(f"[EVAL] Gold passage hit rate : {gold_hit_rate:.1%}  ({len(gold_results)} with gold labels)")
print(f"[EVAL] 'Not found' responses : {not_found}/{len(valid)}")
print(f"\n[EVAL] Per-subset gold hit rates:")
for subset, rows in sorted(by_subset.items()):
    gr = [r for r in rows if r["gold_hit"] is not None]
    if gr:
        rate = sum(r["gold_hit"] for r in gr) / len(gr)
        print(f"  {subset:<12}: {rate:.1%} ({len(gr)} questions)")

# ── STEP 3: RAGAS scoring ─────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas.llms import llm_factory
from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

print(f"\n[RAGAS] Scoring {len(valid)}/{len(results)} results...")

ragas_data = Dataset.from_dict({
    "user_input":         [r["question"]           for r in valid],
    "response":           [r["answer"]             for r in valid],
    "retrieved_contexts": [r["retrieved_contexts"] for r in valid],
    "reference":          [r["ground_truth"]       for r in valid],
})

llm_ragas = llm_factory(
    model  = config.SQL_OPENAI_MODEL,
    client = OpenAI(api_key=config.OPENAI_API_KEY),
)
emb_ragas = LangchainEmbeddingsWrapper(
    LCHuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
)

ragas_result = evaluate(
    dataset          = ragas_data,
    metrics          = [answer_relevancy, faithfulness, context_precision, context_recall],
    llm              = llm_ragas,
    embeddings       = emb_ragas,
    raise_exceptions = False,
)
scores_df = ragas_result.to_pandas()

METRIC_COLS = {
    "answer_relevancy":  "answer_relevancy",
    "faithfulness":      "faithfulness",
    "context_precision": "context_precision",
    "context_recall":    "context_recall",
}
metrics = {
    label: float(scores_df[col].mean())
    for label, col in METRIC_COLS.items()
    if col in scores_df.columns
}

# Per-subset RAGAS breakdown
subset_labels = [r["subset"] for r in valid]
scores_df["subset"] = subset_labels
per_subset_metrics = {}
for subset in scores_df["subset"].unique():
    sub_df = scores_df[scores_df["subset"] == subset]
    per_subset_metrics[subset] = {
        col: float(sub_df[col].mean())
        for col in METRIC_COLS.values()
        if col in sub_df.columns
    }

output = {
    "config":     f"ragbench_{cfg_name}",
    "dataset":    "ragbench",
    "parameters": cfg,
    "overall":    {
        **metrics,
        "gold_hit_rate":      gold_hit_rate,
        "not_found_rate":     not_found / len(valid) if valid else 0,
    },
    "per_subset":  per_subset_metrics,
    "n_evaluated": len(valid),
    "per_question": scores_df.to_dict(orient="records"),
}

with open(RAGAS_OUT, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# ── Final report ──────────────────────────────────────────────────────────────
def bar(v: float) -> str:
    filled = int(v * 20)
    return "#" * filled + "-" * (20 - filled)

print(f"\n{'='*65}")
print(f"  RAGAS REPORT  —  ragbench_{cfg_name}")
print(f"  mode={cfg['retrieval_mode']}  reranker={cfg['reranker']}  "
      f"top_k={cfg['top_k']}  chunk={cfg['chunk_size']}")
print(f"{'='*65}")
print(f"  {'answer_relevancy':<22}  {metrics.get('answer_relevancy',0):.4f}   "
      f"{bar(metrics.get('answer_relevancy',0))}")
print(f"  {'faithfulness':<22}  {metrics.get('faithfulness',0):.4f}   "
      f"{bar(metrics.get('faithfulness',0))}")
print(f"  {'context_precision':<22}  {metrics.get('context_precision',0):.4f}   "
      f"{bar(metrics.get('context_precision',0))}")
print(f"  {'context_recall':<22}  {metrics.get('context_recall',0):.4f}   "
      f"{bar(metrics.get('context_recall',0))}")
print(f"  {'gold_hit_rate':<22}  {gold_hit_rate:.4f}   {bar(gold_hit_rate)}")
print(f"{'='*65}")

if per_subset_metrics:
    print(f"\n  Per-subset context_precision:")
    for subset, m in sorted(per_subset_metrics.items()):
        cp = m.get("context_precision", 0)
        print(f"    {subset:<12}  {cp:.4f}  {bar(cp)}")

print(f"\n  Raw results   →  results/ragbench_{cfg_name}_rag_results.json")
print(f"  RAGAS scores  →  results/ragbench_{cfg_name}_ragas_scores.json")
print(f"{'='*65}\n")
