"""
evaluation/Unstructured/run_config_eval.py
Run one benchmark configuration for the unstructured RAG pipeline.

Steps performed automatically:
  1. Apply config overrides (retrieval_mode, reranker, top_k, chunk_size,
     fetch_k, rerank_pool)
  2. Ingest SQuAD corpus into a chunk-size-specific FAISS index (skipped if
     the index already exists — each chunk size is ingested only once)
  3. Run the first 50 SQuAD questions through RagRetriever + OpenAI generation
  4. Score with RAGAS (answer_relevancy, faithfulness, context_precision,
     context_recall) and save per-config JSON files

Output files (in evaluation/Unstructured/results/):
  <cfg_name>_rag_results.json   — retrieved contexts + generated answers
  <cfg_name>_ragas_scores.json  — RAGAS metrics + per-question breakdown

Usage:
  python evaluation/Unstructured/run_config_eval.py C1
  python evaluation/Unstructured/run_config_eval.py T2a
"""

import os
import sys
import json
import time
from pathlib import Path

# ── Config table ─────────────────────────────────────────────────────────────
CONFIGS = {
    # ── C-series: retrieval mode x reranker x top_k x chunk_size ────────────
    "C1":  dict(retrieval_mode="faiss",  reranker=False, top_k=5,  chunk_size=500),
    "C2":  dict(retrieval_mode="faiss",  reranker=True,  top_k=5,  chunk_size=500),
    "C3":  dict(retrieval_mode="hybrid", reranker=False, top_k=5,  chunk_size=500),
    "C4":  dict(retrieval_mode="hybrid", reranker=True,  top_k=5,  chunk_size=500),  # proposed
    "C5":  dict(retrieval_mode="hybrid", reranker=True,  top_k=3,  chunk_size=500),
    "C6":  dict(retrieval_mode="hybrid", reranker=True,  top_k=9,  chunk_size=500),
    "C7":  dict(retrieval_mode="faiss",  reranker=False, top_k=5,  chunk_size=250),
    "C8":  dict(retrieval_mode="faiss",  reranker=True,  top_k=5,  chunk_size=250),
    "C9":  dict(retrieval_mode="hybrid", reranker=False, top_k=5,  chunk_size=250),
    "C10": dict(retrieval_mode="hybrid", reranker=True,  top_k=5,  chunk_size=250),
    "C11": dict(retrieval_mode="hybrid", reranker=True,  top_k=3,  chunk_size=250),
    "C12": dict(retrieval_mode="hybrid", reranker=True,  top_k=9,  chunk_size=250),
    # ── T-series: fixed retrieve=100, varying rerank_pool and top_k ──────────
    # hybrid + bge-reranker-base, chunk_size=500
    "T2A": dict(retrieval_mode="hybrid", reranker=True, top_k=5,  chunk_size=500, fetch_k=100, rerank_pool=60),
    "T2B": dict(retrieval_mode="hybrid", reranker=True, top_k=10, chunk_size=500, fetch_k=100, rerank_pool=60),
    "T3A": dict(retrieval_mode="hybrid", reranker=True, top_k=5,  chunk_size=500, fetch_k=100, rerank_pool=20),
    "T3B": dict(retrieval_mode="hybrid", reranker=True, top_k=10, chunk_size=500, fetch_k=100, rerank_pool=20),
    "T4A": dict(retrieval_mode="hybrid", reranker=True, top_k=5,  chunk_size=500, fetch_k=60, rerank_pool=60),
    "T4B": dict(retrieval_mode="hybrid", reranker=True, top_k=10, chunk_size=500, fetch_k=60, rerank_pool=60),
    "T5A": dict(retrieval_mode="hybrid", reranker=False, top_k=5,  chunk_size=500, fetch_k=100, rerank_pool=0),
    "T5B": dict(retrieval_mode="hybrid", reranker=False, top_k=10, chunk_size=500, fetch_k=100, rerank_pool=0),
}

# ── Parse argument ────────────────────────────────────────────────────────────
if len(sys.argv) < 2 or sys.argv[1].upper() not in CONFIGS:
    print("Usage: python run_config_eval.py <config_id>")
    print(f"Valid configs: {', '.join(CONFIGS)}")
    sys.exit(1)

cfg_name = sys.argv[1].upper()
cfg = CONFIGS[cfg_name]

SCRIPT_DIR   = Path(__file__).resolve().parent   # .../evaluation/Unstructured/
PROJECT_ROOT = SCRIPT_DIR.parent.parent           # .../Adaptive-Agentic-RAG/
sys.path.insert(0, str(PROJECT_ROOT))

# ── Apply config overrides BEFORE importing pipeline modules ─────────────────
import backend.config as config  # noqa: E402

config.CHUNK_SIZE    = cfg["chunk_size"]
config.CHUNK_OVERLAP = 50 if cfg["chunk_size"] == 250 else 100
config.RAG_RETRIEVAL_MODE         = cfg["retrieval_mode"]
config.RAG_ENABLE_SEMANTIC_RERANK = cfg["reranker"]
config.RAG_TOP_K                  = cfg["top_k"]
config.RAG_FETCH_K                = cfg.get("fetch_k", 0)
config.RAG_RERANK_POOL            = cfg.get("rerank_pool", 0)

# Separate index dir per chunk size so we never mix differently-chunked docs
INDEX_DIR = SCRIPT_DIR / f"squad_index_{cfg['chunk_size']}"
config.INDEX_DIR = INDEX_DIR
INDEX_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUT   = RESULTS_DIR / f"{cfg_name}_rag_results.json"
RAGAS_OUT = RESULTS_DIR / f"{cfg_name}_ragas_scores.json"

print(f"\n{'='*62}")
print(f"  Config         : {cfg_name}")
print(f"  retrieval_mode : {cfg['retrieval_mode']}")
print(f"  reranker       : {cfg['reranker']}")
print(f"  top_k          : {cfg['top_k']}")
print(f"  chunk_size     : {cfg['chunk_size']}  (overlap={config.CHUNK_OVERLAP})")
if cfg.get("fetch_k"):
    print(f"  fetch_k        : {cfg['fetch_k']}  (explicit)")
if cfg.get("rerank_pool"):
    print(f"  rerank_pool    : {cfg['rerank_pool']}")
print(f"  index_dir      : {INDEX_DIR.name}/")
print(f"{'='*62}\n")

# ── Import pipeline (after overrides) ────────────────────────────────────────
from backend.rag import (  # noqa: E402
    TextChunker, SentenceTransformerEmbedder,
    FAISSVectorStore, RagRetriever,
)
from backend.rag.loader import Document  # noqa: E402

embedder      = SentenceTransformerEmbedder(config.EMBEDDING_MODEL_NAME)
lc_embeddings = embedder.get_langchain_embeddings()
vector_store  = FAISSVectorStore(
    index_path    = str(INDEX_DIR / "documents"),
    metadata_path = str(INDEX_DIR / "documents_meta.json"),
    embeddings    = lc_embeddings,
)

# ── STEP 1 · Ingest (only if this chunk-size index doesn't exist yet) ─────────
index_sentinel = INDEX_DIR / "documents.faiss"
if not index_sentinel.exists():
    print(f"[INGEST] Index not found for chunk_size={cfg['chunk_size']} — building now...")

    corpus_path = SCRIPT_DIR / "evaluation" / "squad_corpus.json"
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"[INGEST] {len(corpus)} contexts loaded from corpus.")

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
            metadata = {"title": item["title"], "context_id": item["context_id"]},
        )
        all_chunks.extend(chunker.chunk_documents([doc]))

    print(f"[INGEST] {len(all_chunks)} chunks after splitting.")
    BATCH = 500
    for i in range(0, len(all_chunks), BATCH):
        ingest_retriever.index_chunks(all_chunks[i : i + BATCH])
        print(f"[INGEST]   {min(i + BATCH, len(all_chunks))}/{len(all_chunks)} indexed...")
    print(f"[INGEST] Done. Index saved to {INDEX_DIR.name}/\n")
else:
    print(f"[INGEST] Reusing existing index ({INDEX_DIR.name}/)\n")

# ── STEP 2 · Retrieval + answer generation ────────────────────────────────────
vector_store.load()
retriever = RagRetriever(embedder=embedder, vector_store=vector_store)

from openai import OpenAI  # noqa: E402

_client = OpenAI(api_key=config.OPENAI_API_KEY)

ANSWER_PROMPT = """\
Answer the question in one short phrase using ONLY the context below.
Do not add any information not present in the context.
If the answer cannot be found in the context, reply exactly: "Not found."

Context:
{context}

Question: {question}

Answer:"""

eval_path = SCRIPT_DIR / "evaluation" / "squad_eval_set.json"
with open(eval_path, encoding="utf-8") as f:
    eval_set = json.load(f)[:50]

results = []
for i, item in enumerate(eval_set):
    print(f"[EVAL {cfg_name}] [{i+1:>3}/{len(eval_set)}] {item['question'][:65]}...")

    chunks             = retriever.retrieve(item["question"], top_k=config.RAG_TOP_K)
    retrieved_contexts = [c["text"]          for c in chunks]
    retrieved_sources  = [c.get("source","") for c in chunks]

    answer_in_retrieved = any(
        item["ground_truth"].lower() in ctx.lower()
        for ctx in retrieved_contexts
    )

    context_str = "\n\n---\n\n".join(retrieved_contexts[:cfg["top_k"]])
    try:
        resp = _client.chat.completions.create(
            model       = config.SQL_OPENAI_MODEL,
            messages    = [{"role": "user", "content": ANSWER_PROMPT.format(
                context  = context_str[:2500],
                question = item["question"],
            )}],
            temperature = 0.0,
            max_tokens  = 200,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"[ERROR: {e}]"

    results.append({
        "question":            item["question"],
        "answer":              answer,
        "ground_truth":        item["ground_truth"],
        "all_answers":         item["all_answers"],
        "retrieved_contexts":  retrieved_contexts,
        "reference_contexts":  [item["reference_context"]],
        "context_id":          item["context_id"],
        "retrieved_sources":   retrieved_sources,
        "answer_in_retrieved": answer_in_retrieved,
        "title":               item["title"],
    })
    time.sleep(0.8)

with open(RAW_OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

hit_rate = sum(r["answer_in_retrieved"] for r in results) / len(results)
print(f"\n[EVAL] Saved {len(results)} results  →  {RAW_OUT.name}")
print(f"[EVAL] Retrieval hit rate : {hit_rate:.1%}\n")

# ── STEP 3 · RAGAS scoring ────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from datasets import Dataset  # noqa: E402
from ragas import evaluate    # noqa: E402
from ragas.metrics import (   # noqa: E402
    answer_relevancy, faithfulness,
    context_precision, context_recall,
)
from ragas.llms import llm_factory  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402

valid = [r for r in results if not r["answer"].startswith("[ERROR")]
print(f"[RAGAS] Scoring {len(valid)}/{len(results)} results (errors excluded)...")

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

print("[RAGAS] Running evaluation — this takes a few minutes...")
ragas_result = evaluate(
    dataset         = ragas_data,
    metrics         = [answer_relevancy, faithfulness, context_precision, context_recall],
    llm             = llm_ragas,
    embeddings      = emb_ragas,
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

output = {
    "config":     cfg_name,
    "parameters": cfg,
    "overall":    {**metrics, "retrieval_hit_rate": float(hit_rate)},
    "n_evaluated": len(valid),
    "per_question": scores_df.to_dict(orient="records"),
}
with open(RAGAS_OUT, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# ── Final report ──────────────────────────────────────────────────────────────
def bar(v: float) -> str:
    filled = int(v * 20)
    return "#" * filled + "-" * (20 - filled)

print(f"\n{'='*62}")
print(f"  RAGAS REPORT  —  {cfg_name}  |  {cfg['retrieval_mode']}  |  "
      f"reranker={cfg['reranker']}  |  top_k={cfg['top_k']}  |  chunk={cfg['chunk_size']}")
print(f"{'='*62}")
for name, val in metrics.items():
    print(f"  {name:<22}  {val:.4f}   {bar(val)}")
print(f"  {'retrieval_hit_rate':<22}  {hit_rate:.4f}   {bar(hit_rate)}")
print(f"{'='*62}")
print(f"\n  Raw results   →  results/{RAW_OUT.name}")
print(f"  RAGAS scores  →  results/{RAGAS_OUT.name}")
print(f"{'='*62}\n")
