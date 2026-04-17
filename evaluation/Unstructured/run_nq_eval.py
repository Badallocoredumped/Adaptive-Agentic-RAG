"""
evaluation/NaturalQuestions/run_nq_eval.py

Identical pipeline to run_config_eval.py but pointed at NQ corpus/eval files.
Run only the two configs that matter for the cross-dataset comparison:
  - T5A : hybrid · no rerank · fetch=100 · k=5   (best SQuAD config)
  - T3A : hybrid · rerank    · fetch=100 · pool=20 · k=5  (best reranker config)

This answers: does the reranker recover on NQ vs SQuAD?

Usage:
  python run_nq_eval.py T5A
  python run_nq_eval.py T3A

Output (evaluation/NaturalQuestions/results/):
  NQ_<cfg>_rag_results.json
  NQ_<cfg>_ragas_scores.json
"""

import os
import sys
import json
import time
from pathlib import Path

# ── Config table — only the two comparison configs ────────────────────────────
CONFIGS = {
    "T5A": dict(retrieval_mode="hybrid", reranker=False, top_k=5,
                chunk_size=500, fetch_k=100, rerank_pool=0),
    "T3A": dict(retrieval_mode="hybrid", reranker=True,  top_k=5,
                chunk_size=500, fetch_k=100, rerank_pool=20),
    # Optional: add C1 (faiss baseline) for a clean three-way comparison
    "C1":  dict(retrieval_mode="faiss",  reranker=False, top_k=5,
                chunk_size=500, fetch_k=100, rerank_pool=0),
}

if len(sys.argv) < 2 or sys.argv[1].upper() not in CONFIGS:
    print("Usage: python run_nq_eval.py <T5A|T3A|C1>")
    sys.exit(1)

cfg_name = sys.argv[1].upper()
cfg      = CONFIGS[cfg_name]

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import backend.config as config

config.CHUNK_SIZE                 = cfg["chunk_size"]
config.CHUNK_OVERLAP              = 100
config.RAG_RETRIEVAL_MODE         = cfg["retrieval_mode"]
config.RAG_ENABLE_SEMANTIC_RERANK = cfg["reranker"]
config.RAG_TOP_K                  = cfg["top_k"]
config.RAG_FETCH_K                = cfg["fetch_k"]
config.RAG_RERANK_POOL            = cfg["rerank_pool"]
FETCH_K     = cfg["fetch_k"]
RERANK_POOL = cfg["rerank_pool"]

# Separate NQ index per chunk size — never mix with SQuAD index
INDEX_DIR = SCRIPT_DIR / f"nq_index_{cfg['chunk_size']}"
config.INDEX_DIR = INDEX_DIR
INDEX_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUT   = RESULTS_DIR / f"NQ_{cfg_name}_rag_results.json"
RAGAS_OUT = RESULTS_DIR / f"NQ_{cfg_name}_ragas_scores.json"

# Guard: skip if already done
if RAW_OUT.exists() and RAGAS_OUT.exists():
    print(f"[SKIP] NQ_{cfg_name} results already exist. Delete to re-run.")
    sys.exit(0)

print(f"\n{'='*62}")
print(f"  Config         : NQ_{cfg_name}")
print(f"  retrieval_mode : {cfg['retrieval_mode']}")
print(f"  reranker       : {cfg['reranker']}  (pool={RERANK_POOL})")
print(f"  top_k          : {cfg['top_k']}  fetch_k={FETCH_K}")
print(f"  chunk_size     : {cfg['chunk_size']}")
print(f"  dataset        : Natural Questions (BeIR/nq)")
print(f"{'='*62}\n")

from backend.rag import (
    TextChunker, SentenceTransformerEmbedder,
    FAISSVectorStore, RagRetriever,
)
from backend.rag.loader import Document

embedder      = SentenceTransformerEmbedder(config.EMBEDDING_MODEL_NAME)
lc_embeddings = embedder.get_langchain_embeddings()
vector_store  = FAISSVectorStore(
    index_path    = str(INDEX_DIR / "documents"),
    metadata_path = str(INDEX_DIR / "documents_meta.json"),
    embeddings    = lc_embeddings,
)

# ── Ingest (once per chunk size, shared across configs) ───────────────────────
index_sentinel = INDEX_DIR / "documents.faiss"
if not index_sentinel.exists():
    print("[INGEST] Building NQ index...")
    corpus_path = SCRIPT_DIR / "evaluation" / "nq_corpus.json"
    if not corpus_path.exists():
        print(f"[ERROR] {corpus_path} not found.")
        print("        Run prepare_nq_dataset.py first.")
        sys.exit(1)

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"[INGEST] {len(corpus)} NQ passages loaded.")

    chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    ingest_retriever = RagRetriever(embedder=embedder, vector_store=vector_store)

    all_chunks = []
    for item in corpus:
        doc = Document(
            text     = item["text"],
            source   = item["context_id"],
            metadata = {"title": item.get("title",""), "context_id": item["context_id"]},
        )
        all_chunks.extend(chunker.chunk_documents([doc]))

    print(f"[INGEST] {len(all_chunks)} chunks after splitting.")
    BATCH = 500
    for i in range(0, len(all_chunks), BATCH):
        ingest_retriever.index_chunks(all_chunks[i: i + BATCH])
        print(f"[INGEST]   {min(i+BATCH, len(all_chunks))}/{len(all_chunks)} indexed...")
    print(f"[INGEST] Done.\n")
else:
    print(f"[INGEST] Reusing existing NQ index.\n")

# ── Retrieval + generation ────────────────────────────────────────────────────
vector_store.load()
retriever = RagRetriever(embedder=embedder, vector_store=vector_store)

from openai import OpenAI
_client = OpenAI(api_key=config.OPENAI_API_KEY)

ANSWER_PROMPT = """\
Answer the question in one or two sentences using ONLY the context below.
Do not add information not present in the context.
If the answer cannot be found, reply exactly: "Not found."

Context:
{context}

Question: {question}

Answer:"""

eval_path = SCRIPT_DIR / "evaluation" / "nq_eval_set.json"
if not eval_path.exists():
    print(f"[ERROR] {eval_path} not found. Run prepare_nq_dataset.py first.")
    sys.exit(1)

with open(eval_path, encoding="utf-8") as f:
    eval_set = json.load(f)[:50]

print(f"[EVAL] Running {len(eval_set)} NQ questions through NQ_{cfg_name}...\n")

results = []
for i, item in enumerate(eval_set):
    print(f"[{i+1:>3}/{len(eval_set)}] {item['question'][:65]}...")

    chunks             = retriever.retrieve(item["question"], top_k=cfg["top_k"])
    retrieved_contexts = [c["text"]           for c in chunks]
    retrieved_sources  = [c.get("source", "") for c in chunks]

    # NQ answers can be multi-word — check substring match
    gt = item["ground_truth"].lower()
    answer_in_retrieved = any(gt in ctx.lower() for ctx in retrieved_contexts)

    context_str = "\n\n---\n\n".join(retrieved_contexts[:cfg["top_k"]])
    try:
        resp = _client.chat.completions.create(
            model       = config.SQL_OPENAI_MODEL,
            messages    = [{"role": "user", "content": ANSWER_PROMPT.format(
                context  = context_str[:3000],   # NQ passages are longer than SQuAD
                question = item["question"],
            )}],
            temperature = 0.0,
            max_tokens  = 300,   # NQ answers can be longer than SQuAD
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"[ERROR: {e}]"

    results.append({
        "question":            item["question"],
        "answer":              answer,
        "ground_truth":        item["ground_truth"],
        "all_answers":         item.get("all_answers", [item["ground_truth"]]),
        "retrieved_contexts":  retrieved_contexts,
        "reference_contexts":  [item["reference_context"]],
        "context_id":          item["context_id"],
        "retrieved_sources":   retrieved_sources,
        "answer_in_retrieved": answer_in_retrieved,
        "title":               item.get("title", ""),
    })
    time.sleep(0.3)

with open(RAW_OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

hit_rate = sum(r["answer_in_retrieved"] for r in results) / len(results)
print(f"\n[EVAL] Saved {len(results)} results → {RAW_OUT.name}")
print(f"[EVAL] Retrieval hit rate: {hit_rate:.1%}\n")

# ── RAGAS scoring ─────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas.llms import llm_factory
from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

valid = [r for r in results if not r["answer"].startswith("[ERROR")]
print(f"[RAGAS] Scoring {len(valid)}/{len(results)} results...")

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

output = {
    "config":      f"NQ_{cfg_name}",
    "dataset":     "natural_questions",
    "parameters":  {**cfg, "fetch_k": FETCH_K, "rerank_pool": RERANK_POOL},
    "overall":     {**metrics, "retrieval_hit_rate": float(hit_rate)},
    "n_evaluated": len(valid),
    "per_question": scores_df.to_dict(orient="records"),
}
with open(RAGAS_OUT, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

def bar(v: float) -> str:
    return "#" * int(v * 20) + "-" * (20 - int(v * 20))

print(f"\n{'='*62}")
print(f"  NQ RAGAS REPORT — NQ_{cfg_name}")
print(f"{'='*62}")
for name, val in metrics.items():
    print(f"  {name:<22}  {val:.4f}   {bar(val)}")
print(f"  {'retrieval_hit_rate':<22}  {hit_rate:.4f}   {bar(hit_rate)}")
print(f"{'='*62}")
print(f"\n  Raw    → results/NQ_{cfg_name}_rag_results.json")
print(f"  RAGAS  → results/NQ_{cfg_name}_ragas_scores.json")
print(f"{'='*62}\n")
