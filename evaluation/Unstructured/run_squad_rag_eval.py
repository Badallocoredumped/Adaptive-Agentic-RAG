"""
evaluation/run_squad_rag_eval.py
Runs each SQuAD question through your real RagRetriever.
Records retrieved contexts + OpenAI-generated answer for RAGAS scoring.

Run: python evaluation/run_squad_rag_eval.py
"""
import json, sys, time
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent          # .../evaluation/Unstructured/
PROJECT_ROOT = SCRIPT_DIR.parent.parent                 # .../Adaptive-Agentic-RAG/
sys.path.insert(0, str(PROJECT_ROOT))

import backend.config as config
# ── Use the evaluation index, not your real one ───────────────────────────
config.INDEX_DIR = SCRIPT_DIR / "squad_index"

from backend.rag import (
    SentenceTransformerEmbedder, FAISSVectorStore, RagRetriever
)

from openai import OpenAI

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── OpenAI client — reuses the same key and model as the main system ──────
_openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

# ── Build retriever (loads the eval FAISS index) ──────────────────────────
embedder      = SentenceTransformerEmbedder(config.EMBEDDING_MODEL_NAME)
lc_embeddings = embedder.get_langchain_embeddings()
vector_store  = FAISSVectorStore(
    index_path    = str(config.INDEX_DIR / "documents"),
    metadata_path = str(config.INDEX_DIR / "documents_meta.json"),
    embeddings    = lc_embeddings,
)
vector_store.load()
retriever = RagRetriever(embedder=embedder, vector_store=vector_store)

def call_openai(prompt: str) -> str:
    response = _openai_client.chat.completions.create(
        model=config.SQL_OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

ANSWER_PROMPT = """\
Answer the question in one short phrase using ONLY the context below.
Do not add any information not present in the context.
If the answer cannot be found in the context, reply exactly: "Not found."

Context:
{context}

Question: {question}

Answer:"""

# ── Load eval set ─────────────────────────────────────────────────────────
with open(SCRIPT_DIR / "evaluation" / "squad_eval_set.json") as f:
    eval_set = json.load(f)

results = []
for i, item in enumerate(eval_set):
    print(f"[{i+1}/{len(eval_set)}] {item['question'][:70]}...")

    # 1. Retrieve — uses your exact FAISS + CrossEncoder pipeline
    top_k   = getattr(config, "RAG_TOP_K", 3)
    chunks  = retriever.retrieve(item["question"], top_k=top_k)
    retrieved_contexts = [c["text"] for c in chunks]
    retrieved_sources  = [c.get("source", "") for c in chunks]

    # 2. Check if the correct context was retrieved (retrieval success metric)
    answer_in_retrieved = any(
        item["ground_truth"].lower() in ctx.lower()
        for ctx in retrieved_contexts
    )

    # 3. Generate answer with Gemini
    context_str = "\n\n---\n\n".join(retrieved_contexts[:3])
    try:
        answer = call_openai(ANSWER_PROMPT.format(
            context=context_str[:2500],
            question=item["question"]
        ))
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
        "retrieved_sources":    retrieved_sources,
        "answer_in_retrieved":  answer_in_retrieved,   # bonus metric
        "title":                item["title"],
    })
    time.sleep(0.8)

out_path = RESULTS_DIR / "squad_rag_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

hit_rate = sum(r["answer_in_retrieved"] for r in results) / len(results)
print(f"\nSaved {len(results)} results -> {out_path}")
print(f"Quick retrieval hit rate: {hit_rate:.1%} (answer found in retrieved chunks)")