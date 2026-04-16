"""
evaluation/score_squad_ragas.py
Scores squad_rag_results.json with RAGAS + prints a thesis-ready report.

Run: python evaluation/score_squad_ragas.py
"""
import json, sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR   = Path(__file__).resolve().parent          # .../evaluation/Unstructured/
PROJECT_ROOT = SCRIPT_DIR.parent.parent                 # .../Adaptive-Agentic-RAG/
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR  = SCRIPT_DIR / "results"

# Silence TensorFlow oneDNN warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datasets import Dataset
from ragas import evaluate

# FIX 1: Import metrics from the main module, NOT collections
from ragas.metrics import (
    AnswerRelevancy, Faithfulness, ContextPrecision, ContextRecall,
)

from openai import OpenAI as OpenAIClient
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
import backend.config as config

# ── Load results ──────────────────────────────────────────────────────────
with open(RESULTS_DIR / "squad_rag_results.json") as f:
    results = json.load(f)

# Filter out error results
results = [r for r in results if not r["answer"].startswith("[ERROR")]
print(f"Scoring {len(results)} results...")

# ── Build HuggingFace Dataset for RAGAS ───────────────────────────────────
ragas_data = Dataset.from_dict({
    "user_input":         [r["question"]           for r in results],
    "response":           [r["answer"]             for r in results],
    "retrieved_contexts": [r["retrieved_contexts"] for r in results],
    "reference":          [r["ground_truth"]       for r in results],
})

# ── Configure LLM + Embeddings for RAGAS ─────────────────────────────────
# ragas 0.4.x metrics require ragas-native wrappers
openai_client = OpenAIClient(api_key=config.OPENAI_API_KEY)
llm = llm_factory(model=config.SQL_OPENAI_MODEL, client=openai_client)
embeddings = RagasHFEmbeddings(model=config.EMBEDDING_MODEL_NAME)

# Metrics must be instantiated after llm/embeddings in ragas 0.4.x
METRICS = [
    AnswerRelevancy(llm=llm, embeddings=embeddings),
    Faithfulness(llm=llm),
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
]

# ── Run RAGAS ─────────────────────────────────────────────────────────────
print("Running RAGAS evaluation (this takes a few minutes)...")
ragas_result = evaluate(
    dataset=ragas_data,  # FIX 2: Explicitly define dataset keyword argument
    metrics=METRICS,
    raise_exceptions=False,
)
scores_df = ragas_result.to_pandas()

# Metric column names in ragas 0.4.x match the metric class .name attribute
# (snake_case): answer_relevancy, faithfulness, context_precision, context_recall
METRIC_COLS = {
    "answer_relevancy":  "answer_relevancy",
    "faithfulness":      "faithfulness",
    "context_precision": "context_precision",
    "context_recall":    "context_recall",
}

# ── Also compute retrieval hit rate (our bonus metric) ────────────────────
hit_rate  = sum(r["answer_in_retrieved"] for r in results) / len(results)

# ── Print report ──────────────────────────────────────────────────────────
def bar(v): return "█" * int(v * 20) + "░" * (20 - int(v * 20))

print("\n" + "="*60)
print("  RAGAS REPORT — Unstructured Pipeline (SQuAD v1.1)")
print("="*60)
metrics = {
    label: scores_df[col].mean()
    for label, col in METRIC_COLS.items()
    if col in scores_df.columns
}
for name, val in metrics.items():
    print(f"  {name:<22} {val:.4f}  {bar(val)}")
print(f"  {'retrieval_hit_rate':<22} {hit_rate:.4f}  {bar(hit_rate)}")
print("="*60)

# ── Save ──────────────────────────────────────────────────────────────────
output = {
    "overall": {**metrics, "retrieval_hit_rate": hit_rate},
    "n_evaluated": len(results),
    "per_question": scores_df.to_dict(orient="records"),
}
out_path = RESULTS_DIR / "squad_ragas_scores.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nFull scores → {out_path}")