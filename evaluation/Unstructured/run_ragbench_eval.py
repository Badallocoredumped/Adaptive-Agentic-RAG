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

# Baseline: LLM only, no retrieval
"""     "BASE_NO_RAG": dict(retrieval_mode="none", reranker=False, top_k=0, chunk_size=500),
"""    
# K-Scaling Experiments
"""     "RAG_K3": dict(retrieval_mode="faiss", reranker=False, top_k=3, chunk_size=500),
"RAG_K5": dict(retrieval_mode="faiss", reranker=False, top_k=5, chunk_size=500), """

""" "RAG_K10": dict(retrieval_mode="faiss", reranker=False, top_k=10, chunk_size=500), """

""" # Chunk Density Experiment
"RAG_K5_C250": dict(retrieval_mode="faiss", reranker=False, top_k=5, chunk_size=250), """

""" # Baseline
"RAG_K5_FAISS": dict(retrieval_mode="faiss", reranker=False, top_k=5, chunk_size=500),

# Core reranker comparison
"RAG_K5_BGE_BASE": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=5, chunk_size=500),
"RAG_K5_BGE_LARGE": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-large", top_k=5, chunk_size=500),

# Pool size
"RAG_K5_BGE_BASE_POOL10": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=5, chunk_size=500, rerank_pool=10),
"RAG_K5_BGE_BASE_POOL20": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=5, chunk_size=500, rerank_pool=20),

# Chunk size CONTINUE
"RAG_K5_BGE_BASE_C250": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=5, chunk_size=250),

# Stress test CONTINUE
"RAG_K10_BGE_BASE": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=10, chunk_size=500), """


# ── Config table ──────────────────────────────────────────────────────────────
# Organised in two sections:
#   A) Ablation ladder — new configs, each isolating one variable
#   B) Historical sweep — all previous experiment configs, preserved for re-runs
CONFIGS = {

    # ════════════════════════════════════════════════════════════════════════
    # A) ABLATION LADDER  (run these in order for a clean narrative)
    # ════════════════════════════════════════════════════════════════════════

    # Step 0 — LLM answers from parametric knowledge alone (no retrieval at all).
    # Establishes the floor: how much the LLM already knows without any corpus.
    "LLM_ONLY": dict(
        retrieval_mode="none", reranker=False,
        top_k=0, chunk_size=None,
    ),

    # Step 1 — Baseline RAG: FAISS retrieval, no reranker, 500-char chunks.
    # Shows raw retrieval quality before any re-ranking.
    "RAG_K10_FAISS": dict(
        retrieval_mode="faiss", reranker=False,
        top_k=10, chunk_size=500,
    ),

    # Step 2 — Add the cross-encoder reranker, everything else equal.
    # Isolates the reranker's contribution to precision.
    "RAG_K10_BGE_BASE": dict(
        retrieval_mode="faiss", reranker=True,
        reranker_model="BAAI/bge-reranker-base",
        top_k=10, chunk_size=500,
    ),

    # Step 3 — Switch to 1000-char chunks, reranker stays.
    # Isolates the chunk-size improvement (richer context, less fragmentation).
    "RAG_K10_BGE_BASE_C1000": dict(
        retrieval_mode="faiss", reranker=True,
        reranker_model="BAAI/bge-reranker-base",
        top_k=10, chunk_size=1000,
    ),

    # Step 4 — Full optimised pipeline: K=15 + 1k chunks + pruned rerank pool.
    # Best config from the sweep; used as the production target.
    "RAG_K15_BGE_BASE_C1000_POOL25": dict(
        retrieval_mode="faiss", reranker=True,
        reranker_model="BAAI/bge-reranker-base",
        top_k=15, chunk_size=1000, rerank_pool=25,
    ),

    # ════════════════════════════════════════════════════════════════════════
    # B) HISTORICAL SWEEP  (previous experiment configs — kept for re-runs)
    # ════════════════════════════════════════════════════════════════════════

    # ── K-Scaling (500-char chunks, no reranker) ──
    "RAG_K3":  dict(retrieval_mode="faiss", reranker=False, top_k=3,  chunk_size=500),
    "RAG_K5":  dict(retrieval_mode="faiss", reranker=False, top_k=5,  chunk_size=500),

    # ── Chunk density experiment ──
    "RAG_K5_C250":     dict(retrieval_mode="faiss", reranker=False, top_k=5, chunk_size=250),
    "RAG_K5_BGE_BASE_C250": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=5, chunk_size=250),

    # ── K=5 reranker comparison ──
    "RAG_K5_BGE_BASE":  dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base",  top_k=5, chunk_size=500),
    "RAG_K5_BGE_LARGE": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-large", top_k=5, chunk_size=500),

    # ── K=5 pool-size experiment ──
    "RAG_K5_BGE_BASE_POOL10": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=5, chunk_size=500, rerank_pool=10),
    "RAG_K5_BGE_BASE_POOL20": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=5, chunk_size=500, rerank_pool=20),

    # ── K=10 reranker comparison ──
    "RAG_K10_BGE_LARGE": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-large", top_k=10, chunk_size=500),

    # ── K=10 + 1000-char chunk pool-size experiments ──
    "RAG_K10_BGE_BASE_C1000_POOL25": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=10, chunk_size=1000, rerank_pool=25),
    "RAG_K10_BGE_BASE_C1000_POOL50": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=10, chunk_size=1000, rerank_pool=50),
    "RAG_K10_BGE_BASE_C1000_POOL75": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=10, chunk_size=1000, rerank_pool=75),

    # ── K=15 experiments ──
    "RAG_K15_BGE_BASE":      dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=15, chunk_size=500),
    "RAG_K15_BGE_BASE_C1000": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=15, chunk_size=1000),

    # ── K=15 + pool-size experiments ──
    "RAG_K15_BGE_BASE_C1000_POOL50": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=15, chunk_size=1000, rerank_pool=50),
    "RAG_K15_BGE_BASE_C1000_POOL75": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=15, chunk_size=1000, rerank_pool=75),

    # ── K=20 ──
    "RAG_K20_BGE_BASE_C1000": dict(retrieval_mode="faiss", reranker=True, reranker_model="BAAI/bge-reranker-base", top_k=20, chunk_size=1000),
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

from collections import defaultdict as _defaultdict

def _expand_with_adjacent_chunks(retriever_obj, retrieved_chunks: list[dict]) -> list[dict]:
    """Append sibling chunks (±1 position by chunk_id) for every retrieved source.

    When a passage is split into multiple chunks and the retriever finds chunk N
    (the problem description), the answer often lives in chunk N+1 (the resolution).
    This function adds those neighbors so the LLM has the full context.
    """
    store = retriever_obj.vector_store.store
    if store is None:
        return retrieved_chunks

    # Map source -> sorted list of all docs in docstore for that source
    hit_sources = {c.get("source", "") for c in retrieved_chunks}
    source_doc_map: dict = _defaultdict(list)
    for doc in store.docstore._dict.values():
        src = doc.metadata.get("source", "")
        if src in hit_sources:
            source_doc_map[src].append(doc)

    for src in source_doc_map:
        source_doc_map[src].sort(key=lambda d: d.metadata.get("chunk_id", 0))

    # Collect neighbor chunks not already in the retrieved set
    retrieved_texts = {c["text"] for c in retrieved_chunks}
    retrieved_cids_by_source: dict = _defaultdict(set)
    for c in retrieved_chunks:
        cid = c.get("chunk_id")
        if cid is not None:
            retrieved_cids_by_source[c.get("source", "")].add(cid)

    extra: list[dict] = []
    for src, docs in source_doc_map.items():
        hit_ids = retrieved_cids_by_source[src]
        for idx, doc in enumerate(docs):
            if doc.metadata.get("chunk_id") not in hit_ids:
                continue
            for neighbor_idx in (idx - 1, idx + 1):
                if 0 <= neighbor_idx < len(docs):
                    neighbor = docs[neighbor_idx]
                    if neighbor.page_content not in retrieved_texts:
                        extra.append({
                            "chunk_id": neighbor.metadata.get("chunk_id"),
                            "text":     neighbor.page_content,
                            "source":   src,
                            "score":    0.0,
                            "metadata": dict(neighbor.metadata),
                        })
                        retrieved_texts.add(neighbor.page_content)

    return retrieved_chunks + extra

IS_NO_RAG = cfg["retrieval_mode"] == "none"   # True for LLM_ONLY — skip all retrieval machinery

# Apply retrieval / chunking config overrides
config.RAG_RETRIEVAL_MODE         = cfg["retrieval_mode"]
config.RAG_ENABLE_SEMANTIC_RERANK = cfg["reranker"]
if "reranker_model" in cfg:
    config.RAG_RERANKER_MODEL = cfg["reranker_model"]
config.RAG_TOP_K       = cfg["top_k"]
config.RAG_FETCH_K     = cfg.get("fetch_k", 0)
config.RAG_RERANK_POOL = cfg.get("rerank_pool", 0)

if not IS_NO_RAG:
    # chunk_size is only meaningful when retrieval is active
    config.CHUNK_SIZE    = cfg["chunk_size"]
    config.CHUNK_OVERLAP = 50 if cfg["chunk_size"] == 250 else 100
    INDEX_DIR = SCRIPT_DIR / f"ragbench_index_{cfg['chunk_size']}"
    config.INDEX_DIR = INDEX_DIR
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
else:
    INDEX_DIR = None   # sentinel — ingest / vector-store blocks will be skipped

# ── Output directory routing ──────────────────────────────────────────────────
# Ablation-ladder configs (Section A) → ragbench/v2_ablation/  (post-audit, fixed eval)
# Historical sweep configs (Section B) → ragbench/v1_sweep/    (old prompt, pre-audit)
_ABLATION_CONFIGS = {
    "LLM_ONLY", "RAG_K10_FAISS", "RAG_K10_BGE_BASE",
    "RAG_K10_BGE_BASE_C1000", "RAG_K15_BGE_BASE_C1000_POOL25",
}
if cfg_name in _ABLATION_CONFIGS:
    RESULTS_DIR = SCRIPT_DIR / "results" / "ragbench" / "v2_ablation"
else:
    RESULTS_DIR = SCRIPT_DIR / "results" / "ragbench" / "v1_sweep"
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
if "reranker_model" in cfg:
    print(f"  reranker_model : {cfg['reranker_model']}")
print(f"  top_k          : {cfg['top_k']}")
if not IS_NO_RAG:
    print(f"  chunk_size     : {cfg['chunk_size']}  (overlap={config.CHUNK_OVERLAP})")
    if cfg.get("fetch_k"):
        print(f"  fetch_k        : {cfg['fetch_k']}")
    if cfg.get("rerank_pool"):
        print(f"  rerank_pool    : {cfg['rerank_pool']}")
    print(f"  index_dir      : {INDEX_DIR.name}/")
else:
    print(f"  (no retrieval — LLM answers from parametric knowledge only)")
print(f"{'='*65}\n")

# ── Import pipeline ───────────────────────────────────────────────────────────
from backend.rag import TextChunker, SentenceTransformerEmbedder, FAISSVectorStore, RagRetriever
from backend.rag.loader import Document
from openai import OpenAI

_client = OpenAI(api_key=config.OPENAI_API_KEY)

if not IS_NO_RAG:
    # ── STEP 1: Ingest (once per chunk size) ─────────────────────────────────
    embedder      = SentenceTransformerEmbedder(config.EMBEDDING_MODEL_NAME)
    lc_embeddings = embedder.get_langchain_embeddings()
    vector_store  = FAISSVectorStore(
        index_path    = str(INDEX_DIR / "documents"),
        metadata_path = str(INDEX_DIR / "documents_meta.json"),
        embeddings    = lc_embeddings,
    )

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

    # ── STEP 2a: Load vector store + build retriever ──────────────────────────
    vector_store.load()
    retriever = RagRetriever(embedder=embedder, vector_store=vector_store)
else:
    # LLM_ONLY — no embedding model, no vector store, no retriever needed
    print(f"[INGEST] Skipped — LLM_ONLY config uses no corpus index.\n")
    embedder     = None
    vector_store = None
    retriever    = None

# FIX A — Relaxed abstention prompt (used for all RAG configs)
# Changes from original:
#   - Removed "ONLY" → model can synthesise partial answers from relevant fragments
#   - Added "even partial" clause → hedged answers allowed (reduces FALSE_ABSTENTION)
#   - "Not found." now reserved for genuinely irrelevant context (raises bar for abstention)
ANSWER_PROMPT = """\
Answer the following question in 2-3 sentences using the context below.
Ground every claim in the context. Do NOT add outside knowledge.

If the context contains relevant information — even partial — attempt an
answer and note any uncertainty (e.g., "Based on the context, it appears...").
Only reply exactly "Not found." if the context contains NO information
whatsoever that is relevant to the question.

Context:
{context}

Question: {question}

Answer:"""

# LLM_ONLY baseline prompt — no context provided; model answers from parametric memory.
# "Not found." reserved for questions the model has zero knowledge about.
LLM_ONLY_PROMPT = """\
Answer the following question in 2-3 sentences using your own knowledge.
If you have no knowledge of this topic whatsoever, reply exactly: "Not found."

Question: {question}

Answer:"""

with open(EVAL_PATH, encoding="utf-8") as f:
    eval_set = json.load(f)

# Limit to 50 questions per config
eval_set = eval_set[:3]

print(f"[EVAL] Running {len(eval_set)} questions through ragbench_{cfg_name}...\n")

results = []
for i, item in enumerate(eval_set):
    print(f"[{i+1:>4}/{len(eval_set)}] [{item.get('subset','')}] {item['question'][:65]}...")

    # Retrieve (skipped for LLM_ONLY)
    if not IS_NO_RAG:
        chunks             = retriever.retrieve(item["question"], top_k=cfg["top_k"])
        retrieved_contexts = [c["text"]           for c in chunks]
        retrieved_sources  = [c.get("source", "") for c in chunks]
    else:
        chunks = retrieved_contexts = retrieved_sources = []

    # Gold hit (ID-based): did we retrieve any of the RAGBench-labeled relevant passages?
    gold_ids = set(item.get("gold_passage_ids", []))
    gold_hit = any(src in gold_ids for src in retrieved_sources) if gold_ids else None

    # FIX C — Strict gold hit: retrieved AND content overlaps with ground truth (>10% word overlap).
    # Filters out marginal header/title chunks that match by ID but contain no answer content,
    # preventing them from inflating FALSE_ABSTENTION counts.
    gold_hit_strict: bool | None = None
    if gold_ids and not IS_NO_RAG:
        gold_hit_strict = False
        _gt_words = set(item["ground_truth"].lower().split())
        for _chunk in chunks:
            if _chunk.get("source", "") in gold_ids:
                _chunk_words = set(_chunk["text"].lower().split())
                _overlap     = len(_gt_words & _chunk_words) / max(len(_gt_words), 1)
                if _overlap > 0.10:
                    gold_hit_strict = True
                    break

    # Also check substring match against ground truth (paragraph-level)
    gt_lower = item["ground_truth"].lower()
    answer_in_retrieved = any(
        # check 50+ char overlap — paragraph answers won't be exact substrings
        any(sent.lower() in ctx.lower() for sent in gt_lower.split(". ")[:2] if len(sent) > 30)
        for ctx in retrieved_contexts
    ) if retrieved_contexts else False

    if not IS_NO_RAG:
        expanded_chunks   = _expand_with_adjacent_chunks(retriever, chunks)
        expanded_contexts = [c["text"] for c in expanded_chunks]
        context_str = "\n\n---\n\n".join(expanded_contexts)
    else:
        expanded_chunks   = []
        expanded_contexts = []
        context_str       = ""  # LLM_ONLY: no context provided

    
    try:
        if IS_NO_RAG:
            prompt_text = LLM_ONLY_PROMPT.format(question=item["question"])
        else:
            prompt_text = ANSWER_PROMPT.format(context=context_str, question=item["question"])
        resp = _client.chat.completions.create(
            model       = config.SQL_OPENAI_MODEL,
            messages    = [{"role": "user", "content": prompt_text}],
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
        "expanded_contexts":    expanded_contexts,
        "reference_contexts":   [item["reference_context"]],
        "context_id":           item["context_id"],
        "gold_passage_ids":     item.get("gold_passage_ids", []),
        "retrieved_sources":    retrieved_sources,
        "gold_hit":             gold_hit,            # ID-based: any gold passage retrieved?
        "gold_hit_strict":      gold_hit_strict,     # FIX C: content-overlap validated gold hit
        "unanswerable":         not bool(gold_ids),  # FIX B: True when no gold passage in corpus
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
valid        = [r for r in results if not r["answer"].startswith("[ERROR")]
gold_results = [r for r in valid if r["gold_hit"] is not None]
gold_hit_rate = sum(r["gold_hit"] for r in gold_results) / len(gold_results) if gold_results else 0
not_found    = sum(1 for r in valid if "not found" in r["answer"].lower())

# FIX B — Adjusted not-found rate (excludes NO_GOLD questions where abstaining is correct)
answerable       = [r for r in valid if r.get("gold_passage_ids")]   # has a retrievable answer
no_gold_qs       = [r for r in valid if not r.get("gold_passage_ids")] # no gold passage in corpus
not_found_adj    = sum(1 for r in answerable if "not found" in r["answer"].lower())
not_found_rate_adjusted = not_found_adj / len(answerable) if answerable else 0.0

# Per-subset breakdown
from collections import defaultdict
by_subset = _defaultdict(list)
for r in valid:
    by_subset[r["subset"]].append(r)

print(f"\n[EVAL] Saved {len(results)} results → {RAW_OUT.name}")
print(f"[EVAL] Gold passage hit rate         : {gold_hit_rate:.1%}  ({len(gold_results)} with gold labels)")
print(f"[EVAL] 'Not found' rate (raw)        : {not_found}/{len(valid)}  ({not_found/len(valid)*100:.1f}%)")
print(f"[EVAL] 'Not found' rate (adjusted)   : {not_found_adj}/{len(answerable)} answerable  ({not_found_rate_adjusted:.1%})")  # excludes NO_GOLD
print(f"[EVAL] NO_GOLD questions             : {len(no_gold_qs)} (no gold passage in corpus — correct abstentions)")
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
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

print(f"\n[RAGAS] Scoring {len(valid)}/{len(results)} results...")

if not valid:
    print("\n[ERROR] No valid results to score! Please check your API quota or raw JSON results for errors.")
    sys.exit(1)

ragas_data = Dataset.from_dict({
    "user_input":         [r["question"]           for r in valid],
    "response":           [r["answer"]             for r in valid],
    "retrieved_contexts": [r["retrieved_contexts"] for r in valid],
    "reference":          [r["ground_truth"]       for r in valid],
})

llm_ragas = llm_factory(
    model  = config.SQL_OPENAI_MODEL,
    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY),
)
emb_ragas = LangchainEmbeddingsWrapper(
    LCHuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
)

ragas_result = evaluate(
    dataset          = ragas_data,
    metrics          = [answer_relevancy, faithfulness, context_precision, context_recall],
    llm              = llm_ragas,
    embeddings       = emb_ragas,
    run_config       = RunConfig(
        max_workers  = 8,
        max_wait     = 180,
        max_retries  = 3,
    ),
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
        "gold_hit_rate":           gold_hit_rate,
        "not_found_rate":          not_found / len(valid) if valid else 0,          # raw (all NF)
        "not_found_rate_adjusted": not_found_rate_adjusted,                         # FIX B: excl. NO_GOLD
        "no_gold_count":           len(no_gold_qs),                                 # FIX B: unanswerable Qs
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
