"""
evaluation/ingest_squad_corpus.py
Ingests squad_corpus.json into a FRESH FAISS index for evaluation.
Uses a SEPARATE index directory so it doesn't overwrite your real data index.

Run: python evaluation/ingest_squad_corpus.py
"""
import json, sys
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent          # .../evaluation/Unstructured/
PROJECT_ROOT = SCRIPT_DIR.parent.parent                 # .../Adaptive-Agentic-RAG/
sys.path.insert(0, str(PROJECT_ROOT))

# ── Point to a separate eval index so we don't clobber your real data ────────
import backend.config as config
config.INDEX_DIR = SCRIPT_DIR / "squad_index"
config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

from backend.rag import (
    TextChunker, SentenceTransformerEmbedder,
    FAISSVectorStore, RagRetriever, Chunk
)

# ── Build retriever against the eval index ────────────────────────────────
embedder     = SentenceTransformerEmbedder(config.EMBEDDING_MODEL_NAME)
lc_embeddings = embedder.get_langchain_embeddings()
vector_store  = FAISSVectorStore(
    index_path    = str(config.INDEX_DIR / "documents"),
    metadata_path = str(config.INDEX_DIR / "documents_meta.json"),
    embeddings    = lc_embeddings,
)
retriever = RagRetriever(embedder=embedder, vector_store=vector_store)

# ── Load corpus ───────────────────────────────────────────────────────────
with open(SCRIPT_DIR / "evaluation" / "squad_corpus.json") as f:
    corpus = json.load(f)

print(f"Ingesting {len(corpus)} contexts into FAISS...")

# ── Chunk each context the same way your pipeline chunks PDFs ─────────────
chunker = TextChunker(
    chunk_size    = config.CHUNK_SIZE,
    chunk_overlap = config.CHUNK_OVERLAP,
)

# Build fake Document objects to reuse your chunker
from backend.rag.loader import Document
all_chunks = []
for item in corpus:
    doc = Document(
        text     = item["text"],
        source   = item["context_id"],
        metadata = {"title": item["title"], "context_id": item["context_id"]},
    )
    chunks = chunker.chunk_documents([doc])
    all_chunks.extend(chunks)

print(f"Total chunks after splitting: {len(all_chunks)}")

# ── Index into FAISS ──────────────────────────────────────────────────────
BATCH = 500
for i in range(0, len(all_chunks), BATCH):
    batch = all_chunks[i : i + BATCH]
    retriever.index_chunks(batch)
    print(f"  Indexed {min(i + BATCH, len(all_chunks))}/{len(all_chunks)} chunks...")

print(f"\nDone. Index saved to: {config.INDEX_DIR}")