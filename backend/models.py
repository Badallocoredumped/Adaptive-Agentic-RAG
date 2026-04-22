"""Centralized, thread-safe model singletons.

Every heavy model (SentenceTransformer, HuggingFaceEmbeddings, CrossEncoder)
is loaded exactly once and shared across all modules.  This eliminates the
~900 MB per-duplicate memory cost and the 2-3 s per-duplicate cold-start
latency that occur when table_rag, sql_cache, and embedder each instantiate
the same weights independently.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from backend import config

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder, SentenceTransformer

# ── Embedding model (shared by RAG embedder, TableRAG, and SQL Cache) ─────

_hfe_lock = threading.Lock()
_hfe_instance: HuggingFaceEmbeddings | None = None


def get_shared_hf_embeddings() -> HuggingFaceEmbeddings:
    """Return the singleton HuggingFaceEmbeddings instance (thread-safe).

    The underlying SentenceTransformer is accessible via ``.client``.
    """
    global _hfe_instance
    with _hfe_lock:
        if _hfe_instance is None:
            from langchain_huggingface import HuggingFaceEmbeddings as _HFE

            _hfe_instance = _HFE(model_name=config.EMBEDDING_MODEL_NAME)
    return _hfe_instance


def get_shared_st_model() -> SentenceTransformer:
    """Return the raw SentenceTransformer backing the shared embeddings."""
    return get_shared_hf_embeddings().client


# ── CrossEncoder reranker ─────────────────────────────────────────────────

_ce_lock = threading.Lock()
_ce_instance: CrossEncoder | None = None


def get_shared_cross_encoder() -> CrossEncoder:
    """Return the singleton CrossEncoder instance (thread-safe)."""
    global _ce_instance
    with _ce_lock:
        if _ce_instance is None:
            from sentence_transformers import CrossEncoder as _CE

            _ce_instance = _CE(config.RAG_RERANKER_MODEL)
    return _ce_instance
