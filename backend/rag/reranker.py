"""Cross-encoder reranking for retrieved RAG chunks."""

from __future__ import annotations

from typing import Any

import numpy as np

from backend import config
from backend.models import get_shared_cross_encoder


class Reranker:
    """Reranks text chunks against a query using a CrossEncoder."""

    def __init__(self, model_name: str = config.RAG_RERANKER_MODEL) -> None:
        self.model_name = model_name

    def _log_debug(self, label: str, query: str, docs: list[str], scores: list[float], top_k: int) -> None:
        if label == "before":
            print(f"\n[RERANKER DEBUG] Query: {query!r}")
            print(f"[RERANKER DEBUG] Chunks BEFORE reranking (pool size: {len(docs)}):")
            for i, (doc, score) in enumerate(zip(docs, scores)):
                preview = " ".join(doc.split())[:config.RAG_PREVIEW_CHARS]
                print(f"  {i+1:>2}. {preview}... (score: {score:.4f})")
        else:
            print(f"\n[RERANKER DEBUG] Top {top_k} AFTER reranking:")
            for i, (doc, score) in enumerate(zip(docs[:top_k], scores[:top_k])):
                preview = " ".join(doc.split())[:config.RAG_PREVIEW_CHARS]
                print(f"  {i+1:>2}. {preview}... (score: {score:.4f})")

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict[str, Any]]:
        """Score a list of documents against the query and return the best matches."""
        if not documents:
            return []

        model = get_shared_cross_encoder()
        pairs = [[query, doc] for doc in documents]
        scores = np.atleast_1d(model.predict(pairs)).tolist()

        if config.RAG_RERANK_DEBUG:
            self._log_debug("before", query, documents, scores, top_k)

        doc_scores = sorted(
            [{"text": doc, "score": float(score)} for doc, score in zip(documents, scores)],
            key=lambda x: x["score"],
            reverse=True,
        )

        if config.RAG_RERANK_DEBUG:
            after_docs = [d["text"] for d in doc_scores]
            after_scores = [d["score"] for d in doc_scores]
            self._log_debug("after", query, after_docs, after_scores, top_k)

        return doc_scores[:top_k]
