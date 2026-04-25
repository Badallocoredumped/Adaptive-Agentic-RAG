"""Cross-encoder reranking for retrieved RAG chunks."""

from __future__ import annotations

from typing import Any

from backend import config
from backend.models import get_shared_cross_encoder


class Reranker:
    """Reranks text chunks against a query using a CrossEncoder."""

    def __init__(self, model_name: str = config.RAG_RERANKER_MODEL) -> None:
        self.model_name = model_name

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict[str, Any]]:
        """
        Score a list of documents against the query and return the best matches.
        """
        if not documents:
            return []

        model = get_shared_cross_encoder()

        pairs = [[query, doc] for doc in documents]
        import numpy as np
        scores = np.atleast_1d(model.predict(pairs)).tolist()

        if getattr(config, "RAG_RERANK_DEBUG", False):
            print(f"\n[RERANKER DEBUG] Query: {query!r}")
            print(f"[RERANKER DEBUG] Chunks BEFORE reranking (pool size: {len(documents)}):")
            for i, (doc, score) in enumerate(zip(documents, scores)):
                preview = " ".join(doc.split())[:80]
                print(f"  {i+1:>2}. {preview}... (score: {score:.4f})")

        doc_scores = [
            {"text": doc, "score": float(score)}
            for doc, score in zip(documents, scores)
        ]

        doc_scores.sort(key=lambda x: x["score"], reverse=True)

        if getattr(config, "RAG_RERANK_DEBUG", False):
            print(f"\n[RERANKER DEBUG] Top {top_k} AFTER reranking:")
            for i, ds in enumerate(doc_scores[:top_k]):
                preview = " ".join(ds['text'].split())[:80]
                print(f"  {i+1:>2}. {preview}... (score: {ds['score']:.4f})")

        return doc_scores[:top_k]
