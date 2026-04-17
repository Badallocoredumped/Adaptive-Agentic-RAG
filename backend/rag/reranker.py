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
        scores = model.predict(pairs)

        doc_scores = [
            {"text": doc, "score": float(score)}
            for doc, score in zip(documents, scores)
        ]

        doc_scores.sort(key=lambda x: x["score"], reverse=True)

        return doc_scores[:top_k]
