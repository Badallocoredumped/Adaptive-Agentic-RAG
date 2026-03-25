"""Cross-encoder reranking for retrieved RAG chunks."""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


class Reranker:
    """Reranks text chunks against a query using a CrossEncoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        """Lazy load the CrossEncoder model."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict[str, Any]]:
        """
        Score a list of documents against the query and return the best matches.
        """
        if not documents:
            return []

        model = self._get_model()

        pairs = [[query, doc] for doc in documents]
        scores = model.predict(pairs)

        doc_scores = [
            {"text": doc, "score": float(score)}
            for doc, score in zip(documents, scores)
        ]

        doc_scores.sort(key=lambda x: x["score"], reverse=True)

        return doc_scores[:top_k]
