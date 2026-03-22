"""Cross-encoder reranking for retrieved RAG chunks."""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


class Reranker:
    """Reranks text chunks against a query using a CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
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

        # Step 1: Create pairs of (query, document)
        pairs = [[query, doc] for doc in documents]

        # Step 2: Get relevance scores from the model
        scores = model.predict(pairs)

        # Step 3: Combine documents with their calculated scores
        doc_scores = [
            {"text": doc, "score": float(score)}
            for doc, score in zip(documents, scores)
        ]

        # Step 4: Sort documents by score (descending)
        doc_scores.sort(key=lambda x: x["score"], reverse=True)

        # Step 5: Return top_k
        return doc_scores[:top_k]


if __name__ == "__main__":
    # Simple test block
    print("[1] Initializing Reranker...")
    reranker = Reranker()

    query = "How do I reset my password?"
    docs = [
        "Store hours are 9 AM to 5 PM Monday through Friday.",
        "To easily reset your password, click the 'Forgot Password' link on the login page.",
        "Your password must contain at least one number and one special character.",
        "You can contact support via email at support@example.com.",
        "Password sharing is strictly prohibited by our security policies."
    ]

    print(f"\n[2] Reranking {len(docs)} documents for query: {query!r}")
    results = reranker.rerank(query, docs, top_k=3)

    print("\n[3] Top 3 Results:")
    for i, res in enumerate(results, 1):
        print(f"\n--- Rank {i} (Score: {res['score']:.4f}) ---")
        print(res["text"])
