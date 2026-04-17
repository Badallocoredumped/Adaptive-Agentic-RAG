"""BM25 sparse retrieval index built from the FAISS docstore corpus."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document as LCDocument


class BM25Index:
    """Wraps BM25Okapi for sparse keyword retrieval over a fixed document corpus."""

    def __init__(self) -> None:
        self._bm25 = None
        self._texts: list[str] = []
        self._docs: list[LCDocument] = []

    def build(self, docs: list[LCDocument]) -> None:
        from rank_bm25 import BM25Okapi

        self._docs = docs
        self._texts = [doc.page_content for doc in docs]
        self._bm25 = BM25Okapi([self._tokenize(t) for t in self._texts])

    def search(self, query: str, top_k: int) -> list[tuple[str, LCDocument, float]]:
        """Return (text, doc, score) triples for the top_k BM25 matches."""
        if self._bm25 is None or not self._texts:
            return []

        scores = self._bm25.get_scores(self._tokenize(query))
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            (self._texts[i], self._docs[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())
