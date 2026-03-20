"""Embedding wrapper built on LangChain HuggingFace embeddings."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings


class SentenceTransformerEmbedder:
    """Provides a LangChain-compatible embedding model wrapper."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: HuggingFaceEmbeddings | None = None

    @property
    def model(self) -> HuggingFaceEmbeddings:
        if self._model is None:
            self._model = HuggingFaceEmbeddings(model_name=self.model_name)
        return self._model

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Return the underlying LangChain embeddings object."""
        return self.model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed text chunks using LangChain's embed_documents API."""
        if not texts:
            return []
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query using LangChain's embed_query API."""
        return self.model.embed_query(query)
