"""Embedding wrapper built on LangChain HuggingFace embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend import config
from backend.models import get_shared_hf_embeddings
from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings


class PrefixAwareEmbeddings(Embeddings):
    """Applies model-specific query/document prefixes before embedding."""

    def __init__(self, base_embeddings: HuggingFaceEmbeddings, model_name: str) -> None:
        self.base_embeddings = base_embeddings
        self.model_name = model_name.lower()

    @property
    def _use_e5_prefix(self) -> bool:
        return config.E5_PREFIX_ENABLED and "e5" in self.model_name

    @staticmethod
    def _ensure_prefix(text: str, prefix: str) -> str:
        stripped = text.strip()
        if stripped.lower().startswith(prefix.strip().lower()):
            return stripped
        return f"{prefix}{stripped}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        payload = texts
        if self._use_e5_prefix:
            payload = [self._ensure_prefix(text, config.E5_PASSAGE_PREFIX) for text in texts]
        return self.base_embeddings.embed_documents(payload)

    def embed_query(self, query: str) -> list[float]:
        payload = query
        if self._use_e5_prefix:
            payload = self._ensure_prefix(query, config.E5_QUERY_PREFIX)
        return self.base_embeddings.embed_query(payload)


class SentenceTransformerEmbedder:
    """Provides a LangChain-compatible embedding model wrapper."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @property
    def model(self) -> HuggingFaceEmbeddings:
        """Return the shared HuggingFaceEmbeddings singleton."""
        return get_shared_hf_embeddings()

    def get_langchain_embeddings(self) -> PrefixAwareEmbeddings:
        """Return prefix-aware embeddings used by vector index/retrieval."""
        return PrefixAwareEmbeddings(self.model, self.model_name)

