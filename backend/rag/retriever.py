"""High-level retrieval service for indexing and searching document chunks."""

from __future__ import annotations

import re

from backend import config
from backend.rag.chunker import Chunk
from backend.rag.embedder import SentenceTransformerEmbedder
from backend.rag.vector_store import FAISSVectorStore
from langchain_core.documents import Document as LCDocument


class RagRetriever:
    """Coordinates LangChain document indexing and FAISS retrieval."""

    def __init__(self, embedder: SentenceTransformerEmbedder, vector_store: FAISSVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Convert chunks to LangChain documents and add them to the vector store."""
        if not chunks:
            return

        documents = [
            LCDocument(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    **chunk.metadata,
                },
            )
            for chunk in chunks
        ]

        self.vector_store.add_documents(documents)
        self.vector_store.save()

    def retrieve(self, query: str, top_k: int = config.RAG_TOP_K) -> list[dict]:
        """Return top-k relevant chunks for a query with similarity scores."""
        fetch_k = max(top_k * config.RAG_FETCH_MULTIPLIER, top_k)
        query_domain = self._infer_query_domain(query)

        if query_domain:
            results = self.vector_store.search(
                query,
                fetch_k,
                metadata_filter={"domain": query_domain},
            )
            if len(results) < top_k:
                results = self.vector_store.search(query, fetch_k)
        else:
            results = self.vector_store.search(query, fetch_k)

        if config.RAG_ENABLE_LEXICAL_RERANK:
            ranked_results = self._rerank_results(query, results)
        else:
            ranked_results = [(doc, float(distance), 0.0) for doc, distance in results]

        payload: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for doc, distance, lexical_overlap in ranked_results:
            metadata = dict(doc.metadata)
            chunk_id = metadata.pop("chunk_id", None)
            source = metadata.get("source", "unknown")
            score = 1.0 / (1.0 + float(distance))

            normalized_text = " ".join(doc.page_content.split())
            dedup_key = (str(source), normalized_text)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            payload.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc.page_content,
                    "source": source,
                    "score": round(score, 4),
                    "lexical_overlap": round(lexical_overlap, 4),
                    "metadata": metadata,
                }
            )

            if len(payload) >= top_k:
                break

        return payload

    def _rerank_results(self, query: str, results: list[tuple[LCDocument, float]]) -> list[tuple[LCDocument, float, float]]:
        ranked: list[tuple[float, float, float, LCDocument, float]] = []
        for doc, distance in results:
            base_score = 1.0 / (1.0 + float(distance))
            lexical_overlap = self._lexical_overlap_score(query, doc.page_content)
            combined = base_score + (config.RAG_RERANK_LEXICAL_WEIGHT * lexical_overlap)

            # Hard floor prevents broad semantic matches from dominating when query terms are absent.
            if lexical_overlap < config.RAG_MIN_LEXICAL_OVERLAP:
                combined *= 0.8

            ranked.append((combined, lexical_overlap, base_score, doc, float(distance)))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [(doc, distance, lexical_overlap) for _, lexical_overlap, _, doc, distance in ranked]

    @staticmethod
    def _lexical_overlap_score(query: str, text: str) -> float:
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "how",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "should",
            "the",
            "to",
            "what",
            "where",
            "who",
            "with",
        }

        query_tokens = {
            token
            for token in re.findall(r"\b\w+\b", query.lower())
            if len(token) > 2 and token not in stop_words
        }
        if not query_tokens:
            return 0.0

        text_tokens = set(re.findall(r"\b\w+\b", text.lower()))
        overlap = len(query_tokens.intersection(text_tokens))
        return overlap / float(len(query_tokens))

    @staticmethod
    def _infer_query_domain(query: str) -> str | None:
        tokens = set(re.findall(r"\b\w+\b", query.lower()))
        best_domain: str | None = None
        best_hits = 0

        for domain, keywords in config.DOMAIN_KEYWORDS.items():
            hits = sum(1 for keyword in keywords if keyword in tokens)
            if hits > best_hits:
                best_domain = domain
                best_hits = hits

        return best_domain if best_hits > 0 else None
