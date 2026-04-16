"""High-level retrieval service for indexing and searching document chunks."""

from __future__ import annotations

import re

from backend import config
from backend.rag.chunker import Chunk
from backend.rag.embedder import SentenceTransformerEmbedder
from backend.rag.vector_store import FAISSVectorStore
from langchain_core.documents import Document as LCDocument


def _debug(message: str) -> None:
    if getattr(config, "DEBUG_LOGGING", False):
        print(message)


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

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k relevant chunks for a query using FAISS retrieval + CrossEncoder Reranking."""
        # Dynamically scale the initial fetch size based on the final target output
        fetch_k = max(top_k * config.RAG_FETCH_MULTIPLIER, 20)
        query_domain = self._infer_query_domain(query)

        _debug(f"\n[RAG Pipeline] Query: {query!r}")
        _debug(f"[RAG Pipeline] Fetching top {fetch_k} documents from FAISS...")

        if query_domain:
            results = self.vector_store.search(
                query,
                fetch_k,
                metadata_filter={"domain": query_domain},
            )
            # Fallback if domain filter is too restrictive
            if len(results) < fetch_k:
                results = self.vector_store.search(query, fetch_k)
        else:
            results = self.vector_store.search(query, fetch_k)

        # Deduplicate and map text -> metadata
        doc_map = {}
        for doc, distance in results:
            normalized_text = " ".join(doc.page_content.split())
            if normalized_text not in doc_map:
                doc_map[normalized_text] = doc

        documents_text = list(doc_map.keys())
        _debug(f"[RAG Pipeline] FAISS returned {len(documents_text)} unique chunks.")
        
        # CrossEncoder Reranking Toggle
        if config.RAG_ENABLE_SEMANTIC_RERANK:
            if not hasattr(self, "_reranker") or self._reranker is None:
                from backend.rag.reranker import Reranker
                self._reranker = Reranker()
                
            _debug(f"[RAG Pipeline] Reranking {len(documents_text)} documents using {self._reranker.model_name} CrossEncoder...")
            reranked = self._reranker.rerank(query, documents_text, top_k=top_k)
        else:
            _debug(f"[RAG Pipeline] Semantic Reranking is DISABLED. Returning top {top_k} directly from FAISS.")
            reranked = []
            for i, text in enumerate(documents_text[:top_k]):
                fallback_score = 1.0 / (1.0 + float(results[i][1])) if i < len(results) else 0.0
                reranked.append({"text": text, "score": fallback_score})
        
        _debug(f"\n[RAG Pipeline] --- Top {top_k} Results ---")
        payload: list[dict] = []
        for i, res in enumerate(reranked, 1):
            text = res["text"]
            score = res["score"]
            doc = doc_map[text]
            metadata = dict(doc.metadata)
            chunk_id = metadata.pop("chunk_id", None)
            source = metadata.get("source", "unknown")
            
            _debug(f"  -> Rank {i} | CrossEncoder Score: {score:.4f} | Source: {source}")
            
            payload.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc.page_content,
                    "source": source,
                    "score": round(score, 4),
                    "metadata": metadata,
                }
            )

        return payload

    @staticmethod
    def _infer_query_domain(query: str) -> str | None:
        """Infer domain based on keyword overlap."""
        tokens = set(re.findall(r"\b\w+\b", query.lower()))
        best_domain: str | None = None
        best_hits = 0

        for domain, keywords in config.DOMAIN_KEYWORDS.items():
            hits = sum(1 for keyword in keywords if keyword in tokens)
            if hits > best_hits:
                best_domain = domain
                best_hits = hits

        return best_domain if best_hits > 0 else None
