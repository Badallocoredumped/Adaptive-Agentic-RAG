"""High-level retrieval service for indexing and searching document chunks."""

from __future__ import annotations

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

    def retrieve(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> list[dict]:
        """Return top-k relevant chunks for a query with similarity scores."""
        results = self.vector_store.search(query, max(top_k * 3, top_k))
        payload: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for doc, distance in results:
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
                    "metadata": metadata,
                }
            )

            if len(payload) >= top_k:
                break

        return payload
