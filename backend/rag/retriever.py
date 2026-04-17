"""High-level retrieval service for indexing and searching document chunks."""

from __future__ import annotations

import re
from collections import defaultdict

from backend import config
from backend.rag.bm25_index import BM25Index
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
        self._reranker = None
        self._bm25: BM25Index | None = None

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
        self._bm25 = None  # invalidate BM25 index when new docs are added

    def _get_bm25(self) -> BM25Index:
        """Lazily build BM25 index from the FAISS docstore corpus."""
        if self._bm25 is not None:
            return self._bm25

        store = self.vector_store.store
        if store is None:
            self._bm25 = BM25Index()
            return self._bm25

        all_docs = list(store.docstore._dict.values())
        _debug(f"[RAG Hybrid] Building BM25 index over {len(all_docs)} corpus documents...")
        bm25 = BM25Index()
        bm25.build(all_docs)
        self._bm25 = bm25
        return self._bm25

    @staticmethod
    def _rrf(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
        """Reciprocal Rank Fusion over multiple ranked text lists."""
        scores: dict[str, float] = defaultdict(float)
        for ranked in ranked_lists:
            for rank, text in enumerate(ranked):
                scores[text] += 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda t: scores[t], reverse=True)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k relevant chunks using FAISS + optional BM25 hybrid search, then CrossEncoder reranking."""
        explicit_fetch = getattr(config, "RAG_FETCH_K", 0)
        fetch_k = explicit_fetch if explicit_fetch > 0 else max(top_k * config.RAG_FETCH_MULTIPLIER, 10)
        query_domain = self._infer_query_domain(query)
        query_vector = self.vector_store.embeddings.embed_query(query)

        _debug(f"\n[RAG Pipeline] Query: {query!r}")
        _debug(f"[RAG Pipeline] Fetching top {fetch_k} documents from FAISS...")

        # ── Dense FAISS retrieval ─────────────────────────────────────────
        if query_domain:
            faiss_results = self.vector_store.search_by_vector(
                query_vector, fetch_k, metadata_filter={"domain": query_domain},
            )
            if len(faiss_results) < fetch_k:
                faiss_results = self.vector_store.search_by_vector(query_vector, fetch_k)
        else:
            faiss_results = self.vector_store.search_by_vector(query_vector, fetch_k)

        # Build doc_map from FAISS results (normalized text -> LCDocument)
        doc_map: dict[str, LCDocument] = {}
        faiss_ranked: list[str] = []
        for doc, _ in faiss_results:
            norm = " ".join(doc.page_content.split())
            if norm not in doc_map:
                doc_map[norm] = doc
                faiss_ranked.append(norm)

        _debug(f"[RAG Pipeline] FAISS returned {len(faiss_ranked)} unique chunks.")

        # ── Sparse BM25 retrieval + RRF fusion ───────────────────────────
        retrieval_mode = getattr(config, "RAG_RETRIEVAL_MODE", "faiss")
        if retrieval_mode == "hybrid":
            bm25_raw = self._get_bm25().search(query, top_k=fetch_k)
            bm25_ranked: list[str] = []
            for text, doc, _ in bm25_raw:
                norm = " ".join(text.split())
                if norm not in doc_map:
                    doc_map[norm] = doc  # add BM25-only hits to the map
                bm25_ranked.append(norm)

            _debug(f"[RAG Hybrid] BM25 returned {len(bm25_ranked)} candidates. Applying RRF...")
            rrf_k = getattr(config, "RAG_RRF_K", 60)
            fused = self._rrf([faiss_ranked, bm25_ranked], k=rrf_k)
            documents_text = fused[:fetch_k]
        else:
            documents_text = faiss_ranked

        # ── CrossEncoder reranking ────────────────────────────────────────
        if config.RAG_ENABLE_SEMANTIC_RERANK:
            if self._reranker is None:
                from backend.rag.reranker import Reranker
                reranker_model = getattr(config, "RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
                self._reranker = Reranker(model_name=reranker_model)

            rerank_pool = getattr(config, "RAG_RERANK_POOL", 0)
            candidates = documents_text[:rerank_pool] if rerank_pool > 0 else documents_text
            _debug(f"[RAG Pipeline] Reranking {len(candidates)} documents with {self._reranker.model_name}...")
            reranked = self._reranker.rerank(query, candidates, top_k=top_k)
        else:
            _debug(f"[RAG Pipeline] Reranking DISABLED. Returning top {top_k} from {retrieval_mode} results.")
            reranked = [{"text": t, "score": 0.0} for t in documents_text[:top_k]]

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

            payload.append({
                "chunk_id": chunk_id,
                "text": doc.page_content,
                "source": source,
                "score": round(score, 4),
                "metadata": metadata,
            })

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
