"""High-level retrieval service for indexing and searching document chunks."""

from __future__ import annotations

from collections import defaultdict

from backend import config
from backend.rag.bm25_index import BM25Index
from backend.rag.chunker import Chunk
from backend.rag.embedder import SentenceTransformerEmbedder
from backend.rag.reranker import Reranker
from backend.rag.utils import tokenize
from backend.rag.vector_store import FAISSVectorStore
from langchain_core.documents import Document as LCDocument


class RagRetriever:
    """Coordinates LangChain document indexing and FAISS retrieval."""

    def __init__(self, embedder: SentenceTransformerEmbedder, vector_store: FAISSVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self._reranker: Reranker | None = None
        self._bm25: BM25Index | None = None

    @staticmethod
    def _debug(message: str) -> None:
        if config.DEBUG_LOGGING:
            print(message)

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
        self._debug(f"[RAG Hybrid] Building BM25 index over {len(all_docs)} corpus documents...")
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

    def _fetch_faiss_candidates(
        self, query_vector: list[float], fetch_k: int, query_domain: str | None
    ) -> tuple[list[str], dict, dict, set]:
        """FAISS vector search + deduplication. Returns (ranked_texts, doc_map, text_to_score, seen)."""
        if query_domain:
            results = self.vector_store.search_by_vector(
                query_vector, fetch_k, metadata_filter={"domain": query_domain},
            )
            if len(results) < fetch_k:
                results = self.vector_store.search_by_vector(query_vector, fetch_k)
        else:
            results = self.vector_store.search_by_vector(query_vector, fetch_k)

        doc_map: dict = {}
        text_to_score: dict = {}
        seen_normalized: set = set()
        ranked: list[str] = []

        for doc, distance in results:
            norm = " ".join(doc.page_content.split())
            if norm not in seen_normalized:
                seen_normalized.add(norm)
                doc_map[doc.page_content] = doc
                text_to_score[doc.page_content] = 1.0 / (1.0 + float(distance))
                ranked.append(doc.page_content)

        self._debug(f"[RAG Pipeline] FAISS returned {len(ranked)} unique chunks.")
        return ranked, doc_map, text_to_score, seen_normalized

    def _fuse_bm25(
        self,
        query: str,
        fetch_k: int,
        faiss_ranked: list[str],
        doc_map: dict,
        seen_normalized: set,
    ) -> list[str]:
        """BM25 search + RRF fusion with FAISS results. Returns fused ranked text list."""
        bm25_raw = self._get_bm25().search(query, top_k=fetch_k)
        bm25_ranked: list[str] = []
        for text, doc, _ in bm25_raw:
            norm = " ".join(text.split())
            if norm not in seen_normalized:
                seen_normalized.add(norm)
                doc_map[doc.page_content] = doc
            bm25_ranked.append(doc.page_content)

        self._debug(f"[RAG Hybrid] BM25 returned {len(bm25_ranked)} candidates. Applying RRF...")
        fused = self._rrf([faiss_ranked, bm25_ranked], k=config.RAG_RRF_K)
        return fused[:fetch_k]

    def _apply_reranking(
        self,
        query: str,
        documents_text: list[str],
        top_k: int,
        text_to_score: dict,
        retrieval_mode: str,
    ) -> list[dict]:
        """Apply cross-encoder reranking or threshold fallback. Returns list[{text, score}]."""
        if config.RAG_ENABLE_SEMANTIC_RERANK and config.RAG_RERANKER_MODEL.lower() != "none":
            if self._reranker is None:
                self._reranker = Reranker(model_name=config.RAG_RERANKER_MODEL)
            candidates = documents_text[:config.RAG_RERANK_POOL] if config.RAG_RERANK_POOL > 0 else documents_text
            self._debug(f"[RAG Pipeline] Reranking {len(candidates)} documents with {self._reranker.model_name}...")
            return self._reranker.rerank(query, candidates, top_k=top_k)

        limit = min(top_k, config.RAG_MAX_CHUNKS)

        if retrieval_mode == "hybrid":
            self._debug(f"[RAG Pipeline] Semantic Reranking is DISABLED. Hybrid mode bypasses score thresholding. Taking top {limit}.")
            return [{"text": t, "score": text_to_score.get(t, 0.0)} for t in documents_text[:limit]]

        self._debug(f"[RAG Pipeline] Semantic Reranking is DISABLED. Applying dense threshold (Threshold: {config.RAG_SCORE_THRESHOLD}, Max: {limit}).")
        reranked = [
            {"text": t, "score": s}
            for t in documents_text
            if (s := text_to_score.get(t, 0.0)) >= config.RAG_SCORE_THRESHOLD
        ][:limit]

        if not reranked and documents_text:
            self._debug(f"[RAG Pipeline] No chunks passed threshold, falling back to top {limit}.")
            reranked = [{"text": t, "score": text_to_score.get(t, 0.0)} for t in documents_text[:limit]]

        return reranked

    def _build_payload(self, reranked: list[dict], doc_map: dict) -> list[dict]:
        """Format reranked results into the final output structure."""
        payload: list[dict] = []
        for i, res in enumerate(reranked, 1):
            text = res["text"]
            score = res["score"]
            doc = doc_map[text]
            metadata = dict(doc.metadata)
            chunk_id = metadata.pop("chunk_id", None)
            source = metadata.get("source", "unknown")
            self._debug(f"  -> Rank {i} | Score: {score:.4f} | Source: {source}")
            payload.append({
                "chunk_id": chunk_id,
                "text": doc.page_content,
                "source": source,
                "score": round(score, 4),
                "metadata": metadata,
            })
        return payload

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k relevant chunks using FAISS + optional BM25 hybrid search, then CrossEncoder reranking."""
        fetch_k = config.RAG_FETCH_K if config.RAG_FETCH_K > 0 else max(top_k * config.RAG_FETCH_MULTIPLIER, 10)
        query_domain = self._infer_query_domain(query)
        query_vector = self.vector_store.embeddings.embed_query(query)

        self._debug(f"\n[RAG Pipeline] Query: {query!r}")
        self._debug(f"[RAG Pipeline] Fetching top {fetch_k} documents from FAISS...")

        faiss_ranked, doc_map, text_to_score, seen_normalized = self._fetch_faiss_candidates(
            query_vector, fetch_k, query_domain
        )

        retrieval_mode = config.RAG_RETRIEVAL_MODE
        if retrieval_mode == "hybrid":
            documents_text = self._fuse_bm25(query, fetch_k, faiss_ranked, doc_map, seen_normalized)
        else:
            documents_text = faiss_ranked

        reranked = self._apply_reranking(query, documents_text, top_k, text_to_score, retrieval_mode)

        self._debug(f"\n[RAG Pipeline] --- Top {top_k} Results ---")
        return self._build_payload(reranked, doc_map)

    @staticmethod
    def _infer_query_domain(query: str) -> str | None:
        """Infer domain based on keyword overlap with configured domain keywords."""
        tokens = set(tokenize(query))
        best_domain: str | None = None
        best_hits = 0

        for domain, keywords in config.DOMAIN_KEYWORDS.items():
            hits = sum(1 for keyword in keywords if keyword in tokens)
            if hits > best_hits:
                best_domain = domain
                best_hits = hits

        return best_domain if best_hits > 0 else None
