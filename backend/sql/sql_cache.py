"""Semantic cache for successful SQL queries using FAISS."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend import config
from backend.models import get_shared_st_model


def _debug(message: str) -> None:
    if getattr(config, "DEBUG_LOGGING", False):
        print(message)


class SQLCache:
    """Golden SQL Cache using semantic similarity on the user's question."""

    def __init__(self, index_path: str | Path | None = None, metadata_path: str | Path | None = None) -> None:
        # Default paths if not provided
        if index_path is None:
            index_path = config.INDEX_DIR / "sql_cache.faiss"
        if metadata_path is None:
            metadata_path = config.INDEX_DIR / "sql_cache_texts.json"

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []
        self._lock = threading.Lock()




    def _apply_e5_prefix(self, text: str, is_query: bool) -> str:
        """Apply E5 prefix formatting if configured."""
        if not config.E5_PREFIX_ENABLED or "e5" not in config.EMBEDDING_MODEL_NAME.lower():
            return text

        prefix = config.E5_QUERY_PREFIX if is_query else config.E5_PASSAGE_PREFIX
        stripped = text.strip()
        if stripped.lower().startswith(prefix.strip().lower()):
            return stripped
        return f"{prefix}{stripped}"

    def _embed_texts(self, texts: list[str], is_query: bool) -> np.ndarray:
        """Embed texts and normalize for cosine similarity."""
        model = get_shared_st_model()
        payload = [self._apply_e5_prefix(text, is_query=is_query) for text in texts]
        vectors = model.encode(payload, convert_to_numpy=True, normalize_embeddings=False)
        vectors = np.asarray(vectors, dtype=np.float32)

        # We always want cosine similarity for the cache
        if vectors.size > 0:
            faiss.normalize_L2(vectors)

        return vectors

    def initialize_cache(self) -> None:
        """Initialize an empty FAISS index."""
        # Get dimensions by embedding a dummy string
        dummy_vector = self._embed_texts(["dummy"], is_query=True)
        dim = int(dummy_vector.shape[1])

        # IndexFlatIP with normalized vectors = Cosine Similarity
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def load_cache(self) -> bool:
        """Load index and metadata from disk. Returns True if successful."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False

        try:
            self.index = faiss.read_index(config.win_short_path(self.index_path))
            with self.metadata_path.open("r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            return True
        except Exception as e:
            _debug(f"[SQLCache] Failed to load cache: {e}")
            self.index = None
            self.metadata = []
            return False

    def save_cache(self) -> None:
        """Save index and metadata to disk."""
        with self._lock:
            if self.index is None:
                return

            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, config.win_short_path(self.index_path))

            with self.metadata_path.open("w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=True, indent=2)

    def add_to_cache(self, question: str, sql: str, schema: str | None = None) -> None:
        """Add a new question-SQL pair to the cache."""
        vector = self._embed_texts([question], is_query=True)
        with self._lock:
            if self.index is None:
                self.initialize_cache()
            self.index.add(vector)  # type: ignore
            self.metadata.append({"question": question, "sql": sql, "schema": schema})

    def search_cache(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search for similar past questions and return their SQL and scores."""
        if self.index is None or self.index.ntotal == 0:
            return []

        safe_top_k = min(top_k, self.index.ntotal)
        
        query_vector = self._embed_texts([query], is_query=True)
        scores, indices = self.index.search(query_vector, safe_top_k)

        results = []
        for i in range(safe_top_k):
            idx = int(indices[0][i])
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            score = float(scores[0][i])
            entry = self.metadata[idx].copy()
            entry["score"] = score
            results.append(entry)

        return results

    def check_cache_hit(self, query: str, threshold: float = 0.78) -> dict[str, Any]:
        """Check if a query has a semantic match in the cache above the threshold."""
        results = self.search_cache(query, top_k=1)
        
        if not results:
            _debug(f"[SQLCache] Decision: MISS | No results found for query: '{query}'")
            return {"hit": False}
        
        best_match = results[0]
        score = best_match["score"]
        
        if score >= threshold:
            _debug(f"[SQLCache] Decision: HIT  | Score: {score:.4f} | Query: '{query}'")
            return {
                "hit": True,
                "sql": best_match["sql"],
                "score": score,
                "question": best_match.get("question", query),
                "schema": best_match.get("schema", ""),
            }
        else:
            _debug(f"[SQLCache] Decision: MISS | Score: {score:.4f} (Below threshold {threshold:.4f}) | Query: '{query}'")
            return {"hit": False}
