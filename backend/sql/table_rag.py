"""TableRAG-lite: semantic schema retrieval using SentenceTransformers + FAISS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend import config
from backend.models import get_shared_st_model
from backend.sql.schema import SchemaInfo

_SCHEMA_INDEX_PATH = config.INDEX_DIR / "schema.faiss"
_SCHEMA_TEXTS_PATH = config.INDEX_DIR / "schema_texts.json"




def _apply_e5_prefix(text: str, is_query: bool) -> str:
    """Apply E5 prefix formatting when configured."""
    if not config.E5_PREFIX_ENABLED or "e5" not in config.EMBEDDING_MODEL_NAME.lower():
        return text

    prefix = config.E5_QUERY_PREFIX if is_query else config.E5_PASSAGE_PREFIX
    stripped = text.strip()
    if stripped.lower().startswith(prefix.strip().lower()):
        return stripped
    return f"{prefix}{stripped}"


def _embed_texts(texts: list[str], is_query: bool) -> np.ndarray:
    """Embed texts and return a float32 matrix."""
    model = get_shared_st_model()
    payload = [_apply_e5_prefix(text, is_query=is_query) for text in texts]
    vectors = model.encode(payload, convert_to_numpy=True, normalize_embeddings=False)
    vectors = np.asarray(vectors, dtype=np.float32)

    if config.VECTOR_DISTANCE_METRIC.lower() == "cosine" and vectors.size > 0:
        faiss.normalize_L2(vectors)

    return vectors


def get_schema_texts(schema_info: SchemaInfo) -> list[str]:
    """Convert structured schema properties into retrieval-friendly text lines."""
    return schema_info.to_embedding_texts()


def build_schema_index(schema_info: SchemaInfo) -> None:
    """Build and persist a FAISS index for schema texts."""
    schema_texts = get_schema_texts(schema_info)
    if not schema_texts:
        raise ValueError("schema_info is empty; cannot build schema index.")

    vectors = _embed_texts(schema_texts, is_query=False)
    dim = int(vectors.shape[1])

    if config.VECTOR_DISTANCE_METRIC.lower() == "cosine":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(vectors)

    _SCHEMA_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, config.win_short_path(_SCHEMA_INDEX_PATH))

    with _SCHEMA_TEXTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(schema_texts, f, ensure_ascii=True, indent=2)


def retrieve_relevant_schema(query: str, top_k: int = 3) -> list[str]:
    """Retrieve top-k relevant schema text rows for a natural-language query."""
    if not _SCHEMA_INDEX_PATH.exists() or not _SCHEMA_TEXTS_PATH.exists():
        raise FileNotFoundError(
            "Schema index not found. Run build_schema_index(schema_info) first."
        )

    index = faiss.read_index(config.win_short_path(_SCHEMA_INDEX_PATH))
    with _SCHEMA_TEXTS_PATH.open("r", encoding="utf-8") as f:
        schema_texts: list[str] = json.load(f)

    if not schema_texts:
        return []

    safe_top_k = max(1, min(top_k, len(schema_texts)))

    query_vector = _embed_texts([query], is_query=True)
    _, indices = index.search(query_vector, safe_top_k)

    results: list[str] = []
    for idx in indices[0]:
        if idx < 0:
            continue
        if idx >= len(schema_texts):
            continue
        results.append(schema_texts[idx])

    return results
