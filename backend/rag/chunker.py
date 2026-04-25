"""Document chunking logic for retrieval-ready text segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend import config
from backend.rag.loader import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """A retrieval unit generated from a source document."""

    chunk_id: int
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Character-based chunker with overlap for simple MVP retrieval."""

    def __init__(self, chunk_size: int, chunk_overlap: int, mode: str = config.CHUNKER_MODE) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if mode not in {"fixed", "recursive"}:
            raise ValueError("mode must be 'fixed' or 'recursive'")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.mode = mode

    def _make_chunk(self, next_id: int, text: str, source: str, metadata: dict) -> Chunk:
        return Chunk(chunk_id=next_id, text=text, source=source, metadata=dict(metadata))

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split each document into overlapped chunks while preserving source metadata."""
        if self.mode == "recursive":
            return self._chunk_documents_recursive(documents)

        chunks: list[Chunk] = []
        next_id = 0

        for doc in documents:
            for text_part, start in self._split_text(doc.text):
                meta = dict(doc.metadata)
                meta["start_char"] = start
                chunks.append(self._make_chunk(next_id, text_part, doc.source, meta))
                next_id += 1

        return chunks

    def _split_text(self, text: str) -> list[tuple[str, int]]:
        if not text:
            return []

        step = self.chunk_size - self.chunk_overlap
        parts: list[tuple[str, int]] = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            part = text[start:end].strip()
            if part:
                parts.append((part, start))
            if end >= len(text):
                break
            start += step

        return parts

    def _chunk_documents_recursive(self, documents: list[Document]) -> list[Chunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=config.RECURSIVE_SEPARATORS,
        )

        chunks: list[Chunk] = []
        next_id = 0

        for doc in documents:
            split_parts = splitter.split_text(doc.text)
            for part in split_parts:
                text_part = part.strip()
                if not text_part:
                    continue
                chunks.append(self._make_chunk(next_id, text_part, doc.source, doc.metadata))
                next_id += 1

        return chunks
