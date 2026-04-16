"""Utilities for loading raw documents from text and PDF files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend import config


@dataclass
class Document:
    """Normalized in-memory document object used by the RAG pipeline."""

    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Loads .txt/.md and .pdf files into normalized Document objects."""

    SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}

    def load_documents(self, paths: list[str]) -> list[Document]:
        """Load multiple files into a list of Document instances concurrently."""
        documents: list[Document] = []

        def _load_single(raw_path: str) -> Document | None:
            path = Path(raw_path)
            if not path.exists() or not path.is_file():
                return None

            if path.suffix.lower() == ".pdf":
                text = self._read_pdf(path)
            elif path.suffix.lower() in self.SUPPORTED_TEXT_EXTENSIONS:
                text = self._read_text(path)
            else:
                return None

            if text.strip():
                domain = self._infer_domain(path)
                return Document(
                    text=text,
                    source=str(path),
                    metadata={
                        "filename": path.name,
                        "extension": path.suffix.lower(),
                        "domain": domain,
                    },
                )
            return None

        # Load concurrently using standard ThreadPoolExecutor limits
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_load_single, str(p)) for p in paths]
            for future in as_completed(futures):
                doc = future.result()
                if doc is not None:
                    documents.append(doc)

        return documents

    @staticmethod
    def _infer_domain(path: Path) -> str:
        name = path.name.lower()
        for hint, domain in config.FILENAME_DOMAIN_HINTS.items():
            if hint in name:
                return domain
        return "general"

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _read_pdf(path: Path) -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
