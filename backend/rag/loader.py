"""Utilities for loading raw documents from text and PDF files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
        """Load multiple files into a list of Document instances."""
        documents: list[Document] = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists() or not path.is_file():
                continue

            if path.suffix.lower() == ".pdf":
                text = self._read_pdf(path)
            elif path.suffix.lower() in self.SUPPORTED_TEXT_EXTENSIONS:
                text = self._read_text(path)
            else:
                continue

            if text.strip():
                domain = self._infer_domain(path)
                documents.append(
                    Document(
                        text=text,
                        source=str(path),
                        metadata={
                            "filename": path.name,
                            "extension": path.suffix.lower(),
                            "domain": domain,
                        },
                    )
                )
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
