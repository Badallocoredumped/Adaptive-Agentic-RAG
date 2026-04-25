"""Utilities for loading raw documents from text and PDF files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from backend import config
from backend.rag.utils import tokenize


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
            try:
                if not path.exists() or not path.is_file():
                    print(f"[WARNING] Failed to load {raw_path}: File does not exist or is not a file.")
                    return None

                if path.suffix.lower() == ".pdf":
                    text = self._read_pdf(path)
                elif path.suffix.lower() in self.SUPPORTED_TEXT_EXTENSIONS:
                    text = self._read_text(path)
                else:
                    print(f"[WARNING] Failed to load {raw_path}: Unsupported extension '{path.suffix}'.")
                    return None

                if text.strip():
                    domain = self._infer_domain(path, text)
                    return Document(
                        text=text,
                        source=str(path),
                        metadata={
                            "filename": path.name,
                            "extension": path.suffix.lower(),
                            "domain": domain,
                        },
                    )
                else:
                    print(f"[WARNING] Failed to load {raw_path}: Document is empty.")
                    return None
            except Exception as e:
                print(f"[WARNING] Failed to load {raw_path}: {e}")
                return None

        # Load concurrently using standard ThreadPoolExecutor limits
        with ThreadPoolExecutor() as executor:
            # executor.map yields results strictly in the original order of paths
            results = list(executor.map(_load_single, paths))

        success_count = 0
        failure_count = 0
        for doc in results:
            if doc is not None:
                documents.append(doc)
                success_count += 1
            else:
                failure_count += 1

        print(f"[INFO] Ingestion complete: {success_count} succeeded, {failure_count} failed")
        return documents

    @staticmethod
    def _infer_domain(path: Path, text: str) -> str:
        name = path.name.lower()
        for hint, domain in config.FILENAME_DOMAIN_HINTS.items():
            if hint in name:
                return domain
                
        # Fallback to content-based keyword matching (first 1000 chars)
        tokens = set(tokenize(text[:1000]))
        
        best_domain = "general"
        best_hits = 0
        
        for domain, keywords in config.DOMAIN_KEYWORDS.items():
            hits = sum(1 for keyword in keywords if keyword in tokens)
            if hits > best_hits:
                best_domain = domain
                best_hits = hits

        return best_domain

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _read_pdf(path: Path) -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
