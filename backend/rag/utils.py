"""Shared utilities for the unstructured RAG pipeline."""

from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer used for BM25, domain inference, and query analysis."""
    return re.findall(r"\b\w+\b", text.lower())
