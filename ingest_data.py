"""Standalone ingestion script for indexing files under the data directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend import config
from backend.main import build_system


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def collect_documents(data_dir: Path) -> list[str]:
    """Collect ingestible files recursively while skipping the index folder."""
    index_dir = config.INDEX_DIR.resolve()
    documents: list[str] = []

    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            if index_dir in path.resolve().parents:
                continue
        except Exception:  # noqa: BLE001
            # If resolve fails due to path issues, keep file eligible.
            pass

        documents.append(str(path))

    return sorted(documents)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents under the data directory into the FAISS index."
    )
    parser.add_argument(
        "--data-dir",
        default=str(config.DATA_DIR),
        help="Directory to scan recursively for .pdf/.txt/.md files.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    paths = collect_documents(data_dir)
    if not paths:
        print(f"No supported documents found in: {data_dir}")
        return

    print(f"Found {len(paths)} files to ingest:")
    for path in paths:
        print(f"- {path}")

    system = build_system()
    indexed_chunks = system.ingest_documents(paths)

    print(f"\nIngestion complete. Indexed chunks: {indexed_chunks}")
    print(f"Index directory: {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
