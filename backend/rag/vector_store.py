"""LangChain FAISS vector store wrapper for chunk indexing and retrieval."""

from __future__ import annotations

from pathlib import Path

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument


class FAISSVectorStore:
    """Manages a LangChain FAISS vector store and local persistence."""

    def __init__(self, index_path: str, metadata_path: str, embeddings=None) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embeddings = embeddings
        self.store: FAISS | None = None

    def load(self) -> bool:
        """Load FAISS store from disk if persisted files exist."""
        folder = self.index_path.parent
        index_name = self.index_path.stem
        faiss_file = folder / f"{index_name}.faiss"
        pkl_file = folder / f"{index_name}.pkl"

        if self.embeddings is None or not faiss_file.exists() or not pkl_file.exists():
            return False

        try:
            self.store = FAISS.load_local(
                folder_path=str(folder),
                embeddings=self.embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )
            return True
        except Exception:  # noqa: BLE001
            self.store = None
            return False

    def save(self) -> None:
        """Persist FAISS store locally."""
        if self.store is None:
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.store.save_local(folder_path=str(self.index_path.parent), index_name=self.index_path.stem)

    def add_documents(self, documents: list[LCDocument]) -> None:
        """Add LangChain documents to the vector store."""
        if self.embeddings is None or not documents:
            return

        if self.store is None:
            self.store = FAISS.from_documents(documents=documents, embedding=self.embeddings)
            return

        self.store.add_documents(documents)

    def search(self, query: str, top_k: int) -> list[tuple[LCDocument, float]]:
        """Return top-k similar documents and FAISS distance scores."""
        if self.store is None:
            return []

        return self.store.similarity_search_with_score(query=query, k=top_k)

    @staticmethod
    def create_empty_with_embeddings(embeddings) -> FAISS:
        """Create an empty FAISS store for advanced initialization scenarios."""
        import faiss

        return FAISS(
            embedding_function=embeddings,
            index=faiss.IndexFlatL2(384),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
