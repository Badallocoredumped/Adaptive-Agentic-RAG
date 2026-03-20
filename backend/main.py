"""Main orchestration flow for the Adaptive Agentic RAG MVP."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from backend import config
from backend.rag import (
    DocumentLoader,
    FAISSVectorStore,
    RagRetriever,
    SentenceTransformerEmbedder,
    TextChunker,
)
from backend.router import QueryRouter
from backend.sql import SQLiteDatabase, SQLExecutor, SQLQueryGenerator
from backend.synthesis import ResponseSynthesizer


class AdaptiveAgenticRAGSystem:
    """Coordinates router, SQL pipeline, RAG pipeline, and final synthesis."""

    def __init__(self) -> None:
        self.router = QueryRouter()
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            mode=config.CHUNKER_MODE,
        )
        self.embedder = SentenceTransformerEmbedder(model_name=config.EMBEDDING_MODEL_NAME)
        self.vector_store = FAISSVectorStore(
            index_path=str(config.INDEX_PATH),
            metadata_path=str(config.METADATA_PATH),
            embeddings=self.embedder.get_langchain_embeddings(),
        )
        self.vector_store.load()
        self.retriever = RagRetriever(self.embedder, self.vector_store)

        self.database = SQLiteDatabase(str(config.SQLITE_DB_PATH))
        self.database.initialize_schema()
        self.query_generator = SQLQueryGenerator()
        self.sql_executor = SQLExecutor(self.database)

        self.synthesizer = ResponseSynthesizer()

    def ingest_documents(self, paths: list[str]) -> int:
        """Load, chunk, and index documents into FAISS; returns number of indexed chunks."""
        documents = self.loader.load_documents(paths)
        chunks = self.chunker.chunk_documents(documents)
        self.retriever.index_chunks(chunks)
        return len(chunks)

    def run_query(self, user_query: str) -> str:
        """End-to-end query flow: route -> execute source pipelines -> synthesize answer."""
        if config.ROUTER_MODE == "llm":
            route = self.router.route_with_llm(user_query)
        else:
            route = self.router.route(user_query)

        sql_result: dict | None = None
        rag_result: list[dict] | None = None

        if route in {"sql", "hybrid"}:
            sql_query = self.query_generator.generate_sql(user_query)
            sql_result = self.sql_executor.execute(sql_query)

        if route in {"text", "hybrid"}:
            rag_result = self.retriever.retrieve(user_query, top_k=config.DEFAULT_TOP_K)

        return self.synthesizer.synthesize(
            user_query=user_query,
            route=route,
            sql_result=sql_result,
            rag_result=rag_result,
        )


def build_system() -> AdaptiveAgenticRAGSystem:
    """Factory function for creating the MVP system object."""
    return AdaptiveAgenticRAGSystem()


def run_query(user_query: str) -> str:
    """Convenience function matching the requested function-call based entry mode."""
    system = build_system()
    return system.run_query(user_query)


if __name__ == "__main__":
    demo_question = "What is the total number of orders and summarize related documents?"
    print(run_query(demo_question))
