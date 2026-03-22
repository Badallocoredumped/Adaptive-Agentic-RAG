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
from backend.router import QueryRouter, SubTask
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

    @staticmethod
    def _debug(message: str) -> None:
        if getattr(config, "ROUTER_DEBUG", False):
            print(f"[MAIN DEBUG] {message}")

    def ingest_documents(self, paths: list[str]) -> int:
        """Load, chunk, and index documents into FAISS; returns number of indexed chunks."""
        documents = self.loader.load_documents(paths)
        chunks = self.chunker.chunk_documents(documents)
        self.retriever.index_chunks(chunks)
        return len(chunks)

    def run_query(self, user_query: str) -> str:
        """End-to-end query flow: route -> execute source pipelines -> synthesize answer."""
        self._debug(f"run_query() mode={config.ROUTER_MODE}, query={user_query!r}")
        if config.ROUTER_MODE in {"decompose", "zeroshot"}:
            sub_tasks = self.router.decompose_with_zeroshot(user_query)
            subtask_results = self._execute_subtasks(sub_tasks)
            route = self.router.route_from_subtasks(sub_tasks)
            self._debug(
                "run_query() subtask routes="
                f"{[{'route': t.route, 'sub_query': t.sub_query} for t in sub_tasks]}, final_route={route}"
            )
            return self.synthesizer.synthesize(
                user_query=user_query,
                route=route,
                subtask_results=subtask_results,
            )

        if config.ROUTER_MODE == "llm":
            route = self.router.route_with_llm(user_query)
        else:
            route = self.router.route(user_query)

        sql_result: dict | None = None
        rag_result: list[dict] | None = None

        if route in {"sql", "hybrid"}:
            self._debug(f"run_query() executing SQL pipeline for query={user_query!r}")
            sql_query = self.query_generator.generate_sql(user_query)
            sql_result = self.sql_executor.execute(sql_query)

        if route in {"text", "hybrid"}:
            self._debug(f"run_query() executing RAG pipeline for query={user_query!r}")
            rag_result = self.retriever.retrieve(user_query, top_k=config.RAG_TOP_K)

        return self.synthesizer.synthesize(
            user_query=user_query,
            route=route,
            sql_result=sql_result,
            rag_result=rag_result,
        )

    def _execute_subtasks(self, sub_tasks: list[SubTask]) -> list[dict]:
        """Run each decomposed sub-task on its assigned pipeline(s)."""
        outputs: list[dict] = []
        for task in sub_tasks:
            sql_result: dict | None = None
            rag_result: list[dict] | None = None
            self._debug(f"_execute_subtasks() dispatch route={task.route}, sub_query={task.sub_query!r}")

            if task.route in {"sql", "hybrid"}:
                self._debug("_execute_subtasks() -> SQL pipeline")
                sql_query = self.query_generator.generate_sql(task.sub_query)
                sql_result = self.sql_executor.execute(sql_query)

            if task.route in {"text", "hybrid"}:
                self._debug("_execute_subtasks() -> RAG pipeline")
                rag_result = self.retriever.retrieve(task.sub_query, top_k=config.RAG_TOP_K)

            outputs.append(
                {
                    "sub_query": task.sub_query,
                    "route": task.route,
                    "sql_result": sql_result,
                    "rag_result": rag_result,
                }
            )

        return outputs


def build_system() -> AdaptiveAgenticRAGSystem:
    """Factory function for creating the MVP system object."""
    return AdaptiveAgenticRAGSystem()


def run_query(user_query: str) -> str:
    """Convenience function matching the requested function-call based entry mode."""
    system = build_system()
    return system.run_query(user_query)

if __name__ == "__main__":
    # Example 1: hybrid expansion — same sentence, two pipelines
    q1 = "Show me the revenue from customers of Cairo and explain why those happened."
    print("=" * 60)
    print(f"QUERY 1: {q1}")
    print("=" * 60)
    print(run_query(q1))

    print()

    # Example 2: two distinct sub-tasks (sql + text)
    q2 = (
        "What is the total revenue from all orders, and also summarize "
        "the key business insights mentioned in the sales document about "
        "customer trends and purchasing behavior."
    )
    print("=" * 60)
    print(f"QUERY 2: {q2}")
    print("=" * 60)
    print(run_query(q2))
