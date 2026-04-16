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
from backend.sql import PostgresDatabase, run_table_rag_pipeline
from backend.synthesis import ResponseSynthesizer


class AdaptiveAgenticRAGSystem:
    """Coordinates router, TableRAG SQL pipeline, RAG pipeline, and final synthesis."""

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

        if config.DB_BACKEND == "postgres":
            self.database = PostgresDatabase()
            self.database.initialize_schema()

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

    @staticmethod
    def _run_sql_pipeline(query: str) -> dict:
        """Run the TableRAG SQL pipeline and adapt the result for the synthesizer."""
        pipeline_result = run_table_rag_pipeline(query)

        return {
            "ok": not bool(pipeline_result.get("error")),
            "query": pipeline_result["sql"] or "n/a",
            "error": pipeline_result.get("error"),
            "rows": pipeline_result["result"],
            "row_count": len(pipeline_result["result"]),
            "schema_used": pipeline_result.get("schema_used", []),
            "path": pipeline_result.get("path", "unknown"),
            "latency": pipeline_result.get("latency", 0.0),
        }

    def run_query(self, user_query: str) -> str:
        """End-to-end query flow: route -> execute source pipelines -> synthesize answer.

        Routing modes (ROUTER_MODE)
        ---------------------------
        decompose : LLM decomposes query into routed sub-tasks (project default).
        llm       : Local LLM classifies the whole query into a single route.
        keyword   : Rule-based keyword matching into a single route (fallback).
        """
        self._debug(f"run_query() mode={config.ROUTER_MODE}, query={user_query!r}")

        # --- decompose: LLM decomposition into multiple routed sub-tasks ---
        if config.ROUTER_MODE == "decompose":
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

        # --- single-route modes ---
        if config.ROUTER_MODE == "llm":
            route = self.router.route_with_llm(user_query)
        else:  # "keyword" or any unrecognised value
            route = self.router.route(user_query)

        self._debug(f"run_query() single route={route}")

        sql_result: dict | None = None
        rag_result: list[dict] | None = None

        if route in {"sql", "hybrid"}:
            self._debug(f"run_query() executing TableRAG SQL pipeline for query={user_query!r}")
            sql_result = self._run_sql_pipeline(user_query)

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
                self._debug("_execute_subtasks() -> TableRAG SQL pipeline")
                sql_result = self._run_sql_pipeline(task.sub_query)

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
    import json
    
    print("Building system...")
    system = build_system()

    def print_result(result):
        print(f"\nRaw return value:\n{result}")
        
        # Check for clarification request (whether it is a JSON string or dict/object)
        is_clarification = False
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and "needs_clarification" in parsed:
                    is_clarification = parsed.get("needs_clarification", False)
            except json.JSONDecodeError:
                pass
        elif isinstance(result, dict) and "needs_clarification" in result:
            is_clarification = result.get("needs_clarification", False)
        elif hasattr(result, "needs_clarification"):
            is_clarification = getattr(result, "needs_clarification")

        print(f"Is clarification request? {is_clarification}")
        
        # Extract latency if available on the object or dict
        if hasattr(result, "latency"):
            print(f"Synthesis latency: {result.latency:.3f}s")
        elif hasattr(result, "synthesis_latency"):
            print(f"Synthesis latency: {result.synthesis_latency:.3f}s")
        elif isinstance(result, dict) and "latency" in result:
            print(f"Synthesis latency: {result['latency']:.3f}s")

    # TEST 1 — SQL-only path
    print("\n" + "=" * 80)
    print("TEST 1 — SQL-only path (single route, keyword mode)")
    print("=" * 80)
    config.ROUTER_MODE = "keyword"
    res1 = system.run_query("what is the average amount of orders for customers living in Giza?")
    print_result(res1)

    # TEST 2 — Text-only path
    print("\n" + "=" * 80)
    print("TEST 2 — Text-only path (single route, keyword mode)")
    print("=" * 80)
    config.ROUTER_MODE = "keyword"
    res2 = system.run_query("Summarize the Cairo sales report")
    print_result(res2)

    # TEST 3 — Hybrid path with potential conflict
    print("\n" + "=" * 80)
    print("TEST 3 — Hybrid path with potential conflict (single route, keyword mode)")
    print("=" * 80)
    config.ROUTER_MODE = "keyword"
    res3 = system.run_query("What is the total revenue and what does the sales report say about it?")
    print_result(res3)

    # TEST 4 — Clarification request path
    print("\n" + "=" * 80)
    print("TEST 4 — Clarification request path")
    print("=" * 80)
    config.ROUTER_MODE = "keyword"
    sql_res_empty = {
        "ok": True, "query": "n/a", "rows": [], "row_count": 0,
        "schema_used": [], "path": "fast", "latency": 0.0
    }
    res4 = system.synthesizer.synthesize(
        user_query="What is the market share of our product in Southeast Asia?",
        route="hybrid",
        sql_result=sql_res_empty,
        rag_result=[]
    )
    print_result(res4)

    # TEST 5 — Decompose mode end-to-end
    print("\n" + "=" * 80)
    print("TEST 5 — Decompose mode end-to-end")
    print("=" * 80)
    config.ROUTER_MODE = "decompose"
    res5 = system.run_query("What is the total number of orders and summarize the latest sales report?")
    print_result(res5)

    print("\nRestoring standard decompose mode...")
    config.ROUTER_MODE = "decompose"


