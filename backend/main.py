"""Main orchestration flow for the Adaptive Agentic RAG MVP."""

from __future__ import annotations

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
        self._executor = ThreadPoolExecutor(max_workers=3)
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
        if getattr(config, "DEBUG_LOGGING", False):
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

        if route == "hybrid":
            self._debug(f"run_query() executing BOTH pipelines concurrently for query={user_query!r}")
            # Run sequentially to avoid nested pool starvation
            sql_result = self._run_sql_pipeline(user_query)
            rag_result = self.retriever.retrieve(user_query, top_k=config.RAG_TOP_K)
        else:
            if route == "sql":
                self._debug(f"run_query() executing TableRAG SQL pipeline for query={user_query!r}")
                sql_result = self._run_sql_pipeline(user_query)

            if route == "text":
                self._debug(f"run_query() executing RAG pipeline for query={user_query!r}")
                rag_result = self.retriever.retrieve(user_query, top_k=config.RAG_TOP_K)

        return self.synthesizer.synthesize(
            user_query=user_query,
            route=route,
            sql_result=sql_result,
            rag_result=rag_result,
        )

    def _execute_subtasks(self, sub_tasks: list[SubTask]) -> list[dict]:
        """Run each decomposed sub-task on its assigned pipeline(s) concurrently."""
        outputs: list[dict] = []

        def run_task(task: SubTask) -> dict:
            sql_res: dict | None = None
            rag_res: list[dict] | None = None
            self._debug(f"_execute_subtasks() dispatch route={task.route}, sub_query={task.sub_query!r}")

            if task.route == "hybrid":
                self._debug("_execute_subtasks() -> Executing TableRAG and RAG pipeline concurrently")
                # Run sequentially to avoid nested pool starvation
                sql_res = self._run_sql_pipeline(task.sub_query)
                rag_res = self.retriever.retrieve(task.sub_query, top_k=config.RAG_TOP_K)
            else:
                if task.route in {"sql"}:
                    self._debug("_execute_subtasks() -> TableRAG SQL pipeline")
                    sql_res = self._run_sql_pipeline(task.sub_query)

                if task.route in {"text"}:
                    self._debug("_execute_subtasks() -> RAG pipeline")
                    rag_res = self.retriever.retrieve(task.sub_query, top_k=config.RAG_TOP_K)

            return {
                "sub_query": task.sub_query,
                "route": task.route,
                "sql_result": sql_res,
                "rag_result": rag_res,
            }

        futures = [self._executor.submit(run_task, t) for t in sub_tasks]
        outputs = []
        for f in futures:
            try:
                outputs.append(f.result())
            except Exception as exc:
                outputs.append({
                    "sub_query": None,
                    "route": "error",
                    "sql_result": None,
                    "rag_result": None,
                    "error": str(exc),
                })

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
            if "Clarification required." in result:
                is_clarification = True
            else:
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

    # ── Full Pipeline Performance Test ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TESTING ALL PIPELINES: SQL, RAG, and HYBRID (Decompose Mode)")
    print("=" * 80)
    config.ROUTER_MODE = "decompose"

    test_queries = [
        # SQL-focused queries
        ("SQL 1 (Basic)", "Count the total number of orders in the database."),
        ("SQL 2 (Complex)", "What is the average transaction amount for customers located in Giza?"),
        
        # RAG-focused queries
        ("RAG 1 (Policy)", "What is the company policy on remote work and vacation days?"),
        ("RAG 2 (Support)", "How do I request a hardware upgrade or new laptop?"),

        # Hybrid-focused queries (requires both DB data and text-based policy)
        ("Hybrid 1", "How many support tickets are currently open, and what is the SLA for resolving them?"),
        ("Hybrid 2", "What is the average order amount for customers in Cairo, and what are the shipping terms for those orders?")
    ]

    import time
    total_latency = 0.0

    for i, (label, query) in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {label} ---")
        print(f"Query: {query!r}")
        
        start_time = time.perf_counter()
        try:
            res = system.run_query(query)
            end_time = time.perf_counter()
            duration = end_time - start_time
            total_latency += duration
            
            print_result(res)
            print(f"End-to-End Latency: {duration:.3f}s")
        except Exception as e:
            print(f"Error during execution: {e}")

    print("\n" + "=" * 80)
    print(f"TOTAL TEST SUITE LATENCY: {total_latency:.3f}s")
    print("=" * 80)
    
    print("\nRestoring standard decompose mode...")
    config.ROUTER_MODE = "decompose"


