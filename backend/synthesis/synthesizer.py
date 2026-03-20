"""Response synthesis for combining SQL and RAG outputs."""

from __future__ import annotations

from pathlib import Path

from backend import config


class ResponseSynthesizer:
    """Produces a final natural-language answer from pipeline outputs."""

    def synthesize(
        self,
        user_query: str,
        route: str,
        sql_result: dict | None = None,
        rag_result: list[dict] | None = None,
        subtask_results: list[dict] | None = None,
    ) -> str:
        """Combine SQL and RAG outputs into a single answer string."""
        sections: list[str] = [
            "Request",
            f"  Query : {user_query}",
            f"  Route : {route}",
        ]

        if subtask_results is not None:
            sections.append(self._format_subtask_section(subtask_results))
            return "\n\n".join(sections)

        if sql_result is not None:
            sections.append(self._format_sql_section(sql_result))

        if rag_result is not None:
            sections.append(self._format_rag_section(rag_result))

        if sql_result is None and rag_result is None:
            sections.append("Execution\n  No data source was executed.")

        return "\n\n".join(sections)

    def synthesize_with_llm(
        self,
        user_query: str,
        route: str,
        sql_result: dict | None = None,
        rag_result: list[dict] | None = None,
        subtask_results: list[dict] | None = None,
    ) -> str:
        """Stub hook for future LLM-based synthesis."""
        return self.synthesize(
            user_query,
            route,
            sql_result=sql_result,
            rag_result=rag_result,
            subtask_results=subtask_results,
        )

    def _format_subtask_section(self, subtask_results: list[dict]) -> str:
        if not subtask_results:
            return "Decomposition\n  No sub-tasks generated."

        lines: list[str] = ["Decomposition", f"  Sub-tasks : {len(subtask_results)}"]
        for idx, item in enumerate(subtask_results, start=1):
            sub_query = item.get("sub_query", "")
            route = item.get("route", "unknown")
            lines.append("")
            lines.append(f"  [SubTask {idx}]")
            lines.append(f"    Route : {route}")
            lines.append(f"    Query : {sub_query}")

            sql_result = item.get("sql_result")
            rag_result = item.get("rag_result")

            if sql_result is not None:
                lines.append(self._indent_block(self._format_sql_section(sql_result), prefix="    "))
            if rag_result is not None:
                lines.append(self._indent_block(self._format_rag_section(rag_result), prefix="    "))
            if sql_result is None and rag_result is None:
                lines.append("    Execution: no data source was executed for this sub-task.")

        return "\n".join(lines)

    @staticmethod
    def _format_sql_section(sql_result: dict) -> str:
        if not sql_result.get("ok", False):
            return f"SQL\n  Status : failed\n  Error  : {sql_result.get('error', 'unknown error')}"

        row_count = sql_result.get("row_count", 0)
        rows = sql_result.get("rows", [])
        preview = rows[:3]
        return f"SQL\n  Status : ok\n  Rows   : {row_count}\n  Preview: {preview}"

    @staticmethod
    def _format_rag_section(rag_result: list[dict]) -> str:
        if not rag_result:
            return "RAG\n  Status : no relevant chunks found"

        preview_lines = []
        for idx, item in enumerate(rag_result[:3], start=1):
            snippet = item.get("text", "").strip().replace("\n", " ")
            preview_chars = max(50, int(config.RAG_PREVIEW_CHARS))
            snippet = snippet[:preview_chars] + ("..." if len(snippet) > preview_chars else "")
            source = str(item.get("source", ""))
            source_name = Path(source).name if source else "unknown"
            preview_lines.append(
                f"  {idx}. score={item.get('score')} source={source_name}\n"
                f"     text={snippet}"
            )

        joined = "\n".join(preview_lines)
        return f"RAG\n  Status : ok\n  Chunks : {len(rag_result)}\n{joined}"

    @staticmethod
    def _indent_block(text: str, prefix: str) -> str:
        return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())
