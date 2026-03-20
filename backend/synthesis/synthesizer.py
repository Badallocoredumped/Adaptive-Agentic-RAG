"""Response synthesis for combining SQL and RAG outputs."""

from __future__ import annotations


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
        sections: list[str] = [f"Query: {user_query}", f"Route selected: {route}"]

        if subtask_results is not None:
            sections.append(self._format_subtask_section(subtask_results))
            return "\n\n".join(sections)

        if sql_result is not None:
            sections.append(self._format_sql_section(sql_result))

        if rag_result is not None:
            sections.append(self._format_rag_section(rag_result))

        if sql_result is None and rag_result is None:
            sections.append("No data source was executed.")

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
            return "Decomposition output: no sub-tasks generated."

        lines: list[str] = [f"Decomposition output: {len(subtask_results)} sub-tasks."]
        for idx, item in enumerate(subtask_results, start=1):
            sub_query = item.get("sub_query", "")
            route = item.get("route", "unknown")
            lines.append(f"SubTask {idx}: route={route} query={sub_query}")

            sql_result = item.get("sql_result")
            rag_result = item.get("rag_result")

            if sql_result is not None:
                lines.append(self._format_sql_section(sql_result))
            if rag_result is not None:
                lines.append(self._format_rag_section(rag_result))
            if sql_result is None and rag_result is None:
                lines.append("No data source was executed for this sub-task.")

        return "\n".join(lines)

    @staticmethod
    def _format_sql_section(sql_result: dict) -> str:
        if not sql_result.get("ok", False):
            return f"SQL output: query failed with error: {sql_result.get('error', 'unknown error')}"

        row_count = sql_result.get("row_count", 0)
        rows = sql_result.get("rows", [])
        preview = rows[:3]
        return f"SQL output: {row_count} rows returned. Preview: {preview}"

    @staticmethod
    def _format_rag_section(rag_result: list[dict]) -> str:
        if not rag_result:
            return "RAG output: no relevant chunks found."

        preview_lines = []
        for item in rag_result[:3]:
            snippet = item.get("text", "").strip().replace("\n", " ")
            snippet = snippet[:140] + ("..." if len(snippet) > 140 else "")
            preview_lines.append(
                f"- score={item.get('score')} source={item.get('source')} text={snippet}"
            )

        joined = "\n".join(preview_lines)
        return f"RAG output: {len(rag_result)} chunks retrieved.\n{joined}"
