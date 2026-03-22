"""Response synthesis for combining SQL and RAG outputs."""

from __future__ import annotations

from pathlib import Path

from backend import config

_W = 66  # output width


class ResponseSynthesizer:
    """Produces a final natural-language answer from pipeline outputs."""

    # ── public API ───────────────────────────────────────────────

    def synthesize(
        self,
        user_query: str,
        route: str,
        sql_result: dict | None = None,
        rag_result: list[dict] | None = None,
        subtask_results: list[dict] | None = None,
    ) -> str:
        """Combine SQL and RAG outputs into a single answer string."""
        lines: list[str] = [self._header(user_query, route)]

        if subtask_results is not None:
            lines.append(self._format_subtask_section(subtask_results))
        else:
            if sql_result is not None:
                lines.append(self._format_sql_section(sql_result))
            if rag_result is not None:
                lines.append(self._format_rag_section(rag_result))
            if sql_result is None and rag_result is None:
                lines.append("| [WARN] No data source was executed.")

        lines.append(self._footer())
        return "\n".join(lines)

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

    # ── header / footer ──────────────────────────────────────────

    @staticmethod
    def _header(query: str, route: str) -> str:
        route_label = {"sql": "SQL", "text": "RAG", "hybrid": "HYBRID"}.get(
            route, route.upper()
        )
        return (
            f"\n+{'-' * _W}+\n"
            f"|  QUERY   {query:<{_W - 11}}|\n"
            f"|  ROUTE   {route_label:<{_W - 11}}|\n"
            f"+{'-' * _W}+"
        )

    @staticmethod
    def _footer() -> str:
        return f"+{'-' * _W}+"

    @staticmethod
    def _divider() -> str:
        return f"|{'-' * _W}|"

    # ── subtask decomposition ────────────────────────────────────

    def _format_subtask_section(self, subtask_results: list[dict]) -> str:
        if not subtask_results:
            return "|  No sub-tasks generated."

        lines: list[str] = [
            f"|  DECOMPOSITION  ({len(subtask_results)} sub-tasks)",
            f"+{'-' * _W}+",
        ]

        for idx, item in enumerate(subtask_results, start=1):
            sub_query = item.get("sub_query", "")
            route = item.get("route", "unknown")

            lines.append("|")
            lines.append(f"|  [{route.upper()}]  Sub-task {idx}")
            lines.append(f"|     \"{sub_query}\"")

            sql_result = item.get("sql_result")
            rag_result = item.get("rag_result")

            if sql_result is not None:
                lines.append(self._format_sql_inline(sql_result))
            if rag_result is not None:
                lines.append(self._format_rag_inline(rag_result))
            if sql_result is None and rag_result is None:
                lines.append("|     [WARN] No pipeline executed")

            if idx < len(subtask_results):
                lines.append(self._divider())

        return "\n".join(lines)

    # ── SQL formatting ───────────────────────────────────────────

    @staticmethod
    def _extract_table_names(schema_used: list) -> list[str]:
        table_names = []
        for entry in schema_used:
            if entry.startswith("Table: "):
                table_names.append(entry.split("|")[0].replace("Table:", "").strip())
            else:
                table_names.append(entry)
        return table_names

    def _format_sql_section(self, sql_result: dict) -> str:
        """Full-width SQL section for non-decomposed output."""
        lines: list[str] = ["|", "|  [SQL PIPELINE]"]
        lines.append(self._sql_body(sql_result, prefix="|     "))
        return "\n".join(lines)

    def _format_sql_inline(self, sql_result: dict) -> str:
        """Indented SQL section for subtask output."""
        return self._sql_body(sql_result, prefix="|     ")

    def _sql_body(self, sql_result: dict, prefix: str) -> str:
        query = str(sql_result.get("query", "")).strip()
        schema_used = sql_result.get("schema_used", [])
        lines: list[str] = []

        if schema_used:
            tables = ", ".join(self._extract_table_names(schema_used))
            lines.append(f"{prefix}Tables : {tables}")

        sql_display = query.replace("\n", " ") if query else "n/a"
        if len(sql_display) > 55:
            sql_display = sql_display[:55] + "..."
        lines.append(f"{prefix}SQL    : {sql_display}")

        # Add Path / Latency info
        path = str(sql_result.get("path", "unknown")).upper()
        latency = float(sql_result.get("latency", 0.0))
        lines.append(f"{prefix}Path   : {path} ({latency:.2f}s)")

        if not sql_result.get("ok", False):
            error = sql_result.get("error", "unknown")
            lines.append(f"{prefix}Status : FAILED")
            lines.append(f"{prefix}Error  : {error}")
        else:
            row_count = sql_result.get("row_count", 0)
            rows = sql_result.get("rows", [])
            lines.append(f"{prefix}Status : OK  ({row_count} rows)")
            for row in rows[:3]:
                lines.append(f"{prefix}  > {row}")
            if row_count > 3:
                lines.append(f"{prefix}  ... and {row_count - 3} more")

        return "\n".join(lines)

    # ── RAG formatting ───────────────────────────────────────────

    def _format_rag_section(self, rag_result: list[dict]) -> str:
        """Full-width RAG section for non-decomposed output."""
        lines: list[str] = ["|", "|  [RAG PIPELINE]"]
        lines.append(self._rag_body(rag_result, prefix="|     "))
        return "\n".join(lines)

    def _format_rag_inline(self, rag_result: list[dict]) -> str:
        """Indented RAG section for subtask output."""
        return self._rag_body(rag_result, prefix="|     ")

    @staticmethod
    def _rag_body(rag_result: list[dict], prefix: str) -> str:
        if not rag_result:
            return f"{prefix}Status : No relevant chunks found"

        lines: list[str] = [f"{prefix}Chunks : {len(rag_result)} retrieved"]
        for idx, item in enumerate(rag_result[:3], start=1):
            snippet = item.get("text", "").strip().replace("\n", " ")
            preview_chars = max(50, int(config.RAG_PREVIEW_CHARS))
            snippet = snippet[:preview_chars] + ("..." if len(snippet) > preview_chars else "")
            source = str(item.get("source", ""))
            source_name = Path(source).name if source else "unknown"
            score = item.get("score")
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            lines.append(f"{prefix}  {idx}. [{score_str}] {source_name}")
            lines.append(f"{prefix}     {snippet}")

        return "\n".join(lines)
