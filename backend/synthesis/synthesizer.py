"""LLM-backed response synthesis for SQL + RAG outputs with formatter fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from backend import config

logger = logging.getLogger(__name__)

_W = 66  # output width
_MAX_CHUNKS_FOR_PROMPT = 5
_MAX_CHUNK_CHARS = 1200


@dataclass
class SynthesisResult:
    """Structured synthesis output retained internally for compatibility and tracing."""

    answer: str = ""
    needs_clarification: bool = False
    reason: str | None = None
    question: str | None = None
    sources: list[str] = field(default_factory=list)
    latency: float = 0.0


class ResponseSynthesizer:
    """Produces a final answer from SQL/RAG evidence using LLM synthesis."""

    def __init__(self) -> None:
        self.last_result: SynthesisResult | None = None

    def synthesize(
        self,
        user_query: str,
        route: str,
        sql_result: dict | None = None,
        rag_result: list[dict] | dict | None = None,
        subtask_results: list[dict] | None = None,
    ) -> str:
        """Synthesize final response through LLM, with deterministic formatter fallback."""
        normalized_rag = self._normalize_rag_result(rag_result)
        normalized_subtasks = self._normalize_subtask_results(subtask_results)

        try:
            result = self._synthesize_with_llm(
                original_query=user_query,
                route=route,
                sql_result=sql_result,
                rag_result=normalized_rag,
                subtask_results=normalized_subtasks,
            )
            self.last_result = result
            return self._render_result(result)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Synthesis LLM call failed; using formatter fallback. "
                "query=%r route=%s error=%s",
                user_query,
                route,
                exc,
            )
            return self._format_fallback(
                user_query=user_query,
                route=route,
                sql_result=sql_result,
                rag_result=normalized_rag,
                subtask_results=normalized_subtasks,
            )

    def _synthesize_with_llm(
        self,
        original_query: str,
        route: str,
        sql_result: dict | None,
        rag_result: list[dict],
        subtask_results: list[dict] | None,
    ) -> SynthesisResult:
        chat_openai_cls = self._resolve_chat_openai_class()
        if chat_openai_cls is None:
            raise RuntimeError("ChatOpenAI class unavailable for synthesis")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._build_system_prompt()),
                ("human", "{payload}"),
            ]
        )

        llm = chat_openai_cls(
            model=config.SQL_OPENAI_MODEL,
            temperature=config.SYNTHESIS_TEMPERATURE,
            api_key=config.OPENAI_API_KEY,
            timeout=config.ROUTER_TIMEOUT_SECONDS,
        )

        payload = self._build_user_payload(
            original_query=original_query,
            route=route,
            sql_result=sql_result,
            rag_result=rag_result,
            subtask_results=subtask_results,
        )

        chain = prompt | llm
        start = time.perf_counter()
        response = chain.invoke({"payload": json.dumps(payload, ensure_ascii=True, indent=2)})
        latency = time.perf_counter() - start

        raw_content = str(getattr(response, "content", "")).strip()
        clarification = self._parse_clarification_json(raw_content)

        evidence_sources = self._collect_evidence_sources(
            sql_result=sql_result,
            rag_result=rag_result,
            subtask_results=subtask_results,
        )

        if clarification is not None:
            return SynthesisResult(
                needs_clarification=True,
                reason=clarification["reason"],
                question=clarification["question"],
                sources=evidence_sources,
                latency=latency,
            )

        cited_sources = self._extract_inline_sources(raw_content)
        sources = self._finalize_sources(cited_sources, evidence_sources)
        return SynthesisResult(
            answer=raw_content,
            sources=sources,
            latency=latency,
        )

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are the final synthesis agent for an Adaptive Agentic RAG system.\n"
            "Your job is to answer the original query using only the provided SQL evidence and/or text chunks.\n\n"
            "Rules:\n"
            "1. Answer in clear, concise natural language.\n"
            "2. Ground every claim strictly in the provided evidence. Never invent facts.\n"
            "3. Cite sources inline whenever you reference evidence using this exact format: "
            "[source: <filename_or_table>].\n"
            "4. If SQL evidence and text chunk evidence contradict each other on the same fact, "
            "explicitly flag the conflict in your answer.\n"
            "5. If evidence is insufficient to answer confidently, output only this exact JSON shape "
            "(and nothing else): "
            "{{\"needs_clarification\": true, \"reason\": \"<why>\", \"question\": \"<what to ask the user>\"}}.\n"
            "6. Do not output markdown code fences."
        )

    def _build_user_payload(
        self,
        original_query: str,
        route: str,
        sql_result: dict | None,
        rag_result: list[dict],
        subtask_results: list[dict] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "original_query": original_query,
            "route": route,
            "sql_result": self._to_sql_payload(sql_result),
            "rag_chunks": self._to_chunk_payload(self._top_chunks(rag_result, _MAX_CHUNKS_FOR_PROMPT)),
            "subtask_results": [],
        }

        if subtask_results:
            for item in subtask_results:
                sub_sql = item.get("sql_result")
                sub_rag = self._normalize_rag_result(item.get("rag_result"))
                payload["subtask_results"].append(
                    {
                        "sub_query": item.get("sub_query", ""),
                        "route": item.get("route", "unknown"),
                        "sql_result": self._to_sql_payload(sub_sql if isinstance(sub_sql, dict) else None),
                        "rag_chunks": self._to_chunk_payload(
                            self._top_chunks(sub_rag, _MAX_CHUNKS_FOR_PROMPT)
                        ),
                    }
                )

        return payload

    @staticmethod
    def _resolve_chat_openai_class():
        try:
            module = importlib.import_module("langchain_openai")
            return getattr(module, "ChatOpenAI")
        except Exception:  # noqa: BLE001
            pass

        try:
            module = importlib.import_module("langchain_community.chat_models")
            return getattr(module, "ChatOpenAI")
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _normalize_rag_result(rag_result: list[dict] | dict | None) -> list[dict]:
        if rag_result is None:
            return []
        if isinstance(rag_result, list):
            return [item for item in rag_result if isinstance(item, dict)]
        if isinstance(rag_result, dict):
            chunks = rag_result.get("chunks", [])
            if isinstance(chunks, list):
                return [item for item in chunks if isinstance(item, dict)]
        return []

    def _normalize_subtask_results(self, subtask_results: list[dict] | None) -> list[dict] | None:
        if subtask_results is None:
            return None

        normalized: list[dict] = []
        for item in subtask_results:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "sub_query": item.get("sub_query", ""),
                    "route": item.get("route", "unknown"),
                    "sql_result": item.get("sql_result") if isinstance(item.get("sql_result"), dict) else None,
                    "rag_result": self._normalize_rag_result(item.get("rag_result")),
                }
            )
        return normalized

    @staticmethod
    def _to_sql_payload(sql_result: dict | None) -> dict[str, Any] | None:
        if not sql_result:
            return None

        sql_text = str(sql_result.get("query") or sql_result.get("sql") or "").strip()
        rows = sql_result.get("rows")
        if rows is None:
            rows = sql_result.get("result", [])
        if not isinstance(rows, list):
            rows = []

        return {
            "sql": sql_text,
            "rows": rows,
            "path": sql_result.get("path"),
            "latency": sql_result.get("latency"),
            "schema_used": sql_result.get("schema_used", []),
            "error": sql_result.get("error"),
        }

    def _to_chunk_payload(self, chunks: list[dict]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for chunk in chunks:
            source = self._source_name(str(chunk.get("source", "")))
            text = str(chunk.get("text", "")).strip()
            if len(text) > _MAX_CHUNK_CHARS:
                text = text[:_MAX_CHUNK_CHARS] + "..."
            payload.append(
                {
                    "source": source,
                    "score": chunk.get("score"),
                    "text": text,
                }
            )
        return payload

    @staticmethod
    def _top_chunks(chunks: list[dict], top_k: int) -> list[dict]:
        indexed = list(enumerate(chunks))

        def _rank_key(item: tuple[int, dict]) -> tuple[float, int]:
            idx, chunk = item
            score = chunk.get("score") if isinstance(chunk, dict) else None
            numeric = float(score) if isinstance(score, (int, float)) else float("-inf")
            return (-numeric, idx)

        ranked = sorted(indexed, key=_rank_key)
        return [chunk for _, chunk in ranked[:top_k]]

    def _collect_evidence_sources(
        self,
        sql_result: dict | None,
        rag_result: list[dict],
        subtask_results: list[dict] | None,
    ) -> list[str]:
        sources: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            clean = name.strip()
            if not clean:
                return
            key = clean.lower()
            if key in seen:
                return
            seen.add(key)
            sources.append(clean)

        for table in self._extract_sql_sources(sql_result):
            add(table)

        for chunk in self._top_chunks(rag_result, _MAX_CHUNKS_FOR_PROMPT):
            add(self._source_name(str(chunk.get("source", ""))))

        if subtask_results:
            for task in subtask_results:
                task_sql = task.get("sql_result")
                if isinstance(task_sql, dict):
                    for table in self._extract_sql_sources(task_sql):
                        add(table)

                task_rag = self._normalize_rag_result(task.get("rag_result"))
                for chunk in self._top_chunks(task_rag, _MAX_CHUNKS_FOR_PROMPT):
                    add(self._source_name(str(chunk.get("source", ""))))

        return sources

    def _extract_sql_sources(self, sql_result: dict | None) -> list[str]:
        if not sql_result:
            return []

        sources: list[str] = []
        seen: set[str] = set()

        def add(source: str) -> None:
            clean = source.strip().strip('"')
            if not clean:
                return
            key = clean.lower()
            if key in seen:
                return
            seen.add(key)
            sources.append(clean)

        schema_used = sql_result.get("schema_used", [])
        if isinstance(schema_used, list):
            for table in self._extract_table_names(schema_used):
                add(table)

        sql_text = str(sql_result.get("query") or sql_result.get("sql") or "")
        for match in re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_\.\"]*)", sql_text, flags=re.IGNORECASE):
            add(match.split(".")[-1])

        return sources

    @staticmethod
    def _extract_inline_sources(text: str) -> list[str]:
        matches = re.findall(r"\[source:\s*([^\]]+)\]", text, flags=re.IGNORECASE)
        extracted: list[str] = []
        for raw in matches:
            for part in raw.split(","):
                clean = part.strip()
                if clean:
                    extracted.append(clean)
        return extracted

    @staticmethod
    def _finalize_sources(cited_sources: list[str], evidence_sources: list[str]) -> list[str]:
        canon = {source.lower(): source for source in evidence_sources}
        ordered: list[str] = []
        seen: set[str] = set()

        def add(source: str) -> None:
            key = source.lower()
            if key in seen:
                return
            seen.add(key)
            ordered.append(source)

        if cited_sources:
            for source in cited_sources:
                clean = source.strip()
                if not clean:
                    continue
                add(canon.get(clean.lower(), clean))
            return ordered

        for source in evidence_sources:
            add(source)
        return ordered

    @staticmethod
    def _source_name(source: str) -> str:
        if not source:
            return "unknown"
        return Path(source).name or source

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _parse_clarification_json(self, content: str) -> dict[str, str] | None:
        cleaned = self._strip_code_fences(content)
        if not cleaned:
            return None

        payload: Any = None
        try:
            payload = json.loads(cleaned)
        except Exception:  # noqa: BLE001
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
            except Exception:  # noqa: BLE001
                return None

        if not isinstance(payload, dict):
            return None

        required_keys = {"needs_clarification", "reason", "question"}
        if set(payload.keys()) != required_keys:
            return None

        if payload.get("needs_clarification") is not True:
            return None

        reason = payload.get("reason")
        question = payload.get("question")
        if not isinstance(reason, str) or not isinstance(question, str):
            return None

        return {
            "reason": reason.strip(),
            "question": question.strip(),
        }

    @staticmethod
    def _render_result(result: SynthesisResult) -> str:
        if result.needs_clarification:
            payload = {
                "needs_clarification": True,
                "reason": result.reason or "Insufficient evidence.",
                "question": result.question or "Can you clarify your request?",
            }
            lines = [
                "Clarification required.",
                json.dumps(payload, ensure_ascii=True),
            ]
            lines.append(
                "Sources: " + (", ".join(result.sources) if result.sources else "none")
            )
            lines.append(f"Synthesis latency: {result.latency:.3f}s")
            return "\n".join(lines)

        answer = result.answer.strip() or "No answer generated."
        lines = [answer]
        lines.append("Sources: " + (", ".join(result.sources) if result.sources else "none"))
        lines.append(f"Synthesis latency: {result.latency:.3f}s")
        return "\n".join(lines)

    # Fallback formatter (legacy behavior)

    def _format_fallback(
        self,
        user_query: str,
        route: str,
        sql_result: dict | None,
        rag_result: list[dict],
        subtask_results: list[dict] | None,
    ) -> str:
        """Formatting-only synthesis fallback used when LLM synthesis fails."""
        lines: list[str] = [self._header(user_query, route)]

        if subtask_results is not None:
            lines.append(self._format_subtask_section(subtask_results))
        else:
            if sql_result is not None:
                lines.append(self._format_sql_section(sql_result))
            if rag_result:
                lines.append(self._format_rag_section(rag_result))
            if sql_result is None and not rag_result:
                lines.append("| [WARN] No data source was executed.")

        lines.append(self._footer())
        return "\n".join(lines)

    # Header / footer

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

    # Subtask decomposition formatting

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

            sub_sql_result = item.get("sql_result")
            sub_rag_result = self._normalize_rag_result(item.get("rag_result"))

            if isinstance(sub_sql_result, dict):
                lines.append(self._format_sql_inline(sub_sql_result))
            if sub_rag_result:
                lines.append(self._format_rag_inline(sub_rag_result))
            if not isinstance(sub_sql_result, dict) and not sub_rag_result:
                lines.append("|     [WARN] No pipeline executed")

            if idx < len(subtask_results):
                lines.append(self._divider())

        return "\n".join(lines)

    # SQL formatting

    @staticmethod
    def _extract_table_names(schema_used: list[Any]) -> list[str]:
        table_names: list[str] = []
        for entry in schema_used:
            raw = str(entry)
            if raw.startswith("Table: "):
                table_names.append(raw.split("|")[0].replace("Table:", "").strip())
            else:
                table_names.append(raw.strip())
        return table_names

    def _format_sql_section(self, sql_result: dict) -> str:
        lines: list[str] = ["|", "|  [SQL PIPELINE]"]
        lines.append(self._sql_body(sql_result, prefix="|     "))
        return "\n".join(lines)

    def _format_sql_inline(self, sql_result: dict) -> str:
        return self._sql_body(sql_result, prefix="|     ")

    def _sql_body(self, sql_result: dict, prefix: str) -> str:
        query = str(sql_result.get("query") or sql_result.get("sql") or "").strip()
        schema_used = sql_result.get("schema_used", [])
        lines: list[str] = []

        if isinstance(schema_used, list) and schema_used:
            tables = ", ".join(self._extract_table_names(schema_used))
            lines.append(f"{prefix}Tables : {tables}")

        sql_display = query.replace("\n", " ") if query else "n/a"
        max_sql_len = 55
        if len(sql_display) > max_sql_len:
            wrapped_lines: list[str] = []
            while sql_display:
                if len(sql_display) <= max_sql_len:
                    wrapped_lines.append(sql_display)
                    break
                break_pos = sql_display.rfind(" ", 0, max_sql_len)
                if break_pos == -1:
                    break_pos = max_sql_len
                wrapped_lines.append(sql_display[:break_pos])
                sql_display = sql_display[break_pos:].lstrip()
            for i, line in enumerate(wrapped_lines):
                if i == 0:
                    lines.append(f"{prefix}SQL    : {line}")
                else:
                    lines.append(f"{prefix}         {line}")
        else:
            lines.append(f"{prefix}SQL    : {sql_display}")

        path = str(sql_result.get("path", "unknown")).upper()
        latency = float(sql_result.get("latency", 0.0) or 0.0)
        lines.append(f"{prefix}Path   : {path} ({latency:.2f}s)")

        ok = bool(sql_result.get("ok", not sql_result.get("error")))
        if not ok:
            error = sql_result.get("error", "unknown")
            lines.append(f"{prefix}Status : FAILED")
            lines.append(f"{prefix}Error  : {error}")
        else:
            rows = sql_result.get("rows")
            if rows is None:
                rows = sql_result.get("result", [])
            if not isinstance(rows, list):
                rows = []
            row_count = int(sql_result.get("row_count", len(rows)))
            lines.append(f"{prefix}Status : OK  ({row_count} rows)")
            for row in rows[:3]:
                lines.append(f"{prefix}  > {row}")
            if row_count > 3:
                lines.append(f"{prefix}  ... and {row_count - 3} more")

        return "\n".join(lines)

    # RAG formatting

    def _format_rag_section(self, rag_result: list[dict]) -> str:
        lines: list[str] = ["|", "|  [RAG PIPELINE]"]
        lines.append(self._rag_body(rag_result, prefix="|     "))
        return "\n".join(lines)

    def _format_rag_inline(self, rag_result: list[dict]) -> str:
        return self._rag_body(rag_result, prefix="|     ")

    @staticmethod
    def _rag_body(rag_result: list[dict], prefix: str) -> str:
        if not rag_result:
            return f"{prefix}Status : No relevant chunks found"

        lines: list[str] = [f"{prefix}Chunks : {len(rag_result)} retrieved"]
        for idx, item in enumerate(rag_result[:3], start=1):
            snippet = str(item.get("text", "")).strip().replace("\n", " ")
            preview_chars = max(50, int(config.RAG_PREVIEW_CHARS))
            snippet = snippet[:preview_chars] + ("..." if len(snippet) > preview_chars else "")
            source = str(item.get("source", ""))
            source_name = Path(source).name if source else "unknown"
            score = item.get("score")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            lines.append(f"{prefix}  {idx}. [{score_str}] {source_name}")
            lines.append(f"{prefix}     {snippet}")

        return "\n".join(lines)
