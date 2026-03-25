"""Simple query router to choose SQL, text, or hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
import requests

from backend import config


@dataclass
class SubTask:
    """A decomposed unit of work with an assigned execution route."""

    sub_query: str
    route: str


@dataclass
class QueryRouter:
    """Rule-based router with an extension hook for future LLM routing."""

    default_route: str = config.DEFAULT_ROUTE

    @staticmethod
    def _debug(message: str) -> None:
        if getattr(config, "ROUTER_DEBUG", False):
            print(f"[ROUTER DEBUG] {message}")

    def route(self, query: str) -> str:
        """Return one of: 'sql', 'text', or 'hybrid'."""
        cleaned = query.lower()
        sql_hits = self._keyword_hits(cleaned, config.SQL_KEYWORDS)
        text_hits = self._keyword_hits(cleaned, config.TEXT_KEYWORDS)
        self._debug(f"route() keyword hits -> sql={sql_hits}, text={text_hits}, query={query!r}")

        if sql_hits and text_hits:
            self._debug("route() selected: hybrid")
            return "hybrid"
        if sql_hits:
            self._debug("route() selected: sql")
            return "sql"
        if text_hits:
            self._debug("route() selected: text")
            return "text"
        self._debug(f"route() selected default: {self.default_route}")
        return self.default_route

    def route_with_llm(self, query: str) -> str:
        """Classify query intent using local llama.cpp server; fallback to rule routing on failure."""
        try:
            prompt = (
                "Classify the following user query into exactly one label: sql, text, or hybrid. "
                "Return only the label with no extra words. "
                "Use sql for database/aggregation intent, text for document retrieval intent, "
                "and hybrid when both are clearly required.\n\n"
                f"User query: {query}"
            )

            content = self._call_local_llm(prompt)
            label = self._normalize_label(content)
            self._debug(f"route_with_llm() raw response={content!r}, normalized={label!r}")
            if label in {"sql", "text", "hybrid"}:
                return label
        except Exception as exc:  # noqa: BLE001
            self._debug(
                "route_with_llm() failed, falling back to keyword route(); "
                f"error={type(exc).__name__}: {exc}"
            )

        return self.route(query)

    def decompose(self, query: str) -> list[SubTask]:
        """Fallback decomposition: keep one sub-task using rule routing."""
        route = self.route(query)
        self._debug(f"decompose() fallback route={route} for query={query!r}")
        expanded = self._expand_hybrid_subtasks([SubTask(sub_query=query, route=route)])
        self._debug(
            "decompose() expanded subtasks="
            f"{[{'route': t.route, 'sub_query': t.sub_query} for t in expanded]}"
        )
        return expanded

    def decompose_with_zeroshot(self, query: str) -> list[SubTask]:
        """Decompose a query into multiple routed sub-tasks using LangChain chat model."""
        chat_openai_cls = self._resolve_chat_openai_class()
        if chat_openai_cls is None:
            self._debug("decompose_with_zeroshot() ChatOpenAI class unavailable, using decompose() fallback")
            return self.decompose(query)

        self._debug(f"decompose_with_zeroshot() using chat class={chat_openai_cls.__module__}.{chat_openai_cls.__name__}")

        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a decomposition and routing agent for a hybrid SQL + RAG system. "
                        "Break the user query into meaningful sub-queries and assign one route for each. "
                        "Allowed routes are exactly: sql, text. "
                        "Return strict JSON only in this shape: "
                        "[{{\"sub_query\": \"...\", \"route\": \"sql|text\"}}]. "
                        "If the user asks for mixed intent, split into multiple sub-queries so each sub-query is either sql or text. "
                        "The sub_query field must be a natural-language request, never raw SQL (no SELECT/INSERT/UPDATE/DELETE statements). "
                        "Do not include markdown, comments, or extra keys.",
                    ),
                    (
                        "human",
                        "User query: {query}\n"
                        "Max sub-tasks: {max_subtasks}\n"
                        "If one sub-task is enough, return a one-item JSON list.",
                    ),
                ]
            )

            llm = chat_openai_cls(
                model=config.ROUTER_MODEL,
                temperature=config.ROUTER_LLM_TEMPERATURE,
                base_url=f"{config.ROUTER_BASE_URL.rstrip('/')}/v1",
                api_key=config.ROUTER_API_KEY,
                timeout=config.ROUTER_TIMEOUT_SECONDS,
            )

            chain = prompt | llm
            response = chain.invoke(
                {
                    "query": query,
                    "max_subtasks": str(config.ROUTER_DECOMPOSE_MAX_SUBTASKS),
                }
            )
            self._debug(f"decompose_with_zeroshot() response type={type(response).__name__}")
            content = str(getattr(response, "content", ""))
            self._debug(f"decompose_with_zeroshot() raw LLM content={content!r}")

            sub_tasks = self._parse_decomposition_output(content)
            if sub_tasks:
                expanded = self._expand_hybrid_subtasks(sub_tasks)
                self._debug(
                    "decompose_with_zeroshot() parsed subtasks="
                    f"{[{'route': t.route, 'sub_query': t.sub_query} for t in sub_tasks]}"
                )
                self._debug(
                    "decompose_with_zeroshot() expanded subtasks="
                    f"{[{'route': t.route, 'sub_query': t.sub_query} for t in expanded]}"
                )
                return expanded
            self._debug("decompose_with_zeroshot() no valid subtasks parsed, using fallback")
        except Exception as exc:  # noqa: BLE001
            self._debug(
                "decompose_with_zeroshot() failed, using decompose() fallback; "
                f"error={type(exc).__name__}: {exc}"
            )

        return self.decompose(query)

    def route_from_subtasks(self, sub_tasks: list[SubTask]) -> str:
        """Derive an overall route from routed sub-tasks."""
        routes = {task.route for task in sub_tasks}
        if "sql" in routes and "text" in routes:
            self._debug("route_from_subtasks() selected: hybrid")
            return "hybrid"
        if "sql" in routes:
            self._debug("route_from_subtasks() selected: sql")
            return "sql"
        if "text" in routes:
            self._debug("route_from_subtasks() selected: text")
            return "text"
        self._debug(f"route_from_subtasks() selected default: {self.default_route}")
        return self.default_route

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

    def _parse_decomposition_output(self, raw_output: str) -> list[SubTask]:
        payload = self._safe_json_parse(raw_output)
        if isinstance(payload, dict):
            # Accept common model wrapper shapes like {"subtasks": [...]}.
            payload = payload.get("subtasks") or payload.get("tasks")
        if not isinstance(payload, list):
            self._debug("_parse_decomposition_output() payload is not a list")
            return []

        cleaned: list[SubTask] = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            sub_query = str(item.get("sub_query", "")).strip()
            route = self._normalize_label(str(item.get("route", "")))

            if not sub_query:
                continue
            if route not in {"sql", "text", "hybrid"}:
                self._debug(
                    "_parse_decomposition_output() invalid route from LLM, "
                    f"fallback to keyword route() for sub_query={sub_query!r}"
                )
                route = self.route(sub_query)

            cleaned.append(SubTask(sub_query=sub_query, route=route))
            if len(cleaned) >= config.ROUTER_DECOMPOSE_MAX_SUBTASKS:
                break

        return cleaned

    def _expand_hybrid_subtasks(self, sub_tasks: list[SubTask]) -> list[SubTask]:
        """Normalize hybrid subtasks into explicit sql and text execution units."""
        expanded: list[SubTask] = []
        for task in sub_tasks:
            if task.route == "hybrid":
                expanded.append(
                    SubTask(sub_query=self._specialize_sub_query(task.sub_query, "sql"), route="sql")
                )
                expanded.append(
                    SubTask(sub_query=self._specialize_sub_query(task.sub_query, "text"), route="text")
                )
            else:
                expanded.append(task)

        deduped: list[SubTask] = []
        seen: set[tuple[str, str]] = set()
        for task in expanded:
            key = (task.sub_query.strip().lower(), task.route)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(task)
            if len(deduped) >= config.ROUTER_DECOMPOSE_MAX_SUBTASKS:
                break

        return deduped

    def _specialize_sub_query(self, query: str, target_route: str) -> str:
        """Extract a route-focused clause from a mixed-intent query when possible."""
        if target_route not in {"sql", "text"}:
            return query.strip()

        candidates = self._split_on_connectors(query)
        best = query.strip()
        best_hits = -1

        for candidate in candidates:
            cleaned = candidate.strip(" ,.;")
            if not cleaned:
                continue

            hits = self._keyword_hits(
                cleaned.lower(),
                config.SQL_KEYWORDS if target_route == "sql" else config.TEXT_KEYWORDS,
            )
            if hits > best_hits:
                best = cleaned
                best_hits = hits

        return best

    @staticmethod
    def _split_on_connectors(query: str) -> list[str]:
        # Split on common coordination patterns used in mixed requests.
        parts = re.split(r"\b(?:and|also|then|plus|, then|, and)\b", query, flags=re.IGNORECASE)
        cleaned = [part.strip() for part in parts if part.strip()]
        return cleaned if cleaned else [query]

    @staticmethod
    def _safe_json_parse(raw_output: str) -> Any:
        text = raw_output.strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except Exception:  # noqa: BLE001
            pass

        # Fallback for models that wrap JSON with prose.
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _keyword_hits(query: str, keywords: set[str]) -> int:
        tokens = set(re.findall(r"\b\w+\b", query.lower()))
        return sum(1 for word in keywords if word in tokens)

    @staticmethod
    def _normalize_label(value: str) -> str | None:
        cleaned = value.strip().lower()
        if cleaned in {"sql", "text", "hybrid"}:
            return cleaned

        match = re.search(r"\b(sql|text|hybrid)\b", cleaned)
        if match:
            return match.group(1)

        return None

    @staticmethod
    def _call_local_llm(prompt: str) -> str:
        base_url = config.ROUTER_BASE_URL.rstrip("/")
        timeout = config.ROUTER_TIMEOUT_SECONDS

        chat_url = f"{base_url}{config.ROUTER_CHAT_ENDPOINT}"
        chat_payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict classifier. Output only one of: sql, text, hybrid.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": config.ROUTER_LLM_TEMPERATURE,
            "max_tokens": 8,
        }
        if config.ROUTER_MODEL:
            chat_payload["model"] = config.ROUTER_MODEL

        try:
            response = requests.post(chat_url, json=chat_payload, timeout=timeout)
            response.raise_for_status()
            payload = response.json()

            choices = payload.get("choices")
            if not choices:
                raise ValueError(f"Chat endpoint returned no 'choices': {payload}")

            first = choices[0]
            content = (first.get("message") or {}).get("content") or first.get("text")
            if not content:
                raise ValueError(f"Chat endpoint choice missing 'content'/'text': {first}")

            return str(content)

        except Exception:  # noqa: BLE001
            pass  # intentional fallback to completion endpoint

        completion_url = f"{base_url}{config.ROUTER_COMPLETION_ENDPOINT}"
        completion_payload = {
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": config.ROUTER_LLM_TEMPERATURE,
        }
        if config.ROUTER_MODEL:
            completion_payload["model"] = config.ROUTER_MODEL

        response = requests.post(completion_url, json=completion_payload, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        if "content" in payload:
            return str(payload["content"])

        choices = payload.get("choices") or []
        if choices and "text" in choices[0]:
            return str(choices[0]["text"])

        raise ValueError(f"Completion endpoint returned unrecognisable payload: {payload}")
