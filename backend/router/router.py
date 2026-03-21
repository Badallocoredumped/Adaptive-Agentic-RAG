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

    def route(self, query: str) -> str:
        """Return one of: 'sql', 'text', or 'hybrid'."""
        cleaned = query.lower()
        sql_hits = self._keyword_hits(cleaned, config.SQL_KEYWORDS)
        text_hits = self._keyword_hits(cleaned, config.TEXT_KEYWORDS)

        if sql_hits and text_hits:
            return "hybrid"
        if sql_hits:
            return "sql"
        if text_hits:
            return "text"
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
            if label in {"sql", "text", "hybrid"}:
                return label
        except Exception:  # noqa: BLE001
            pass

        return self.route(query)

    def route_with_zeroshot(self, query: str) -> str:
        """Classify query intent using LangChain + OpenAI-compatible chat model."""
        chat_openai_cls = self._resolve_chat_openai_class()
        if chat_openai_cls is None:
            return self.route(query)

        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a zero-shot intent classifier for retrieval routing. "
                        "Classify each query into exactly one label: sql, text, or hybrid. "
                        "Return only the label with no explanation. "
                        "Use sql for database/aggregation intents, text for document retrieval/summarization intents, "
                        "and hybrid when both are clearly needed.",
                    ),
                    ("human", "User query: {query}"),
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
            response = chain.invoke({"query": query})
            content = getattr(response, "content", "")
            label = self._normalize_label(str(content))
            if label in {"sql", "text", "hybrid"}:
                return label
        except Exception:  # noqa: BLE001
            pass

        return self.route(query)

    def decompose(self, query: str) -> list[SubTask]:
        """Fallback decomposition: keep one sub-task using rule routing."""
        route = self.route(query)
        return self._expand_hybrid_subtasks([SubTask(sub_query=query, route=route)])

    def decompose_with_zeroshot(self, query: str) -> list[SubTask]:
        """Decompose a query into multiple routed sub-tasks using LangChain chat model."""
        chat_openai_cls = self._resolve_chat_openai_class()
        if chat_openai_cls is None:
            return self.decompose(query)

        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a decomposition and routing agent for a hybrid SQL + RAG system. "
                        "Break the user query into meaningful sub-queries and assign one route for each. "
                        "Allowed routes are exactly: sql, text, hybrid. "
                        "Return strict JSON only in this shape: "
                        "[{\"sub_query\": \"...\", \"route\": \"sql|text|hybrid\"}]. "
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
            content = str(getattr(response, "content", ""))

            sub_tasks = self._parse_decomposition_output(content)
            if sub_tasks:
                return self._expand_hybrid_subtasks(sub_tasks)
        except Exception:  # noqa: BLE001
            pass

        return self.decompose(query)

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
        if not isinstance(payload, list):
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
