"""Simple query router to choose SQL, text, or hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import re

from langchain_core.prompts import ChatPromptTemplate
import requests

from backend import config


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
            choices = payload.get("choices") or []
            if choices:
                first = choices[0]
                message = first.get("message") or {}
                content = message.get("content")
                if content:
                    return str(content)
                text = first.get("text")
                if text:
                    return str(text)
        except Exception:  # noqa: BLE001
            pass

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

        return str(payload)
