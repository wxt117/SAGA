from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    model: str = "v4pro"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_key: str | None = None
    timeout_s: int = 120
    temperature: float = 0.2
    max_tokens: int = 4096
    extra_body: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "LLMConfig":
        llm = raw.get("llm", raw)
        return cls(
            provider=llm.get("provider", "deepseek"),
            model=llm.get("model", "v4pro"),
            base_url=llm.get("base_url", "https://api.deepseek.com"),
            api_key_env=llm.get("api_key_env", "DEEPSEEK_API_KEY"),
            api_key=llm.get("api_key"),
            timeout_s=int(llm.get("timeout_s", 120)),
            temperature=float(llm.get("temperature", 0.2)),
            max_tokens=int(llm.get("max_tokens", 4096)),
            extra_body=llm.get("extra_body") or {},
        )


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def chat(self, messages: list[dict[str, str]], response_format: dict[str, Any] | None = None) -> str:
        api_key = self.config.api_key or os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set {self.config.api_key_env} in the environment; credentials are not read from repository configs."
            )
        url = self.config.base_url.rstrip("/") + "/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.extra_body:
            payload.update(self.config.extra_body)
        if response_format:
            payload["response_format"] = response_format
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self.config.timeout_s,
        )
        if not response.ok:
            raise RuntimeError(
                f"LLM request failed with HTTP {response.status_code}: {response.text[:2000]}"
            )
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        if not content and message.get("reasoning_content"):
            reasoning_len = len(message.get("reasoning_content") or "")
            finish_reason = choice.get("finish_reason")
            raise RuntimeError(
                "LLM returned empty message.content but non-empty reasoning_content. "
                f"finish_reason={finish_reason}, reasoning_len={reasoning_len}. "
                "Try a smaller inspection sample, a non-reasoning/flash model, or a larger output budget."
            )
        return content
