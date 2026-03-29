from __future__ import annotations

import json
from typing import Any

import httpx

from .config import ModelConfig


class ProviderUnavailable(RuntimeError):
    pass


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    raise ProviderUnavailable("Model response content is not text-like.")


def _extract_json_blob(text: str) -> Any:
    candidate = text.strip()
    if candidate.startswith("```"):
        chunks = [chunk.strip() for chunk in candidate.split("```") if chunk.strip()]
        for chunk in chunks:
            if chunk.startswith("json"):
                candidate = chunk[4:].strip()
                break
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ProviderUnavailable("Model did not return parseable JSON.") from None
        return json.loads(candidate[start : end + 1])


class OpenAICompatibleProvider:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @property
    def configured(self) -> bool:
        return self.config.configured

    def complete_json(self, system_prompt: str, user_prompt: str) -> Any:
        if not self.configured:
            raise ProviderUnavailable("No model provider configured.")

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=self.config.timeout_seconds, trust_env=False) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        message = data["choices"][0]["message"]["content"]
        return _extract_json_blob(_content_to_text(message))
