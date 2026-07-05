"""Ollama provider over the shared httpx client (POST /api/chat, ADR-011)."""

import logging
from typing import Any

import httpx

from agent.llm.base import LLMError, LLMResult, ModelTier

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://ollama:11434"


class OllamaProvider:
    provider_name = "ollama"

    def __init__(
        self,
        base_url: str | None,
        model: str,
        cheap_model: str,
        http_client: httpx.AsyncClient,
        timeout_seconds: float,
    ) -> None:
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        # A single local model serves both tiers unless one is configured.
        self._models: dict[ModelTier, str] = {"quality": model, "cheap": cheap_model or model}
        self._client = http_client
        self._timeout = timeout_seconds

    def model_for(self, tier: ModelTier) -> str:
        return self._models[tier]

    async def complete(
        self, *, system: str, prompt: str, tier: ModelTier = "quality", max_tokens: int | None = None
    ) -> LLMResult:
        model = self.model_for(tier)
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        if max_tokens is not None:
            body["options"] = {"num_predict": max_tokens}
        try:
            response = await self._client.post(f"{self._base_url}/api/chat", json=body, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama is unavailable: {exc}") from exc
        if response.status_code != 200:
            raise LLMError(f"Ollama returned {response.status_code}: {response.text[:200]}")
        try:
            payload = response.json()
            text = str(payload["message"]["content"])
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("Ollama returned a malformed chat response") from exc
        if not text:
            raise LLMError("Ollama returned no text content")
        return LLMResult(
            text=text,
            model=str(payload.get("model", model)),
            provider=self.provider_name,
            input_tokens=_int_or_none(payload.get("prompt_eval_count")),
            output_tokens=_int_or_none(payload.get("eval_count")),
        )

    async def is_healthy(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/tags", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        """The httpx client is shared and owned by the app lifespan."""


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None
