"""Anthropic provider over the official SDK (httpx-based, respx-mockable).

The SDK retries 429/5xx with backoff internally (max_retries), so no extra
retry wrapper is needed on this path.
"""

import logging
from collections.abc import AsyncIterator

import anthropic

from agent.llm.base import LLMError, LLMResult, LLMStreamChunk, ModelTier

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_CHEAP_MODEL = "claude-haiku-4-5"  # summaries + alert descriptions (ADR-011 tiering)


class AnthropicProvider:
    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        model: str,
        cheap_model: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> None:
        self._api_key = api_key
        self._models: dict[ModelTier, str] = {"quality": model, "cheap": cheap_model or DEFAULT_CHEAP_MODEL}
        self._max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or "unset",
            base_url=base_url or DEFAULT_BASE_URL,
            timeout=timeout_seconds,
            max_retries=2,
        )

    def model_for(self, tier: ModelTier) -> str:
        return self._models[tier]

    async def complete(
        self, *, system: str, prompt: str, tier: ModelTier = "quality", max_tokens: int | None = None
    ) -> LLMResult:
        if not self._api_key:
            raise LLMError("ANTHROPIC_API_KEY is not configured")
        model = self.model_for(tier)
        try:
            message = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens or self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc
        text = "".join(block.text for block in message.content if block.type == "text")
        if not text:
            raise LLMError("Anthropic returned no text content")
        return LLMResult(
            text=text,
            model=message.model,
            provider=self.provider_name,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )

    async def stream(
        self, *, system: str, prompt: str, tier: ModelTier = "quality", max_tokens: int | None = None
    ) -> AsyncIterator[LLMStreamChunk]:
        if not self._api_key:
            raise LLMError("ANTHROPIC_API_KEY is not configured")
        model = self.model_for(tier)
        try:
            async with self._client.messages.stream(
                model=model,
                max_tokens=max_tokens or self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for delta in stream.text_stream:
                    if delta:
                        yield LLMStreamChunk(text=delta)
                message = await stream.get_final_message()
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc
        text = "".join(block.text for block in message.content if block.type == "text")
        if not text:
            raise LLMError("Anthropic returned no text content")
        yield LLMStreamChunk(
            text="",
            final=LLMResult(
                text=text,
                model=message.model,
                provider=self.provider_name,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
        )

    async def is_healthy(self) -> bool:
        if not self._api_key:
            return False
        try:
            await self._client.models.list(limit=1)
        except anthropic.APIError:
            return False
        except Exception:  # noqa: BLE001 - health probes never raise
            return False
        return True

    async def aclose(self) -> None:
        await self._client.close()
