"""Provider construction from settings — the entire ADR-011 config switch."""

import httpx

from agent.config import Settings
from agent.llm.anthropic_provider import AnthropicProvider
from agent.llm.base import LLMProvider
from agent.llm.ollama_provider import OllamaProvider


def create_llm_provider(settings: Settings, http_client: httpx.AsyncClient) -> LLMProvider:
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            cheap_model=settings.llm_model_cheap,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    if provider == "ollama":
        return OllamaProvider(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            cheap_model=settings.llm_model_cheap,
            http_client=http_client,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r} (expected 'anthropic' or 'ollama')")
