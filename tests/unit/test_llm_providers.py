"""Provider request/response shaping against respx-mocked endpoints."""

import httpx
import pytest
import respx

from agent.config import Settings
from agent.llm.anthropic_provider import AnthropicProvider
from agent.llm.base import LLMError
from agent.llm.factory import create_llm_provider
from agent.llm.ollama_provider import OllamaProvider

ANTHROPIC_URL = "http://anthropic.test"
OLLAMA_URL = "http://ollama.test"


def anthropic_provider() -> AnthropicProvider:
    return AnthropicProvider(
        api_key="test-key",
        base_url=ANTHROPIC_URL,
        model="claude-opus-4-8",
        cheap_model="",
        max_tokens=1024,
        timeout_seconds=5.0,
    )


def message_payload(text: str) -> dict[str, object]:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


class TestAnthropicProvider:
    @respx.mock
    async def test_complete_extracts_text_and_usage(self) -> None:
        route = respx.post(f"{ANTHROPIC_URL}/v1/messages").mock(
            return_value=httpx.Response(200, json=message_payload("## Analysis"))
        )
        provider = anthropic_provider()
        result = await provider.complete(system="sys", prompt="user prompt")
        assert result.text == "## Analysis"
        assert result.provider == "anthropic"
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        body = route.calls[0].request.content
        assert b"user prompt" in body
        assert b'"claude-opus-4-8"' in body
        await provider.aclose()

    async def test_missing_api_key_raises_llm_error(self) -> None:
        provider = AnthropicProvider(
            api_key=None, base_url=ANTHROPIC_URL, model="m", cheap_model="", max_tokens=10, timeout_seconds=1.0
        )
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            await provider.complete(system="s", prompt="p")
        assert await provider.is_healthy() is False
        await provider.aclose()

    def test_cheap_tier_defaults_to_haiku(self) -> None:
        assert anthropic_provider().model_for("cheap") == "claude-haiku-4-5"


class TestOllamaProvider:
    @respx.mock
    async def test_complete_parses_chat_response(self) -> None:
        respx.post(f"{OLLAMA_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "llama3.1:8b",
                    "message": {"role": "assistant", "content": "## Summary"},
                    "done": True,
                    "prompt_eval_count": 800,
                    "eval_count": 300,
                },
            )
        )
        async with httpx.AsyncClient() as client:
            provider = OllamaProvider(OLLAMA_URL, "llama3.1:8b", "", client, timeout_seconds=5.0)
            result = await provider.complete(system="sys", prompt="p", tier="cheap")
        assert result.text == "## Summary"
        assert result.provider == "ollama"
        assert result.input_tokens == 800
        assert result.output_tokens == 300

    @respx.mock
    async def test_malformed_response_raises_llm_error(self) -> None:
        respx.post(f"{OLLAMA_URL}/api/chat").mock(return_value=httpx.Response(200, json={"done": True}))
        async with httpx.AsyncClient() as client:
            provider = OllamaProvider(OLLAMA_URL, "m", "", client, timeout_seconds=5.0)
            with pytest.raises(LLMError, match="malformed"):
                await provider.complete(system="s", prompt="p")

    @respx.mock
    async def test_http_error_raises_llm_error(self) -> None:
        respx.post(f"{OLLAMA_URL}/api/chat").mock(return_value=httpx.Response(500, text="boom"))
        async with httpx.AsyncClient() as client:
            provider = OllamaProvider(OLLAMA_URL, "m", "", client, timeout_seconds=5.0)
            with pytest.raises(LLMError, match="500"):
                await provider.complete(system="s", prompt="p")

    def test_cheap_tier_falls_back_to_quality_model(self) -> None:
        provider = OllamaProvider(OLLAMA_URL, "phi3:mini", "", httpx.AsyncClient(), timeout_seconds=1.0)
        assert provider.model_for("cheap") == "phi3:mini"


class TestFactory:
    async def test_config_only_switch(self) -> None:
        async with httpx.AsyncClient() as client:
            anthropic = create_llm_provider(Settings(llm_provider="anthropic", anthropic_api_key="k"), client)
            ollama = create_llm_provider(Settings(llm_provider="ollama"), client)
            assert anthropic.provider_name == "anthropic"
            assert ollama.provider_name == "ollama"
            await anthropic.aclose()
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            create_llm_provider(Settings(llm_provider="gpt"), httpx.AsyncClient())
