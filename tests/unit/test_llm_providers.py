"""Provider request/response shaping against respx-mocked endpoints."""

import json

import httpx
import pytest
import respx

from agent.config import Settings
from agent.llm.anthropic_provider import AnthropicProvider
from agent.llm.base import LLMError, LLMStreamChunk
from agent.llm.factory import create_llm_provider
from agent.llm.ollama_provider import OllamaProvider

ANTHROPIC_URL = "http://anthropic.test"
OLLAMA_URL = "http://ollama.test"


async def collect(stream) -> list[LLMStreamChunk]:
    return [chunk async for chunk in stream]


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


def anthropic_sse_body(deltas: list[str]) -> bytes:
    events: list[tuple[str, dict[str, object]]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_01",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-4-8",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
    ]
    events.extend(
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": d}})
        for d in deltas
    )
    events.extend(
        [
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 20},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    return "".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events).encode()


class TestAnthropicStreaming:
    @respx.mock
    async def test_stream_yields_deltas_then_final_usage(self) -> None:
        respx.post(f"{ANTHROPIC_URL}/v1/messages").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=anthropic_sse_body(["## Ana", "lysis"]),
            )
        )
        provider = anthropic_provider()
        chunks = await collect(provider.stream(system="sys", prompt="user prompt"))
        assert [c.text for c in chunks[:-1]] == ["## Ana", "lysis"]
        final = chunks[-1].final
        assert final is not None
        assert final.text == "## Analysis"
        assert final.provider == "anthropic"
        assert final.input_tokens == 10
        assert final.output_tokens == 20
        assert all(c.final is None for c in chunks[:-1])
        await provider.aclose()

    async def test_stream_missing_api_key_raises_before_first_chunk(self) -> None:
        provider = AnthropicProvider(
            api_key=None, base_url=ANTHROPIC_URL, model="m", cheap_model="", max_tokens=10, timeout_seconds=1.0
        )
        stream = provider.stream(system="s", prompt="p")
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            await anext(stream)
        await provider.aclose()


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


def ollama_ndjson(deltas: list[str], done_line: dict[str, object] | None = None) -> bytes:
    lines = [
        json.dumps({"model": "phi3:mini", "message": {"role": "assistant", "content": d}, "done": False})
        for d in deltas
    ]
    lines.append(
        json.dumps(
            done_line
            or {
                "model": "phi3:mini",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 800,
                "eval_count": 300,
            }
        )
    )
    return ("\n".join(lines) + "\n").encode()


class TestOllamaStreaming:
    @respx.mock
    async def test_stream_yields_deltas_then_final_counts(self) -> None:
        respx.post(f"{OLLAMA_URL}/api/chat").mock(
            return_value=httpx.Response(200, content=ollama_ndjson(["## Sum", "mary"]))
        )
        async with httpx.AsyncClient() as client:
            provider = OllamaProvider(OLLAMA_URL, "phi3:mini", "", client, timeout_seconds=5.0)
            chunks = await collect(provider.stream(system="sys", prompt="p"))
        assert [c.text for c in chunks[:-1]] == ["## Sum", "mary"]
        final = chunks[-1].final
        assert final is not None
        assert final.text == "## Summary"
        assert final.provider == "ollama"
        assert final.input_tokens == 800
        assert final.output_tokens == 300

    @respx.mock
    async def test_stream_http_error_raises_llm_error(self) -> None:
        respx.post(f"{OLLAMA_URL}/api/chat").mock(return_value=httpx.Response(500, text="boom"))
        async with httpx.AsyncClient() as client:
            provider = OllamaProvider(OLLAMA_URL, "m", "", client, timeout_seconds=5.0)
            with pytest.raises(LLMError, match="500"):
                await collect(provider.stream(system="s", prompt="p"))

    @respx.mock
    async def test_stream_malformed_line_raises_llm_error(self) -> None:
        respx.post(f"{OLLAMA_URL}/api/chat").mock(return_value=httpx.Response(200, content=b"not json\n"))
        async with httpx.AsyncClient() as client:
            provider = OllamaProvider(OLLAMA_URL, "m", "", client, timeout_seconds=5.0)
            with pytest.raises(LLMError, match="malformed"):
                await collect(provider.stream(system="s", prompt="p"))

    @respx.mock
    async def test_stream_without_text_raises_llm_error(self) -> None:
        respx.post(f"{OLLAMA_URL}/api/chat").mock(return_value=httpx.Response(200, content=ollama_ndjson([])))
        async with httpx.AsyncClient() as client:
            provider = OllamaProvider(OLLAMA_URL, "m", "", client, timeout_seconds=5.0)
            with pytest.raises(LLMError, match="no text content"):
                await collect(provider.stream(system="s", prompt="p"))


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
