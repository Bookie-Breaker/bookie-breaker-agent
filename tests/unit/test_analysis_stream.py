"""AnalysisService streaming semantics with a hand-rolled fake provider.

Persistence only on completion, nothing persisted on mid-stream failure,
cached analyses replay without touching the provider.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest

from agent.api.errors import ApiError, DependencyError, UnprocessableError
from agent.core.analysis import AnalysisService, StreamEvent
from agent.db.repository import AnalysisRecord
from agent.llm.base import LLMError, LLMResult, LLMStreamChunk, ModelTier


class FakeProvider:
    provider_name = "fake"

    def __init__(self, deltas: list[str] | None = None, fail_after: int | None = None) -> None:
        self.deltas = deltas if deltas is not None else ["Hello ", "world"]
        self.fail_after = fail_after  # raise LLMError after yielding this many deltas
        self.stream_calls = 0

    def model_for(self, tier: ModelTier) -> str:
        return "fake-model"

    async def complete(self, *, system: str, prompt: str, tier: ModelTier = "quality", max_tokens=None) -> LLMResult:
        raise AssertionError("streaming tests must not call complete()")

    async def stream(
        self, *, system: str, prompt: str, tier: ModelTier = "quality", max_tokens=None
    ) -> AsyncIterator[LLMStreamChunk]:
        self.stream_calls += 1
        yielded = 0
        for delta in self.deltas:
            if self.fail_after is not None and yielded >= self.fail_after:
                raise LLMError("provider blew up mid-stream")
            yield LLMStreamChunk(text=delta)
            yielded += 1
        if self.fail_after is not None and yielded >= self.fail_after:
            raise LLMError("provider blew up mid-stream")
        yield LLMStreamChunk(
            text="",
            final=LLMResult(
                text="".join(self.deltas), model="fake-model", provider="fake", input_tokens=11, output_tokens=22
            ),
        )

    async def is_healthy(self) -> bool:
        return True

    async def aclose(self) -> None:  # pragma: no cover - unused
        pass


class DegradedProvider(FakeProvider):
    async def stream(self, *, system, prompt, tier="quality", max_tokens=None) -> AsyncIterator[LLMStreamChunk]:
        self.stream_calls += 1
        raise LLMError("no provider configured")
        yield  # pragma: no cover - makes this an async generator


class DownstreamDown:
    """Every context client call fails; _maybe() tolerates ApiError."""

    def __getattr__(self, name: str) -> Any:
        async def _fail(*args: Any, **kwargs: Any) -> Any:
            raise ApiError("downstream unavailable")

        return _fail


class FakeAnalysisRepo:
    def __init__(self) -> None:
        self.records: dict[uuid.UUID, AnalysisRecord] = {}
        self.insert_calls = 0

    async def insert(self, values: dict[str, Any]) -> AnalysisRecord:
        self.insert_calls += 1
        record = AnalysisRecord(
            id=uuid.uuid4(),
            analysis_type=values["analysis_type"],
            game_id=values["game_id"],
            edge_id=values["edge_id"],
            title=values["title"],
            content=values["content"],
            question=values["question"],
            model_used=values["model_used"],
            provider=values["provider"],
            input_summary=values["input_summary"],
            input_tokens=values["input_tokens"],
            output_tokens=values["output_tokens"],
            created_at=datetime.now(tz=UTC),
        )
        self.records[record.id] = record
        return record

    async def get(self, analysis_id: uuid.UUID) -> AnalysisRecord | None:
        return self.records.get(analysis_id)


class FakeEdgeRepo:
    async def get(self, edge_id: uuid.UUID) -> None:
        return None

    async def active_edges(self) -> list[Any]:
        return []


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value


def make_service(provider: FakeProvider) -> tuple[AnalysisService, FakeAnalysisRepo, FakeRedis]:
    analysis_repo = FakeAnalysisRepo()
    redis = FakeRedis()
    down = DownstreamDown()
    service = AnalysisService(
        llm=provider,  # type: ignore[arg-type]
        statistics=down,  # type: ignore[arg-type]
        lines=down,  # type: ignore[arg-type]
        prediction=down,  # type: ignore[arg-type]
        emulator=down,  # type: ignore[arg-type]
        edge_repo=FakeEdgeRepo(),  # type: ignore[arg-type]
        analysis_repo=analysis_repo,  # type: ignore[arg-type]
        reconciler=down,  # type: ignore[arg-type]
        redis_client=redis,  # type: ignore[arg-type]
        cache_ttl_seconds=60,
    )
    return service, analysis_repo, redis


async def run(service: AnalysisService, stream: Any) -> list[StreamEvent]:
    return [event async for event in service.run_stream(stream)]


class TestStreaming:
    async def test_deltas_then_done_persists_once_and_caches(self) -> None:
        provider = FakeProvider(deltas=["Hello ", "world"])
        service, repo, redis = make_service(provider)
        game_id = uuid.uuid4()

        stream = await service.prepare_stream("GAME_PREVIEW", game_id, None, None)
        events = await run(service, stream)

        assert [e.text for e in events if e.type == "chunk"] == ["Hello ", "world"]
        done = events[-1]
        assert done.type == "done"
        assert done.cached is False
        assert done.record is not None
        assert done.record.content == "Hello world"
        assert done.record.input_tokens == 11
        assert done.record.output_tokens == 22
        assert repo.insert_calls == 1
        assert redis.values[f"agent:analysis:GAME_PREVIEW:{game_id}"] == str(done.record.id)

    async def test_mid_stream_failure_persists_nothing(self) -> None:
        provider = FakeProvider(deltas=["partial "], fail_after=1)
        service, repo, redis = make_service(provider)

        stream = await service.prepare_stream("GAME_PREVIEW", uuid.uuid4(), None, None)
        with pytest.raises(LLMError, match="mid-stream"):
            await run(service, stream)
        assert repo.insert_calls == 0
        assert redis.values == {}

    async def test_cached_analysis_replays_without_provider_call(self) -> None:
        provider = FakeProvider()
        service, repo, redis = make_service(provider)
        game_id = uuid.uuid4()

        first = await service.prepare_stream("GAME_PREVIEW", game_id, None, None)
        first_events = await run(service, first)
        record = first_events[-1].record
        assert record is not None
        assert provider.stream_calls == 1

        second = await service.prepare_stream("GAME_PREVIEW", game_id, None, None)
        events = await run(service, second)
        assert provider.stream_calls == 1  # cache hit: provider untouched
        assert repo.insert_calls == 1
        assert [e.type for e in events] == ["chunk", "done"]
        assert events[0].text == record.content  # full content in one chunk
        assert events[-1].cached is True
        assert events[-1].record is not None and events[-1].record.id == record.id

    async def test_question_bypasses_cache(self) -> None:
        provider = FakeProvider()
        service, repo, _ = make_service(provider)
        game_id = uuid.uuid4()
        await run(service, await service.prepare_stream("GAME_PREVIEW", game_id, None, None))
        await run(service, await service.prepare_stream("GAME_PREVIEW", game_id, None, "why?"))
        assert provider.stream_calls == 2
        assert repo.insert_calls == 2

    async def test_degraded_provider_fails_before_streaming(self) -> None:
        service, repo, _ = make_service(DegradedProvider())
        with pytest.raises(DependencyError, match="LLM analysis failed"):
            await service.prepare_stream("GAME_PREVIEW", uuid.uuid4(), None, None)
        assert repo.insert_calls == 0

    async def test_validation_errors_raise_before_streaming(self) -> None:
        service, _, _ = make_service(FakeProvider())
        with pytest.raises(UnprocessableError, match="edge_id is required"):
            await service.prepare_stream("EDGE_BREAKDOWN", None, None, None)
        with pytest.raises(UnprocessableError, match="Unknown analysis_type"):
            await service.prepare_stream("HOT_TAKES", None, None, None)
