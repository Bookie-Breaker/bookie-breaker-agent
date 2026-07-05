"""AlertService: descriptions, LLM cap, persistence, and fire-and-forget."""

import json
from typing import Any

from agent.core.alerts import AlertService
from agent.llm.base import LLMError, LLMResult
from agent.llm.prompts import fallback_alert_description
from tests.unit.factories import FakeRedis, make_edge_record


class FakeLLM:
    provider_name = "anthropic"

    def __init__(self, text: str = "Sharp edge alert.", fail: bool = False) -> None:
        self._text = text
        self._fail = fail
        self.calls = 0

    def model_for(self, tier: str) -> str:
        return "claude-haiku-4-5"

    async def complete(self, *, system: str, prompt: str, tier: str = "quality", max_tokens: int | None = None):
        self.calls += 1
        if self._fail:
            raise LLMError("down")
        return LLMResult(
            text=self._text, model="claude-haiku-4-5", provider="anthropic", input_tokens=1, output_tokens=2
        )

    async def is_healthy(self) -> bool:
        return not self._fail

    async def aclose(self) -> None:
        return None


class FakeAlertRepo:
    def __init__(self, fail: bool = False) -> None:
        self.inserted: list[dict[str, Any]] = []
        self._fail = fail

    async def insert(self, values: dict[str, Any]) -> dict[str, Any]:
        if self._fail:
            raise RuntimeError("db down")
        self.inserted.append(values)
        return values


def make_service(
    llm: FakeLLM | None,
    repo: FakeAlertRepo | None = None,
    enabled: bool = True,
    max_per_run: int = 10,
) -> tuple[AlertService, FakeRedis, FakeAlertRepo]:
    redis = FakeRedis()
    repo = repo or FakeAlertRepo()
    service = AlertService(redis, repo, llm, llm_descriptions_enabled=enabled, llm_max_per_run=max_per_run)  # type: ignore[arg-type]
    return service, redis, repo


class TestDispatchAll:
    async def test_publishes_with_llm_description_and_persists(self) -> None:
        llm = FakeLLM(text="Model likes the Lakers tonight.")
        service, redis, repo = make_service(llm)
        edge = make_edge_record()

        await service.dispatch_all([edge])

        assert len(redis.published) == 1
        channel, raw = redis.published[0]
        assert channel == "events:edge.detected"
        payload = json.loads(raw)
        assert payload["description"] == "Model likes the Lakers tonight."
        assert payload["priority"] == "HIGH"  # confidence 0.78
        assert len(repo.inserted) == 1
        row = repo.inserted[0]
        assert row["edge_id"] == edge.id
        assert row["priority"] == "HIGH"
        assert row["message"] == "Model likes the Lakers tonight."
        assert row["payload"]["edge_id"] == str(edge.id)

    async def test_llm_failure_falls_back_to_template(self) -> None:
        service, redis, repo = make_service(FakeLLM(fail=True))
        edge = make_edge_record()

        await service.dispatch_all([edge])

        payload = json.loads(redis.published[0][1])
        assert payload["description"] == fallback_alert_description(edge)
        assert repo.inserted[0]["message"] == fallback_alert_description(edge)

    async def test_per_run_llm_cap(self) -> None:
        llm = FakeLLM()
        service, redis, _ = make_service(llm, max_per_run=2)
        edges = [make_edge_record() for _ in range(5)]

        await service.dispatch_all(edges)

        assert llm.calls == 2
        descriptions = [json.loads(raw)["description"] for _, raw in redis.published]
        assert descriptions[:2] == ["Sharp edge alert.", "Sharp edge alert."]
        assert descriptions[2] == fallback_alert_description(edges[2])

    async def test_descriptions_disabled_uses_template(self) -> None:
        llm = FakeLLM()
        service, redis, _ = make_service(llm, enabled=False)
        edge = make_edge_record()

        await service.dispatch_all([edge])

        assert llm.calls == 0
        assert json.loads(redis.published[0][1])["description"] == fallback_alert_description(edge)

    async def test_persistence_failure_does_not_block_publish(self) -> None:
        service, redis, _ = make_service(FakeLLM(), repo=FakeAlertRepo(fail=True))

        await service.dispatch_all([make_edge_record()])

        assert len(redis.published) == 1
