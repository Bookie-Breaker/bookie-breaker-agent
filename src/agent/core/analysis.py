"""LLM analysis generation per api-contracts/agent-api.md (POST/GET /analysis).

Context gathering is best-effort: missing upstream sources are noted in
input_summary rather than failing the request. Results are persisted to
agent.analyses and cached in Redis (agent:analysis:{type}:{scope}) so
repeated requests for the same subject reuse the stored record instead of
spending tokens; free-form questions bypass the cache.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

import redis.asyncio as aioredis

from agent.api.errors import ApiError, DependencyError, NotFoundError, UnprocessableError
from agent.clients.emulator import EmulatorClient
from agent.clients.lines import LinesClient
from agent.clients.prediction import PredictionClient
from agent.clients.reconcile import GameReconciler
from agent.clients.statistics import Game, StatisticsClient
from agent.db.repository import AnalysisRecord, AnalysisRepository, EdgeRecord, EdgeRepository
from agent.llm.base import LLMError, LLMProvider, LLMResult, LLMStreamChunk
from agent.llm.prompts import (
    RenderedPrompt,
    build_edge_breakdown,
    build_game_preview,
    build_performance_review,
)

logger = logging.getLogger(__name__)

CACHEABLE_TYPES = ("GAME_PREVIEW", "EDGE_BREAKDOWN")


@dataclass(frozen=True)
class _Prepared:
    """Validated request state shared by the JSON and streaming paths."""

    analysis_type: str
    game_id: uuid.UUID | None
    edge_id: uuid.UUID | None
    question: str | None
    edge: EdgeRecord | None
    cache_key: str | None
    cached: AnalysisRecord | None


@dataclass
class AnalysisStream:
    """A prepared stream: either a cached record replay or a live LLM stream.

    prepare_stream() has already awaited the first provider chunk, so
    constructing this object means pre-stream failures (degraded LLM,
    validation) have been raised as JSON-mappable errors.
    """

    prepared: _Prepared
    cached_record: AnalysisRecord | None = None
    rendered: RenderedPrompt | None = None
    iterator: AsyncIterator[LLMStreamChunk] | None = None
    first_chunk: LLMStreamChunk | None = None


@dataclass(frozen=True)
class StreamEvent:
    """What run_stream yields: text deltas, then one terminal done event."""

    type: Literal["chunk", "done"]
    text: str = ""
    record: AnalysisRecord | None = None
    cached: bool = False


class AnalysisService:
    def __init__(
        self,
        llm: LLMProvider,
        statistics: StatisticsClient,
        lines: LinesClient,
        prediction: PredictionClient,
        emulator: EmulatorClient,
        edge_repo: EdgeRepository,
        analysis_repo: AnalysisRepository,
        reconciler: GameReconciler,
        redis_client: "aioredis.Redis",
        cache_ttl_seconds: int,
    ) -> None:
        self._llm = llm
        self._statistics = statistics
        self._lines = lines
        self._prediction = prediction
        self._emulator = emulator
        self._edge_repo = edge_repo
        self._analysis_repo = analysis_repo
        self._reconciler = reconciler
        self._redis = redis_client
        self._cache_ttl = cache_ttl_seconds

    async def create(
        self,
        analysis_type: str,
        game_id: uuid.UUID | None,
        edge_id: uuid.UUID | None,
        question: str | None,
    ) -> tuple[AnalysisRecord, bool]:
        """Generate (or reuse) an analysis. Returns (record, from_cache)."""
        prepared = await self._prepare(analysis_type, game_id, edge_id, question)
        if prepared.cached is not None:
            return prepared.cached, True

        rendered = await self._render(prepared.analysis_type, prepared.game_id, prepared.edge, prepared.question)
        try:
            result = await self._llm.complete(system=rendered.system, prompt=rendered.prompt, tier=rendered.tier)
        except LLMError as exc:
            raise DependencyError(f"LLM analysis failed: {exc}") from exc

        record = await self._persist(prepared, rendered, result)
        return record, False

    async def prepare_stream(
        self,
        analysis_type: str,
        game_id: uuid.UUID | None,
        edge_id: uuid.UUID | None,
        question: str | None,
    ) -> AnalysisStream:
        """Phase A of streaming: validate, check the cache, and await the
        first provider chunk so degraded-LLM/validation failures surface as
        ordinary JSON-mappable errors before any SSE bytes are sent."""
        prepared = await self._prepare(analysis_type, game_id, edge_id, question)
        if prepared.cached is not None:
            return AnalysisStream(prepared=prepared, cached_record=prepared.cached)

        rendered = await self._render(prepared.analysis_type, prepared.game_id, prepared.edge, prepared.question)
        iterator = self._llm.stream(system=rendered.system, prompt=rendered.prompt, tier=rendered.tier)
        try:
            first_chunk = await anext(iterator)
        except LLMError as exc:
            raise DependencyError(f"LLM analysis failed: {exc}") from exc
        except StopAsyncIteration as exc:
            raise DependencyError("LLM analysis failed: provider stream was empty") from exc
        return AnalysisStream(prepared=prepared, rendered=rendered, iterator=iterator, first_chunk=first_chunk)

    async def run_stream(self, stream: AnalysisStream) -> AsyncIterator[StreamEvent]:
        """Phase B of streaming: yield deltas, then persist and yield done.

        Cached analyses replay as a single chunk. Mid-stream LLMErrors
        propagate to the caller with nothing persisted or cached, as does
        cancellation on client disconnect.
        """
        if stream.cached_record is not None:
            yield StreamEvent(type="chunk", text=stream.cached_record.content)
            yield StreamEvent(type="done", record=stream.cached_record, cached=True)
            return

        assert stream.iterator is not None and stream.first_chunk is not None and stream.rendered is not None
        final: LLMResult | None = None
        chunk = stream.first_chunk
        while True:
            if chunk.text:
                yield StreamEvent(type="chunk", text=chunk.text)
            if chunk.final is not None:
                final = chunk.final
                break
            try:
                chunk = await anext(stream.iterator)
            except StopAsyncIteration:
                break
        if final is None:
            raise LLMError("provider stream ended without a final result")

        record = await self._persist(stream.prepared, stream.rendered, final)
        yield StreamEvent(type="done", record=record, cached=False)

    async def get(self, analysis_id: uuid.UUID) -> AnalysisRecord | None:
        return await self._analysis_repo.get(analysis_id)

    async def _prepare(
        self,
        analysis_type: str,
        game_id: uuid.UUID | None,
        edge_id: uuid.UUID | None,
        question: str | None,
    ) -> _Prepared:
        edge: EdgeRecord | None = None
        if analysis_type == "EDGE_BREAKDOWN":
            if edge_id is None:
                raise UnprocessableError("edge_id is required for EDGE_BREAKDOWN")
            edge = await self._edge_repo.get(edge_id)
            if edge is None:
                raise NotFoundError(f"Edge {edge_id} not found")
            game_id = edge.game_id
        elif analysis_type == "GAME_PREVIEW":
            if game_id is None:
                raise UnprocessableError("game_id is required for GAME_PREVIEW")
        elif analysis_type != "PERFORMANCE_REVIEW":
            raise UnprocessableError(f"Unknown analysis_type: {analysis_type}")

        cache_key = self._cache_key(analysis_type, game_id, edge_id) if question is None else None
        cached = await self._cached_record(cache_key) if cache_key is not None else None
        return _Prepared(
            analysis_type=analysis_type,
            game_id=game_id,
            edge_id=edge_id,
            question=question,
            edge=edge,
            cache_key=cache_key,
            cached=cached,
        )

    async def _persist(self, prepared: _Prepared, rendered: RenderedPrompt, result: LLMResult) -> AnalysisRecord:
        record = await self._analysis_repo.insert(
            {
                "analysis_type": prepared.analysis_type,
                "game_id": prepared.game_id,
                "edge_id": prepared.edge_id if prepared.analysis_type == "EDGE_BREAKDOWN" else None,
                "title": rendered.title,
                "content": result.text,
                "question": prepared.question,
                "model_used": result.model,
                "provider": result.provider,
                "input_summary": rendered.input_summary,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
        )
        if prepared.cache_key is not None:
            await self._set_cached(prepared.cache_key, record.id)
        return record

    # ------------------------------------------------------------------
    # Context gathering

    async def _render(
        self,
        analysis_type: str,
        game_id: uuid.UUID | None,
        edge: EdgeRecord | None,
        question: str | None,
    ) -> RenderedPrompt:
        if analysis_type == "PERFORMANCE_REVIEW":
            performance = await self._maybe(self._performance_context())
            recent = await self._maybe(self._recent_edges_context())
            return build_performance_review(performance, recent, question)

        game = await self._maybe(self._game_context(game_id)) if game_id is not None else None
        game_payload = game.model_dump() if isinstance(game, Game) else None

        if analysis_type == "EDGE_BREAKDOWN":
            assert edge is not None  # validated in create()
            lines = await self._maybe(self._edge_lines_context(edge))
            prediction = await self._maybe(self._prediction_context(str(edge.game_id)))
            return build_edge_breakdown(edge, game_payload, None, lines, prediction, question)

        lines = await self._maybe(self._game_lines_context(game)) if isinstance(game, Game) else None
        predictions = await self._maybe(self._prediction_context(str(game_id)))
        return build_game_preview(game_payload, None, predictions, lines, question)

    async def _game_context(self, game_id: uuid.UUID) -> Game:
        return await self._statistics.get_game(str(game_id))

    async def _edge_lines_context(self, edge: EdgeRecord) -> list[dict[str, Any]]:
        snapshots = await self._lines.game_lines(edge.game_external_id, market_type=edge.market_type)
        return [snapshot.model_dump() for snapshot in snapshots]

    async def _game_lines_context(self, game: Game) -> list[dict[str, Any]] | None:
        external_id = await self._reconciler.resolve(game)
        if external_id is None:
            return None
        snapshots = await self._lines.game_lines(external_id)
        return [snapshot.model_dump() for snapshot in snapshots]

    async def _prediction_context(self, game_id: str) -> list[dict[str, Any]]:
        items = await self._prediction.latest_for_game(game_id)
        return [item.model_dump() for item in items]

    async def _performance_context(self) -> dict[str, Any]:
        performance = await self._emulator.performance()
        return performance.model_dump()

    async def _recent_edges_context(self) -> list[dict[str, Any]]:
        records = await self._edge_repo.active_edges()
        return [
            {
                "selection": record.selection,
                "league": record.league,
                "market_type": record.market_type,
                "edge_percentage": record.edge_percentage,
                "sportsbook": record.sportsbook_key,
                "has_paper_bet": record.paper_bet_id is not None,
            }
            for record in records[:25]
        ]

    @staticmethod
    async def _maybe(coro: Any) -> Any:
        """Await a context fetch, tolerating upstream failures (returns None)."""
        try:
            return await coro
        except ApiError as exc:
            logger.info("analysis context source unavailable: %s", exc.message)
            return None

    # ------------------------------------------------------------------
    # Redis cache (value = analysis id; record itself lives in Postgres)

    @staticmethod
    def _cache_key(analysis_type: str, game_id: uuid.UUID | None, edge_id: uuid.UUID | None) -> str | None:
        if analysis_type == "EDGE_BREAKDOWN" and edge_id is not None:
            return f"agent:analysis:EDGE_BREAKDOWN:{edge_id}"
        if analysis_type == "GAME_PREVIEW" and game_id is not None:
            return f"agent:analysis:GAME_PREVIEW:{game_id}"
        return None

    async def _cached_record(self, cache_key: str) -> AnalysisRecord | None:
        try:
            value = await self._redis.get(cache_key)
        except Exception:  # noqa: BLE001 - cache is best-effort
            return None
        if not value:
            return None
        try:
            return await self._analysis_repo.get(uuid.UUID(str(value)))
        except ValueError:
            return None

    async def _set_cached(self, cache_key: str, analysis_id: uuid.UUID) -> None:
        try:
            await self._redis.set(cache_key, str(analysis_id), ex=self._cache_ttl)
        except Exception:  # noqa: BLE001 - cache is best-effort
            logger.warning("failed to cache analysis %s", analysis_id, exc_info=True)
