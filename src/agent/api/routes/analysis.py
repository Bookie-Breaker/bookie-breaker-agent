"""LLM analysis endpoints per api-contracts/agent-api.md (Phase 4 + 5).

The streaming variant emits SSE events: `chunk` ({"text": delta})
repeated, then `done` carrying the persisted AnalysisData envelope with
meta.cached, or `error` on a mid-stream provider failure. Pre-stream
failures (validation, unknown ids, degraded LLM) return the standard
JSON error envelope — the response never switches to SSE.
"""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Response, status
from fastapi.responses import StreamingResponse

from agent.api.dependencies import get_analysis_service
from agent.api.envelope import Envelope, envelope, make_meta
from agent.api.errors import NotFoundError, UnprocessableError
from agent.api.schemas import AnalysisData, AnalysisRequest
from agent.core.analysis import AnalysisService, AnalysisStream
from agent.db.repository import AnalysisRecord
from agent.llm.base import LLMError

router = APIRouter(tags=["analysis"])

AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]


def _to_data(record: AnalysisRecord) -> AnalysisData:
    return AnalysisData(
        id=str(record.id),
        analysis_type=record.analysis_type,
        game_id=str(record.game_id) if record.game_id else None,
        edge_id=str(record.edge_id) if record.edge_id else None,
        title=record.title,
        content=record.content,
        model_used=record.model_used,
        input_summary=record.input_summary,
        created_at=record.created_at.isoformat().replace("+00:00", "Z"),
    )


def _parse_uuid(value: str | None, field: str) -> uuid.UUID | None:
    if value is None:
        return None
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise UnprocessableError(f"{field} is not a valid UUID") from exc


@router.post(
    "/analysis",
    response_model=Envelope[AnalysisData],
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_200_OK: {"model": Envelope[AnalysisData], "description": "Reused a cached analysis."}},
)
async def create_analysis(
    body: AnalysisRequest,
    service: AnalysisServiceDep,
    response: Response,
) -> Envelope[AnalysisData]:
    """Generate an LLM analysis for a game, edge, or performance period.

    Cached analyses for the same subject are reused (200) instead of
    re-generated; free-form questions always generate fresh output (201).
    """
    record, from_cache = await service.create(
        analysis_type=body.analysis_type,
        game_id=_parse_uuid(body.game_id, "game_id"),
        edge_id=_parse_uuid(body.edge_id, "edge_id"),
        question=body.question,
    )
    if from_cache:
        response.status_code = status.HTTP_200_OK
    return envelope(_to_data(record))


def _sse(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def _sse_events(service: AnalysisService, stream: AnalysisStream) -> AsyncIterator[str]:
    try:
        async for item in service.run_stream(stream):
            if item.type == "chunk":
                yield _sse("chunk", {"text": item.text})
            else:
                assert item.record is not None
                meta = make_meta().model_dump(mode="json") | {"cached": item.cached}
                yield _sse("done", {"data": _to_data(item.record).model_dump(mode="json"), "meta": meta})
    except LLMError as exc:
        # Nothing was persisted or cached; the client should retry.
        yield _sse("error", {"code": "DEPENDENCY_ERROR", "message": f"LLM analysis failed: {exc}"})


@router.post(
    "/analysis/stream",
    responses={
        status.HTTP_200_OK: {
            "description": 'SSE stream: `chunk` events ({"text": delta}) followed by a terminal '
            "`done` event carrying the persisted analysis envelope (meta.cached indicates a cache "
            "replay), or `error` on a mid-stream LLM failure.",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
async def create_analysis_stream(body: AnalysisRequest, service: AnalysisServiceDep) -> StreamingResponse:
    """Generate an LLM analysis, streaming text deltas over SSE.

    Same request body and semantics as POST /analysis; cached analyses
    replay as a single chunk. Validation, unknown-id, and degraded-LLM
    failures return the standard JSON error envelope before any SSE bytes.
    """
    stream = await service.prepare_stream(
        analysis_type=body.analysis_type,
        game_id=_parse_uuid(body.game_id, "game_id"),
        edge_id=_parse_uuid(body.edge_id, "edge_id"),
        question=body.question,
    )
    return StreamingResponse(
        _sse_events(service, stream),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/analysis/{analysis_id}", response_model=Envelope[AnalysisData])
async def get_analysis(
    analysis_id: Annotated[uuid.UUID, Path(description="The analysis identifier.")],
    service: AnalysisServiceDep,
) -> Envelope[AnalysisData]:
    """Fetch a previously generated analysis."""
    record = await service.get(analysis_id)
    if record is None:
        raise NotFoundError(f"Analysis {analysis_id} not found")
    return envelope(_to_data(record))
