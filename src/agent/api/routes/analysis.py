"""LLM analysis endpoints per api-contracts/agent-api.md (Phase 4)."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Response, status

from agent.api.dependencies import get_analysis_service
from agent.api.envelope import Envelope, envelope
from agent.api.errors import NotFoundError, UnprocessableError
from agent.api.schemas import AnalysisData, AnalysisRequest
from agent.core.analysis import AnalysisService
from agent.db.repository import AnalysisRecord

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


@router.post("/analysis", response_model=Envelope[AnalysisData], status_code=status.HTTP_201_CREATED)
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
