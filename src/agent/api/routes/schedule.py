"""Pipeline schedule configuration per api-contracts/agent-api.md (Phase 4)."""

from datetime import UTC, datetime
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from fastapi import APIRouter, Depends, Response, status

from agent.api.dependencies import get_schedule_repo, get_scheduler
from agent.api.envelope import Envelope, envelope
from agent.api.errors import UnprocessableError
from agent.api.schemas import ScheduleData, ScheduleListData, ScheduleRequest
from agent.core.scheduler import PipelineScheduler, next_fire
from agent.db.repository import ScheduleRecord, ScheduleRepository

router = APIRouter(tags=["schedule"])

ScheduleRepoDep = Annotated[ScheduleRepository, Depends(get_schedule_repo)]
SchedulerDep = Annotated[PipelineScheduler, Depends(get_scheduler)]

LEAGUES = ("NFL", "NBA", "MLB", "NCAA_FB", "NCAA_BB", "NCAA_BSB")


def _iso(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _to_data(record: ScheduleRecord) -> ScheduleData:
    return ScheduleData(
        id=str(record.id),
        league=record.league,
        cron_expression=record.cron_expression,
        timezone=record.timezone,
        description=record.description,
        enabled=record.enabled,
        last_run_at=_iso(record.last_run_at),
        next_run_at=_iso(record.next_run_at),
        simulation_config=record.simulation_config,
        auto_bet=record.auto_bet,
        min_edge_threshold=record.min_edge_threshold,
    )


@router.get("/schedule", response_model=Envelope[ScheduleListData])
async def list_schedules(repo: ScheduleRepoDep) -> Envelope[ScheduleListData]:
    """Current pipeline schedule configuration for all leagues."""
    records = await repo.list_all()
    return envelope(ScheduleListData(schedules=[_to_data(record) for record in records]))


@router.post(
    "/schedule",
    response_model=Envelope[ScheduleData],
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_200_OK: {"model": Envelope[ScheduleData], "description": "Updated the league's schedule."}},
)
async def upsert_schedule(
    body: ScheduleRequest,
    repo: ScheduleRepoDep,
    scheduler: SchedulerDep,
    response: Response,
) -> Envelope[ScheduleData]:
    """Create (201) or update (200) the league's pipeline schedule."""
    league = body.league.upper()
    if league not in LEAGUES:
        raise UnprocessableError(f"Unknown league: {body.league}")
    if not croniter.is_valid(body.cron_expression):
        raise UnprocessableError(f"Invalid cron expression: {body.cron_expression}")
    try:
        ZoneInfo(body.timezone)
    except ZoneInfoNotFoundError as exc:
        raise UnprocessableError(f"Unknown timezone: {body.timezone}") from exc

    now = datetime.now(tz=UTC)
    record, created = await repo.upsert_for_league(
        {
            "league": league,
            "cron_expression": body.cron_expression,
            "timezone": body.timezone,
            "description": body.description,
            "enabled": body.enabled,
            "simulation_config": body.simulation_config,
            "auto_bet": body.auto_bet,
            "min_edge_threshold": body.min_edge_threshold,
            "next_run_at": next_fire(body.cron_expression, body.timezone, now) if body.enabled else None,
        }
    )
    scheduler.wake()
    if not created:
        response.status_code = status.HTTP_200_OK
    return envelope(_to_data(record))
