"""Health endpoint aggregating all downstream dependency status.

Always returns 200 -- even when degraded -- so container healthchecks
measure the agent itself rather than its dependencies. The LLM dependency
key follows the configured provider: anthropic_api or ollama.
"""

import asyncio
import time
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Request

from agent import __version__
from agent.api.dependencies import (
    get_edge_repo,
    get_emulator_client,
    get_event_subscriber,
    get_lines_client,
    get_llm_provider,
    get_prediction_client,
    get_run_repo,
    get_schedule_repo,
    get_simulation_client,
    get_statistics_client,
)
from agent.api.envelope import Envelope, envelope
from agent.api.schemas import HealthData, HealthPipeline
from agent.clients.emulator import EmulatorClient
from agent.clients.lines import LinesClient
from agent.clients.prediction import PredictionClient
from agent.clients.simulation import SimulationClient
from agent.clients.statistics import StatisticsClient
from agent.db.repository import EdgeRepository, PipelineRunRepository, ScheduleRepository
from agent.events.subscriber import EventSubscriber
from agent.llm.base import LLMProvider

router = APIRouter(tags=["health"])

_STARTED_MONOTONIC = time.monotonic()


async def _redis_ok(redis_client: "aioredis.Redis") -> bool:
    try:
        return bool(await redis_client.ping())
    except Exception:  # noqa: BLE001 - any redis failure means unhealthy
        return False


def _label(ok: bool) -> str:
    return "healthy" if ok else "unhealthy"


@router.get("/health", response_model=Envelope[HealthData])
async def get_health(
    request: Request,
    statistics: Annotated[StatisticsClient, Depends(get_statistics_client)],
    lines: Annotated[LinesClient, Depends(get_lines_client)],
    simulation: Annotated[SimulationClient, Depends(get_simulation_client)],
    prediction: Annotated[PredictionClient, Depends(get_prediction_client)],
    emulator: Annotated[EmulatorClient, Depends(get_emulator_client)],
    edge_repo: Annotated[EdgeRepository, Depends(get_edge_repo)],
    run_repo: Annotated[PipelineRunRepository, Depends(get_run_repo)],
    subscriber: Annotated[EventSubscriber, Depends(get_event_subscriber)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
    schedule_repo: Annotated[ScheduleRepository, Depends(get_schedule_repo)],
) -> Envelope[HealthData]:
    """Agent liveness plus per-dependency status (200 even when degraded)."""
    lines_ok, stats_ok, sim_ok, predict_ok, emulator_ok, postgres_ok, redis_ok, llm_ok = await asyncio.gather(
        lines.health(),
        statistics.health(),
        simulation.health(),
        prediction.health(),
        emulator.health(),
        edge_repo.is_healthy(),
        _redis_ok(request.app.state.redis),
        llm.is_healthy(),
    )
    subscriber_ok = subscriber.is_healthy()
    llm_key = "anthropic_api" if llm.provider_name == "anthropic" else llm.provider_name
    dependencies = {
        "lines_service": _label(lines_ok),
        "statistics_service": _label(stats_ok),
        "simulation_engine": _label(sim_ok),
        "prediction_engine": _label(predict_ok),
        "bookie_emulator": _label(emulator_ok),
        "postgres": _label(postgres_ok),
        "redis": _label(redis_ok),
        "event_subscriber": _label(subscriber_ok),
        llm_key: _label(llm_ok),
    }
    healthy = all((lines_ok, stats_ok, sim_ok, predict_ok, emulator_ok, postgres_ok, redis_ok, subscriber_ok, llm_ok))

    last_run = await run_repo.last_run() if postgres_ok else None
    next_scheduled: str | None = None
    if postgres_ok:
        try:
            value = await schedule_repo.min_next_run()
            next_scheduled = value.isoformat().replace("+00:00", "Z") if value else None
        except Exception:  # noqa: BLE001 - health must not fail on schedule reads
            next_scheduled = None
    pipeline = HealthPipeline(
        last_run_status=last_run.status.lower() if last_run else None,
        last_run_at=(
            last_run.finished_at.isoformat().replace("+00:00", "Z") if last_run and last_run.finished_at else None
        ),
        next_scheduled_run=next_scheduled,
    )
    return envelope(
        HealthData(
            status="healthy" if healthy else "degraded",
            version=__version__,
            uptime_seconds=int(time.monotonic() - _STARTED_MONOTONIC),
            dependencies=dependencies,
            pipeline=pipeline,
        )
    )
