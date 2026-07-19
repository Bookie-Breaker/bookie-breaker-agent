"""FastAPI application entry point.

Startup is crash-loop-safe: the engine, Redis client, and HTTP client are
constructed lazily (no eager connections), and the event subscriber retries
with capped backoff instead of failing startup, so the container keeps
serving /health while dependencies come up.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI

from agent import __version__
from agent.api.envelope import RequestIDMiddleware
from agent.api.errors import register_error_handlers
from agent.api.routes import alerts, analysis, dashboard, edges, health, parlays, pipeline, schedule, slate
from agent.clients.emulator import EmulatorClient
from agent.clients.lines import LinesClient
from agent.clients.prediction import PredictionClient
from agent.clients.reconcile import GameReconciler
from agent.clients.simulation import SimulationClient
from agent.clients.statistics import StatisticsClient
from agent.config import Settings, get_settings
from agent.core.alerts import AlertService
from agent.core.analysis import AnalysisService
from agent.core.bettor import AutoBettor
from agent.core.dashboard import DashboardService
from agent.core.edge_detector import EdgeDetector
from agent.core.parlay import ParlayEvaluator
from agent.core.parlay_scanner import ParlayScanner
from agent.core.pipeline import PipelineRunner
from agent.core.rerun import RerunCoordinator
from agent.core.scheduler import PipelineScheduler
from agent.core.slate import SlateService
from agent.core.summary import DailySummaryService
from agent.db.engine import create_engine
from agent.db.repository import (
    AnalysisRepository,
    EdgeAlertRepository,
    EdgeRepository,
    ParlayRepository,
    PipelineRunRepository,
    ScheduleRepository,
)
from agent.events.subscriber import EventSubscriber
from agent.llm import create_llm_provider
from agent.telemetry import configure_telemetry


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings.database_url)
        redis_client: aioredis.Redis = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
        # simulations can take tens of seconds; keep a generous read timeout
        http_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=60.0))

        statistics = StatisticsClient(settings.statistics_service_url, http_client)
        lines = LinesClient(settings.lines_service_url, http_client)
        simulation = SimulationClient(settings.simulation_engine_url, http_client)
        prediction = PredictionClient(settings.prediction_engine_url, http_client)
        emulator = EmulatorClient(settings.bookie_emulator_url, http_client)
        reconciler = GameReconciler(lines, redis_client, ttl_seconds=settings.game_map_ttl_seconds)

        run_repo = PipelineRunRepository(engine)
        edge_repo = EdgeRepository(engine)
        analysis_repo = AnalysisRepository(engine)
        alert_repo = EdgeAlertRepository(engine)
        schedule_repo = ScheduleRepository(engine)
        parlay_repo = ParlayRepository(engine)
        detector = EdgeDetector(
            devig_method=settings.devig_method,
            kelly_multiplier=settings.kelly_multiplier,
            max_bet_pct=settings.max_bet_pct,
        )
        bettor = AutoBettor(
            emulator, edge_repo, max_total_exposure=settings.max_total_exposure, parlay_repo=parlay_repo
        )
        parlay_evaluator = ParlayEvaluator(
            edge_repo,
            parlay_repo,
            statistics,
            lines,
            prediction,
            simulation,
            reconciler,
            bettor,
            redis_client,
            kelly_multiplier=settings.kelly_multiplier,
            max_bet_pct=settings.max_bet_pct,
        )
        # The scanner only enters the pipeline when PARLAY_SCAN_ENABLED, and
        # only auto-bets when PARLAY_AUTO_BET (both default off).
        parlay_scanner = ParlayScanner(
            edge_repo,
            parlay_evaluator,
            min_edge_ratio=settings.parlay_scan_min_edge_pct,
            bettor=bettor if settings.parlay_auto_bet else None,
        )
        # LLM construction is lazy (no eager connection): startup stays
        # crash-loop-safe when Anthropic/Ollama are unreachable.
        llm = create_llm_provider(settings, http_client)
        alert_service = AlertService(
            redis_client,
            alert_repo,
            llm,
            llm_descriptions_enabled=settings.alert_llm_descriptions,
            llm_max_per_run=settings.alert_llm_max_per_run,
        )
        runner = PipelineRunner(
            run_repo,
            edge_repo,
            statistics,
            simulation,
            prediction,
            lines,
            reconciler,
            detector,
            bettor,
            alert_service,
            redis_client,
            concurrency=settings.pipeline_concurrency,
            parlay_scanner=parlay_scanner if settings.parlay_scan_enabled else None,
        )
        summary_service = DailySummaryService(edge_repo, emulator, llm, analysis_repo)
        scheduler = PipelineScheduler(schedule_repo, runner, summary_service, settings)
        rerun = RerunCoordinator(runner, schedule_repo, settings)
        subscriber = EventSubscriber(redis_client, edge_repo, rerun=rerun)

        app.state.redis = redis_client
        app.state.statistics_client = statistics
        app.state.lines_client = lines
        app.state.simulation_client = simulation
        app.state.prediction_client = prediction
        app.state.emulator_client = emulator
        app.state.run_repo = run_repo
        app.state.edge_repo = edge_repo
        app.state.alert_repo = alert_repo
        app.state.schedule_repo = schedule_repo
        app.state.event_subscriber = subscriber
        app.state.parlay_evaluator = parlay_evaluator
        app.state.llm_provider = llm
        app.state.scheduler = scheduler
        app.state.pipeline_runner = runner
        app.state.analysis_service = AnalysisService(
            llm,
            statistics,
            lines,
            prediction,
            emulator,
            edge_repo,
            analysis_repo,
            reconciler,
            redis_client,
            cache_ttl_seconds=settings.analysis_cache_ttl_seconds,
        )
        app.state.slate_service = SlateService(
            statistics, prediction, edge_repo, redis_client, ttl_seconds=settings.slate_cache_ttl_seconds
        )
        app.state.dashboard_service = DashboardService(
            edge_repo,
            run_repo,
            emulator,
            redis_client,
            ttl_seconds=settings.dashboard_cache_ttl_seconds,
            schedule_repo=schedule_repo,
        )

        subscriber.start()
        scheduler.start()
        try:
            yield
        finally:
            await scheduler.stop()  # stop creating runs first
            await rerun.stop()
            await subscriber.stop()
            await llm.aclose()
            await http_client.aclose()
            await redis_client.aclose()
            await engine.dispose()

    app = FastAPI(
        title="BookieBreaker Agent",
        version=__version__,
        description="Pipeline orchestration, edge detection, and dashboard aggregation across all backend services.",
        contact={
            "name": "BookieBreaker",
            "url": "https://github.com/Bookie-Breaker",
            "email": "jsamuelsen11@gmail.com",
        },
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        servers=[{"url": "http://localhost:8006", "description": "Local development"}],
        # alphabetical by name: the docs repo's Spectral ruleset requires it
        openapi_tags=[
            {"name": "alerts", "description": "Delivered edge alerts with acknowledgement."},
            {"name": "analysis", "description": "LLM-generated edge, game, and performance analyses."},
            {"name": "dashboard", "description": "Aggregated edges, performance, and pipeline status."},
            {"name": "edges", "description": "Detected positive-EV edges against de-vigged market prices."},
            {"name": "health", "description": "Service health and downstream dependency status."},
            {"name": "parlays", "description": "Correlation-aware parlay evaluation over detected edges."},
            {"name": "pipeline", "description": "Trigger and inspect prediction pipeline runs."},
            {"name": "schedule", "description": "Cron schedules for automated pipeline runs."},
            {"name": "slate", "description": "A date's games with prediction summaries and active edges."},
        ],
        lifespan=lifespan,
    )
    app.add_middleware(RequestIDMiddleware)
    register_error_handlers(app)
    app.include_router(pipeline.router, prefix="/api/v1/agent")
    app.include_router(edges.router, prefix="/api/v1/agent")
    app.include_router(slate.router, prefix="/api/v1/agent")
    app.include_router(dashboard.router, prefix="/api/v1/agent")
    app.include_router(analysis.router, prefix="/api/v1/agent")
    app.include_router(alerts.router, prefix="/api/v1/agent")
    app.include_router(schedule.router, prefix="/api/v1/agent")
    app.include_router(parlays.router, prefix="/api/v1/agent")
    app.include_router(health.router, prefix="/api/v1/agent")
    configure_telemetry(app, settings)
    return app


app = create_app()
