"""Schedule endpoints, repository SQL, and a real-DB scheduler tick."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from agent.config import Settings
from agent.core.pipeline import RunParams
from agent.core.scheduler import PipelineScheduler
from agent.db.engine import create_engine
from agent.db.repository import PipelineRunRepository, ScheduleRepository
from tests.integration.conftest import execute_sql, run_async


@pytest.fixture(autouse=True)
def clean_schedules(migrated_database_url: str) -> Any:
    yield
    execute_sql(migrated_database_url, "DELETE FROM agent.pipeline_schedules")


class TestScheduleApi:
    def test_upsert_create_then_update(self, client) -> None:
        body = {
            "league": "NBA",
            "cron_expression": "0 10,14,18 * * *",
            "description": "Thrice daily",
            "simulation_config": {"iterations": 10000},
            "min_edge_threshold": 4.0,
        }
        created = client.post("/api/v1/agent/schedule", json=body)
        assert created.status_code == 201, created.text
        data = created.json()["data"]
        assert data["league"] == "NBA"
        assert data["enabled"] is True
        assert data["auto_bet"] is True
        assert data["min_edge_threshold"] == 4.0
        assert data["timezone"] == "UTC"
        assert data["next_run_at"] is not None

        updated = client.post(
            "/api/v1/agent/schedule",
            json={"league": "NBA", "cron_expression": "0 9 * * *", "enabled": False, "timezone": "America/New_York"},
        )
        assert updated.status_code == 200, updated.text
        upd = updated.json()["data"]
        assert upd["id"] == data["id"]
        assert upd["cron_expression"] == "0 9 * * *"
        assert upd["enabled"] is False
        assert upd["timezone"] == "America/New_York"

        listing = client.get("/api/v1/agent/schedule")
        schedules = listing.json()["data"]["schedules"]
        assert [s["id"] for s in schedules] == [data["id"]]

    def test_validation_errors(self, client) -> None:
        bad_cron = client.post("/api/v1/agent/schedule", json={"league": "NBA", "cron_expression": "99 99 * *"})
        assert bad_cron.status_code == 422
        bad_league = client.post("/api/v1/agent/schedule", json={"league": "XFL", "cron_expression": "0 9 * * *"})
        assert bad_league.status_code == 422
        bad_tz = client.post(
            "/api/v1/agent/schedule",
            json={"league": "NBA", "cron_expression": "0 9 * * *", "timezone": "Mars/Olympus"},
        )
        assert bad_tz.status_code == 422

    def test_dashboard_and_health_surface_next_run(self, client, upstream, redis_url) -> None:
        import redis as sync_redis
        from httpx import Response

        # downstream services degrade gracefully; only next_run matters here
        for host in ("stats.test", "lines.test", "sim.test", "predict.test", "emulator.test", "anthropic.test"):
            upstream.route(host=host).mock(return_value=Response(500, json={"error": {}, "meta": {}}))

        created = client.post("/api/v1/agent/schedule", json={"league": "MLB", "cron_expression": "0 11 * * *"})
        assert created.status_code == 201
        expected = created.json()["data"]["next_run_at"]

        redis_client = sync_redis.Redis.from_url(redis_url, decode_responses=True)
        for key in redis_client.scan_iter(match="agent:dashboard:*"):
            redis_client.delete(key)

        dashboard = client.get("/api/v1/agent/dashboard")
        assert dashboard.status_code == 200
        assert dashboard.json()["data"]["pipeline_status"]["next_scheduled_run"] == expected

        health = client.get("/api/v1/agent/health")
        assert health.json()["data"]["pipeline"]["next_scheduled_run"] == expected


class TestScheduleRepositorySql:
    def test_mark_ran_and_min_next_run(self, migrated_database_url) -> None:
        async def scenario() -> tuple[datetime | None, datetime | None]:
            engine = create_engine(migrated_database_url)
            try:
                repo = ScheduleRepository(engine)
                record, created = await repo.upsert_for_league(
                    {
                        "league": "NFL",
                        "cron_expression": "0 8 * * *",
                        "timezone": "UTC",
                        "enabled": True,
                        "auto_bet": True,
                        "min_edge_threshold": 3.0,
                        "next_run_at": datetime(2026, 7, 5, 8, 0, tzinfo=UTC),
                    }
                )
                assert created is True
                now = datetime.now(tz=UTC)
                await repo.mark_ran(record.id, now, datetime(2026, 7, 6, 8, 0, tzinfo=UTC))
                refreshed = [s for s in await repo.list_all() if s.id == record.id][0]
                return refreshed.last_run_at, await repo.min_next_run()
            finally:
                await engine.dispose()

        last_run_at, min_next = run_async(scenario())
        assert last_run_at is not None
        assert min_next == datetime(2026, 7, 6, 8, 0, tzinfo=UTC)


class TestSchedulerTickAgainstRealDb:
    def test_due_schedule_creates_scheduled_pipeline_run(self, migrated_database_url) -> None:
        """A real-DB tick: due schedule fires, run row carries trigger=SCHEDULED,
        and next_run_at rolls forward — exercising the CHECK constraint and
        all scheduler SQL end-to-end."""

        class RecordingRunner:
            def __init__(self, run_repo: PipelineRunRepository) -> None:
                self._run_repo = run_repo
                self.created: list[uuid.UUID] = []

            async def start_run(self, params: RunParams, trigger: str = "MANUAL") -> tuple[Any, int]:
                run = await self._run_repo.create_running(params.league, trigger, params.as_json())
                self.created.append(run.id)
                return run, 0

        async def scenario() -> None:
            engine = create_engine(migrated_database_url)
            try:
                schedule_repo = ScheduleRepository(engine)
                run_repo = PipelineRunRepository(engine)
                now = datetime.now(tz=UTC)
                record, _ = await schedule_repo.upsert_for_league(
                    {
                        "league": "NCAA_BB",
                        "cron_expression": "*/5 * * * *",
                        "timezone": "UTC",
                        "enabled": True,
                        "auto_bet": False,
                        "min_edge_threshold": 5.0,
                        "next_run_at": now - timedelta(seconds=10),
                    }
                )
                runner = RecordingRunner(run_repo)
                scheduler = PipelineScheduler(
                    schedule_repo,
                    runner,  # type: ignore[arg-type]
                    None,
                    Settings(daily_summary_enabled=False),
                )
                await scheduler._tick(now)

                assert len(runner.created) == 1
                run = await run_repo.get(runner.created[0])
                assert run is not None
                assert run.trigger == "SCHEDULED"
                assert run.league == "NCAA_BB"
                assert run.params["min_edge_threshold"] == 5.0

                refreshed = [s for s in await schedule_repo.list_all() if s.id == record.id][0]
                assert refreshed.last_run_at is not None
                assert refreshed.next_run_at is not None
                assert refreshed.next_run_at > now
                await run_repo.finish(runner.created[0], "COMPLETED")
            finally:
                await engine.dispose()

        run_async(scenario())
