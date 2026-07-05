"""Scheduler cron math, due-firing, misfire grace, and summary cadence."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from agent.api.errors import DuplicateResourceError
from agent.config import Settings
from agent.core.pipeline import RunParams
from agent.core.scheduler import PipelineScheduler, next_fire
from agent.db.repository import ScheduleRecord

NOW = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)


def make_schedule(**overrides: Any) -> ScheduleRecord:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "league": "NBA",
        "cron_expression": "0 10,14,18 * * *",
        "timezone": "UTC",
        "description": None,
        "enabled": True,
        "simulation_config": {"iterations": 10000},
        "auto_bet": True,
        "min_edge_threshold": 3.0,
        "last_run_at": None,
        "next_run_at": NOW - timedelta(seconds=30),
        "created_at": NOW - timedelta(days=1),
        "updated_at": NOW - timedelta(days=1),
    }
    defaults.update(overrides)
    return ScheduleRecord(**defaults)


class FakeScheduleRepo:
    def __init__(self, schedules: list[ScheduleRecord]) -> None:
        self.schedules = schedules
        self.marked: list[tuple[uuid.UUID, datetime, datetime]] = []
        self.next_runs: list[tuple[uuid.UUID, datetime | None]] = []

    async def list_enabled(self) -> list[ScheduleRecord]:
        return [s for s in self.schedules if s.enabled]

    async def mark_ran(self, schedule_id: uuid.UUID, last_run_at: datetime, next_run_at: datetime) -> None:
        self.marked.append((schedule_id, last_run_at, next_run_at))

    async def set_next_run(self, schedule_id: uuid.UUID, next_run_at: datetime | None) -> None:
        self.next_runs.append((schedule_id, next_run_at))


class FakeRunner:
    def __init__(self, duplicate: bool = False) -> None:
        self.calls: list[tuple[RunParams, str]] = []
        self._duplicate = duplicate

    async def start_run(self, params: RunParams, trigger: str = "MANUAL") -> tuple[Any, int]:
        self.calls.append((params, trigger))
        if self._duplicate:
            raise DuplicateResourceError("already running")

        class _Run:
            id = uuid.uuid4()

        return _Run(), 3


class FakeSummary:
    def __init__(self) -> None:
        self.dates: list[Any] = []

    async def generate(self, summary_date: Any) -> None:
        self.dates.append(summary_date)


def make_scheduler(
    repo: FakeScheduleRepo,
    runner: FakeRunner,
    summary: FakeSummary | None = None,
    **settings_overrides: Any,
) -> PipelineScheduler:
    settings = Settings(daily_summary_enabled=summary is not None, **settings_overrides)
    return PipelineScheduler(repo, runner, summary, settings)  # type: ignore[arg-type]


class TestNextFire:
    def test_utc_cron(self) -> None:
        after = datetime(2026, 7, 4, 12, 30, tzinfo=UTC)
        assert next_fire("0 14 * * *", "UTC", after) == datetime(2026, 7, 4, 14, 0, tzinfo=UTC)

    def test_timezone_conversion(self) -> None:
        # 10:00 America/New_York on 2026-07-04 (EDT, UTC-4) is 14:00 UTC
        after = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)
        assert next_fire("0 10 * * *", "America/New_York", after) == datetime(2026, 7, 4, 14, 0, tzinfo=UTC)

    def test_unknown_timezone_falls_back_to_utc(self) -> None:
        after = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)
        assert next_fire("0 14 * * *", "Not/AZone", after) == datetime(2026, 7, 4, 14, 0, tzinfo=UTC)

    def test_strictly_after(self) -> None:
        exactly = datetime(2026, 7, 4, 14, 0, tzinfo=UTC)
        assert next_fire("0 14 * * *", "UTC", exactly) == datetime(2026, 7, 5, 14, 0, tzinfo=UTC)


class TestTick:
    async def test_due_schedule_fires_scheduled_run(self) -> None:
        schedule = make_schedule()
        repo = FakeScheduleRepo([schedule])
        runner = FakeRunner()
        scheduler = make_scheduler(repo, runner)

        await scheduler._tick(NOW)

        assert len(runner.calls) == 1
        params, trigger = runner.calls[0]
        assert trigger == "SCHEDULED"
        assert params.league == "NBA"
        assert params.min_edge_threshold == 3.0
        assert params.simulation_config == {"iterations": 10000}
        assert len(repo.marked) == 1
        _, last_run, next_run = repo.marked[0]
        assert last_run == NOW
        assert next_run == datetime(2026, 7, 4, 14, 0, tzinfo=UTC)

    async def test_misfire_beyond_grace_skips_run(self) -> None:
        schedule = make_schedule(next_run_at=NOW - timedelta(hours=2))
        repo = FakeScheduleRepo([schedule])
        runner = FakeRunner()
        scheduler = make_scheduler(repo, runner, schedule_misfire_grace_seconds=300)

        await scheduler._tick(NOW)

        assert runner.calls == []
        assert repo.next_runs == [(schedule.id, datetime(2026, 7, 4, 14, 0, tzinfo=UTC))]
        assert repo.marked == []

    async def test_future_schedule_returns_wakeup_time(self) -> None:
        fire_at = NOW + timedelta(minutes=45)
        repo = FakeScheduleRepo([make_schedule(next_run_at=fire_at)])
        runner = FakeRunner()
        scheduler = make_scheduler(repo, runner)

        assert await scheduler._tick(NOW) == fire_at
        assert runner.calls == []

    async def test_missing_next_run_is_backfilled_not_fired(self) -> None:
        schedule = make_schedule(next_run_at=None)
        repo = FakeScheduleRepo([schedule])
        runner = FakeRunner()
        scheduler = make_scheduler(repo, runner)

        await scheduler._tick(NOW)

        assert runner.calls == []
        assert repo.next_runs == [(schedule.id, datetime(2026, 7, 4, 14, 0, tzinfo=UTC))]

    async def test_duplicate_run_is_swallowed_and_rolled_forward(self) -> None:
        schedule = make_schedule()
        repo = FakeScheduleRepo([schedule])
        runner = FakeRunner(duplicate=True)
        scheduler = make_scheduler(repo, runner)

        await scheduler._tick(NOW)

        assert len(runner.calls) == 1
        assert len(repo.marked) == 1  # still rolls forward past the duplicate

    async def test_invalid_cron_is_skipped(self) -> None:
        schedule = make_schedule(cron_expression="not a cron", next_run_at=None)
        repo = FakeScheduleRepo([schedule])
        runner = FakeRunner()
        scheduler = make_scheduler(repo, runner)

        assert await scheduler._tick(NOW) is None
        assert runner.calls == []


class TestDailySummary:
    async def test_summary_fires_on_its_cron(self) -> None:
        repo = FakeScheduleRepo([])
        runner = FakeRunner()
        summary = FakeSummary()
        scheduler = make_scheduler(repo, runner, summary, daily_summary_cron="0 12 * * *")

        # First tick just computes the next fire (13:00 was missed? no: next after 11:59 is 12:00)
        before = NOW - timedelta(minutes=1)
        next_wakeup = await scheduler._tick(before)
        assert next_wakeup == datetime(2026, 7, 4, 12, 0, tzinfo=UTC)
        assert summary.dates == []

        # At 12:00:30 the summary is due and within grace
        await scheduler._tick(NOW + timedelta(seconds=30))
        assert summary.dates == [NOW.date()]

    async def test_summary_disabled_never_fires(self) -> None:
        scheduler = make_scheduler(FakeScheduleRepo([]), FakeRunner(), None)
        assert await scheduler._tick(NOW) is None
