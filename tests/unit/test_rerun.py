"""RerunCoordinator debounce coalescing, cooldown, and schedule gating."""

import asyncio
import uuid
from typing import Any

from agent.api.errors import DuplicateResourceError
from agent.config import Settings
from agent.core.pipeline import RunParams
from agent.core.rerun import RerunCoordinator


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

        return _Run(), 2


class FakeScheduleRepo:
    def __init__(self, enabled_leagues: set[str]) -> None:
        self.enabled = enabled_leagues

    async def has_enabled_for_league(self, league: str) -> bool:
        return league in self.enabled


def make_coordinator(
    runner: FakeRunner,
    enabled_leagues: set[str],
    debounce: float = 0.02,
    cooldown: float = 10.0,
    enabled: bool = True,
) -> RerunCoordinator:
    settings = Settings(
        event_reruns_enabled=enabled,
        rerun_debounce_seconds=debounce,
        rerun_cooldown_seconds=cooldown,
    )
    return RerunCoordinator(runner, FakeScheduleRepo(enabled_leagues), settings)  # type: ignore[arg-type]


class TestRerunCoordinator:
    async def test_burst_coalesces_into_one_run(self) -> None:
        runner = FakeRunner()
        coordinator = make_coordinator(runner, {"NBA"})

        for _ in range(5):
            coordinator.request("NBA")
            await asyncio.sleep(0.005)  # keep re-arming inside the debounce window
        await asyncio.sleep(0.06)

        assert len(runner.calls) == 1
        params, trigger = runner.calls[0]
        assert trigger == "EVENT"
        assert params.league == "NBA"

    async def test_cooldown_suppresses_second_run(self) -> None:
        runner = FakeRunner()
        coordinator = make_coordinator(runner, {"NBA"}, cooldown=10.0)

        coordinator.request("NBA")
        await asyncio.sleep(0.05)
        coordinator.request("NBA")
        await asyncio.sleep(0.05)

        assert len(runner.calls) == 1

    async def test_league_without_schedule_is_gated(self) -> None:
        runner = FakeRunner()
        coordinator = make_coordinator(runner, enabled_leagues=set())

        coordinator.request("NBA")
        await asyncio.sleep(0.05)

        assert runner.calls == []

    async def test_globally_disabled(self) -> None:
        runner = FakeRunner()
        coordinator = make_coordinator(runner, {"NBA"}, enabled=False)

        coordinator.request("NBA")
        await asyncio.sleep(0.05)

        assert runner.calls == []

    async def test_leagues_debounce_independently(self) -> None:
        runner = FakeRunner()
        coordinator = make_coordinator(runner, {"NBA", "MLB"})

        coordinator.request("NBA")
        coordinator.request("MLB")
        await asyncio.sleep(0.06)

        leagues = {params.league for params, _ in runner.calls}
        assert leagues == {"NBA", "MLB"}

    async def test_duplicate_run_swallowed(self) -> None:
        runner = FakeRunner(duplicate=True)
        coordinator = make_coordinator(runner, {"NBA"})

        coordinator.request("NBA")
        await asyncio.sleep(0.05)

        assert len(runner.calls) == 1  # attempted once, error swallowed

    async def test_stop_cancels_pending_timers(self) -> None:
        runner = FakeRunner()
        coordinator = make_coordinator(runner, {"NBA"}, debounce=5.0)

        coordinator.request("NBA")
        await coordinator.stop()
        await asyncio.sleep(0.02)

        assert runner.calls == []
