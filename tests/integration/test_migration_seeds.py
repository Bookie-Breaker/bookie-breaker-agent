"""Migrations 0004/0005/0006: idempotent pipeline schedule seeds (ADR-026/027).

Migration 0004 seeds the soccer schedules (FIFA_WC, EPL); migration 0005
seeds the baseball schedules (MLB, NCAA_BSB); migration 0006 seeds the
football, hockey, and NCAA basketball schedules (NFL, NCAA_FB, NHL,
NCAA_BB). The session conftest applies head then clears the seeded rows,
so every test here re-runs the relevant seed upgrade itself and cleans up
after, leaving the shared DB schedule-free for the other modules.
"""

from datetime import UTC, datetime
from typing import Any

import pytest
from alembic.config import Config

from agent.config import Settings
from agent.core.scheduler import PipelineScheduler
from agent.db.engine import create_engine
from agent.db.repository import ScheduleRepository
from alembic import command
from tests.integration.conftest import execute_sql, run_async

SEED_LEAGUES = ("FIFA_WC", "EPL")
BASEBALL_SEED_LEAGUES = ("MLB", "NCAA_BSB")
WAVE345_SEED_LEAGUES = ("NFL", "NCAA_FB", "NHL", "NCAA_BB")
DELETE_SEEDS = (
    "DELETE FROM agent.pipeline_schedules WHERE league IN"
    " ('FIFA_WC', 'EPL', 'MLB', 'NCAA_BSB', 'NFL', 'NCAA_FB', 'NHL', 'NCAA_BB')"
)


def _reapply_0004(migrated_database_url: str) -> Config:
    """Roll back to 0003 and re-apply head, exercising a fresh 0004 run."""
    config = Config("alembic.ini")
    command.downgrade(config, "0003")
    command.upgrade(config, "head")
    return config


def _reapply_0005(migrated_database_url: str) -> Config:
    """Roll back to 0004 and re-apply head, exercising a fresh 0005 run."""
    config = Config("alembic.ini")
    command.downgrade(config, "0004")
    command.upgrade(config, "head")
    return config


def _reapply_0006(migrated_database_url: str) -> Config:
    """Roll back to 0005 and re-apply head, exercising a fresh 0006 run."""
    config = Config("alembic.ini")
    command.downgrade(config, "0005")
    command.upgrade(config, "head")
    return config


def _seed_rows(database_url: str) -> dict[str, Any]:
    rows = execute_sql(
        database_url,
        "SELECT league::text AS league, cron_expression, timezone, enabled, auto_bet,"
        " min_edge_threshold::float8 AS threshold, last_run_at"
        " FROM agent.pipeline_schedules WHERE league IN ('FIFA_WC', 'EPL')",
    )
    return {row["league"]: dict(row) for row in rows}


def _baseball_seed_rows(database_url: str) -> dict[str, Any]:
    rows = execute_sql(
        database_url,
        "SELECT league::text AS league, cron_expression, timezone, enabled, auto_bet,"
        " min_edge_threshold::float8 AS threshold, last_run_at"
        " FROM agent.pipeline_schedules WHERE league IN ('MLB', 'NCAA_BSB')",
    )
    return {row["league"]: dict(row) for row in rows}


def _wave345_seed_rows(database_url: str) -> dict[str, Any]:
    rows = execute_sql(
        database_url,
        "SELECT league::text AS league, cron_expression, timezone, enabled, auto_bet,"
        " min_edge_threshold::float8 AS threshold, last_run_at"
        " FROM agent.pipeline_schedules WHERE league IN ('NFL', 'NCAA_FB', 'NHL', 'NCAA_BB')",
    )
    return {row["league"]: dict(row) for row in rows}


@pytest.fixture(autouse=True)
def isolated_seeds(migrated_database_url: str) -> Any:
    execute_sql(migrated_database_url, DELETE_SEEDS)
    yield
    execute_sql(migrated_database_url, DELETE_SEEDS)


class TestSoccerScheduleSeeds:
    def test_fresh_upgrade_seeds_expected_rows(self, migrated_database_url: str) -> None:
        _reapply_0004(migrated_database_url)

        rows = _seed_rows(migrated_database_url)
        assert set(rows) == set(SEED_LEAGUES)

        fifa = rows["FIFA_WC"]
        assert fifa["cron_expression"] == "0 9 * * *"
        assert fifa["timezone"] == "America/New_York"
        assert fifa["enabled"] is True
        assert fifa["auto_bet"] is False
        assert fifa["threshold"] == 4.0  # matches MIN_EV_PCT_BY_LEAGUE
        assert fifa["last_run_at"] is None

        epl = rows["EPL"]
        assert epl["cron_expression"] == "0 9 * * *"
        assert epl["timezone"] == "Europe/London"
        assert epl["enabled"] is False  # season starts mid-August
        assert epl["auto_bet"] is False
        assert epl["threshold"] == 3.5

    def test_reapplied_upgrade_is_idempotent_and_preserves_edits(self, migrated_database_url: str) -> None:
        config = _reapply_0004(migrated_database_url)
        execute_sql(
            migrated_database_url,
            "UPDATE agent.pipeline_schedules SET cron_expression = '0 7 * * *' WHERE league = 'FIFA_WC'",
        )

        # Downgrade drops only the untouched EPL row; the customized
        # FIFA_WC row is operator state and survives.
        command.downgrade(config, "0003")
        rows = _seed_rows(migrated_database_url)
        assert set(rows) == {"FIFA_WC"}
        assert rows["FIFA_WC"]["cron_expression"] == "0 7 * * *"

        # Re-upgrading reseeds EPL and leaves the customization alone
        # (INSERT ... ON CONFLICT (league) DO NOTHING).
        command.upgrade(config, "head")
        rows = _seed_rows(migrated_database_url)
        assert set(rows) == set(SEED_LEAGUES)
        assert rows["FIFA_WC"]["cron_expression"] == "0 7 * * *"
        assert rows["EPL"]["cron_expression"] == "0 9 * * *"

        counts = execute_sql(
            migrated_database_url,
            "SELECT COUNT(*) AS n FROM agent.pipeline_schedules WHERE league IN ('FIFA_WC', 'EPL') GROUP BY league",
        )
        assert [row["n"] for row in counts] == [1, 1]

    def test_scheduler_picks_up_seeded_fifa_wc_schedule(self, migrated_database_url: str) -> None:
        _reapply_0004(migrated_database_url)

        async def scenario() -> None:
            engine = create_engine(migrated_database_url)
            try:
                repo = ScheduleRepository(engine)
                enabled = {schedule.league for schedule in await repo.list_enabled()}
                assert "FIFA_WC" in enabled
                assert "EPL" not in enabled  # disabled seed stays invisible to the scheduler

                # A real tick backfills next_run_at for the seeded row (it
                # ships NULL) without firing: next_fire is strictly in the
                # future, so the runner is never touched.
                scheduler = PipelineScheduler(
                    repo,
                    None,  # type: ignore[arg-type]
                    None,
                    Settings(daily_summary_enabled=False),
                )
                now = datetime.now(tz=UTC)
                next_wakeup = await scheduler._tick(now)

                fifa = next(s for s in await repo.list_all() if s.league == "FIFA_WC")
                assert fifa.next_run_at is not None
                assert fifa.next_run_at > now
                assert fifa.last_run_at is None
                assert next_wakeup is not None
                assert next_wakeup <= fifa.next_run_at
            finally:
                await engine.dispose()

        run_async(scenario())


class TestBaseballScheduleSeeds:
    def test_fresh_upgrade_seeds_expected_rows(self, migrated_database_url: str) -> None:
        _reapply_0005(migrated_database_url)

        rows = _baseball_seed_rows(migrated_database_url)
        assert set(rows) == set(BASEBALL_SEED_LEAGUES)

        mlb = rows["MLB"]
        assert mlb["cron_expression"] == "0 9 * * *"
        assert mlb["timezone"] == "America/New_York"
        assert mlb["enabled"] is True
        assert mlb["auto_bet"] is False
        assert mlb["threshold"] == 2.5  # matches MIN_EV_PCT_BY_LEAGUE
        assert mlb["last_run_at"] is None

        ncaa = rows["NCAA_BSB"]
        assert ncaa["cron_expression"] == "0 9 * * *"
        assert ncaa["timezone"] == "America/New_York"
        assert ncaa["enabled"] is False  # dormant until February 2027
        assert ncaa["auto_bet"] is False
        assert ncaa["threshold"] == 2.0

    def test_reapplied_upgrade_is_idempotent_and_preserves_edits(self, migrated_database_url: str) -> None:
        config = _reapply_0005(migrated_database_url)
        execute_sql(
            migrated_database_url,
            "UPDATE agent.pipeline_schedules SET cron_expression = '0 7 * * *' WHERE league = 'MLB'",
        )

        # Downgrade drops only the untouched NCAA_BSB row; the customized
        # MLB row is operator state and survives.
        command.downgrade(config, "0004")
        rows = _baseball_seed_rows(migrated_database_url)
        assert set(rows) == {"MLB"}
        assert rows["MLB"]["cron_expression"] == "0 7 * * *"

        # Re-upgrading reseeds NCAA_BSB and leaves the customization alone
        # (INSERT ... ON CONFLICT (league) DO NOTHING).
        command.upgrade(config, "head")
        rows = _baseball_seed_rows(migrated_database_url)
        assert set(rows) == set(BASEBALL_SEED_LEAGUES)
        assert rows["MLB"]["cron_expression"] == "0 7 * * *"
        assert rows["NCAA_BSB"]["cron_expression"] == "0 9 * * *"

        counts = execute_sql(
            migrated_database_url,
            "SELECT COUNT(*) AS n FROM agent.pipeline_schedules WHERE league IN ('MLB', 'NCAA_BSB') GROUP BY league",
        )
        assert [row["n"] for row in counts] == [1, 1]

    def test_scheduler_picks_up_seeded_mlb_schedule(self, migrated_database_url: str) -> None:
        _reapply_0005(migrated_database_url)

        async def scenario() -> None:
            engine = create_engine(migrated_database_url)
            try:
                repo = ScheduleRepository(engine)
                enabled = {schedule.league for schedule in await repo.list_enabled()}
                assert "MLB" in enabled
                assert "NCAA_BSB" not in enabled  # disabled seed stays invisible to the scheduler

                # A real tick backfills next_run_at for the seeded row (it
                # ships NULL) without firing: next_fire is strictly in the
                # future, so the runner is never touched.
                scheduler = PipelineScheduler(
                    repo,
                    None,  # type: ignore[arg-type]
                    None,
                    Settings(daily_summary_enabled=False),
                )
                now = datetime.now(tz=UTC)
                next_wakeup = await scheduler._tick(now)

                mlb = next(s for s in await repo.list_all() if s.league == "MLB")
                assert mlb.next_run_at is not None
                assert mlb.next_run_at > now
                assert mlb.last_run_at is None
                assert next_wakeup is not None
                assert next_wakeup <= mlb.next_run_at
            finally:
                await engine.dispose()

        run_async(scenario())


class TestFootballHockeyBballScheduleSeeds:
    def test_fresh_upgrade_seeds_expected_rows(self, migrated_database_url: str) -> None:
        _reapply_0006(migrated_database_url)

        rows = _wave345_seed_rows(migrated_database_url)
        assert set(rows) == set(WAVE345_SEED_LEAGUES)

        # thresholds mirror MIN_EV_PCT_BY_LEAGUE
        thresholds = {"NFL": 3.0, "NCAA_FB": 2.0, "NHL": 3.0, "NCAA_BB": 2.0}
        for league, threshold in thresholds.items():
            row = rows[league]
            assert row["cron_expression"] == "0 9 * * *"
            assert row["timezone"] == "America/New_York"
            assert row["enabled"] is False  # every league is off-season; flips on at season start
            assert row["auto_bet"] is False
            assert row["threshold"] == threshold
            assert row["last_run_at"] is None

    def test_reapplied_upgrade_is_idempotent_and_preserves_edits(self, migrated_database_url: str) -> None:
        config = _reapply_0006(migrated_database_url)
        execute_sql(
            migrated_database_url,
            "UPDATE agent.pipeline_schedules SET cron_expression = '0 7 * * *' WHERE league = 'NFL'",
        )

        # Downgrade drops only the untouched NCAA_FB/NHL/NCAA_BB rows; the
        # customized NFL row is operator state and survives.
        command.downgrade(config, "0005")
        rows = _wave345_seed_rows(migrated_database_url)
        assert set(rows) == {"NFL"}
        assert rows["NFL"]["cron_expression"] == "0 7 * * *"

        # Re-upgrading reseeds the other three and leaves the customization
        # alone (INSERT ... ON CONFLICT (league) DO NOTHING).
        command.upgrade(config, "head")
        rows = _wave345_seed_rows(migrated_database_url)
        assert set(rows) == set(WAVE345_SEED_LEAGUES)
        assert rows["NFL"]["cron_expression"] == "0 7 * * *"
        assert rows["NCAA_FB"]["cron_expression"] == "0 9 * * *"
        assert rows["NHL"]["cron_expression"] == "0 9 * * *"
        assert rows["NCAA_BB"]["cron_expression"] == "0 9 * * *"

        counts = execute_sql(
            migrated_database_url,
            "SELECT COUNT(*) AS n FROM agent.pipeline_schedules"
            " WHERE league IN ('NFL', 'NCAA_FB', 'NHL', 'NCAA_BB') GROUP BY league",
        )
        assert [row["n"] for row in counts] == [1, 1, 1, 1]

    def test_scheduler_sees_seed_only_after_enabling(self, migrated_database_url: str) -> None:
        _reapply_0006(migrated_database_url)

        async def scenario() -> None:
            engine = create_engine(migrated_database_url)
            try:
                repo = ScheduleRepository(engine)

                # All four rows ship disabled, so none is visible to the
                # scheduler out of the box.
                enabled = {schedule.league for schedule in await repo.list_enabled()}
                assert enabled.isdisjoint(WAVE345_SEED_LEAGUES)

                # Season start: the operator enables NFL through the
                # repository (the `bb pipeline schedule set` path).
                await repo.upsert_for_league(
                    {
                        "league": "NFL",
                        "cron_expression": "0 9 * * *",
                        "timezone": "America/New_York",
                        "enabled": True,
                        "auto_bet": False,
                        "min_edge_threshold": 3.0,
                    }
                )
                enabled = {schedule.league for schedule in await repo.list_enabled()}
                assert "NFL" in enabled
                assert enabled.isdisjoint({"NCAA_FB", "NHL", "NCAA_BB"})

                # A real tick backfills next_run_at for the newly enabled
                # row (it carries NULL) without firing: next_fire is
                # strictly in the future, so the runner is never touched.
                scheduler = PipelineScheduler(
                    repo,
                    None,  # type: ignore[arg-type]
                    None,
                    Settings(daily_summary_enabled=False),
                )
                now = datetime.now(tz=UTC)
                next_wakeup = await scheduler._tick(now)

                nfl = next(s for s in await repo.list_all() if s.league == "NFL")
                assert nfl.next_run_at is not None
                assert nfl.next_run_at > now
                assert nfl.last_run_at is None
                assert next_wakeup is not None
                assert next_wakeup <= nfl.next_run_at
            finally:
                await engine.dispose()

        run_async(scenario())
