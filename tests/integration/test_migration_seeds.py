"""Migration 0004: idempotent soccer pipeline schedule seeds (ADR-026/027).

The session conftest applies head then clears the seeded rows, so every
test here re-runs the 0003 -> 0004 upgrade itself and cleans up after,
leaving the shared DB schedule-free for the other modules.
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
DELETE_SEEDS = "DELETE FROM agent.pipeline_schedules WHERE league IN ('FIFA_WC', 'EPL')"


def _reapply_0004(migrated_database_url: str) -> Config:
    """Roll back to 0003 and re-apply head, exercising a fresh 0004 run."""
    config = Config("alembic.ini")
    command.downgrade(config, "0003")
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
