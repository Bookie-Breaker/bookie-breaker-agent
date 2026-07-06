"""Phase 6 Wave 1: seed soccer pipeline schedules (FIFA_WC, EPL).

ADR-026 brings the soccer competitions into scope; each gets a daily
pipeline schedule row. FIFA_WC (2026 World Cup, live now) is seeded
enabled; EPL is seeded disabled until the season starts mid-August.
auto_bet stays off for both until the soccer models are validated, and
min_edge_threshold mirrors MIN_EV_PCT_BY_LEAGUE in agent.edges.ev.

Seeding is idempotent (INSERT ... ON CONFLICT (league) DO NOTHING), so
re-running the upgrade never duplicates rows or overwrites operator
edits. The downgrade deletes the two rows only while they still carry
exactly the seeded values and have never fired; customized or used
schedules survive a downgrade untouched.

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-05

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (league, cron_expression, timezone, description, enabled, auto_bet, min_edge_threshold)
_SEEDS: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    (
        "FIFA_WC",
        "0 9 * * *",
        "America/New_York",
        "Daily FIFA World Cup slate (Phase 6 Wave 1 seed)",
        "TRUE",
        "FALSE",
        "4.0",
    ),
    (
        "EPL",
        "0 9 * * *",
        "Europe/London",
        "Daily Premier League slate (Phase 6 Wave 1 seed; season starts mid-August)",
        "FALSE",
        "FALSE",
        "3.5",
    ),
)


def upgrade() -> None:
    for league, cron, timezone, description, enabled, auto_bet, threshold in _SEEDS:
        op.execute(
            "INSERT INTO agent.pipeline_schedules"
            " (league, cron_expression, timezone, description, enabled, auto_bet, min_edge_threshold)"
            f" VALUES ('{league}', '{cron}', '{timezone}', '{description}', {enabled}, {auto_bet}, {threshold})"
            " ON CONFLICT (league) DO NOTHING"
        )


def downgrade() -> None:
    # Remove only pristine seed rows: every seeded value must still match
    # and the schedule must never have run. Anything else is operator
    # state and survives the downgrade.
    for league, cron, timezone, description, enabled, auto_bet, threshold in _SEEDS:
        op.execute(
            "DELETE FROM agent.pipeline_schedules"
            f" WHERE league = '{league}'"
            f" AND cron_expression = '{cron}'"
            f" AND timezone = '{timezone}'"
            f" AND description = '{description}'"
            f" AND enabled IS {enabled}"
            f" AND auto_bet IS {auto_bet}"
            f" AND min_edge_threshold = {threshold}"
            " AND last_run_at IS NULL"
        )
