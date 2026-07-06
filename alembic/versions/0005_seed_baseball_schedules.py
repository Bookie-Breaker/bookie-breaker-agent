"""Phase 6 Wave 2: seed baseball pipeline schedules (MLB, NCAA_BSB).

ADR-026 brings the baseball leagues into scope; each gets a daily
pipeline schedule row. MLB (season live now) is seeded enabled;
NCAA_BSB is seeded disabled until the college season starts in
February 2027. auto_bet stays off for both until the baseball models
are validated, and min_edge_threshold mirrors MIN_EV_PCT_BY_LEAGUE in
agent.edges.ev.

Seeding is idempotent (INSERT ... ON CONFLICT (league) DO NOTHING), so
re-running the upgrade never duplicates rows or overwrites operator
edits. The downgrade deletes the two rows only while they still carry
exactly the seeded values and have never fired; customized or used
schedules survive a downgrade untouched.

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-05

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (league, cron_expression, timezone, description, enabled, auto_bet, min_edge_threshold)
_SEEDS: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    (
        "MLB",
        "0 9 * * *",
        "America/New_York",
        "Daily MLB slate (Phase 6 Wave 2 seed)",
        "TRUE",
        "FALSE",
        "2.5",
    ),
    (
        "NCAA_BSB",
        "0 9 * * *",
        "America/New_York",
        "Daily NCAA Baseball slate (Phase 6 Wave 2 seed; dormant until February 2027)",
        "FALSE",
        "FALSE",
        "2.0",
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
