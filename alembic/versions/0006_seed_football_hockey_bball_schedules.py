"""Phase 6 Waves 3-5: seed football, hockey, and NCAA basketball pipeline schedules.

ADR-026 brings the remaining leagues into scope; each gets a daily
pipeline schedule row. All four leagues are off-season during the July
2026 build window, so every row is seeded disabled — they flip on at
season start (NFL/NCAA_FB in September, NHL in October, NCAA_BB in
November). auto_bet stays off until the per-league models are
validated, and min_edge_threshold mirrors MIN_EV_PCT_BY_LEAGUE in
agent.edges.ev.

Seeding is idempotent (INSERT ... ON CONFLICT (league) DO NOTHING), so
re-running the upgrade never duplicates rows or overwrites operator
edits. The downgrade deletes the four rows only while they still carry
exactly the seeded values and have never fired; customized or used
schedules survive a downgrade untouched.

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-06

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (league, cron_expression, timezone, description, enabled, auto_bet, min_edge_threshold)
_SEEDS: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    (
        "NFL",
        "0 9 * * *",
        "America/New_York",
        "Daily NFL slate (Phase 6 Waves 3-5 seed; dormant until September 2026)",
        "FALSE",
        "FALSE",
        "3.0",
    ),
    (
        "NCAA_FB",
        "0 9 * * *",
        "America/New_York",
        "Daily NCAA Football slate (Phase 6 Waves 3-5 seed; dormant until September 2026)",
        "FALSE",
        "FALSE",
        "2.0",
    ),
    (
        "NHL",
        "0 9 * * *",
        "America/New_York",
        "Daily NHL slate (Phase 6 Waves 3-5 seed; dormant until October 2026)",
        "FALSE",
        "FALSE",
        "3.0",
    ),
    (
        "NCAA_BB",
        "0 9 * * *",
        "America/New_York",
        "Daily NCAA Basketball slate (Phase 6 Waves 3-5 seed; dormant until November 2026)",
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
