"""Phase 6 Wave 0: widen chk_edges_side to allow the DRAW side.

Three-way soccer moneylines are represented as MONEYLINE with a third
DRAW side (ADR-027), so the agent.edges side vocabulary gains 'DRAW'.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-05

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("chk_edges_side", "edges", schema="agent", type_="check")
    op.create_check_constraint(
        "chk_edges_side",
        "edges",
        "side IS NULL OR side IN ('HOME', 'AWAY', 'DRAW', 'OVER', 'UNDER')",
        schema="agent",
    )


def downgrade() -> None:
    # Fails if DRAW rows exist; delete them before downgrading.
    op.drop_constraint("chk_edges_side", "edges", schema="agent", type_="check")
    op.create_check_constraint(
        "chk_edges_side",
        "edges",
        "side IS NULL OR side IN ('HOME', 'AWAY', 'OVER', 'UNDER')",
        schema="agent",
    )
