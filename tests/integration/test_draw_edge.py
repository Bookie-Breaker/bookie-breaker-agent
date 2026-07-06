"""Migration 0003: agent.edges accepts DRAW sides and new league values."""

import pytest
from sqlalchemy.exc import IntegrityError

from tests.integration.conftest import execute_sql, insert_edge


class TestDrawEdgeRows:
    def test_draw_moneyline_edge_row_accepted(self, migrated_database_url: str) -> None:
        edge_id = insert_edge(
            migrated_database_url,
            league="FIFA_WC",
            market_type="MONEYLINE",
            selection="Draw",
            side="DRAW",
            odds_american=220,
        )
        rows = execute_sql(
            migrated_database_url,
            "SELECT side, league::text AS league FROM agent.edges WHERE id = $1::uuid",
            edge_id,
        )
        assert rows[0]["side"] == "DRAW"
        assert rows[0]["league"] == "FIFA_WC"

    def test_unknown_side_still_rejected(self, migrated_database_url: str) -> None:
        with pytest.raises(IntegrityError, match="chk_edges_side"):
            insert_edge(migrated_database_url, side="TIE")
