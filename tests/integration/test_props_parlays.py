"""Migration 0007: prop/live edge columns, YES/NO sides, and parlay tables."""

import uuid
from datetime import timedelta
from typing import Any

import pytest
from alembic.config import Config
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError

from agent.db.engine import create_engine
from agent.db.repository import EdgeRepository
from agent.db.tables import parlay_legs, parlays
from alembic import command
from tests.integration.conftest import execute_sql, insert_edge, run_async, utc_now

PROP_EDGE_OVERRIDES = {
    "market_type": "PLAYER_PROP",
    "selection": "LeBron James Over 27.5 Points",
    "side": "OVER",
    "line_value": 27.5,
    "player_external_id": "player-lebron-james",
    "stat_type": "points",
    "prop_type": "over_under",
    "is_live": True,
}


def _parlay_values(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "league": "NBA",
        "combined_odds_american": 264,
        "combined_odds_decimal": 3.6364,
        "joint_probability": 0.32,
        "independent_probability": 0.29,
        "correlation_edge": 0.03,
        "expected_value": 0.164,
        "kelly_fraction": 0.021,
        "recommended_stake": 1.5,
        "confidence": 0.7,
        "is_same_game": True,
        "leg_count": 2,
        "correlations": {"pairs": [{"legs": [0, 1], "rho": 0.18}]},
        "expires_at": utc_now() + timedelta(days=1),
    }
    values.update(overrides)
    return values


def _leg_values(parlay_id: uuid.UUID, leg_index: int, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "parlay_id": parlay_id,
        "leg_index": leg_index,
        "game_id": uuid.uuid4(),
        "game_external_id": f"ext-{uuid.uuid4()}",
        "league": "NBA",
        "market_type": "MONEYLINE",
        "selection": "Los Angeles Lakers",
        "side": "HOME",
        "line_value": None,
        "odds_american": -140,
        "odds_decimal": 1.7143,
        "predicted_probability": 0.70,
    }
    values.update(overrides)
    return values


class TestMigration0007Cycle:
    def test_downgrade_and_reapply(self, migrated_database_url: str) -> None:
        """0007 downgrades to 0006 and re-applies cleanly (fresh run)."""
        # Downgrade restores the pre-0007 side vocabulary; YES/NO rows from
        # other tests in this module would violate the recreated constraint.
        execute_sql(migrated_database_url, "DELETE FROM agent.edges WHERE side IN ('YES', 'NO')")
        config = Config("alembic.ini")
        command.downgrade(config, "0006")

        columns = execute_sql(
            migrated_database_url,
            "SELECT column_name FROM information_schema.columns WHERE table_schema = 'agent' AND table_name = 'edges'",
        )
        assert "player_external_id" not in {row["column_name"] for row in columns}
        tables = execute_sql(
            migrated_database_url,
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'agent'",
        )
        table_names = {row["table_name"] for row in tables}
        assert "parlays" not in table_names
        assert "parlay_legs" not in table_names

        command.upgrade(config, "head")

        columns = execute_sql(
            migrated_database_url,
            "SELECT column_name FROM information_schema.columns WHERE table_schema = 'agent' AND table_name = 'edges'",
        )
        assert {"player_external_id", "stat_type", "prop_type", "is_live"} <= {row["column_name"] for row in columns}
        tables = execute_sql(
            migrated_database_url,
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'agent'",
        )
        table_names = {row["table_name"] for row in tables}
        assert {"parlays", "parlay_legs"} <= table_names


class TestPropEdgeRows:
    def test_yes_side_accepted(self, migrated_database_url: str) -> None:
        edge_id = insert_edge(
            migrated_database_url,
            market_type="PLAYER_PROP",
            selection="LeBron James To Record A Triple-Double",
            side="YES",
            odds_american=450,
        )
        rows = execute_sql(
            migrated_database_url,
            "SELECT side FROM agent.edges WHERE id = $1::uuid",
            edge_id,
        )
        assert rows[0]["side"] == "YES"

    def test_unknown_side_still_rejected(self, migrated_database_url: str) -> None:
        with pytest.raises(IntegrityError, match="chk_edges_side"):
            insert_edge(migrated_database_url, side="MAYBE")

    def test_prop_fields_round_trip_through_repository(self, migrated_database_url: str) -> None:
        edge_id = insert_edge(migrated_database_url, **PROP_EDGE_OVERRIDES)

        async def _get() -> Any:
            engine = create_engine(migrated_database_url)
            try:
                return await EdgeRepository(engine).get(uuid.UUID(edge_id))
            finally:
                await engine.dispose()

        record = run_async(_get())
        assert record is not None
        assert record.player_external_id == "player-lebron-james"
        assert record.stat_type == "points"
        assert record.prop_type == "over_under"
        assert record.is_live is True

    def test_non_prop_edge_defaults(self, migrated_database_url: str) -> None:
        """Inserts without the prop columns keep NULL/FALSE defaults."""
        edge_id = insert_edge(migrated_database_url)

        async def _get() -> Any:
            engine = create_engine(migrated_database_url)
            try:
                return await EdgeRepository(engine).get(uuid.UUID(edge_id))
            finally:
                await engine.dispose()

        record = run_async(_get())
        assert record is not None
        assert record.player_external_id is None
        assert record.stat_type is None
        assert record.prop_type is None
        assert record.is_live is False


class TestParlayRows:
    def test_parlay_with_legs_insert_and_read(self, migrated_database_url: str) -> None:
        async def _round_trip() -> tuple[Any, list[Any]]:
            engine = create_engine(migrated_database_url)
            try:
                async with engine.begin() as conn:
                    parlay_row = (
                        await conn.execute(insert(parlays).values(**_parlay_values()).returning(parlays))
                    ).one()
                    await conn.execute(insert(parlay_legs).values(_leg_values(parlay_row.id, 0)))
                    await conn.execute(
                        insert(parlay_legs).values(
                            _leg_values(
                                parlay_row.id,
                                1,
                                market_type="PLAYER_PROP",
                                selection="LeBron James Over 27.5 Points",
                                side="OVER",
                                line_value=27.5,
                                player_external_id="player-lebron-james",
                                stat_type="points",
                                prop_type="over_under",
                                odds_american=-115,
                                odds_decimal=1.8696,
                                predicted_probability=0.61,
                            )
                        )
                    )
                async with engine.connect() as conn:
                    leg_rows = (
                        await conn.execute(
                            select(parlay_legs)
                            .where(parlay_legs.c.parlay_id == parlay_row.id)
                            .order_by(parlay_legs.c.leg_index)
                        )
                    ).fetchall()
                return parlay_row, list(leg_rows)
            finally:
                await engine.dispose()

        parlay_row, leg_rows = run_async(_round_trip())
        assert parlay_row.leg_count == 2
        assert parlay_row.is_same_game is True
        assert float(parlay_row.joint_probability) == 0.32
        assert dict(parlay_row.correlations) == {"pairs": [{"legs": [0, 1], "rho": 0.18}]}
        assert [row.leg_index for row in leg_rows] == [0, 1]
        assert leg_rows[0].side == "HOME"
        assert leg_rows[0].player_external_id is None
        assert leg_rows[1].side == "OVER"
        assert leg_rows[1].player_external_id == "player-lebron-james"
        assert leg_rows[1].stat_type == "points"
        assert leg_rows[1].prop_type == "over_under"

    def test_single_leg_parlay_rejected(self, migrated_database_url: str) -> None:
        async def _insert() -> None:
            engine = create_engine(migrated_database_url)
            try:
                async with engine.begin() as conn:
                    await conn.execute(insert(parlays).values(**_parlay_values(leg_count=1)))
            finally:
                await engine.dispose()

        with pytest.raises(IntegrityError, match="chk_parlays_leg_count"):
            run_async(_insert())

    def test_duplicate_leg_index_rejected(self, migrated_database_url: str) -> None:
        async def _insert() -> None:
            engine = create_engine(migrated_database_url)
            try:
                async with engine.begin() as conn:
                    parlay_row = (
                        await conn.execute(insert(parlays).values(**_parlay_values()).returning(parlays))
                    ).one()
                    await conn.execute(insert(parlay_legs).values(_leg_values(parlay_row.id, 0)))
                    await conn.execute(insert(parlay_legs).values(_leg_values(parlay_row.id, 0)))
            finally:
                await engine.dispose()

        with pytest.raises(IntegrityError, match="uq_parlay_legs_parlay_leg_index"):
            run_async(_insert())
