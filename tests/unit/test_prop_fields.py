"""Phase 7 Wave 0: player-prop/live fields thread through models and inserts."""

import uuid

from agent.clients.lines import LineSnapshot
from agent.clients.prediction import PredictionItem
from agent.core.pipeline import PipelineRunner
from agent.db.tables import edges
from tests.unit.factories import make_candidate, make_edge_record

PROP_OVERRIDES = {
    "market_type": "PLAYER_PROP",
    "selection": "LeBron James Over 27.5 Points",
    "side": "OVER",
    "player_external_id": "player-lebron-james",
    "stat_type": "points",
    "prop_type": "over_under",
}


class TestEdgeCandidatePropFields:
    def test_defaults_to_non_prop(self) -> None:
        candidate = make_candidate()
        assert candidate.player_external_id is None
        assert candidate.stat_type is None
        assert candidate.prop_type is None
        assert candidate.is_live is False

    def test_prop_fields_settable(self) -> None:
        candidate = make_candidate(**PROP_OVERRIDES, is_live=True)
        assert candidate.player_external_id == "player-lebron-james"
        assert candidate.stat_type == "points"
        assert candidate.prop_type == "over_under"
        assert candidate.is_live is True


class TestEdgeValues:
    def test_includes_prop_keys(self) -> None:
        values = PipelineRunner._edge_values(uuid.uuid4(), make_candidate(), 2.5)
        assert {"player_external_id", "stat_type", "prop_type", "is_live"} <= set(values)
        assert values["player_external_id"] is None
        assert values["stat_type"] is None
        assert values["prop_type"] is None
        assert values["is_live"] is False

    def test_prop_candidate_values_thread_through(self) -> None:
        values = PipelineRunner._edge_values(uuid.uuid4(), make_candidate(**PROP_OVERRIDES, is_live=True), 2.5)
        assert values["player_external_id"] == "player-lebron-james"
        assert values["stat_type"] == "points"
        assert values["prop_type"] == "over_under"
        assert values["is_live"] is True

    def test_all_keys_are_edges_columns(self) -> None:
        values = PipelineRunner._edge_values(uuid.uuid4(), make_candidate(), 2.5)
        assert set(values) <= set(edges.columns.keys())


class TestEdgeRecordPropFields:
    def test_defaults_to_non_prop(self) -> None:
        record = make_edge_record()
        assert record.player_external_id is None
        assert record.stat_type is None
        assert record.prop_type is None
        assert record.is_live is False

    def test_prop_fields_settable(self) -> None:
        record = make_edge_record(
            player_external_id="player-lebron-james", stat_type="points", prop_type="over_under", is_live=True
        )
        assert record.player_external_id == "player-lebron-james"
        assert record.stat_type == "points"
        assert record.prop_type == "over_under"
        assert record.is_live is True


class TestPredictionItemPropFields:
    def test_parses_payload_without_prop_fields(self) -> None:
        item = PredictionItem.model_validate(
            {
                "id": str(uuid.uuid4()),
                "market_type": "MONEYLINE",
                "selection": "Los Angeles Lakers ML",
                "predicted_probability": 0.70,
            }
        )
        assert item.player_external_id is None
        assert item.stat_type is None
        assert item.prop_type is None

    def test_parses_payload_with_prop_fields(self) -> None:
        item = PredictionItem.model_validate(
            {
                "id": str(uuid.uuid4()),
                "market_type": "PLAYER_PROP",
                "selection": "LeBron James Over 27.5 Points",
                "side": "OVER",
                "predicted_probability": 0.61,
                "player_external_id": "player-lebron-james",
                "stat_type": "points",
                "prop_type": "over_under",
            }
        )
        assert item.player_external_id == "player-lebron-james"
        assert item.stat_type == "points"
        assert item.prop_type == "over_under"


class TestLineSnapshotPropFields:
    def test_parses_payload_without_prop_fields(self) -> None:
        snapshot = LineSnapshot.model_validate({"id": str(uuid.uuid4()), "game_id": "ext-game-1"})
        assert snapshot.player_external_id is None
        assert snapshot.stat_type is None
        assert snapshot.prop_type is None

    def test_parses_payload_with_prop_fields(self) -> None:
        snapshot = LineSnapshot.model_validate(
            {
                "id": str(uuid.uuid4()),
                "game_id": "ext-game-1",
                "market_type": "PLAYER_PROP",
                "selection": "LeBron James Over 27.5 Points",
                "side": "OVER",
                "line_value": 27.5,
                "odds_american": -115,
                "player_external_id": "player-lebron-james",
                "stat_type": "points",
                "prop_type": "over_under",
            }
        )
        assert snapshot.player_external_id == "player-lebron-james"
        assert snapshot.stat_type == "points"
        assert snapshot.prop_type == "over_under"
