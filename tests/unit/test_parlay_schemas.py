"""Parlay request schema validation (leg bounds, market restrictions)."""

import pytest
from pydantic import ValidationError

from agent.api.schemas import ParlayEvaluateRequest, ParlayLegRequest


def leg(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {"game_external_id": "ext-game-1", "market_type": "MONEYLINE", "side": "HOME"}
    values.update(overrides)
    return values


class TestParlayLegRequest:
    def test_team_markets_accepted_and_uppercased(self) -> None:
        for market in ("SPREAD", "total", "Moneyline"):
            assert ParlayLegRequest.model_validate(leg(market_type=market)).market_type == market.upper()

    def test_prop_market_rejected_with_wave3_message(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ParlayLegRequest.model_validate(leg(market_type="PLAYER_PROP"))
        assert "Wave 3" in str(excinfo.value)
        assert "team markets only" in str(excinfo.value)

    def test_unknown_market_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ParlayLegRequest.model_validate(leg(market_type="FUTURE"))

    def test_optional_fields_default(self) -> None:
        parsed = ParlayLegRequest.model_validate(leg())
        assert parsed.line_value is None
        assert parsed.sportsbook_key is None


class TestParlayEvaluateRequest:
    def test_minimum_two_legs(self) -> None:
        with pytest.raises(ValidationError):
            ParlayEvaluateRequest.model_validate({"legs": [leg()]})

    def test_maximum_six_legs(self) -> None:
        with pytest.raises(ValidationError):
            ParlayEvaluateRequest.model_validate({"legs": [leg() for _ in range(7)]})

    def test_valid_request_defaults(self) -> None:
        parsed = ParlayEvaluateRequest.model_validate({"legs": [leg(), leg(market_type="TOTAL", side="OVER")]})
        assert parsed.parlay_odds_american is None
        assert parsed.persist is False

    def test_parlay_odds_and_persist_roundtrip(self) -> None:
        parsed = ParlayEvaluateRequest.model_validate(
            {
                "legs": [leg(), leg(market_type="TOTAL", side="OVER", line_value=220.5)],
                "parlay_odds_american": 264,
                "persist": True,
            }
        )
        assert parsed.parlay_odds_american == 264
        assert parsed.persist is True
