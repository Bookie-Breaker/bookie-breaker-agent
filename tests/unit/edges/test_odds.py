"""Golden-value tests from algorithms/edge-detection.md section 1."""

import pytest

from agent.edges.odds import american_to_decimal, american_to_implied_prob, decimal_to_american


class TestAmericanToImpliedProb:
    def test_favorite_minus_150(self) -> None:
        assert american_to_implied_prob(-150) == pytest.approx(0.600)

    def test_underdog_plus_150(self) -> None:
        assert american_to_implied_prob(150) == pytest.approx(0.400)

    def test_standard_juice_minus_110(self) -> None:
        assert american_to_implied_prob(-110) == pytest.approx(110 / 210)

    def test_even_odds(self) -> None:
        assert american_to_implied_prob(100) == pytest.approx(0.500)
        assert american_to_implied_prob(-100) == pytest.approx(0.500)

    def test_zero_odds_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be 0"):
            american_to_implied_prob(0)


class TestAmericanToDecimal:
    def test_favorite(self) -> None:
        assert american_to_decimal(-150) == pytest.approx(1 + 100 / 150)

    def test_underdog(self) -> None:
        assert american_to_decimal(150) == pytest.approx(2.5)

    def test_minus_110(self) -> None:
        assert american_to_decimal(-110) == pytest.approx(1.909, abs=1e-3)


class TestDecimalToAmerican:
    def test_underdog(self) -> None:
        assert decimal_to_american(2.5) == 150

    def test_favorite(self) -> None:
        assert decimal_to_american(1.909) == pytest.approx(-110, abs=1)

    def test_round_trip(self) -> None:
        for odds in (-250, -150, -110, 100, 120, 150, 300):
            assert decimal_to_american(american_to_decimal(odds)) == odds

    def test_invalid_decimal_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than 1.0"):
            decimal_to_american(1.0)
