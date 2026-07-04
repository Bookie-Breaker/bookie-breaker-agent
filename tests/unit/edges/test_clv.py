"""Golden-value tests from algorithms/edge-detection.md section 4 (CLV)."""

import pytest

from agent.edges.clv import calculate_clv


class TestCalculateCLV:
    def test_doc_example_minus105_closes_minus115(self) -> None:
        # Bet at -105 (51.2%), closes -115 (53.5%) -> ~2.3% CLV
        assert calculate_clv(-105, -115) == pytest.approx(2.3, abs=0.05)

    def test_negative_clv_when_line_moves_against(self) -> None:
        assert calculate_clv(-115, -105) == pytest.approx(-2.3, abs=0.05)

    def test_zero_clv_when_line_unchanged(self) -> None:
        assert calculate_clv(-110, -110) == 0.0

    def test_underdog_odds(self) -> None:
        # Bet +150 (40.0%), closes +130 (43.5%) -> positive CLV
        assert calculate_clv(150, 130) == pytest.approx(3.48, abs=0.05)
