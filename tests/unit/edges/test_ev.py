"""Golden-value tests from algorithms/edge-detection.md section 2."""

import pytest

from agent.edges.ev import calculate_ev, calculate_ev_pct, meets_ev_threshold, min_ev_pct_for_league


class TestCalculateEV:
    def test_doc_example_58pct_at_minus120(self) -> None:
        # decimal = 1.833..., EV = 0.58 * 1.833 - 1.0 = 0.063 (6.3%)
        assert calculate_ev(0.58, -120) == pytest.approx(0.063, abs=1e-3)
        assert calculate_ev_pct(0.58, -120) == pytest.approx(6.3, abs=0.1)

    def test_negative_ev(self) -> None:
        # 50% at -110 is a losing bet (vig)
        assert calculate_ev(0.50, -110) == pytest.approx(-0.0455, abs=1e-3)

    def test_zero_ev_at_fair_odds(self) -> None:
        # 40% at +150 (decimal 2.5) is exactly break-even
        assert calculate_ev(0.40, 150) == pytest.approx(0.0)


class TestThresholds:
    def test_league_minimums(self) -> None:
        assert min_ev_pct_for_league("NBA") == 3.0
        assert min_ev_pct_for_league("NFL") == 3.0
        assert min_ev_pct_for_league("MLB") == 2.5
        assert min_ev_pct_for_league("NCAA_BB") == 2.0
        assert min_ev_pct_for_league("NCAA_FB") == 2.0
        assert min_ev_pct_for_league("NCAA_BSB") == 2.0

    def test_unknown_league_uses_default(self) -> None:
        assert min_ev_pct_for_league("XFL") == 3.0

    def test_meets_threshold(self) -> None:
        assert meets_ev_threshold(3.0, "NBA")
        assert not meets_ev_threshold(2.9, "NBA")
        assert meets_ev_threshold(2.0, "NCAA_BB")
        assert meets_ev_threshold(2.5, "MLB")
        assert not meets_ev_threshold(2.4, "MLB")

    def test_case_insensitive_league(self) -> None:
        assert min_ev_pct_for_league("nba") == 3.0
