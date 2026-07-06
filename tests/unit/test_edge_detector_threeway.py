"""EdgeDetector three-way MONEYLINE grouping and side-based prediction
matching (Phase 6 Wave 0, ADR-027)."""

from agent.core.edge_detector import EdgeDetector
from agent.edges import american_to_implied_prob, multiplicative_devig_n
from tests.unit.factories import make_game, make_line, make_prediction


def detector() -> EdgeDetector:
    return EdgeDetector(devig_method="multiplicative", kelly_multiplier=0.25, max_bet_pct=0.05)


def soccer_game():
    return make_game(league="FIFA_WC")


def three_way_moneyline(book: str = "draftkings", home: int = 150, draw: int = 220, away: int = 200) -> list:
    return [
        make_line(sportsbook_key=book, selection="Los Angeles Lakers", side="HOME", odds_american=home),
        make_line(sportsbook_key=book, selection="Draw", side="DRAW", odds_american=draw),
        make_line(sportsbook_key=book, selection="Boston Celtics", side="AWAY", odds_american=away),
    ]


def sided_predictions(home: float = 0.45, draw: float = 0.32, away: float = 0.30) -> list:
    return [
        make_prediction(selection="Los Angeles Lakers ML", side="HOME", predicted_probability=home),
        make_prediction(selection="Draw", side="DRAW", predicted_probability=draw),
        make_prediction(selection="Boston Celtics ML", side="AWAY", predicted_probability=away),
    ]


class TestThreeWayGrouping:
    def test_complete_three_way_moneyline_detected(self) -> None:
        candidates = detector().detect(soccer_game(), "ext-game-1", sided_predictions(), three_way_moneyline())

        by_side = {c.side: c for c in candidates}
        assert "HOME" in by_side
        assert "DRAW" in by_side
        # away prediction (0.30) is below the devigged implied: no edge
        assert "AWAY" not in by_side

        raws = [american_to_implied_prob(odds) for odds in (150, 220, 200)]
        implied_home, implied_draw, _ = multiplicative_devig_n(raws)
        assert abs(by_side["HOME"].implied_probability - round(implied_home, 5)) < 1e-9
        assert abs(by_side["DRAW"].implied_probability - round(implied_draw, 5)) < 1e-9

    def test_draw_candidate_flows_through_ev_kelly_quality(self) -> None:
        candidates = detector().detect(soccer_game(), "ext-game-1", sided_predictions(), three_way_moneyline())
        draw = next(c for c in candidates if c.side == "DRAW")
        assert draw.market_type == "MONEYLINE"
        assert draw.league == "FIFA_WC"
        # EV = 0.32 * 3.2 - 1 = 2.4%: positive but below FIFA_WC's 4% floor
        assert draw.expected_value > 0
        assert not draw.meets_threshold
        assert draw.kelly_fraction >= 0
        assert 0 <= draw.confidence <= 1

    def test_two_of_three_moneyline_skipped(self) -> None:
        lines = three_way_moneyline()[:2]  # HOME + DRAW, no AWAY
        assert detector().detect(soccer_game(), "ext-game-1", sided_predictions(), lines) == []

    def test_three_way_sides_on_non_moneyline_market_skipped(self) -> None:
        lines = [
            make_line(market_type="SPREAD", selection="A", side=side, line_value=0.5, odds_american=150)
            for side in ("HOME", "DRAW", "AWAY")
        ]
        predictions = [make_prediction(market_type="SPREAD", selection="A", side="HOME", predicted_probability=0.6)]
        assert detector().detect(soccer_game(), "ext-game-1", predictions, lines) == []

    def test_nba_two_way_pair_still_detected(self) -> None:
        game = make_game()
        predictions = [make_prediction(selection="Los Angeles Lakers ML", predicted_probability=0.70)]
        lines = [
            make_line(selection="Los Angeles Lakers", side="HOME", odds_american=-150),
            make_line(selection="Boston Celtics", side="AWAY", odds_american=130),
        ]
        candidates = detector().detect(game, "ext-game-1", predictions, lines)
        assert {c.side for c in candidates} == {"HOME"}


class TestSideBasedMatching:
    def test_side_match_wins_over_selection_mismatch(self) -> None:
        # selections do not line up with the lines-service strings at all;
        # the (market_type, side) key still matches each row
        predictions = [
            make_prediction(selection="home-team-to-win", side="HOME", predicted_probability=0.45),
            make_prediction(selection="match-drawn", side="DRAW", predicted_probability=0.32),
            make_prediction(selection="away-team-to-win", side="AWAY", predicted_probability=0.30),
        ]
        candidates = detector().detect(soccer_game(), "ext-game-1", predictions, three_way_moneyline())
        assert {c.side for c in candidates} == {"HOME", "DRAW"}

    def test_selection_fallback_when_predictions_carry_no_side(self) -> None:
        predictions = [
            make_prediction(selection="Los Angeles Lakers ML", predicted_probability=0.45),
            make_prediction(selection="Draw", predicted_probability=0.32),
            make_prediction(selection="Boston Celtics ML", predicted_probability=0.30),
        ]
        candidates = detector().detect(soccer_game(), "ext-game-1", predictions, three_way_moneyline())
        assert {c.side for c in candidates} == {"HOME", "DRAW"}

    def test_no_complement_fallback_in_three_way_groups(self) -> None:
        # Only a HOME prediction exists. In a two-way market the AWAY side
        # would get 1 - P; in a three-way group the DRAW and AWAY sides must
        # be skipped instead (1 - P covers two outcomes, not one).
        predictions = [make_prediction(selection="Los Angeles Lakers ML", side="HOME", predicted_probability=0.30)]
        candidates = detector().detect(soccer_game(), "ext-game-1", predictions, three_way_moneyline())
        assert candidates == []

    def test_complement_fallback_still_applies_to_two_way(self) -> None:
        game = make_game()
        predictions = [make_prediction(selection="Los Angeles Lakers ML", predicted_probability=0.45)]
        lines = [
            make_line(selection="Los Angeles Lakers", side="HOME", odds_american=-150),
            make_line(selection="Boston Celtics", side="AWAY", odds_american=130),
        ]
        candidates = detector().detect(game, "ext-game-1", predictions, lines)
        away = next(c for c in candidates if c.side == "AWAY")
        assert abs(away.predicted_probability - 0.55) < 1e-9
