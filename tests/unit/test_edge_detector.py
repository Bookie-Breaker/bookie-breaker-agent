"""EdgeDetector: devig grouping, best-price selection, and graceful skips."""

from agent.core.edge_detector import EdgeDetector
from agent.edges import american_to_implied_prob, calculate_ev_pct, multiplicative_devig
from tests.unit.factories import make_game, make_line, make_prediction


def detector() -> EdgeDetector:
    return EdgeDetector(devig_method="multiplicative", kelly_multiplier=0.25, max_bet_pct=0.05)


def moneyline_pair(book: str, home_odds: int, away_odds: int) -> list:
    return [
        make_line(sportsbook_key=book, selection="Los Angeles Lakers", side="HOME", odds_american=home_odds),
        make_line(sportsbook_key=book, selection="Boston Celtics", side="AWAY", odds_american=away_odds),
    ]


class TestDevigGrouping:
    def test_moneyline_edge_uses_devigged_implied(self) -> None:
        game = make_game()
        predictions = [make_prediction(selection="Los Angeles Lakers ML", predicted_probability=0.70)]
        lines = moneyline_pair("draftkings", -150, +130)

        candidates = detector().detect(game, "ext-game-1", predictions, lines)

        assert len(candidates) == 1
        edge = candidates[0]
        expected_implied, _ = multiplicative_devig(american_to_implied_prob(-150), american_to_implied_prob(+130))
        assert edge.side == "HOME"
        assert abs(edge.implied_probability - round(expected_implied, 5)) < 1e-9
        assert abs(edge.edge_percentage - round((0.70 - expected_implied) * 100, 3)) < 1e-6
        assert abs(edge.expected_value - round(calculate_ev_pct(0.70, -150) / 100, 5)) < 1e-9
        assert edge.meets_threshold
        assert edge.devig_method == "multiplicative"

    def test_opposite_side_uses_complement_probability(self) -> None:
        # only a home prediction exists; a mispriced away side (bigger
        # complement than devigged implied) still yields an away edge
        game = make_game()
        predictions = [make_prediction(selection="Los Angeles Lakers ML", predicted_probability=0.45)]
        lines = moneyline_pair("draftkings", -150, +130)

        candidates = detector().detect(game, "ext-game-1", predictions, lines)

        sides = {c.side for c in candidates}
        assert "AWAY" in sides
        away = next(c for c in candidates if c.side == "AWAY")
        assert abs(away.predicted_probability - 0.55) < 1e-9

    def test_missing_opposite_side_skips_market(self) -> None:
        # one-sided total: no devig possible, market skipped gracefully
        game = make_game()
        predictions = [make_prediction(market_type="TOTAL", selection="Over 224.5", predicted_probability=0.65)]
        lines = [
            make_line(market_type="TOTAL", selection="Over 224.5", side="OVER", line_value=224.5, odds_american=-110)
        ]
        assert detector().detect(game, "ext-game-1", predictions, lines) == []

    def test_negative_vig_pair_skipped(self) -> None:
        # raw probabilities summing below 1.0 (arbitrage) cannot be devigged
        game = make_game()
        predictions = [make_prediction(selection="Los Angeles Lakers ML", predicted_probability=0.70)]
        lines = moneyline_pair("draftkings", +105, +105)
        assert detector().detect(game, "ext-game-1", predictions, lines) == []

    def test_spread_pair_grouped_by_absolute_line(self) -> None:
        game = make_game()
        predictions = [
            make_prediction(market_type="SPREAD", selection="Los Angeles Lakers -3.5", predicted_probability=0.58)
        ]
        lines = [
            make_line(
                market_type="SPREAD",
                selection="Los Angeles Lakers -3.5",
                side="HOME",
                line_value=-3.5,
                odds_american=-110,
            ),
            make_line(
                market_type="SPREAD",
                selection="Boston Celtics +3.5",
                side="AWAY",
                line_value=3.5,
                odds_american=-110,
            ),
        ]
        candidates = detector().detect(game, "ext-game-1", predictions, lines)
        assert len(candidates) == 1
        assert candidates[0].market_type == "SPREAD"
        assert candidates[0].line_value == -3.5
        # symmetric -110/-110 devigs to 0.5
        assert abs(candidates[0].implied_probability - 0.5) < 1e-9


class TestBestPriceSelection:
    def test_best_price_across_books_wins(self) -> None:
        game = make_game()
        predictions = [make_prediction(selection="Los Angeles Lakers ML", predicted_probability=0.70)]
        lines = moneyline_pair("draftkings", -150, +130) + moneyline_pair("fanduel", -140, +120)

        candidates = detector().detect(game, "ext-game-1", predictions, lines)

        home = [c for c in candidates if c.side == "HOME"]
        assert len(home) == 1
        assert home[0].sportsbook_key == "fanduel"
        assert home[0].odds_american == -140

    def test_no_prediction_no_candidates(self) -> None:
        game = make_game()
        lines = moneyline_pair("draftkings", -150, +130)
        assert detector().detect(game, "ext-game-1", [], lines) == []

    def test_no_edge_when_market_beats_model(self) -> None:
        game = make_game()
        predictions = [make_prediction(selection="Los Angeles Lakers ML", predicted_probability=0.50)]
        lines = moneyline_pair("draftkings", -150, +130)
        candidates = detector().detect(game, "ext-game-1", predictions, lines)
        assert all(c.side != "HOME" for c in candidates)


class TestMetadata:
    def test_expires_at_is_scheduled_start(self) -> None:
        game = make_game(scheduled_start="2026-07-05T00:30:00Z")
        predictions = [make_prediction(selection="Los Angeles Lakers ML", predicted_probability=0.70)]
        candidates = detector().detect(game, "ext-game-1", predictions, moneyline_pair("draftkings", -150, +130))
        assert candidates[0].expires_at.isoformat() == "2026-07-05T00:30:00+00:00"

    def test_unparseable_start_skips_game(self) -> None:
        game = make_game(scheduled_start="not-a-date")
        predictions = [make_prediction(selection="Los Angeles Lakers ML", predicted_probability=0.70)]
        assert detector().detect(game, "ext-game-1", predictions, moneyline_pair("draftkings", -150, +130)) == []

    def test_prediction_and_simulation_ids_recorded(self) -> None:
        game = make_game()
        prediction = make_prediction(selection="Los Angeles Lakers ML", predicted_probability=0.70)
        sim_id = "0a627939-14b7-4b34-8d3a-7a2f8b9d3f11"
        candidates = detector().detect(
            game, "ext-game-1", [prediction], moneyline_pair("draftkings", -150, +130), simulation_run_id=sim_id
        )
        assert candidates[0].prediction_id == prediction.id
        assert candidates[0].simulation_run_id == sim_id
