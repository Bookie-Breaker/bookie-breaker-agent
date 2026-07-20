"""EdgeDetector prop paths: per-player grouping, single-sided YES/NO, matching."""

from datetime import UTC, datetime
from typing import Any

from agent.clients.lines import LineSnapshot
from agent.clients.prediction import PredictionItem
from agent.core.edge_detector import EdgeDetector
from agent.edges import (
    american_to_implied_prob,
    calculate_ev_pct,
    edge_quality_score,
    market_efficiency,
    multiplicative_devig,
)
from tests.unit.factories import make_game, make_line, make_prediction

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
NOW_ISO = NOW.isoformat().replace("+00:00", "Z")

MBAPPE = "kylian-mbappe"
GIROUD = "olivier-giroud"


def detector(haircut: float = 0.06) -> EdgeDetector:
    return EdgeDetector(
        devig_method="multiplicative", kelly_multiplier=0.25, max_bet_pct=0.05, single_sided_vig_haircut=haircut
    )


def prop_line(player: str, stat: str, side: str, odds: int, line: float | None, **overrides: Any) -> LineSnapshot:
    defaults: dict[str, Any] = {
        "market_type": "PLAYER_PROP",
        "selection": f"{player} {side.title()} {line} {stat}",
        "side": side,
        "line_value": line,
        "player_external_id": player,
        "stat_type": stat,
        "prop_type": "YES_NO" if side in ("YES", "NO") else "OVER_UNDER",
        "odds_american": odds,
        "timestamp": NOW_ISO,
    }
    defaults.update(overrides)
    return make_line(**defaults)


def prop_prediction(player: str, stat: str, side: str, line: float | None, probability: float) -> PredictionItem:
    return make_prediction(
        market_type="PLAYER_PROP",
        selection=f"{player} {side.title()} {line} {stat}",
        side=side,
        predicted_probability=probability,
        player_external_id=player,
        stat_type=stat,
        prop_type="YES_NO" if side in ("YES", "NO") else "OVER_UNDER",
        prop_line=line,
        confidence_lower=probability - 0.04,
        confidence_upper=probability + 0.04,
    )


def shots_pair(player: str, over_odds: int = -110, under_odds: int = -110, book: str = "draftkings") -> list:
    return [
        prop_line(player, "player_shots", "OVER", over_odds, 2.5, sportsbook_key=book),
        prop_line(player, "player_shots", "UNDER", under_odds, 2.5, sportsbook_key=book),
    ]


class TestPropGrouping:
    def test_over_under_pair_devigs_per_player_stat_line(self) -> None:
        game = make_game(league="EPL")
        predictions = [prop_prediction(MBAPPE, "player_shots", "OVER", 2.5, 0.62)]
        lines = shots_pair(MBAPPE, over_odds=-105, under_odds=-115)

        candidates = detector().detect(game, "ext-game-1", predictions, lines, now=NOW)

        assert len(candidates) == 1
        edge = candidates[0]
        expected_implied, _ = multiplicative_devig(american_to_implied_prob(-105), american_to_implied_prob(-115))
        assert edge.market_type == "PLAYER_PROP"
        assert edge.side == "OVER"
        assert edge.player_external_id == MBAPPE
        assert edge.stat_type == "player_shots"
        assert edge.prop_type == "OVER_UNDER"
        assert edge.line_value == 2.5
        assert abs(edge.implied_probability - round(expected_implied, 5)) < 1e-9
        assert edge.devig_method == "multiplicative"

    def test_under_side_uses_complement_of_over_prediction(self) -> None:
        # only an OVER row exists; a mispriced UNDER still yields an edge
        game = make_game(league="EPL")
        predictions = [prop_prediction(MBAPPE, "player_shots", "OVER", 2.5, 0.40)]
        candidates = detector().detect(game, "ext-game-1", predictions, shots_pair(MBAPPE), now=NOW)

        assert [c.side for c in candidates] == ["UNDER"]
        assert abs(candidates[0].predicted_probability - 0.60) < 1e-9
        assert candidates[0].player_external_id == MBAPPE

    def test_two_players_same_stat_do_not_cross_group(self) -> None:
        # one side per player: neither forms a complete pair -> no candidates
        game = make_game(league="EPL")
        predictions = [
            prop_prediction(MBAPPE, "player_shots", "OVER", 2.5, 0.65),
            prop_prediction(GIROUD, "player_shots", "UNDER", 2.5, 0.65),
        ]
        lines = [
            prop_line(MBAPPE, "player_shots", "OVER", -110, 2.5),
            prop_line(GIROUD, "player_shots", "UNDER", -110, 2.5),
        ]
        assert detector().detect(game, "ext-game-1", predictions, lines, now=NOW) == []

    def test_two_players_same_stat_pair_independently(self) -> None:
        game = make_game(league="EPL")
        predictions = [
            prop_prediction(MBAPPE, "player_shots", "OVER", 2.5, 0.62),
            prop_prediction(GIROUD, "player_shots", "OVER", 2.5, 0.60),
        ]
        lines = shots_pair(MBAPPE) + shots_pair(GIROUD)
        candidates = detector().detect(game, "ext-game-1", predictions, lines, now=NOW)

        assert {c.player_external_id for c in candidates} == {MBAPPE, GIROUD}
        for edge in candidates:
            expected = predictions[0] if edge.player_external_id == MBAPPE else predictions[1]
            assert edge.predicted_probability == round(expected.predicted_probability, 5)

    def test_prediction_for_other_player_never_matches(self) -> None:
        game = make_game(league="EPL")
        predictions = [prop_prediction(GIROUD, "player_shots", "OVER", 2.5, 0.70)]
        assert detector().detect(game, "ext-game-1", predictions, shots_pair(MBAPPE), now=NOW) == []

    def test_different_lines_do_not_match(self) -> None:
        game = make_game(league="EPL")
        predictions = [prop_prediction(MBAPPE, "player_shots", "OVER", 3.5, 0.70)]
        assert detector().detect(game, "ext-game-1", predictions, shots_pair(MBAPPE), now=NOW) == []

    def test_best_price_across_books_per_player_prop(self) -> None:
        game = make_game(league="EPL")
        predictions = [prop_prediction(MBAPPE, "player_shots", "OVER", 2.5, 0.62)]
        lines = shots_pair(MBAPPE, over_odds=-115, book="draftkings") + shots_pair(
            MBAPPE, over_odds=-105, book="fanduel"
        )
        candidates = detector().detect(game, "ext-game-1", predictions, lines, now=NOW)

        assert len(candidates) == 1
        assert candidates[0].sportsbook_key == "fanduel"
        assert candidates[0].odds_american == -105

    def test_prop_predictions_never_leak_into_team_markets(self) -> None:
        # a PLAYER_PROP OVER row must not match a TOTAL OVER/UNDER market
        game = make_game(league="EPL")
        predictions = [prop_prediction(MBAPPE, "player_shots", "OVER", 2.5, 0.70)]
        lines = [
            make_line(market_type="TOTAL", selection="Over 2.5", side="OVER", line_value=2.5, odds_american=-110),
            make_line(market_type="TOTAL", selection="Under 2.5", side="UNDER", line_value=2.5, odds_american=-110),
        ]
        assert detector().detect(game, "ext-game-1", predictions, lines, now=NOW) == []


class TestSingleSidedYesNo:
    def yes_line(self, odds: int = +150) -> LineSnapshot:
        return prop_line(MBAPPE, "player_goal_scorer_anytime", "YES", odds, None)

    def yes_prediction(self, probability: float = 0.50) -> PredictionItem:
        return prop_prediction(MBAPPE, "player_goal_scorer_anytime", "YES", None, probability)

    def test_haircut_math_and_devig_method(self) -> None:
        game = make_game(league="EPL")
        candidates = detector().detect(game, "ext-game-1", [self.yes_prediction()], [self.yes_line()], now=NOW)

        assert len(candidates) == 1
        edge = candidates[0]
        # p_true_est = implied_raw / (1 + haircut)
        expected_implied = american_to_implied_prob(+150) / 1.06
        assert abs(edge.implied_probability - round(expected_implied, 5)) < 1e-9
        assert edge.devig_method == "single_sided"
        assert edge.side == "YES"
        assert edge.prop_type == "YES_NO"
        assert edge.player_external_id == MBAPPE

    def test_haircut_is_configurable(self) -> None:
        game = make_game(league="EPL")
        candidates = detector(haircut=0.0).detect(
            game, "ext-game-1", [self.yes_prediction()], [self.yes_line()], now=NOW
        )
        assert candidates[0].implied_probability == round(american_to_implied_prob(+150), 5)

    def test_confidence_penalty_applied(self) -> None:
        game = make_game(league="EPL")
        prediction = self.yes_prediction()
        candidates = detector().detect(game, "ext-game-1", [prediction], [self.yes_line()], now=NOW)

        base = edge_quality_score(
            ev_pct=calculate_ev_pct(prediction.predicted_probability, +150),
            prediction_confidence=0.08,  # 0.46..0.54 CI
            market_efficiency=market_efficiency("EPL", "PLAYER_PROP"),
            line_freshness_hours=0.0,
            model_calibration_error=0.03,
        )
        assert candidates[0].confidence == round(base * 0.8, 3)

    def test_yes_no_pair_devigs_two_way_without_penalty(self) -> None:
        # when both YES and NO are quoted, the normal two-way path applies
        game = make_game(league="EPL")
        lines = [
            prop_line(MBAPPE, "player_goal_scorer_anytime", "YES", +120, None),
            prop_line(MBAPPE, "player_goal_scorer_anytime", "NO", -150, None),
        ]
        candidates = detector().detect(game, "ext-game-1", [self.yes_prediction(0.55)], lines, now=NOW)

        assert len(candidates) == 1
        edge = candidates[0]
        expected_implied, _ = multiplicative_devig(american_to_implied_prob(+120), american_to_implied_prob(-150))
        assert abs(edge.implied_probability - round(expected_implied, 5)) < 1e-9
        assert edge.devig_method == "multiplicative"

    def test_lone_over_prop_still_skipped(self) -> None:
        # single-sidedness is a YES/NO privilege: a lone OVER stays incomplete
        game = make_game(league="EPL")
        predictions = [prop_prediction(MBAPPE, "player_shots", "OVER", 2.5, 0.70)]
        lines = [prop_line(MBAPPE, "player_shots", "OVER", -110, 2.5)]
        assert detector().detect(game, "ext-game-1", predictions, lines, now=NOW) == []

    def test_lone_yes_without_yes_no_prop_type_skipped(self) -> None:
        game = make_game(league="EPL")
        line = prop_line(MBAPPE, "player_goal_scorer_anytime", "YES", +150, None, prop_type="OVER_UNDER")
        assert detector().detect(game, "ext-game-1", [self.yes_prediction()], [line], now=NOW) == []

    def test_no_prediction_no_single_sided_candidate(self) -> None:
        game = make_game(league="EPL")
        assert detector().detect(game, "ext-game-1", [], [self.yes_line()], now=NOW) == []

    def test_meets_threshold_uses_league_min_ev(self) -> None:
        game = make_game(league="EPL")  # min EV 3.5%
        candidates = detector().detect(game, "ext-game-1", [self.yes_prediction(0.50)], [self.yes_line()], now=NOW)
        # EV = 0.5 * 2.5 - 1 = 25% -> actionable
        assert candidates[0].meets_threshold
