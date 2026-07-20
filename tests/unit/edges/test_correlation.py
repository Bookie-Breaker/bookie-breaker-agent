"""Parlay correlation math (algorithms/edge-detection.md section 5)."""

import math

import pytest

from agent.edges import (
    american_to_decimal,
    correlated_kelly,
    correlated_parlay_ev,
    correlation_prior,
    estimate_correlation,
    kelly_fraction,
    multi_leg_parlay_prob,
    scaled_joint_probability,
)


class TestEstimateCorrelation:
    def test_perfectly_correlated(self) -> None:
        assert estimate_correlation([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1.0)

    def test_perfectly_anticorrelated(self) -> None:
        assert estimate_correlation([0, 1, 0, 1], [1, 0, 1, 0]) == pytest.approx(-1.0)

    def test_independent(self) -> None:
        assert estimate_correlation([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)

    def test_zero_variance_returns_zero(self) -> None:
        assert estimate_correlation([1, 1, 1], [0, 1, 0]) == 0.0
        assert estimate_correlation([0, 1, 0], [1, 1, 1]) == 0.0

    def test_known_phi_value(self) -> None:
        # 2x2 table: n11=3, n10=1, n01=1, n00=3 over 8 observations
        outcomes_a = [1, 1, 1, 1, 0, 0, 0, 0]
        outcomes_b = [1, 1, 1, 0, 1, 0, 0, 0]
        # phi = (3*3 - 1*1) / sqrt(4*4*4*4) = 8/16 = 0.5
        assert estimate_correlation(outcomes_a, outcomes_b) == pytest.approx(0.5)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            estimate_correlation([0, 1], [0, 1, 0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            estimate_correlation([], [])


class TestCorrelatedParlayEv:
    def test_doc_formula_without_parlay_odds(self) -> None:
        prob_a, prob_b, rho = 0.55, 0.60, 0.15
        result = correlated_parlay_ev(prob_a, prob_b, -110, -110, rho)

        expected_joint = prob_a * prob_b + rho * math.sqrt(prob_a * 0.45 * prob_b * 0.40)
        expected_decimal = american_to_decimal(-110) ** 2
        assert result["joint_probability"] == pytest.approx(expected_joint)
        assert result["independent_probability"] == pytest.approx(0.33)
        assert result["correlation_edge"] == pytest.approx(expected_joint - 0.33)
        assert result["parlay_decimal_odds"] == pytest.approx(expected_decimal)
        assert result["ev"] == pytest.approx(expected_joint * expected_decimal - 1.0)
        assert result["ev_pct"] == pytest.approx(result["ev"] * 100)
        assert "implied_probability" not in result

    def test_doc_formula_with_sgp_odds(self) -> None:
        result = correlated_parlay_ev(0.55, 0.60, -110, -110, 0.15, parlay_odds=250)

        expected_joint = 0.33 + 0.15 * math.sqrt(0.55 * 0.45 * 0.60 * 0.40)
        assert result["implied_probability"] == pytest.approx(100 / 350)
        assert result["ev"] == pytest.approx(expected_joint * 3.5 - 1.0)
        assert "parlay_decimal_odds" not in result

    def test_zero_rho_matches_independence(self) -> None:
        result = correlated_parlay_ev(0.5, 0.5, 100, 100, 0.0)
        assert result["joint_probability"] == pytest.approx(0.25)
        assert result["correlation_edge"] == pytest.approx(0.0)

    def test_frechet_upper_clamp(self) -> None:
        # raw joint = 0.18 + sqrt(0.09 * 0.16) = 0.30 > min(0.9, 0.2)
        result = correlated_parlay_ev(0.9, 0.2, -110, -110, 1.0)
        assert result["joint_probability"] == pytest.approx(0.2)

    def test_frechet_lower_clamp(self) -> None:
        # raw joint = 0.81 - 0.09 = 0.72 < max(0, 0.9 + 0.9 - 1) = 0.8
        result = correlated_parlay_ev(0.9, 0.9, -110, -110, -1.0)
        assert result["joint_probability"] == pytest.approx(0.8)


class TestMultiLegParlayProb:
    def test_no_correlations_is_product(self) -> None:
        assert multi_leg_parlay_prob([0.6, 0.55, 0.5], {}) == pytest.approx(0.6 * 0.55 * 0.5)

    def test_first_order_adjustment(self) -> None:
        probs = [0.6, 0.55, 0.5]
        base = 0.6 * 0.55 * 0.5
        pair_adj = 0.1 * math.sqrt(0.6 * 0.4 * 0.55 * 0.45)
        expected = base + pair_adj * (base / (0.6 * 0.55))
        assert multi_leg_parlay_prob(probs, {(0, 1): 0.1}) == pytest.approx(expected)

    def test_two_legs_matches_correlated_parlay_ev(self) -> None:
        joint = multi_leg_parlay_prob([0.55, 0.60], {(0, 1): 0.15})
        assert joint == pytest.approx(correlated_parlay_ev(0.55, 0.60, -110, -110, 0.15)["joint_probability"])

    def test_frechet_clamp(self) -> None:
        # extreme rho drives the approximation past min(p_i)
        assert multi_leg_parlay_prob([0.9, 0.2], {(0, 1): 1.0}) == pytest.approx(0.2)


class TestScaledJointProbability:
    def test_identity_when_calibrated_equals_sim(self) -> None:
        assert scaled_joint_probability(0.42, [0.7, 0.6], [0.7, 0.6]) == pytest.approx(0.42)

    def test_scales_by_marginal_ratios(self) -> None:
        expected = 0.42 * (0.72 / 0.70) * (0.57 / 0.60)
        assert scaled_joint_probability(0.42, [0.70, 0.60], [0.72, 0.57]) == pytest.approx(expected)

    def test_zero_sim_marginal_is_guarded(self) -> None:
        # floored at 1e-6, then Frechet-clamped to min(calibrated)
        assert scaled_joint_probability(0.1, [0.0, 0.6], [0.5, 0.6]) == pytest.approx(0.5)

    def test_frechet_clamp_on_upscale(self) -> None:
        # 0.55 * (0.9/0.5) = 0.99 > min(0.9, 0.9)
        assert scaled_joint_probability(0.55, [0.5, 0.9], [0.9, 0.9]) == pytest.approx(0.9)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            scaled_joint_probability(0.4, [0.7], [0.7, 0.6])


class TestCorrelationPrior:
    def test_spread_plus_over(self) -> None:
        assert correlation_prior("SPREAD", "HOME", "TOTAL", "OVER", same_game=True) == pytest.approx(0.15)

    def test_spread_plus_under_flips_sign(self) -> None:
        assert correlation_prior("SPREAD", "HOME", "TOTAL", "UNDER", same_game=True) == pytest.approx(-0.15)

    def test_moneyline_plus_over(self) -> None:
        assert correlation_prior("MONEYLINE", "AWAY", "TOTAL", "OVER", same_game=True) == pytest.approx(0.15)

    def test_moneyline_spread_same_side(self) -> None:
        assert correlation_prior("MONEYLINE", "HOME", "SPREAD", "HOME", same_game=True) == pytest.approx(0.30)

    def test_moneyline_spread_opposite_sides_flips_sign(self) -> None:
        assert correlation_prior("MONEYLINE", "HOME", "SPREAD", "AWAY", same_game=True) == pytest.approx(-0.30)

    def test_moneyline_plus_player_prop_over(self) -> None:
        assert correlation_prior("MONEYLINE", "HOME", "PLAYER_PROP", "OVER", same_game=True) == pytest.approx(0.20)

    def test_moneyline_plus_player_prop_under_flips_sign(self) -> None:
        assert correlation_prior("MONEYLINE", "HOME", "PLAYER_PROP", "UNDER", same_game=True) == pytest.approx(-0.20)

    def test_cross_game_defaults_to_zero(self) -> None:
        assert correlation_prior("SPREAD", "HOME", "TOTAL", "OVER", same_game=False) == 0.0
        assert correlation_prior("MONEYLINE", "HOME", "MONEYLINE", "HOME", same_game=False) == 0.0

    def test_unknown_pair_defaults_to_zero(self) -> None:
        assert correlation_prior("MONEYLINE", "HOME", "MONEYLINE", "HOME", same_game=True) == 0.0

    def test_same_market_same_direction_totals(self) -> None:
        assert correlation_prior("TOTAL", "OVER", "TOTAL", "OVER", same_game=True) == pytest.approx(0.45)

    def test_opposite_sides_same_market_raises(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive|cannot be parlayed"):
            correlation_prior("TOTAL", "OVER", "TOTAL", "UNDER", same_game=True)
        with pytest.raises(ValueError, match="cannot be parlayed"):
            correlation_prior("MONEYLINE", "HOME", "MONEYLINE", "AWAY", same_game=True)

    def test_case_insensitive(self) -> None:
        assert correlation_prior("spread", "home", "total", "over", same_game=True) == pytest.approx(0.15)


class TestPropPriorSignConventions:
    """Phase 7 Wave 4: team-agreement x prop-direction signs.

    rho(ML/SPREAD side S, prop of player P) = base * agree(S, team(P)) *
    dir(prop side), with agree = +1 same team / -1 opposite team (unknown
    team assumes the bet-on team) and dir = +1 OVER/YES, -1 UNDER/NO.
    """

    def test_ml_same_team_yes_positive(self) -> None:
        rho = correlation_prior("MONEYLINE", "HOME", "PLAYER_PROP", "YES", same_game=True, player_team_b="HOME")
        assert rho == pytest.approx(0.20)

    def test_ml_opposite_team_flips_sign(self) -> None:
        rho = correlation_prior("MONEYLINE", "AWAY", "PLAYER_PROP", "YES", same_game=True, player_team_b="HOME")
        assert rho == pytest.approx(-0.20)

    def test_no_side_flips_direction(self) -> None:
        rho = correlation_prior("MONEYLINE", "HOME", "PLAYER_PROP", "NO", same_game=True, player_team_b="HOME")
        assert rho == pytest.approx(-0.20)

    def test_opposite_team_under_double_flip(self) -> None:
        rho = correlation_prior("MONEYLINE", "AWAY", "PLAYER_PROP", "UNDER", same_game=True, player_team_b="HOME")
        assert rho == pytest.approx(0.20)

    def test_prop_leg_first_argument_order_irrelevant(self) -> None:
        rho = correlation_prior("PLAYER_PROP", "YES", "SPREAD", "AWAY", same_game=True, player_team_a="AWAY")
        assert rho == pytest.approx(0.15)

    def test_draw_side_has_no_prior(self) -> None:
        rho = correlation_prior("MONEYLINE", "DRAW", "PLAYER_PROP", "YES", same_game=True, player_team_b="HOME")
        assert rho == 0.0

    def test_unknown_player_team_keeps_wave1_sign(self) -> None:
        assert correlation_prior("MONEYLINE", "HOME", "PLAYER_PROP", "YES", same_game=True) == pytest.approx(0.20)

    def test_two_players_props_do_not_raise_and_default_zero(self) -> None:
        rho = correlation_prior(
            "PLAYER_PROP",
            "YES",
            "PLAYER_PROP",
            "NO",
            same_game=True,
            player_a="bukayo-saka",
            player_b="cole-palmer",
            stat_a="player_goal_scorer_anytime",
            stat_b="player_goal_scorer_anytime",
        )
        assert rho == 0.0

    def test_same_player_same_stat_opposite_sides_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be parlayed"):
            correlation_prior(
                "PLAYER_PROP",
                "YES",
                "PLAYER_PROP",
                "NO",
                same_game=True,
                player_a="bukayo-saka",
                player_b="bukayo-saka",
                stat_a="player_goal_scorer_anytime",
                stat_b="player_goal_scorer_anytime",
            )


class TestCorrelatedKelly:
    def test_matches_kelly_fraction_for_single_leg(self) -> None:
        for odds in (-140, -110, 120, 250):
            assert correlated_kelly(0.55, american_to_decimal(odds)) == pytest.approx(kelly_fraction(0.55, odds))

    def test_no_edge_returns_zero(self) -> None:
        # joint 0.25 at decimal 3.0 -> full Kelly negative
        assert correlated_kelly(0.25, 3.0) == 0.0

    def test_capped_at_max_bet_pct(self) -> None:
        assert correlated_kelly(0.9, 10.0) == 0.05

    def test_quarter_kelly_default(self) -> None:
        # b = 2.0, full Kelly = (2*0.4 - 0.6) / 2 = 0.1 -> quarter = 0.025
        assert correlated_kelly(0.4, 3.0) == pytest.approx(0.025)

    def test_degenerate_odds_return_zero(self) -> None:
        assert correlated_kelly(0.9, 1.0) == 0.0
        assert correlated_kelly(0.9, 0.5) == 0.0
