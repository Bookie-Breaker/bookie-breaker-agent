"""Parlay correlation math.

Implements algorithms/edge-detection.md section 5: phi-coefficient
correlation estimation, correlation-adjusted joint probabilities for
two-leg and multi-leg parlays, the Monte-Carlo scaled-joint path, and the
documented prior correlation estimates for common parlay shapes.

All joint probabilities are clamped to the Frechet bounds: for events with
marginals p_1..p_n the joint probability of all occurring must lie in
[max(0, sum(p_i) - (n - 1)), min(p_i)]. The correlation adjustment is an
approximation and can otherwise leave that feasible region.
"""

import math
from collections.abc import Mapping, Sequence

from agent.edges.odds import american_to_decimal, american_to_implied_prob

TEAM_MARKETS = frozenset({"SPREAD", "MONEYLINE"})

# Common correlation estimates from algorithms/edge-detection.md section 5
# (midpoints of the documented ranges), keyed by the unordered market pair
# and a same_game flag. Magnitudes only: correlation_prior() applies the
# sign conventions documented there.
#
# Sign conventions (same game):
# - SPREAD/MONEYLINE + TOTAL: covering/winning is positively correlated
#   with the OVER (winning teams score more); the UNDER flips the sign.
# - MONEYLINE + SPREAD: the same team winning and covering are strongly
#   positively correlated; opposite teams flip the sign. The magnitude is
#   capped at 0.30 -- the first-order approximation's validity limit --
#   and the simulation path should be preferred for this pair.
# - MONEYLINE/SPREAD + PLAYER_PROP (Phase 7 Wave 4 sign conventions):
#   team success is positively correlated with a star's OVER/YES (doc: ML +
#   player points over ~ +0.20) WHEN the player plays for the bet-on team;
#   betting the opposite team flips the sign (the player's team losing means
#   fewer touches for the star), and UNDER/NO props flip it again. The
#   player's team comes from the simulation player-distributions payload
#   (player_team_* kwargs); when it is unknown the player is assumed to be
#   on the bet-on team (the Wave 1 behavior). A DRAW moneyline side has no
#   documented prior against player props and returns 0.0.
# - PLAYER_PROP + PLAYER_PROP (two different players, or one player across
#   two stats) has no documented prior and returns 0.0; such pairs never
#   raise the mutual-exclusion error unless they are the same player AND
#   the same stat.
# - Cross-game pairs default to 0.0 (independent matchups). The doc's
#   same-division (+0.035) and weather (+0.10) priors need schedule and
#   weather context this accessor does not have.
CORRELATION_PRIORS: dict[tuple[frozenset[str], bool], float] = {
    (frozenset({"SPREAD", "TOTAL"}), True): 0.15,
    (frozenset({"MONEYLINE", "TOTAL"}), True): 0.15,
    (frozenset({"MONEYLINE", "SPREAD"}), True): 0.30,
    (frozenset({"MONEYLINE", "PLAYER_PROP"}), True): 0.20,
    (frozenset({"SPREAD", "PLAYER_PROP"}), True): 0.15,
    # Same market, same direction, different lines (e.g. OVER 210.5 +
    # OVER 215.5): nested outcomes, strongly correlated. Midpoint of the
    # doc's "over + first-half over" +0.40..+0.50 row as nearest analog.
    (frozenset({"TOTAL"}), True): 0.45,
}


def _frechet_clamp(joint: float, marginals: Sequence[float]) -> float:
    """Clamp a joint probability to its Frechet bounds."""
    lower = max(0.0, sum(marginals) - (len(marginals) - 1))
    upper = min(marginals)
    return min(max(joint, lower), upper)


def estimate_correlation(outcomes_a: Sequence[int], outcomes_b: Sequence[int]) -> float:
    """Phi coefficient between two binary outcome series.

    Equivalent to the Pearson correlation for 0/1 data. Returns 0.0 when
    either series has zero variance (correlation is undefined there).
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("outcome series must have equal length")
    n = len(outcomes_a)
    if n == 0:
        raise ValueError("outcome series must be non-empty")

    mean_a = sum(outcomes_a) / n
    mean_b = sum(outcomes_b) / n
    var_a = sum((a - mean_a) ** 2 for a in outcomes_a)
    var_b = sum((b - mean_b) ** 2 for b in outcomes_b)
    if var_a == 0.0 or var_b == 0.0:
        return 0.0
    covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(outcomes_a, outcomes_b, strict=True))
    return covariance / math.sqrt(var_a * var_b)


def correlated_parlay_ev(
    prob_a: float,
    prob_b: float,
    odds_a: int,
    odds_b: int,
    rho: float,
    parlay_odds: int | None = None,
) -> dict[str, float]:
    """EV of a correlated two-leg parlay (edge-detection.md section 5).

    The correlation-adjusted joint probability is clamped to the Frechet
    bounds [max(0, p_a + p_b - 1), min(p_a, p_b)] -- the adjustment formula
    can otherwise produce infeasible joints for extreme rho.

    With parlay_odds (an offered SGP price), the result carries the price's
    implied_probability; without it, standard multiplicative parlay pricing
    (parlay_decimal_odds) is used.
    """
    joint_prob = prob_a * prob_b + rho * math.sqrt(prob_a * (1 - prob_a) * prob_b * (1 - prob_b))
    joint_prob = _frechet_clamp(joint_prob, (prob_a, prob_b))
    independent_prob = prob_a * prob_b

    if parlay_odds is not None:
        implied_prob = american_to_implied_prob(parlay_odds)
        ev = joint_prob * american_to_decimal(parlay_odds) - 1.0
        return {
            "joint_probability": joint_prob,
            "independent_probability": independent_prob,
            "correlation_edge": joint_prob - independent_prob,
            "implied_probability": implied_prob,
            "ev": ev,
            "ev_pct": ev * 100,
        }

    parlay_decimal = american_to_decimal(odds_a) * american_to_decimal(odds_b)
    parlay_ev = joint_prob * parlay_decimal - 1.0
    return {
        "joint_probability": joint_prob,
        "independent_probability": independent_prob,
        "correlation_edge": joint_prob - independent_prob,
        "parlay_decimal_odds": parlay_decimal,
        "ev": parlay_ev,
        "ev_pct": parlay_ev * 100,
    }


def multi_leg_parlay_prob(
    probs: Sequence[float],
    correlations: Mapping[tuple[int, int], float],
) -> float:
    """Approximate joint probability for a multi-leg correlated parlay.

    First-order approximation per edge-detection.md section 5:

        P(all) ~= prod(P_i)
                + sum_{i<j} rho_ij * sqrt(P_i(1-P_i)P_j(1-P_j)) * prod_{k!=i,j} P_k

    Caveat (from the doc): this breaks down for large correlations
    (rho > 0.3) or same-game parlays with 3+ correlated legs -- prefer the
    simulation path (scaled_joint_probability) there. The result is
    clamped to the Frechet bounds.
    """
    base = math.prod(probs)
    adjustment = 0.0
    for (i, j), rho in correlations.items():
        pair_adj = rho * math.sqrt(probs[i] * (1 - probs[i]) * probs[j] * (1 - probs[j]))
        other_prod = base / (probs[i] * probs[j]) if probs[i] * probs[j] > 0 else 0.0
        adjustment += pair_adj * other_prod
    return _frechet_clamp(base + adjustment, probs)


def scaled_joint_probability(
    sim_joint: float,
    sim_marginals: Sequence[float],
    calibrated_marginals: Sequence[float],
) -> float:
    """Ride the simulation's joint structure with calibrated marginals.

    The Monte-Carlo path for same-game parlays: the simulation provides a
    joint probability and per-leg marginals from the same draw, so its
    correlation structure is exact for the simulated distribution. The ML
    calibration layer adjusts each marginal, and the joint is rescaled by
    the product of those per-leg ratios:

        joint = sim_joint * prod(calibrated_i / sim_i)

    Simulation marginals are floored at 1e-6 to guard division, and the
    result is clamped to the calibrated marginals' Frechet bounds.
    """
    if len(sim_marginals) != len(calibrated_marginals):
        raise ValueError("marginal sequences must have equal length")
    scale = math.prod(cal / max(sim, 1e-6) for sim, cal in zip(sim_marginals, calibrated_marginals, strict=True))
    return _frechet_clamp(sim_joint * scale, calibrated_marginals)


def _total_direction(market_type: str, side: str | None) -> float:
    """+1 for OVER/YES-like outcomes, -1 for UNDER/NO; +1 for team markets."""
    if market_type == "PLAYER_PROP" and side in ("UNDER", "NO"):
        return -1.0
    if market_type == "TOTAL" and side == "UNDER":
        return -1.0
    return 1.0


def correlation_prior(
    market_a: str,
    side_a: str | None,
    market_b: str,
    side_b: str | None,
    same_game: bool,
    *,
    player_a: str | None = None,
    player_b: str | None = None,
    stat_a: str | None = None,
    stat_b: str | None = None,
    player_team_a: str | None = None,
    player_team_b: str | None = None,
) -> float:
    """Prior correlation for a leg pair when no simulation data is available.

    Looks up CORRELATION_PRIORS by the unordered market pair, then applies
    the documented sign conventions (see the table's comment). Cross-game
    pairs return 0.0 (independent matchups default).

    The keyword arguments carry player-prop identity (Phase 7 Wave 4):
    ``player_*``/``stat_*`` scope the mutual-exclusion check to one prop
    instance, and ``player_team_*`` (HOME/AWAY, from the simulation
    player-distributions payload) drives the team-agreement sign for
    team-market x player-prop pairs.

    Raises ValueError for opposite sides of the same market in the same
    game -- those outcomes are mutually exclusive and must never be
    combined into a parlay. For PLAYER_PROP pairs that only applies to the
    same player's same stat; distinct players (or stats) legitimately
    combine with any side mix.
    """
    market_a = market_a.upper()
    market_b = market_b.upper()
    side_a = side_a.upper() if side_a else None
    side_b = side_b.upper() if side_b else None

    same_prop_instance = (
        player_a is not None and player_a == player_b and (stat_a or "").lower() == (stat_b or "").lower()
    )
    if same_game and market_a == market_b and side_a != side_b and (market_a != "PLAYER_PROP" or same_prop_instance):
        raise ValueError(f"opposite sides of the same {market_a} market cannot be parlayed in the same game")
    if not same_game:
        return 0.0

    base = CORRELATION_PRIORS.get((frozenset({market_a, market_b}), True))
    if base is None:
        return 0.0

    # Team-side agreement drives the sign for team-market pairs
    # (ML + spread); the total direction drives it for total/prop pairs.
    if market_a in TEAM_MARKETS and market_b in TEAM_MARKETS:
        return base if side_a == side_b else -base

    # Team-market x player-prop: sign = team-agreement x prop direction.
    prop_pairs = ((market_a, side_a, player_team_b), (market_b, side_b, player_team_a))
    for team_market, team_side, player_team in prop_pairs:
        if team_market in TEAM_MARKETS:
            if team_side == "DRAW":
                return 0.0
            agreement = 1.0
            if team_side in ("HOME", "AWAY") and player_team in ("HOME", "AWAY"):
                agreement = 1.0 if team_side == player_team else -1.0
            return base * agreement * _total_direction(market_a, side_a) * _total_direction(market_b, side_b)

    return base * _total_direction(market_a, side_a) * _total_direction(market_b, side_b)
