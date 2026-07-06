"""Edge detection math for BookieBreaker.

Standalone, dependency-free implementation of algorithms/edge-detection.md
(sections 1-4 and 6). Parlay correlation (section 5) is deferred until the
Phase 7 advanced bet types work, where simulation output is available.
"""

from agent.edges.clv import calculate_clv
from agent.edges.decay import HALF_LIVES, BetDecision, estimate_edge_remaining, should_bet_now
from agent.edges.detect import Edge, detect_edge
from agent.edges.devig import (
    DevigMethod,
    additive_devig,
    additive_devig_n,
    devig,
    devig_many,
    multiplicative_devig,
    multiplicative_devig_n,
    shin_devig,
    shin_devig_n,
)
from agent.edges.ev import (
    MIN_EV_PCT_BY_LEAGUE,
    calculate_ev,
    calculate_ev_pct,
    meets_ev_threshold,
    min_ev_pct_for_league,
)
from agent.edges.kelly import BetSizing, kelly_fraction, scale_simultaneous_bets
from agent.edges.odds import american_to_decimal, american_to_implied_prob, decimal_to_american
from agent.edges.quality import MARKET_EFFICIENCY, edge_quality_score, is_line_stale, market_efficiency

__all__ = [
    "HALF_LIVES",
    "MARKET_EFFICIENCY",
    "MIN_EV_PCT_BY_LEAGUE",
    "BetDecision",
    "BetSizing",
    "DevigMethod",
    "Edge",
    "additive_devig",
    "additive_devig_n",
    "american_to_decimal",
    "american_to_implied_prob",
    "calculate_clv",
    "calculate_ev",
    "calculate_ev_pct",
    "decimal_to_american",
    "detect_edge",
    "devig",
    "devig_many",
    "edge_quality_score",
    "estimate_edge_remaining",
    "is_line_stale",
    "kelly_fraction",
    "market_efficiency",
    "meets_ev_threshold",
    "min_ev_pct_for_league",
    "multiplicative_devig",
    "multiplicative_devig_n",
    "scale_simultaneous_bets",
    "shin_devig",
    "shin_devig_n",
    "should_bet_now",
]
