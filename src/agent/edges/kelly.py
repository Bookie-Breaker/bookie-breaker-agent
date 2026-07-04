"""Kelly criterion position sizing.

Implements algorithms/edge-detection.md section 3: quarter Kelly by default,
a 5% per-bet hard cap, and proportional scaling so simultaneous bets never
exceed 15% total bankroll exposure.
"""

from dataclasses import dataclass, replace

from agent.edges.odds import american_to_decimal


def kelly_fraction(
    predicted_prob: float,
    american_odds: int,
    kelly_multiplier: float = 0.25,
    max_bet_pct: float = 0.05,
) -> float:
    """Fractional Kelly bet size as a fraction of bankroll (0.0 if no edge)."""
    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1.0  # net odds
    p = predicted_prob
    q = 1.0 - p

    full_kelly = (b * p - q) / b

    if full_kelly <= 0:
        return 0.0  # no edge, no bet

    fractional = full_kelly * kelly_multiplier
    return min(fractional, max_bet_pct)


@dataclass(frozen=True)
class BetSizing:
    """A sized bet, before or after simultaneous-exposure scaling."""

    game_id: str
    kelly_fraction: float
    scaled: bool = False
    scale_factor: float = 1.0


def scale_simultaneous_bets(
    bets: list[BetSizing],
    max_total_exposure: float = 0.15,
) -> list[BetSizing]:
    """Scale bet sizes proportionally when total exposure exceeds the cap.

    Returns new BetSizing instances; inputs are not mutated.
    """
    total_exposure = sum(bet.kelly_fraction for bet in bets)

    if total_exposure <= max_total_exposure:
        return list(bets)

    scale_factor = max_total_exposure / total_exposure
    return [
        replace(bet, kelly_fraction=bet.kelly_fraction * scale_factor, scaled=True, scale_factor=scale_factor)
        for bet in bets
    ]
