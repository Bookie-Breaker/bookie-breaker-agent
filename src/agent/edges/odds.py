"""Odds format conversions.

Implements the conversions from algorithms/edge-detection.md section 1.
"""


def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds < 0:
        return 1 + 100 / abs(odds)
    return 1 + odds / 100


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))
