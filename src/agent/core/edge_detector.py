"""Edge detection over calibrated predictions and de-vigged market prices.

Composition per algorithms/edge-detection.md sections 1-4:

1. Group the game's lines by (sportsbook, market, line) and de-vig each
   complete market (multiplicative by default): {HOME, AWAY}, {OVER, UNDER},
   or -- MONEYLINE only -- the three-way {HOME, AWAY, DRAW} (ADR-027).
   Markets missing a side are skipped -- de-vigging requires every price.
2. ``edge_percentage`` is (predicted - devigged implied) in percentage
   points; ``expected_value`` is computed from the raw quoted odds
   (``predicted * decimal_odds - 1``) since that is what a bet pays.
3. The best-priced offer per (market, side, line) across books wins.
4. Threshold by the league minimum EV; size with fractional Kelly; score
   quality with the section-4 composite.

The agent.edges module raw-implied ``detect_edge`` is not used directly
because the contract requires the de-vigged implied probability; the same
underlying math (calculate_ev_pct, kelly_fraction, edge_quality_score,
min_ev_pct_for_league) is composed here instead.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from agent.clients.lines import LineSnapshot
from agent.clients.prediction import PredictionItem
from agent.clients.statistics import Game
from agent.edges import (
    DevigMethod,
    american_to_decimal,
    american_to_implied_prob,
    calculate_ev_pct,
    devig,
    devig_many,
    edge_quality_score,
    kelly_fraction,
    market_efficiency,
    min_ev_pct_for_league,
)

logger = logging.getLogger(__name__)

# Side sets that make a market complete and de-viggable. Three-way applies
# to MONEYLINE only (ADR-027: soccer moneylines carry a DRAW side).
TWO_WAY_SIDE_SETS = (frozenset({"HOME", "AWAY"}), frozenset({"OVER", "UNDER"}))
THREE_WAY_MONEYLINE_SIDES = frozenset({"HOME", "AWAY", "DRAW"})

# Target model expected calibration error (algorithms/edge-detection.md
# section 2: "~3% ECE target"); feeds the quality score until per-model
# metrics are wired through in Phase 4.
DEFAULT_CALIBRATION_ERROR = 0.03

# CI width assumed when the prediction carries no confidence interval.
DEFAULT_CI_WIDTH = 0.10


@dataclass(frozen=True)
class EdgeCandidate:
    """A detected edge, ready to persist once stakes are sized."""

    game_id: str
    game_external_id: str
    league: str
    market_type: str
    selection: str
    side: str | None
    line_value: float | None
    sportsbook_key: str
    odds_american: int
    predicted_probability: float
    implied_probability: float  # de-vigged
    edge_percentage: float  # percentage points: (predicted - implied) * 100
    expected_value: float  # EV per unit staked (fraction)
    kelly_fraction: float
    confidence: float
    devig_method: str
    prediction_id: str | None
    simulation_run_id: str | None
    expires_at: datetime
    meets_threshold: bool
    # Player-prop metadata (Phase 7 Wave 0); None/False for non-prop edges.
    player_external_id: str | None = None
    stat_type: str | None = None
    prop_type: str | None = None
    is_live: bool = False


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _prediction_key(market_type: str, selection: str) -> tuple[str, str]:
    normalized = _normalize(selection)
    if market_type.upper() == "MONEYLINE" and normalized.endswith(" ml"):
        # prediction-engine moneyline selections carry an " ML" suffix that
        # lines-service selections do not
        normalized = normalized[: -len(" ml")]
    return market_type.upper(), normalized


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _parse_uuid(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


class EdgeDetector:
    def __init__(
        self,
        devig_method: str = "multiplicative",
        kelly_multiplier: float = 0.25,
        max_bet_pct: float = 0.05,
    ) -> None:
        self._devig_method = DevigMethod(devig_method)
        self._kelly_multiplier = kelly_multiplier
        self._max_bet_pct = max_bet_pct

    def detect(
        self,
        game: Game,
        game_external_id: str,
        predictions: list[PredictionItem],
        lines: list[LineSnapshot],
        simulation_run_id: str | None = None,
        now: datetime | None = None,
        mark_live: bool = False,
    ) -> list[EdgeCandidate]:
        """Detect positive-EV edges for one game.

        Returns one candidate per (market_type, side, line_value) at the
        best available price across books. Candidates are positive edges
        (predicted beats the de-vigged implied probability with positive
        EV); ``meets_threshold`` marks the actionable subset. mark_live
        flags every built candidate is_live=True (Phase 7 Wave 2 in-game
        detection over live lines).
        """
        now = now or datetime.now(tz=UTC)
        expires_at = _parse_datetime(game.scheduled_start)
        if expires_at is None:
            logger.warning("game %s has unparseable scheduled_start %r; skipping", game.id, game.scheduled_start)
            return []

        prediction_map = {_prediction_key(p.market_type, p.selection): p for p in predictions}
        side_map = {(p.market_type.upper(), p.side.upper()): p for p in predictions if p.side}

        candidates: dict[tuple[str, str, float | None], EdgeCandidate] = {}
        for group in self._group_markets(lines):
            devigged = self._devig_group(list(group.values()))
            if devigged is None:
                continue
            for line, implied in devigged:
                matched = self._match_prediction(prediction_map, side_map, line, group)
                if matched is None:
                    continue
                predicted, prediction = matched
                candidate = self._build_candidate(
                    game,
                    game_external_id,
                    line,
                    predicted,
                    implied,
                    prediction,
                    simulation_run_id,
                    expires_at,
                    now,
                    is_live=mark_live,
                )
                if candidate is None:
                    continue
                key = (candidate.market_type, candidate.side or "", candidate.line_value)
                existing = candidates.get(key)
                if existing is None or american_to_decimal(candidate.odds_american) > american_to_decimal(
                    existing.odds_american
                ):
                    candidates[key] = candidate
        return list(candidates.values())

    def _group_markets(self, lines: list[LineSnapshot]) -> list[dict[str, LineSnapshot]]:
        """Group lines into complete (sportsbook, market, line) markets.

        Spreads are grouped by absolute line value since the two sides carry
        mirrored values (-3.5 / +3.5). A market is complete when its sides
        form exactly {HOME, AWAY}, {OVER, UNDER}, or -- MONEYLINE only --
        the three-way {HOME, AWAY, DRAW}. Incomplete markets are skipped:
        de-vigging needs every price.
        """
        groups: dict[tuple[str, str, float | None], dict[str, LineSnapshot]] = {}
        for line in lines:
            if not line.side or line.odds_american == 0:
                continue
            line_key = abs(line.line_value) if line.line_value is not None else None
            key = (line.sportsbook_key, line.market_type.upper(), line_key)
            groups.setdefault(key, {})[line.side] = line

        complete: list[dict[str, LineSnapshot]] = []
        for key, sides in groups.items():
            side_set = frozenset(sides)
            if side_set in TWO_WAY_SIDE_SETS or (key[1] == "MONEYLINE" and side_set == THREE_WAY_MONEYLINE_SIDES):
                complete.append(sides)
            else:
                logger.debug("skipping incomplete market %s (sides: %s)", key, sorted(sides))
        return complete

    def _devig_group(self, group_lines: list[LineSnapshot]) -> list[tuple[LineSnapshot, float]] | None:
        raws = [american_to_implied_prob(line.odds_american) for line in group_lines]
        try:
            if len(group_lines) == 2:
                true_probs: tuple[float, ...] = devig(raws[0], raws[1], self._devig_method)
            else:
                true_probs = devig_many(raws, self._devig_method)
        except ValueError:
            logger.debug(
                "devig failed for %s %s (%s); skipping market",
                group_lines[0].sportsbook_key,
                group_lines[0].market_type,
                "/".join(str(line.odds_american) for line in group_lines),
            )
            return None
        return list(zip(group_lines, true_probs, strict=True))

    @staticmethod
    def _match_prediction(
        prediction_map: dict[tuple[str, str], PredictionItem],
        side_map: dict[tuple[str, str], PredictionItem],
        line: LineSnapshot,
        group: dict[str, LineSnapshot],
    ) -> tuple[float, PredictionItem] | None:
        """Find the calibrated probability for a line's side.

        Prediction rows with an explicit side match by (market_type, side);
        otherwise the selection string matches one side of the market. The
        complement fallback (P(other) = 1 - P) applies only to two-sided
        markets -- in a three-way group every side needs its own prediction
        row or that side is skipped (ADR-027).
        """
        if line.side:
            by_side = side_map.get((line.market_type.upper(), line.side.upper()))
            if by_side is not None:
                return by_side.predicted_probability, by_side
        direct = prediction_map.get(_prediction_key(line.market_type, line.selection))
        if direct is not None:
            return direct.predicted_probability, direct
        if len(group) != 2:
            return None
        for side, other in group.items():
            if side == line.side:
                continue
            opposite = prediction_map.get(_prediction_key(other.market_type, other.selection))
            if opposite is not None:
                return 1.0 - opposite.predicted_probability, opposite
        return None

    def _build_candidate(
        self,
        game: Game,
        game_external_id: str,
        line: LineSnapshot,
        predicted: float,
        implied: float,
        prediction: PredictionItem,
        simulation_run_id: str | None,
        expires_at: datetime,
        now: datetime,
        is_live: bool = False,
    ) -> EdgeCandidate | None:
        if not 0.0 < predicted < 1.0:
            return None
        ev_pct = calculate_ev_pct(predicted, line.odds_american)
        if predicted <= implied or ev_pct <= 0.0:
            return None

        ci_width = DEFAULT_CI_WIDTH
        if prediction.confidence_lower is not None and prediction.confidence_upper is not None:
            ci_width = prediction.confidence_upper - prediction.confidence_lower

        line_timestamp = _parse_datetime(line.timestamp)
        freshness_hours = max((now - line_timestamp).total_seconds() / 3600, 0.0) if line_timestamp else 0.0

        confidence = edge_quality_score(
            ev_pct=ev_pct,
            prediction_confidence=ci_width,
            market_efficiency=market_efficiency(game.league, line.market_type),
            line_freshness_hours=freshness_hours,
            model_calibration_error=DEFAULT_CALIBRATION_ERROR,
        )

        return EdgeCandidate(
            game_id=game.id,
            game_external_id=game_external_id,
            league=game.league,
            market_type=line.market_type.upper(),
            selection=line.selection,
            side=line.side or None,
            line_value=line.line_value,
            sportsbook_key=line.sportsbook_key,
            odds_american=line.odds_american,
            predicted_probability=round(predicted, 5),
            implied_probability=round(implied, 5),
            edge_percentage=round((predicted - implied) * 100, 3),
            expected_value=round(ev_pct / 100, 5),
            kelly_fraction=kelly_fraction(predicted, line.odds_american, self._kelly_multiplier, self._max_bet_pct),
            confidence=confidence,
            devig_method=self._devig_method.value,
            prediction_id=_parse_uuid(prediction.id),
            simulation_run_id=_parse_uuid(simulation_run_id),
            expires_at=expires_at,
            meets_threshold=ev_pct >= min_ev_pct_for_league(game.league),
            is_live=is_live,
        )
