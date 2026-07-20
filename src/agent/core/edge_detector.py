"""Edge detection over calibrated predictions and de-vigged market prices.

Composition per algorithms/edge-detection.md sections 1-4:

1. Group the game's lines by (sportsbook, market, line) and de-vig each
   complete market (multiplicative by default): {HOME, AWAY}, {OVER, UNDER},
   {YES, NO}, or -- MONEYLINE only -- the three-way {HOME, AWAY, DRAW}
   (ADR-027). Markets missing a side are skipped -- de-vigging requires
   every price -- with one exception: single-sided YES/NO props (below).
   Player props additionally group per (player, stat): an OVER/UNDER pair
   only ever de-vigs against the same player's same stat at the same line
   (ADR-029), and prop predictions match on the exact
   (player, stat, side, line) tuple.

   Single-sided YES/NO props (Phase 7 Wave 3): YES-only quotes (e.g.
   anytime goalscorer) have no complement price, so no two-way de-vig is
   possible. Instead the raw implied probability is deflated
   multiplicatively: ``p_true_est = implied_raw / (1 + haircut)`` with
   haircut = SINGLE_SIDED_VIG_HAIRCUT (default 0.06). This treats the
   single quote as one leg of a hypothetical market whose prices sum to
   (1 + haircut) -- i.e. a multiplicative de-vig under an assumed book
   margin -- mirroring the DRAW no-complement precedent (ADR-027) but with
   an explicit margin assumption instead of sibling prices. Because that
   assumption replaces arithmetic over real complement prices, such
   candidates carry devig_method="single_sided" and their edge-quality
   confidence is discounted by SINGLE_SIDED_CONFIDENCE_PENALTY (0.8).
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
# to MONEYLINE only (ADR-027: soccer moneylines carry a DRAW side); YES/NO
# pairs are prop markets (ADR-029).
TWO_WAY_SIDE_SETS = (frozenset({"HOME", "AWAY"}), frozenset({"OVER", "UNDER"}), frozenset({"YES", "NO"}))
THREE_WAY_MONEYLINE_SIDES = frozenset({"HOME", "AWAY", "DRAW"})

# Lone YES/NO quotes are complete only for YES_NO props (single-sided path,
# see the module docstring); a lone OVER or HOME stays incomplete.
SINGLE_SIDED_PROP_SIDES = (frozenset({"YES"}), frozenset({"NO"}))
PROP_TYPE_YES_NO = "YES_NO"

# Default assumed book margin on a single-sided quote:
# p_true_est = implied_raw / (1 + haircut). Overridable via config
# (SINGLE_SIDED_VIG_HAIRCUT).
DEFAULT_SINGLE_SIDED_VIG_HAIRCUT = 0.06

# Confidence multiplier for single-sided candidates: the de-vig there is an
# assumption (flat haircut), not arithmetic over real complement prices.
SINGLE_SIDED_CONFIDENCE_PENALTY = 0.8

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


def _is_prop_market(market_type: str) -> bool:
    return market_type.upper().endswith("_PROP")


def _line_key(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _prop_prediction_key(
    player: str | None, stat: str | None, side: str | None, line: float | None
) -> tuple[str, str, str, float | None]:
    """Exact-match key for prop predictions: (player, stat, side, line).

    Both sides of the bridge use the ADR-029 name slug for ``player`` by the
    time detection runs (core/props.py rewrites prediction rows to slugs).
    """
    return (player or "", (stat or "").lower(), (side or "").upper(), _line_key(line))


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
        single_sided_vig_haircut: float = DEFAULT_SINGLE_SIDED_VIG_HAIRCUT,
    ) -> None:
        self._devig_method = DevigMethod(devig_method)
        self._kelly_multiplier = kelly_multiplier
        self._max_bet_pct = max_bet_pct
        self._single_sided_haircut = single_sided_vig_haircut

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

        # Prop predictions match only through the exact-tuple prop map:
        # keeping them out of the team maps means two players with the same
        # stat (or a prop OVER next to a TOTAL OVER) can never cross-match.
        team_predictions = [p for p in predictions if not _is_prop_market(p.market_type)]
        prediction_map = {_prediction_key(p.market_type, p.selection): p for p in team_predictions}
        side_map = {(p.market_type.upper(), p.side.upper()): p for p in team_predictions if p.side}
        prop_map = {
            _prop_prediction_key(p.player_external_id, p.stat_type, p.side, p.prop_line): p
            for p in predictions
            if _is_prop_market(p.market_type) and p.player_external_id and p.side
        }

        candidates: dict[tuple[str, str, float | None, str, str], EdgeCandidate] = {}
        for group in self._group_markets(lines):
            group_lines = list(group.values())
            single_sided = len(group_lines) == 1
            if single_sided:
                # YES/NO prop with no complement (module docstring):
                # p_true_est = implied_raw / (1 + haircut).
                only = group_lines[0]
                implied_raw = american_to_implied_prob(only.odds_american)
                devigged: list[tuple[LineSnapshot, float]] | None = [
                    (only, implied_raw / (1.0 + self._single_sided_haircut))
                ]
            else:
                devigged = self._devig_group(group_lines)
            if devigged is None:
                continue
            for line, implied in devigged:
                matched = self._match_prediction(prediction_map, side_map, prop_map, line, group)
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
                    devig_method_override="single_sided" if single_sided else None,
                    confidence_multiplier=SINGLE_SIDED_CONFIDENCE_PENALTY if single_sided else 1.0,
                )
                if candidate is None:
                    continue
                key = (
                    candidate.market_type,
                    candidate.side or "",
                    candidate.line_value,
                    candidate.player_external_id or "",
                    candidate.stat_type or "",
                )
                existing = candidates.get(key)
                if existing is None or american_to_decimal(candidate.odds_american) > american_to_decimal(
                    existing.odds_american
                ):
                    candidates[key] = candidate
        return list(candidates.values())

    def _group_markets(self, lines: list[LineSnapshot]) -> list[dict[str, LineSnapshot]]:
        """Group lines into complete (sportsbook, market, line) markets.

        Spreads are grouped by absolute line value since the two sides carry
        mirrored values (-3.5 / +3.5). Prop markets extend the group key
        with (player, stat) so an OVER/UNDER pair only ever pairs the same
        player's same stat at the same line (ADR-029). A market is complete
        when its sides form exactly {HOME, AWAY}, {OVER, UNDER}, {YES, NO},
        or -- MONEYLINE only -- the three-way {HOME, AWAY, DRAW}. A lone
        YES or NO is complete too, but only for YES_NO props (single-sided
        path). Other incomplete markets are skipped: de-vigging needs every
        price.
        """
        groups: dict[tuple[str, str, float | None, str, str], dict[str, LineSnapshot]] = {}
        for line in lines:
            if not line.side or line.odds_american == 0:
                continue
            line_key = abs(line.line_value) if line.line_value is not None else None
            market = line.market_type.upper()
            is_prop = _is_prop_market(market)
            player = (line.player_external_id or "") if is_prop else ""
            stat = (line.stat_type or "").lower() if is_prop else ""
            key = (line.sportsbook_key, market, line_key, player, stat)
            groups.setdefault(key, {})[line.side] = line

        complete: list[dict[str, LineSnapshot]] = []
        for key, sides in groups.items():
            side_set = frozenset(sides)
            if (
                side_set in TWO_WAY_SIDE_SETS
                or (key[1] == "MONEYLINE" and side_set == THREE_WAY_MONEYLINE_SIDES)
                or (side_set in SINGLE_SIDED_PROP_SIDES and self._is_yes_no_prop(sides))
            ):
                complete.append(sides)
            else:
                logger.debug("skipping incomplete market %s (sides: %s)", key, sorted(sides))
        return complete

    @staticmethod
    def _is_yes_no_prop(sides: dict[str, LineSnapshot]) -> bool:
        line = next(iter(sides.values()))
        return _is_prop_market(line.market_type) and (line.prop_type or "").upper() == PROP_TYPE_YES_NO

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
        prop_map: dict[tuple[str, str, str, float | None], PredictionItem],
        line: LineSnapshot,
        group: dict[str, LineSnapshot],
    ) -> tuple[float, PredictionItem] | None:
        """Find the calibrated probability for a line's side.

        Prop lines match only through the exact (player, stat, side, line)
        tuple, with the two-sided complement fallback applied within the
        same player/stat group. Team prediction rows with an explicit side
        match by (market_type, side); otherwise the selection string matches
        one side of the market. The complement fallback (P(other) = 1 - P)
        applies only to two-sided markets -- in a three-way group every side
        needs its own prediction row or that side is skipped (ADR-027).
        """
        if _is_prop_market(line.market_type):
            direct_prop = prop_map.get(
                _prop_prediction_key(line.player_external_id, line.stat_type, line.side, line.line_value)
            )
            if direct_prop is not None:
                return direct_prop.predicted_probability, direct_prop
            if len(group) != 2:
                return None
            for side, other in group.items():
                if side == line.side:
                    continue
                opposite_prop = prop_map.get(
                    _prop_prediction_key(other.player_external_id, other.stat_type, other.side, other.line_value)
                )
                if opposite_prop is not None:
                    return 1.0 - opposite_prop.predicted_probability, opposite_prop
            return None
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
        devig_method_override: str | None = None,
        confidence_multiplier: float = 1.0,
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
        if confidence_multiplier != 1.0:
            # Single-sided candidates: the implied probability rests on an
            # assumed margin, not real complement prices -- discount quality.
            confidence = round(confidence * confidence_multiplier, 3)

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
            devig_method=devig_method_override or self._devig_method.value,
            prediction_id=_parse_uuid(prediction.id),
            simulation_run_id=_parse_uuid(simulation_run_id),
            expires_at=expires_at,
            meets_threshold=ev_pct >= min_ev_pct_for_league(game.league),
            # Prop identity comes from the LINE, so player_external_id is
            # the ADR-029 name slug -- the cross-system prop identity the
            # emulator grades by (never the engine-internal player UUID).
            player_external_id=line.player_external_id,
            stat_type=line.stat_type,
            prop_type=line.prop_type,
            is_live=is_live,
        )
