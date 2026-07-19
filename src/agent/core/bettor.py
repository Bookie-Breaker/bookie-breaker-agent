"""Auto-bet gating, sizing, and idempotent placement via the bookie-emulator.

Gating: auto_bet enabled, EV clears the league threshold, and the section-6
timing framework says BET_NOW. Sizing: fractional Kelly scaled so the run's
simultaneous bets never exceed the exposure cap minus what is already open.
Placement is idempotent: the X-Idempotency-Key is a UUIDv5 over the bet's
identity, so retries of the same priced edge never double-bet.
"""

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from agent.api.errors import ApiError
from agent.clients.emulator import EmulatorClient
from agent.core.edge_detector import EdgeCandidate
from agent.db.repository import EdgeRecord, EdgeRepository, ParlayRepository
from agent.edges import BetSizing, scale_simultaneous_bets, should_bet_now

if TYPE_CHECKING:
    from agent.core.parlay import ParlayEvaluation

logger = logging.getLogger(__name__)

# Fixed namespace so identical bet identities always produce identical
# idempotency keys across processes and restarts.
BET_IDEMPOTENCY_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "bets.agent.bookie-breaker")

# Bankroll assumed when the emulator is unreachable at sizing time (the
# emulator's documented starting bankroll).
FALLBACK_BANKROLL_UNITS = 100.0


def candidate_key(candidate: "EdgeCandidate | EdgeRecord") -> str:
    return (
        f"{candidate.game_id}:{candidate.market_type}:{candidate.selection}"
        f":{candidate.sportsbook_key}:{candidate.line_value}:{candidate.odds_american}"
    )


def idempotency_key(edge: "EdgeCandidate | EdgeRecord") -> uuid.UUID:
    return uuid.uuid5(BET_IDEMPOTENCY_NAMESPACE, candidate_key(edge))


def parlay_identity(evaluation: "ParlayEvaluation") -> str:
    """Order-independent identity for a parlay (sorted leg identities)."""
    legs = sorted(
        f"{leg.game_external_id}:{leg.market_type}:{leg.side}:{leg.line_value}:{leg.sportsbook_key}:{leg.odds_american}"
        for leg in evaluation.legs
    )
    return "parlay|" + "|".join(legs)


def parlay_idempotency_key(evaluation: "ParlayEvaluation") -> uuid.UUID:
    return uuid.uuid5(BET_IDEMPOTENCY_NAMESPACE, parlay_identity(evaluation))


@dataclass(frozen=True)
class BetPlan:
    """Sizing outcome for one pipeline run's candidates.

    ``stakes`` maps every candidate key to its recommended stake in units
    (exposure-scaled for the to-bet subset, plain fractional Kelly
    otherwise); ``to_bet`` lists the keys that passed the gating checks;
    ``kelly`` maps candidate keys to the (possibly scaled) Kelly fraction.
    """

    stakes: dict[str, float]
    kelly: dict[str, float]
    to_bet: list[str]


class AutoBettor:
    def __init__(
        self,
        emulator: EmulatorClient,
        edge_repo: EdgeRepository,
        max_total_exposure: float = 0.15,
        parlay_repo: ParlayRepository | None = None,
    ) -> None:
        self._emulator = emulator
        self._edge_repo = edge_repo
        self._max_total_exposure = max_total_exposure
        self._parlay_repo = parlay_repo

    async def fetch_bankroll(self) -> tuple[float, float]:
        """Current (bankroll_units, open_exposure_units); safe fallback when down."""
        try:
            bankroll = await self._emulator.bankroll()
        except ApiError:
            logger.warning(
                "bookie-emulator bankroll unavailable; using fallback of %.0f units", FALLBACK_BANKROLL_UNITS
            )
            return FALLBACK_BANKROLL_UNITS, 0.0
        return bankroll.bankroll_units, bankroll.open_bets_exposure_units

    def should_bet(self, candidate: EdgeCandidate, now: datetime) -> bool:
        """Gating: actionable EV threshold plus the BET_NOW timing decision."""
        if not candidate.meets_threshold:
            return False
        hours_until_game = (candidate.expires_at - now).total_seconds() / 3600
        if hours_until_game <= 0:
            return False
        decision = should_bet_now(
            edge_pct=candidate.edge_percentage,
            edge_quality=candidate.confidence,
            hours_until_game=hours_until_game,
            league=candidate.league,
            market_type=candidate.market_type,
        )
        return decision == "BET_NOW"

    def plan(
        self,
        candidates: Sequence[EdgeCandidate],
        bankroll_units: float,
        open_exposure_units: float,
        auto_bet: bool,
        now: datetime,
        min_edge_threshold: float | None = None,
    ) -> BetPlan:
        """Size every candidate and select the subset to bet.

        The exposure cap available to this run is max_total_exposure minus
        the fraction of bankroll already tied up in open bets; the to-bet
        subset is proportionally scaled to fit it.
        """
        stakes: dict[str, float] = {}
        kelly: dict[str, float] = {}
        for candidate in candidates:
            key = candidate_key(candidate)
            stakes[key] = round(candidate.kelly_fraction * bankroll_units, 2)
            kelly[key] = candidate.kelly_fraction

        if not auto_bet:
            return BetPlan(stakes=stakes, kelly=kelly, to_bet=[])

        bettable = [c for c in candidates if self.should_bet(c, now)]
        if min_edge_threshold is not None:
            bettable = [c for c in bettable if c.edge_percentage >= min_edge_threshold]
        if not bettable:
            return BetPlan(stakes=stakes, kelly=kelly, to_bet=[])

        open_fraction = open_exposure_units / bankroll_units if bankroll_units > 0 else 1.0
        available_exposure = max(self._max_total_exposure - open_fraction, 0.0)
        sizings = scale_simultaneous_bets(
            [BetSizing(game_id=candidate_key(c), kelly_fraction=c.kelly_fraction) for c in bettable],
            max_total_exposure=available_exposure,
        )
        to_bet: list[str] = []
        for sizing in sizings:
            stake = round(sizing.kelly_fraction * bankroll_units, 2)
            stakes[sizing.game_id] = stake
            kelly[sizing.game_id] = sizing.kelly_fraction
            if stake > 0:
                to_bet.append(sizing.game_id)
        return BetPlan(stakes=stakes, kelly=kelly, to_bet=to_bet)

    async def place_bet(self, edge: EdgeRecord, stake: float, kelly_used: float) -> str | None:
        """Place one paper bet and link it back to the edge row.

        Returns the paper bet id, or None when the edge already carries one.
        Emulator failures propagate as ApiError for per-game error recording.
        """
        if edge.paper_bet_id is not None:
            logger.info("edge %s already has paper bet %s; skipping", edge.id, edge.paper_bet_id)
            return None
        body = {
            "game_id": str(edge.game_id),
            "game_external_id": edge.game_external_id,
            "edge_id": str(edge.id),
            "prediction_id": str(edge.prediction_id) if edge.prediction_id else None,
            "market_type": edge.market_type,
            "selection": edge.selection,
            "side": edge.side,
            "sportsbook_key": edge.sportsbook_key,
            "predicted_probability": edge.predicted_probability,
            "edge_percentage": edge.edge_percentage,
            "stake": stake,
            "kelly_fraction": kelly_used,
            "reasoning": self._reasoning(edge),
        }
        bet = await self._emulator.place_bet(body, idempotency_key=str(idempotency_key(edge)))
        await self._edge_repo.set_paper_bet(edge.id, uuid.UUID(bet.id))
        return bet.id

    async def place_parlay(self, evaluation: "ParlayEvaluation") -> str | None:
        """Place one paper parlay via the emulator's parlay endpoint.

        Callers gate this exactly like singles (auto_bet config; the
        scanner only wires it when PARLAY_AUTO_BET is on). The idempotency
        key is a UUIDv5 over the sorted leg identity, so retries of the
        same priced parlay never double-bet. Returns the paper bet id, or
        None for zero-stake or below-threshold evaluations.
        """
        if not evaluation.meets_threshold or evaluation.recommended_stake <= 0:
            return None
        body = {
            "legs": [
                {
                    "game_id": leg.game_id,
                    "game_external_id": leg.game_external_id,
                    "market_type": leg.market_type,
                    "selection": leg.selection,
                    "side": leg.side,
                    "line_value": leg.line_value,
                    "sportsbook_key": leg.sportsbook_key,
                }
                for leg in evaluation.legs
            ],
            "stake": evaluation.recommended_stake,
            "predicted_probability": evaluation.joint_probability,
            "edge_percentage": round((evaluation.joint_probability - 1.0 / evaluation.combined_odds_decimal) * 100, 3),
            "kelly_fraction": evaluation.kelly_fraction,
            "reasoning": self._parlay_reasoning(evaluation),
        }
        bet = await self._emulator.place_parlay(body, idempotency_key=str(parlay_idempotency_key(evaluation)))
        if self._parlay_repo is not None and evaluation.parlay_id is not None:
            await self._parlay_repo.set_paper_bet(uuid.UUID(evaluation.parlay_id), uuid.UUID(bet.id))
        return bet.id

    @staticmethod
    def _parlay_reasoning(evaluation: "ParlayEvaluation") -> str:
        return (
            f"Auto-parlay ({len(evaluation.legs)} legs, {evaluation.method}): joint probability "
            f"{evaluation.joint_probability:.3f} vs independent {evaluation.independent_probability:.3f} "
            f"at {evaluation.combined_odds_american:+d}; EV {evaluation.ev_pct:.1f}% of stake, "
            f"correlation edge {evaluation.correlation_edge:+.3f}."
        )

    @staticmethod
    def _reasoning(edge: EdgeRecord) -> str:
        return (
            f"Auto-bet: {edge.selection} at {edge.odds_american:+d} ({edge.sportsbook_key}). "
            f"EV {edge.expected_value * 100:.1f}% of stake, edge {edge.edge_percentage:.1f}pp over the "
            f"de-vigged market, quality {edge.confidence if edge.confidence is not None else 0.0:.2f}."
        )
