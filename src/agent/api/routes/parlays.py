"""Parlay evaluation endpoint (Phase 7 Wave 1)."""

from typing import Annotated

from fastapi import APIRouter, Depends

from agent.api.dependencies import get_parlay_evaluator
from agent.api.envelope import Envelope, envelope
from agent.api.schemas import ParlayEvaluateRequest, ParlayEvaluationData, ParlayLegData
from agent.core.parlay import ParlayEvaluation, ParlayEvaluator, ParlayLegSpec

router = APIRouter(tags=["parlays"])

EvaluatorDep = Annotated[ParlayEvaluator, Depends(get_parlay_evaluator)]


def _to_evaluation_data(evaluation: ParlayEvaluation) -> ParlayEvaluationData:
    return ParlayEvaluationData(
        parlay_id=evaluation.parlay_id,
        league=evaluation.league,
        legs=[
            ParlayLegData(
                game_external_id=leg.game_external_id,
                game_id=leg.game_id,
                market_type=leg.market_type,
                selection=leg.selection,
                side=leg.side,
                line_value=leg.line_value,
                sportsbook_key=leg.sportsbook_key,
                odds_american=leg.odds_american,
                odds_decimal=leg.odds_decimal,
                predicted_probability=leg.predicted_probability,
                sim_leg_key=leg.sim_leg_key,
                player_external_id=leg.player_external_id,
                stat_type=leg.stat_type,
                prop_type=leg.prop_type,
            )
            for leg in evaluation.legs
        ],
        is_same_game=evaluation.is_same_game,
        joint_probability=evaluation.joint_probability,
        independent_probability=evaluation.independent_probability,
        correlation_edge=evaluation.correlation_edge,
        combined_odds_american=evaluation.combined_odds_american,
        combined_odds_decimal=evaluation.combined_odds_decimal,
        expected_value=evaluation.expected_value,
        ev_pct=evaluation.ev_pct,
        kelly_fraction=evaluation.kelly_fraction,
        recommended_stake=evaluation.recommended_stake,
        meets_threshold=evaluation.meets_threshold,
        method=evaluation.method,
        correlations=evaluation.correlations,
        expires_at=evaluation.expires_at.isoformat().replace("+00:00", "Z"),
    )


@router.post("/parlays/evaluate", response_model=Envelope[ParlayEvaluationData])
async def evaluate_parlay(request: ParlayEvaluateRequest, evaluator: EvaluatorDep) -> Envelope[ParlayEvaluationData]:
    """Evaluate a 2-6 leg parlay with correlation-aware math.

    Legs are team markets (SPREAD/TOTAL/MONEYLINE) or -- since Phase 7
    Wave 4 -- PLAYER_PROP legs carrying the ADR-029 slug identity
    (player_external_id + stat_type; line_value for OVER/UNDER stats).
    Same-game legs use the simulation engine's joint outcome structure
    (falling back to documented correlation priors, e.g. when the latest
    run captured no player distributions); distinct games multiply as
    independent. meets_threshold evaluations are persisted and published
    to events:parlay.detected; persist=true also stores below-threshold
    evaluations. Returns 422 for mixed-league leg sets, mutually exclusive
    legs, incomplete prop identities, or unsupported (team/game prop)
    markets.
    """
    specs = [
        ParlayLegSpec(
            game_external_id=leg.game_external_id,
            market_type=leg.market_type,
            side=leg.side.upper(),
            line_value=leg.line_value,
            sportsbook_key=leg.sportsbook_key,
            player_external_id=leg.player_external_id,
            stat_type=leg.stat_type,
            prop_type=leg.prop_type,
        )
        for leg in request.legs
    ]
    evaluation = await evaluator.evaluate(
        specs, parlay_odds_american=request.parlay_odds_american, persist=request.persist
    )
    return envelope(_to_evaluation_data(evaluation))
