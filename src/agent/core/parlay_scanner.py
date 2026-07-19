"""Same-game parlay scanning over freshly detected edges.

After edge detection, the scanner enumerates 2-3 leg same-game
combinations of near-actionable team-market edges and runs each through
the ParlayEvaluator; meets_threshold results are persisted and published
by the evaluator itself. Disabled by default (PARLAY_SCAN_ENABLED) and
never auto-bets unless a bettor is wired in (PARLAY_AUTO_BET).
"""

import logging
from itertools import combinations

from agent.api.errors import ApiError
from agent.core.bettor import AutoBettor
from agent.core.parlay import ALLOWED_MARKETS, ParlayEvaluation, ParlayEvaluator, ParlayLegSpec
from agent.db.repository import EdgeRecord, EdgeRepository
from agent.edges import min_ev_pct_for_league

logger = logging.getLogger(__name__)

# Combination guards: at most C(6, 3) evaluations per game, from at most
# 6 candidate edges, in 2-3 leg combinations.
MAX_CANDIDATE_EDGES = 6
MAX_COMBINATIONS = 20
COMBO_SIZES = (2, 3)


def _leg_key(edge: EdgeRecord) -> tuple[str, str | None, float | None]:
    return (edge.market_type, edge.side, edge.line_value)


class ParlayScanner:
    def __init__(
        self,
        edge_repo: EdgeRepository,
        evaluator: ParlayEvaluator,
        min_edge_ratio: float = 0.6,
        bettor: AutoBettor | None = None,
    ) -> None:
        self._edge_repo = edge_repo
        self._evaluator = evaluator
        self._min_edge_ratio = min_edge_ratio
        self._bettor = bettor

    async def scan_league(self, league: str) -> list[ParlayEvaluation]:
        """Scan every game with fresh edges in a league."""
        edges = await self._edge_repo.active_edges([league])
        results: list[ParlayEvaluation] = []
        for game_external_id in sorted({edge.game_external_id for edge in edges}):
            results.extend(await self.scan_game(game_external_id))
        return results

    async def scan_game(self, game_external_id: str) -> list[ParlayEvaluation]:
        """Evaluate same-game 2-3 leg combinations; keep actionable results.

        meets_threshold evaluations are persisted and published by the
        evaluator; when a bettor is wired (PARLAY_AUTO_BET), they are also
        placed as paper parlays.
        """
        edges = await self._edge_repo.active_for_game_external(game_external_id)
        candidates = self.select_candidates(edges)
        results: list[ParlayEvaluation] = []
        for combo in self.enumerate_combinations(candidates):
            specs = [
                ParlayLegSpec(
                    game_external_id=edge.game_external_id,
                    market_type=edge.market_type,
                    side=edge.side or "",
                    line_value=edge.line_value,
                    sportsbook_key=edge.sportsbook_key,
                    edge_id=str(edge.id),
                )
                for edge in combo
            ]
            try:
                evaluation = await self._evaluator.evaluate(specs)
            except ApiError as exc:
                logger.info("parlay evaluation failed for game %s (%s); skipping combo", game_external_id, exc.message)
                continue
            if not evaluation.meets_threshold:
                continue
            results.append(evaluation)
            if self._bettor is not None:
                try:
                    await self._bettor.place_parlay(evaluation)
                except ApiError as exc:
                    logger.warning("parlay auto-bet failed for parlay %s: %s", evaluation.parlay_id, exc.message)
        return results

    def select_candidates(self, edges: list[EdgeRecord]) -> list[EdgeRecord]:
        """Near-actionable, deduplicated team-market edges, best-EV first.

        An edge enters the scan when its EV reaches min_edge_ratio of the
        league minimum (default 60%): slightly sub-threshold legs can
        still combine into an actionable correlated parlay. Duplicate
        (market, side, line) legs keep only the best-EV offer, and the
        pool is capped at MAX_CANDIDATE_EDGES.
        """
        best_by_key: dict[tuple[str, str | None, float | None], EdgeRecord] = {}
        for edge in edges:
            if edge.market_type not in ALLOWED_MARKETS or edge.side is None:
                continue
            if edge.expected_value * 100 < self._min_edge_ratio * min_ev_pct_for_league(edge.league):
                continue
            key = _leg_key(edge)
            current = best_by_key.get(key)
            if current is None or edge.expected_value > current.expected_value:
                best_by_key[key] = edge
        ranked = sorted(best_by_key.values(), key=lambda edge: edge.expected_value, reverse=True)
        return ranked[:MAX_CANDIDATE_EDGES]

    @staticmethod
    def enumerate_combinations(candidates: list[EdgeRecord]) -> list[tuple[EdgeRecord, ...]]:
        """2-3 leg combinations with guards.

        Skips combos that repeat a market type -- that excludes both
        mutually-exclusive opposite sides and near-duplicate same-side
        legs of one market -- dedupes by leg set, and caps the total at
        MAX_COMBINATIONS.
        """
        combos: list[tuple[EdgeRecord, ...]] = []
        seen: set[frozenset[tuple[str, str | None, float | None]]] = set()
        for size in COMBO_SIZES:
            for combo in combinations(candidates, size):
                if len(combos) >= MAX_COMBINATIONS:
                    return combos
                markets = [edge.market_type for edge in combo]
                if len(set(markets)) != len(markets):
                    continue
                identity = frozenset(_leg_key(edge) for edge in combo)
                if identity in seen:
                    continue
                seen.add(identity)
                combos.append(combo)
        return combos
