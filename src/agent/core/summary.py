"""Daily edge summary generation (Phase 4).

Runs on the scheduler's daily cron: gathers active edges and a performance
snapshot, writes a cheap-tier LLM summary to agent.analyses
(analysis_type=DAILY_SUMMARY, retrievable via GET /analysis/{id}). Returns
None instead of raising — a failed summary must never wound the scheduler.
"""

import logging
from datetime import date as date_type

from agent.api.errors import ApiError
from agent.clients.emulator import EmulatorClient
from agent.db.repository import AnalysisRecord, AnalysisRepository, EdgeRepository
from agent.llm.base import LLMError, LLMProvider
from agent.llm.prompts import build_daily_summary

logger = logging.getLogger(__name__)


class DailySummaryService:
    def __init__(
        self,
        edge_repo: EdgeRepository,
        emulator: EmulatorClient,
        llm: LLMProvider,
        analysis_repo: AnalysisRepository,
    ) -> None:
        self._edge_repo = edge_repo
        self._emulator = emulator
        self._llm = llm
        self._analysis_repo = analysis_repo

    async def generate(self, summary_date: date_type) -> AnalysisRecord | None:
        try:
            edges = await self._edge_repo.active_edges()
        except Exception:  # noqa: BLE001 - summary generation is best-effort
            logger.warning("daily summary skipped: could not load active edges", exc_info=True)
            return None
        if not edges:
            logger.info("daily summary skipped for %s: no active edges", summary_date.isoformat())
            return None

        edges_by_league: dict[str, list[dict[str, object]]] = {}
        for edge in edges:
            edges_by_league.setdefault(edge.league, []).append(
                {
                    "selection": edge.selection,
                    "market_type": edge.market_type,
                    "sportsbook": edge.sportsbook_key,
                    "odds_american": edge.odds_american,
                    "edge_percentage": edge.edge_percentage,
                    "confidence": edge.confidence,
                    "has_paper_bet": edge.paper_bet_id is not None,
                }
            )

        performance: dict[str, object] | None
        try:
            performance = (await self._emulator.performance()).model_dump()
        except ApiError:
            performance = None

        rendered = build_daily_summary(summary_date, edges_by_league, performance)
        try:
            result = await self._llm.complete(system=rendered.system, prompt=rendered.prompt, tier=rendered.tier)
        except LLMError as exc:
            logger.warning("daily summary skipped: LLM unavailable (%s)", exc)
            return None

        record = await self._analysis_repo.insert(
            {
                "analysis_type": "DAILY_SUMMARY",
                "game_id": None,
                "edge_id": None,
                "title": rendered.title,
                "content": result.text,
                "question": None,
                "model_used": result.model,
                "provider": result.provider,
                "input_summary": rendered.input_summary,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
        )
        logger.info("daily summary %s generated for %s", record.id, summary_date.isoformat())
        return record
