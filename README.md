# bookie-breaker-agent

Central FastAPI coordinator for BookieBreaker (port 8006). Runs the prediction pipeline
(simulation → prediction → edge_detection → bet_placement), detects +EV edges against de-vigged
market prices, sizes stakes with fractional Kelly, and places paper bets through the
bookie-emulator. Per-league cron schedules (croniter) persist in `agent.pipeline_schedules`,
event-triggered re-runs fire off Redis pub/sub channels, and a daily LLM summary recaps
performance. An LLM analyst — Ollama by default, Anthropic optional — narrates edges and slates.
Parlay evaluation and live-edge detection landed in Phase 7.

## Quickstart

### With Docker Compose (recommended)

```bash
task up  # from BookieBreaker/ root
```

### Standalone

```bash
cp .env.example .env  # fill in values
task bootstrap
task dev
```

## API

Interactive docs at `http://localhost:8006/docs` when running. All endpoints live under
`/api/v1/agent`.

Full contract:
[agent-api.md](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/api-contracts/agent-api.md)

## Architecture Decisions

- [Local LLM Strategy (ADR-011)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/011-local-llm-strategy.md)
- [Pipeline Scheduler (ADR-015)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/015-pipeline-scheduler.md)
- [Streaming Analysis Transport (ADR-024)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/024-streaming-analysis-transport.md)
- [Parlay Joint Probability and Correlated Kelly (ADR-030)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/030-parlay-joint-probability-and-correlated-kelly.md)
- [Live Ingestion Transport (ADR-031)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/031-live-ingestion-transport.md)

Operating the pipeline and schedules:
[Pipeline and Scheduling playbook](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/playbooks/05-pipeline-and-scheduling.md)

## Environment Variables

See `.env.example` for all variables with descriptions. Key ones: `DATABASE_URL`, `REDIS_URL`,
the downstream service URLs (`*_SERVICE_URL`, `*_ENGINE_URL`, `BOOKIE_EMULATOR_URL`), and the
LLM layer (`LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, `ANTHROPIC_API_KEY`).
