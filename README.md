# bookie-breaker-agent

[![CI](https://img.shields.io/github/actions/workflow/status/Bookie-Breaker/bookie-breaker-agent/ci.yml?branch=main&label=CI&logo=githubactions&logoColor=white)](https://github.com/Bookie-Breaker/bookie-breaker-agent/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/codecov/c/github/Bookie-Breaker/bookie-breaker-agent?logo=codecov&logoColor=white)](https://app.codecov.io/gh/Bookie-Breaker/bookie-breaker-agent)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-LLM-191919?logo=anthropic&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white)

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
