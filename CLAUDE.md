# bookie-breaker-agent

## Service Purpose

Python/FastAPI orchestration agent. Coordinates the prediction pipeline (stats -> sim -> predict -> edges -> paper
bets), detects betting edges against de-vigged market prices, and aggregates the slate/dashboard views. Central
coordinator for the entire system. LLM-powered analysis and cron scheduling arrive in Phase 4.

## Language & Conventions

- **Language:** Python 3.12
- **Framework:** FastAPI + uvicorn (Anthropic SDK for LLM lands in Phase 4)
- **Project layout:** `src/agent/` package, `main.py` FastAPI entry point
- **Naming:** `snake_case.py` files, `snake_case` functions, `PascalCase` classes
- **Package manager:** uv
- **Testing:** pytest in `tests/` (`unit/`, `integration/` via testcontainers, `e2e/` gated by `BB_E2E_STACK=1`)

## Key Files

- `src/agent/main.py` — FastAPI app entry point and lifespan wiring
- `src/agent/api/` — Route handlers, envelope/error/pagination helpers, schemas
- `src/agent/core/` — Pipeline orchestration, edge detection, auto-betting, slate/dashboard assembly
- `src/agent/edges/` — Standalone edge-detection math (devig, EV, Kelly, quality, decay)
- `src/agent/clients/` — Typed clients for the five downstream services + game id reconciliation
- `src/agent/db/` — SQLAlchemy Core tables and repositories (agent schema)
- `src/agent/events/` — Redis pub/sub publisher and background subscriber
- `alembic/` — Migrations for `agent.pipeline_runs` and `agent.edges`
- `pyproject.toml` — Dependencies and tool config

## Service-Specific Commands

```bash
task dev          # uvicorn with --reload on port 8006
task lint         # ruff check + format
task test         # pytest --cov (unit + integration)
task typecheck    # mypy src/
task db:migrate   # alembic upgrade head
task spec:export  # export OpenAPI spec to bookie-breaker-docs
```

## Dependencies

- **All backend services** — Orchestrates the full pipeline:
  - statistics-service (8002), lines-service (8001), simulation-engine (8003), prediction-engine (8004), bookie-emulator (8005)
- **PostgreSQL** (agent schema) — Persisted edges and pipeline run history
- **Redis** — Pub/sub events, dashboard/slate caching, game id mapping cache
- **Anthropic API / Ollama** — Phase 4 (LLM analysis)

## Environment Variables

See `.env.example`. Key: `DATABASE_URL`, all `*_SERVICE_URL`/`*_ENGINE_URL`/`BOOKIE_EMULATOR_URL` vars, `REDIS_URL`,
`PORT=8006`. `ANTHROPIC_API_KEY` is Phase 4.
