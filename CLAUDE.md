# bookie-breaker-agent

## Service Purpose

Python/FastAPI LLM-powered orchestration agent. Coordinates the prediction pipeline (stats -> sim -> predict -> edges), detects betting edges, generates analysis, and manages alerts. Central coordinator for the entire system.

## Language & Conventions

- **Language:** Python 3.12
- **Framework:** FastAPI + uvicorn, Anthropic SDK for LLM
- **Project layout:** `src/agent/` package, `main.py` FastAPI entry point
- **Naming:** `snake_case.py` files, `snake_case` functions, `PascalCase` classes
- **Package manager:** uv
- **Testing:** pytest in `tests/`

## Key Files

- `src/agent/main.py` — FastAPI app entry point
- `src/agent/api/` — Route handlers
- `src/agent/core/` — Pipeline orchestration, edge detection, analysis
- `pyproject.toml` — Dependencies and tool config

## Service-Specific Commands

```bash
task dev          # uvicorn with --reload on port 8006
task lint         # ruff check + format
task test         # pytest --cov
task typecheck    # mypy src/
```

## Dependencies

- **All backend services** — Orchestrates the full pipeline:
  - statistics-service (8002), lines-service (8001), simulation-engine (8003), prediction-engine (8004), bookie-emulator (8005)
- **Redis** — Pub/sub for events, caching dashboard data
- **Anthropic API / Ollama** — LLM for analysis generation

## Environment Variables

See `.env.example`. Key: `ANTHROPIC_API_KEY`, all `*_SERVICE_URL` vars, `REDIS_URL`, `PORT=8006`.
