"""Runtime configuration via environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    port: int = 8006
    log_level: str = "info"
    database_url: str = "postgres://agent_svc:localdev@localhost:5432/bookiebreaker?search_path=agent,public"
    redis_url: str = "redis://localhost:6379"
    lines_service_url: str = "http://localhost:8001"
    statistics_service_url: str = "http://localhost:8002"
    simulation_engine_url: str = "http://localhost:8003"
    prediction_engine_url: str = "http://localhost:8004"
    bookie_emulator_url: str = "http://localhost:8005"

    dashboard_cache_ttl_seconds: int = 300  # agent:dashboard:{league} per redis-schemas.md
    slate_cache_ttl_seconds: int = 300  # agent:slate:{league}:{date} per redis-schemas.md
    game_map_ttl_seconds: int = 86_400  # statistics<->lines game id mapping cache

    kelly_multiplier: float = 0.25  # quarter Kelly (algorithms/edge-detection.md section 3)
    max_bet_pct: float = 0.05  # per-bet hard cap as a fraction of bankroll
    max_total_exposure: float = 0.15  # simultaneous-bet exposure cap
    devig_method: str = "multiplicative"  # multiplicative | additive | shin
    pipeline_concurrency: int = 4  # bounded per-game concurrency inside a run

    # Parlays (Phase 7 Wave 1). Scanning and auto-betting are off by
    # default; the evaluate API is always available.
    parlay_scan_enabled: bool = False  # run the same-game parlay scan after edge detection
    parlay_scan_min_edge_pct: float = 0.6  # fraction of the league min EV an edge needs to enter the scan
    parlay_auto_bet: bool = False  # let the scanner place paper parlays for meets_threshold results

    # Player-prop edges (Phase 7 Wave 3). The prop step (simulate with
    # include_player_props, bridge slugs to engine UUIDs, request
    # PLAYER_PROP predictions) runs only for leagues in this list; an
    # empty list disables prop edge detection entirely.
    prop_edges_leagues: list[str] = ["FIFA_WC", "EPL", "MLB"]
    # Single-sided YES/NO props have no complement price to de-vig
    # against; the raw implied probability is deflated multiplicatively
    # instead: p_true_est = implied / (1 + haircut). See core/edge_detector.
    single_sided_vig_haircut: float = 0.06

    # Live edges (Phase 7 Wave 2). Off by default: is_live lines.updated
    # events trigger a trimmed per-game re-evaluation behind an aggressive
    # per-game debounce (sub-second live frames must coalesce).
    live_edges_enabled: bool = False
    live_debounce_seconds: float = 5.0  # per-game coalesce window for live line bursts
    live_edge_ttl_seconds: int = 120  # live edges expire fast: in-game prices move constantly

    llm_provider: str = "anthropic"  # anthropic | ollama (ADR-011: config-only switch)
    anthropic_api_key: str | None = None
    llm_base_url: str | None = None  # None -> provider default (api.anthropic.com / http://ollama:11434)
    llm_model: str = "claude-opus-4-8"  # quality tier: edge breakdowns, previews, reviews
    llm_model_cheap: str = ""  # cheap tier; empty -> provider default (claude-haiku-4-5 / llm_model)
    llm_max_tokens: int = 2048
    llm_timeout_seconds: float = 60.0
    analysis_cache_ttl_seconds: int = 3600  # agent:analysis:{type}:{scope} per redis-schemas.md

    schedule_misfire_grace_seconds: int = 300  # missed cron fires older than this roll forward unrun
    daily_summary_enabled: bool = True
    daily_summary_cron: str = "0 12 * * *"
    daily_summary_timezone: str = "UTC"
    event_reruns_enabled: bool = True
    rerun_debounce_seconds: float = 120.0  # quiet period after the last triggering event
    rerun_cooldown_seconds: float = 600.0  # minimum spacing between EVENT runs per league
    alert_llm_descriptions: bool = True
    alert_llm_max_per_run: int = 10  # LLM-written alert descriptions per pipeline run (cost cap)

    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "agent"


@lru_cache
def get_settings() -> Settings:
    return Settings()
