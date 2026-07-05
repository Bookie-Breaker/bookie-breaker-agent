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

    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "agent"


@lru_cache
def get_settings() -> Settings:
    return Settings()
