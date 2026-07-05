"""POST/GET /analysis against real Postgres/Redis with mocked LLM + services.

The parametrized-provider test (Ollama app) is the ADR-011 config-only
switch verification: identical request, different provider, no code path
changes.
"""

import uuid
from collections.abc import Iterator

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response

from agent.config import Settings
from agent.main import create_app
from tests.integration.conftest import (
    EMULATOR_URL,
    LINES_URL,
    OLLAMA_URL,
    PREDICT_URL,
    SIM_URL,
    STATS_URL,
    enveloped,
    error_enveloped,
    game_lines_payload,
    game_payload,
    insert_edge,
    mock_anthropic_messages,
)


def mock_edge_context(router: respx.MockRouter, game_id: str, game_external_id: str) -> None:
    """Stats + lines + prediction context for an EDGE_BREAKDOWN."""
    router.get(f"{STATS_URL}/api/v1/stats/games/{game_id}").mock(
        return_value=Response(200, json=enveloped(game_payload(game_id)))
    )
    router.get(f"{LINES_URL}/api/v1/lines/game/{game_external_id}").mock(
        return_value=Response(200, json=enveloped(game_lines_payload(game_external_id)))
    )
    router.get(f"{PREDICT_URL}/api/v1/predict/games/{game_id}/latest").mock(
        return_value=Response(404, json=error_enveloped("RESOURCE_NOT_FOUND", "no predictions"))
    )


class TestCreateAnalysis:
    def test_edge_breakdown_created_and_cached(self, client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"ext-{uuid.uuid4().hex[:10]}"
        edge_id = insert_edge(migrated_database_url, game_id=uuid.UUID(game_id), game_external_id=game_external_id)
        mock_edge_context(upstream, game_id, game_external_id)
        llm_route = mock_anthropic_messages(upstream, "## Summary\n\nThe model likes this edge.")

        response = client.post("/api/v1/agent/analysis", json={"analysis_type": "EDGE_BREAKDOWN", "edge_id": edge_id})
        assert response.status_code == 201, response.text
        data = response.json()["data"]
        assert data["analysis_type"] == "EDGE_BREAKDOWN"
        assert data["edge_id"] == edge_id
        assert data["game_id"] == game_id
        assert data["content"].startswith("## Summary")
        assert data["model_used"] == "claude-opus-4-8"
        assert data["title"].startswith("Edge Analysis:")
        assert "Edge" in data["input_summary"]
        assert llm_route.call_count == 1

        # Second identical request reuses the persisted analysis (200, no new LLM call)
        cached = client.post("/api/v1/agent/analysis", json={"analysis_type": "EDGE_BREAKDOWN", "edge_id": edge_id})
        assert cached.status_code == 200, cached.text
        assert cached.json()["data"]["id"] == data["id"]
        assert llm_route.call_count == 1

        # A free-form question bypasses the cache
        asked = client.post(
            "/api/v1/agent/analysis",
            json={"analysis_type": "EDGE_BREAKDOWN", "edge_id": edge_id, "question": "Why?"},
        )
        assert asked.status_code == 201
        assert llm_route.call_count == 2

        # Edge detail now surfaces the newest analysis summary
        detail = client.get(f"/api/v1/agent/edges/{edge_id}")
        assert detail.status_code == 200
        analysis = detail.json()["data"]["analysis"]
        assert analysis is not None
        assert analysis["title"].startswith("Edge Analysis:")

    def test_get_analysis_roundtrip_and_404(self, client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"ext-{uuid.uuid4().hex[:10]}"
        edge_id = insert_edge(migrated_database_url, game_id=uuid.UUID(game_id), game_external_id=game_external_id)
        mock_edge_context(upstream, game_id, game_external_id)
        mock_anthropic_messages(upstream)

        created = client.post("/api/v1/agent/analysis", json={"analysis_type": "EDGE_BREAKDOWN", "edge_id": edge_id})
        analysis_id = created.json()["data"]["id"]

        fetched = client.get(f"/api/v1/agent/analysis/{analysis_id}")
        assert fetched.status_code == 200
        assert fetched.json()["data"]["content"] == created.json()["data"]["content"]

        missing = client.get(f"/api/v1/agent/analysis/{uuid.uuid4()}")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "RESOURCE_NOT_FOUND"

    def test_validation_errors(self, client) -> None:
        no_game = client.post("/api/v1/agent/analysis", json={"analysis_type": "GAME_PREVIEW"})
        assert no_game.status_code == 422
        no_edge = client.post("/api/v1/agent/analysis", json={"analysis_type": "EDGE_BREAKDOWN"})
        assert no_edge.status_code == 422
        bad_type = client.post("/api/v1/agent/analysis", json={"analysis_type": "HOT_TAKE"})
        assert bad_type.status_code == 400  # pydantic literal -> VALIDATION_ERROR

    def test_unknown_edge_404(self, client) -> None:
        response = client.post(
            "/api/v1/agent/analysis", json={"analysis_type": "EDGE_BREAKDOWN", "edge_id": str(uuid.uuid4())}
        )
        assert response.status_code == 404

    def test_llm_failure_maps_to_dependency_error(self, client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"ext-{uuid.uuid4().hex[:10]}"
        edge_id = insert_edge(migrated_database_url, game_id=uuid.UUID(game_id), game_external_id=game_external_id)
        mock_edge_context(upstream, game_id, game_external_id)
        upstream.post("http://anthropic.test/v1/messages").mock(
            return_value=Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}})
        )

        response = client.post("/api/v1/agent/analysis", json={"analysis_type": "EDGE_BREAKDOWN", "edge_id": edge_id})
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "DEPENDENCY_ERROR"


@pytest.fixture(scope="module")
def ollama_client(migrated_database_url: str, redis_url: str) -> Iterator[TestClient]:
    """A second app instance configured for Ollama (ADR-011 switch)."""
    settings = Settings(
        database_url=migrated_database_url,
        redis_url=redis_url,
        statistics_service_url=STATS_URL,
        lines_service_url=LINES_URL,
        simulation_engine_url=SIM_URL,
        prediction_engine_url=PREDICT_URL,
        bookie_emulator_url=EMULATOR_URL,
        llm_provider="ollama",
        llm_base_url=OLLAMA_URL,
        llm_model="llama3.1:8b",
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


class TestOllamaProviderSwitch:
    def test_analysis_via_ollama(self, ollama_client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"ext-{uuid.uuid4().hex[:10]}"
        edge_id = insert_edge(migrated_database_url, game_id=uuid.UUID(game_id), game_external_id=game_external_id)
        mock_edge_context(upstream, game_id, game_external_id)
        ollama_route = upstream.post(f"{OLLAMA_URL}/api/chat").mock(
            return_value=Response(
                200,
                json={
                    "model": "llama3.1:8b",
                    "message": {"role": "assistant", "content": "## Summary\n\nLocal model take."},
                    "done": True,
                    "prompt_eval_count": 500,
                    "eval_count": 200,
                },
            )
        )

        response = ollama_client.post(
            "/api/v1/agent/analysis", json={"analysis_type": "EDGE_BREAKDOWN", "edge_id": edge_id}
        )
        assert response.status_code == 201, response.text
        data = response.json()["data"]
        assert data["model_used"] == "llama3.1:8b"
        assert data["content"].startswith("## Summary")
        assert ollama_route.called

    def test_health_reports_ollama_dependency(self, ollama_client, upstream) -> None:
        upstream.get(f"{OLLAMA_URL}/api/tags").mock(return_value=Response(200, json={"models": []}))
        for host in ("stats.test", "lines.test", "sim.test", "predict.test", "emulator.test"):
            upstream.route(host=host).mock(return_value=Response(500, json={"error": {}, "meta": {}}))
        response = ollama_client.get("/api/v1/agent/health")
        assert response.status_code == 200
        deps = response.json()["data"]["dependencies"]
        assert deps["ollama"] == "healthy"
        assert "anthropic_api" not in deps
