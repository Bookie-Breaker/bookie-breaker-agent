"""POST /analysis/stream over real Postgres/Redis with respx-mocked LLM SSE.

Asserts the SSE frame grammar (chunk* then done|error), persistence and
cache-replay semantics, and that pre-stream failures stay JSON.
"""

import json
import uuid
from typing import Any

import respx
from httpx import Response

from tests.integration.conftest import (
    ANTHROPIC_URL,
    enveloped,
    game_payload,
    insert_edge,
)
from tests.integration.test_analysis_api import mock_edge_context
from tests.unit.test_llm_providers import anthropic_sse_body

STREAM_PATH = "/api/v1/agent/analysis/stream"


def mock_anthropic_stream(router: respx.MockRouter, deltas: list[str]) -> Any:
    return router.post(f"{ANTHROPIC_URL}/v1/messages").mock(
        return_value=Response(200, headers={"content-type": "text/event-stream"}, content=anthropic_sse_body(deltas))
    )


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        name = ""
        data = ""
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        events.append((name, json.loads(data)))
    return events


def stream_events(client: Any, payload: dict[str, Any]) -> tuple[int, str, list[tuple[str, dict[str, Any]]]]:
    with client.stream("POST", STREAM_PATH, json=payload) as response:
        content_type = response.headers.get("content-type", "")
        body = response.read().decode()
    if "text/event-stream" not in content_type:
        return response.status_code, content_type, []
    return response.status_code, content_type, parse_sse(body)


class TestAnalysisStreamApi:
    def test_chunks_then_done_persists_and_replays_cached(self, client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"ext-{uuid.uuid4().hex[:10]}"
        edge_id = insert_edge(migrated_database_url, game_id=uuid.UUID(game_id), game_external_id=game_external_id)
        mock_edge_context(upstream, game_id, game_external_id)
        llm_route = mock_anthropic_stream(upstream, ["## Sum", "mary of ", "the edge"])

        status, content_type, events = stream_events(client, {"analysis_type": "EDGE_BREAKDOWN", "edge_id": edge_id})
        assert status == 200
        assert "text/event-stream" in content_type
        assert [name for name, _ in events] == ["chunk", "chunk", "chunk", "done"]
        assert "".join(payload["text"] for name, payload in events[:-1]) == "## Summary of the edge"
        done = events[-1][1]
        assert done["meta"]["cached"] is False
        assert done["meta"]["request_id"]
        data = done["data"]
        assert data["analysis_type"] == "EDGE_BREAKDOWN"
        assert data["edge_id"] == edge_id
        assert data["content"] == "## Summary of the edge"
        assert llm_route.call_count == 1

        # the done event's id resolves via GET /analysis/{id}
        fetched = client.get(f"/api/v1/agent/analysis/{data['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["data"]["content"] == "## Summary of the edge"

        # identical request replays from cache: one full-content chunk, no LLM call
        status, _, cached_events = stream_events(client, {"analysis_type": "EDGE_BREAKDOWN", "edge_id": edge_id})
        assert status == 200
        assert [name for name, _ in cached_events] == ["chunk", "done"]
        assert cached_events[0][1]["text"] == "## Summary of the edge"
        assert cached_events[-1][1]["meta"]["cached"] is True
        assert cached_events[-1][1]["data"]["id"] == data["id"]
        assert llm_route.call_count == 1

    def test_question_streams_fresh_analysis(self, client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        upstream.get(f"http://stats.test/api/v1/stats/games/{game_id}").mock(
            return_value=Response(200, json=enveloped(game_payload(game_id)))
        )
        upstream.get(f"http://predict.test/api/v1/predict/games/{game_id}/latest").mock(return_value=Response(500))
        upstream.route(host="lines.test").mock(return_value=Response(500))
        llm_route = mock_anthropic_stream(upstream, ["Preview take"])

        status, _, events = stream_events(
            client, {"analysis_type": "GAME_PREVIEW", "game_id": game_id, "question": "Key injuries?"}
        )
        assert status == 200
        assert events[-1][0] == "done"
        assert llm_route.call_count == 1

    def test_pre_stream_failures_return_json_envelopes(self, client) -> None:
        missing = client.post(STREAM_PATH, json={"analysis_type": "EDGE_BREAKDOWN"})
        assert missing.status_code == 422
        assert missing.headers["content-type"].startswith("application/json")
        assert missing.json()["error"]["code"] == "UNPROCESSABLE_ENTITY"

        unknown = client.post(STREAM_PATH, json={"analysis_type": "EDGE_BREAKDOWN", "edge_id": str(uuid.uuid4())})
        assert unknown.status_code == 404
        assert unknown.json()["error"]["code"] == "RESOURCE_NOT_FOUND"

    def test_llm_down_is_a_json_502_not_sse(self, client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"ext-{uuid.uuid4().hex[:10]}"
        edge_id = insert_edge(migrated_database_url, game_id=uuid.UUID(game_id), game_external_id=game_external_id)
        mock_edge_context(upstream, game_id, game_external_id)
        upstream.post(f"{ANTHROPIC_URL}/v1/messages").mock(
            return_value=Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}})
        )

        response = client.post(STREAM_PATH, json={"analysis_type": "EDGE_BREAKDOWN", "edge_id": edge_id})
        assert response.status_code == 502
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"]["code"] == "DEPENDENCY_ERROR"

    def test_mid_stream_error_event_and_no_persistence(self, client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"ext-{uuid.uuid4().hex[:10]}"
        edge_id = insert_edge(migrated_database_url, game_id=uuid.UUID(game_id), game_external_id=game_external_id)
        mock_edge_context(upstream, game_id, game_external_id)
        # a valid stream start followed by an in-band Anthropic error event
        body = anthropic_sse_body(["partial "]).decode()
        head, _, _ = body.partition("event: content_block_stop")
        error_data = json.dumps({"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}})
        broken = head + f"event: error\ndata: {error_data}\n\n"
        upstream.post(f"{ANTHROPIC_URL}/v1/messages").mock(
            return_value=Response(200, headers={"content-type": "text/event-stream"}, content=broken.encode())
        )

        status, content_type, events = stream_events(client, {"analysis_type": "EDGE_BREAKDOWN", "edge_id": edge_id})
        assert status == 200  # headers were already sent; the error is in-band
        assert "text/event-stream" in content_type
        assert events[0][0] == "chunk"
        assert events[-1][0] == "error"
        assert events[-1][1]["code"] == "DEPENDENCY_ERROR"

        # nothing was persisted or cached: a retry hits the LLM again
        mock_anthropic_stream(upstream, ["recovered"])
        status, _, retry_events = stream_events(client, {"analysis_type": "EDGE_BREAKDOWN", "edge_id": edge_id})
        assert retry_events[-1][0] == "done"
        assert retry_events[-1][1]["meta"]["cached"] is False
