"""End-to-end pipeline flow against the full docker-compose stack.

Run with: BB_E2E_STACK=1 uv run pytest tests/e2e -m e2e
Excluded from the default pytest run (testpaths covers unit + integration).
"""

import os
import time

import httpx
import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.environ.get("BB_E2E_STACK") != "1", reason="requires the full service stack"),
]

AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8006")
EMULATOR_URL = os.environ.get("BOOKIE_EMULATOR_URL", "http://localhost:8005")
LEAGUE = os.environ.get("BB_E2E_LEAGUE", "NBA")


@pytest.fixture(scope="module")
def http() -> httpx.Client:
    with httpx.Client(timeout=httpx.Timeout(10.0, read=120.0)) as client:
        yield client


def test_full_pipeline(http: httpx.Client) -> None:
    # 1. the slate has games for today
    slate = http.get(f"{AGENT_URL}/api/v1/agent/slate", params={"league": LEAGUE})
    assert slate.status_code == 200, slate.text
    games = slate.json()["data"]["games"]
    if not games:
        pytest.skip(f"no {LEAGUE} games on today's slate")

    # 2. trigger a pipeline run with auto-bet
    run_response = http.post(f"{AGENT_URL}/api/v1/agent/pipeline/run", json={"league": LEAGUE, "auto_bet": True})
    assert run_response.status_code == 202, run_response.text
    run_id = run_response.json()["data"]["pipeline_run_id"]

    # 3. poll until terminal
    deadline = time.monotonic() + 600
    run = None
    while time.monotonic() < deadline:
        run = http.get(f"{AGENT_URL}/api/v1/agent/pipeline/runs/{run_id}").json()["data"]
        if run["status"] not in ("QUEUED", "RUNNING"):
            break
        time.sleep(2.0)
    assert run is not None
    assert run["status"] in ("COMPLETED", "COMPLETED_WITH_ERRORS"), run

    # 4. edges were persisted
    edges = http.get(f"{AGENT_URL}/api/v1/agent/edges", params={"league": LEAGUE}).json()["data"]
    if not edges:
        pytest.skip("pipeline found no edges on today's market (nothing to grade)")

    bet_edges = [edge for edge in edges if edge["paper_bet_id"]]
    if not bet_edges:
        pytest.skip("no edge cleared the auto-bet gate (nothing to grade)")
    bet_id = bet_edges[0]["paper_bet_id"]

    # 5. the emulator shows the bet
    bet = http.get(f"{EMULATOR_URL}/api/v1/emulator/bets/{bet_id}")
    assert bet.status_code == 200, bet.text
    assert bet.json()["data"]["result"] in ("PENDING", "WIN", "LOSS", "PUSH", "VOID")

    # 6. force-grade and confirm performance reflects the grade
    before = http.get(f"{EMULATOR_URL}/api/v1/emulator/performance").json()["data"]
    grade = http.post(f"{EMULATOR_URL}/api/v1/emulator/bets/{bet_id}/grade", json={"force": True})
    if grade.status_code == 422:
        pytest.skip("game has not completed yet; grading unavailable")
    assert grade.status_code == 200, grade.text
    graded = grade.json()["data"]
    assert graded["result"] in ("WIN", "LOSS", "PUSH", "VOID")

    after = http.get(f"{EMULATOR_URL}/api/v1/emulator/performance").json()["data"]
    assert after["total_bets"] >= before["total_bets"]
    assert (
        after["total_wins"] + after["total_losses"] + after["total_pushes"]
        >= before["total_wins"] + before["total_losses"] + before["total_pushes"]
    )
