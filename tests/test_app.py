"""The web app: every dashboard renders from the committed artefacts, every API
answers, bad input is refused, and nothing ships invalid JSON.

The deployed container has models_out/ but no raw lap data, so the key tests
run with the processed-data directory pointed somewhere empty.
"""

from __future__ import annotations

import json
import re

import pytest
from starlette.testclient import TestClient

from app import main
from pitwall import config

PAGES = {
    "/": "A place on track",
    "/circuits": "Every circuit, measured",
    "/tyres": "Tyre wear, circuit by circuit",
    "/validation": "The validation report",
    "/simulator": "Run a race, 600 times",
}

needs_artefacts = pytest.mark.skipif(
    not (config.MODELS_OUT / "metrics.json").exists(), reason="fitted artefacts not built"
)


@pytest.fixture(scope="module")
def client():
    """App as deployed: artefacts present, raw lap tables absent."""
    original = config.PROCESSED
    config.PROCESSED = original.parent / "__absent_in_deployment__"
    try:
        with TestClient(main.app) as c:
            yield c
    finally:
        config.PROCESSED = original


def _page_data(html: str) -> dict:
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert m, "page is missing its data block"
    return json.loads(m.group(1))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


@needs_artefacts
@pytest.mark.parametrize("path,heading", PAGES.items())
def test_every_dashboard_renders_with_valid_data(client, path, heading):
    r = client.get(path)
    assert r.status_code == 200
    assert heading in r.text
    assert "Medhansh Shekhawat" in r.text
    assert re.search(r"pitwall\.css\?v=[0-9a-f]{10}", r.text), "asset URLs must be versioned"
    _page_data(r.text)  # raises on NaN or malformed JSON


@needs_artefacts
def test_no_ai_attribution_on_any_page(client):
    for path in PAGES:
        text = client.get(path).text.lower()
        for word in ("claude", "anthropic", "chatgpt", "openai"):
            assert word not in text, f"{word} appears on {path}"


@needs_artefacts
def test_withheld_audit_is_not_published(client):
    """The strategy audit failed its gate, so team and driver verdicts stay off the site."""
    for path in PAGES:
        data = _page_data(client.get(path).text)
        assert "audit_by_team" not in json.dumps(data)
        assert "team" not in (data.get("gate") or {})


@needs_artefacts
def test_json_apis_are_valid(client):
    for path in ("/api/track-position", "/api/degradation"):
        r = client.get(path)
        assert r.status_code == 200
        json.loads(r.text)  # strict: NaN would fail here


@needs_artefacts
def test_circuit_api(client):
    ok = client.get("/api/circuit/monza")
    assert ok.status_code == 200
    body = ok.json()
    assert body["race_laps"] > 40 and body["pit_loss_s"] > 10
    assert client.get("/api/circuit/not_a_track").status_code == 404


@needs_artefacts
def test_simulate_returns_a_distribution(client):
    r = client.post(
        "/api/simulate",
        json={
            "circuit": "monza",
            "stops": [[26, 2]],
            "start_rank": 1,
            "grid_position": 5,
            "n_sims": 200,
        },
    )
    assert r.status_code == 200
    d = r.json()
    assert sum(d["position_histogram"]) == 200
    assert 1 <= d["mean_position"] <= 12
    assert 0 <= d["win_rate"] <= d["podium_rate"] <= 1


@needs_artefacts
@pytest.mark.parametrize(
    "payload",
    [
        {"circuit": "monza", "stops": [[0, 2]]},
        {"circuit": "monza", "stops": [[26, 7]]},
        {"circuit": "monza", "n_sims": 50},
        {"circuit": "monza", "field_size": 40},
        {"circuit": "nowhere"},
    ],
)
def test_simulate_rejects_bad_input(client, payload):
    assert client.post("/api/simulate", json=payload).status_code in (400, 404, 422)


@needs_artefacts
def test_optimiser_returns_legal_plans(client):
    r = client.post("/api/optimise", json={"circuit": "monza", "grid_position": 5, "stops": []})
    assert r.status_code == 200
    cands = r.json()["candidates"]
    assert cands
    for c in cands:
        laps = [s[0] for s in c["stops"]]
        assert laps == sorted(laps)
        assert all(1 <= lap < 53 for lap in laps)
