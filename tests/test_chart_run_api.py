"""Re-running a saved chart: what a live chart on Home asks the server for.

No model and no history: the executor is replaced, so these pin down the
endpoint's contract - runs a spec, never reads a question, pulls fresh plays
first, and refuses a spec it cannot draw or no longer recognises.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from musicshare.api import app as app_mod
from musicshare.spec.chartrun import ChartData

client = TestClient(app_mod.app)


def _fake(monkeypatch, data: ChartData | None = None) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(app_mod.live_mod, "pull_if_stale", lambda: calls.append("pull"))
    answer = data or ChartData(["00", "01"], [0.0, 1.5], "hour of day", "hours", "today", "bar")
    monkeypatch.setattr(app_mod, "run_chart", lambda spec: answer)
    return calls


def test_a_saved_spec_runs_without_a_model(monkeypatch):
    def no_model(*a, **k):
        raise AssertionError("re-running a chart must not read the question again")

    calls = _fake(monkeypatch)
    monkeypatch.setattr(app_mod, "generate", no_model)
    r = client.post("/api/chart/run", json={"range": "today", "dimension": "hour_of_day"})
    body = r.json()
    assert r.status_code == 200
    assert body["understood"] and not body["empty"]
    assert body["data"]["values"] == [0.0, 1.5]
    assert "cumulative" in body["data"] and "partial_last" in body["data"]
    assert calls == ["pull"], "a live chart has to include plays captured a minute ago"


def test_a_spec_that_cannot_be_drawn_is_refused_not_run(monkeypatch):
    calls = _fake(monkeypatch)
    r = client.post("/api/chart/run", json={"dimension": "artist", "series": "artist"})
    assert r.status_code == 200
    assert r.json()["understood"] is False
    assert calls == [], "nothing is pulled for a chart that will not be drawn"


def test_a_spec_from_an_older_schema_is_rejected_before_it_runs(monkeypatch):
    """The page turns this into 'ask it again' instead of a broken Home."""
    _fake(monkeypatch)
    assert client.post("/api/chart/run", json={"range": "fortnight"}).status_code == 422


def test_nothing_played_yet_is_an_answer_not_a_failure(monkeypatch):
    _fake(monkeypatch, ChartData([], [], "track", "hours", "top songs today", "bar"))
    body = client.post("/api/chart/run", json={"range": "today", "dimension": "track"}).json()
    assert body["understood"] and body["empty"]
