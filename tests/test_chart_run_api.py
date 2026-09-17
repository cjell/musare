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
    """Replace the executor, recording how it was called. `back` is recorded too:
    a live chart of today asks for earlier days through the same endpoint."""
    calls: list[str] = []
    monkeypatch.setattr(app_mod.live_mod, "pull_if_stale", lambda: calls.append("pull"))
    answer = data or ChartData(["00", "01"], [0.0, 1.5], "hour of day", "hours", "today", "bar")

    def run(spec, back=0):
        calls.append(f"run:{back}")
        return answer

    monkeypatch.setattr(app_mod, "run_chart", run)
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
    assert calls == ["pull", "run:0"], "fresh plays are pulled before the chart is run"


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


def test_an_earlier_day_is_asked_for_through_the_same_endpoint(monkeypatch):
    """A past day is recomputed, not stored, so it is the same call with an offset."""
    calls = _fake(monkeypatch)
    r = client.post("/api/chart/run?back=2", json={"range": "today", "dimension": "hour_of_day"})
    assert r.status_code == 200 and "run:2" in calls


def test_the_offset_defaults_to_today_and_refuses_nonsense(monkeypatch):
    calls = _fake(monkeypatch)
    client.post("/api/chart/run", json={"range": "today", "dimension": "hour_of_day"})
    assert "run:0" in calls
    bad = client.post("/api/chart/run?back=-1", json={"range": "today", "dimension": "hour_of_day"})
    assert bad.status_code == 422
