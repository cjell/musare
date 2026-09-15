"""Changing a chart: what reaches the model, and what comes back. No model, no history."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from musicshare.api import app as app_mod
from musicshare.spec.chart import ChartSpec
from musicshare.spec.chartrun import ChartData
from musicshare.spec.generate import REFINE, Generated, refine_input

client = TestClient(app_mod.app)
BASE = {"dimension": "artist", "range": "30d", "metric": "plays", "limit": 5}


def _fake(monkeypatch, answer: ChartSpec) -> tuple[dict, list]:
    seen: dict = {}
    runs: list = []

    def fake_generate(text, task=None, model=None):
        seen["text"], seen["task"] = text, task
        return Generated(answer, "fake", 0, 0, 0)

    def fake_run(spec):
        runs.append(spec)
        return ChartData(["a"], [1.0], "artist", "plays", "t", "bar")

    monkeypatch.setattr(app_mod, "generate", fake_generate)
    monkeypatch.setattr(app_mod, "run_chart", fake_run)
    return seen, runs


def test_the_model_sees_the_chart_as_it_stands_and_the_change(monkeypatch):
    seen, runs = _fake(monkeypatch, ChartSpec(**{**BASE, "range": "7d"}))
    r = client.post("/api/chart/refine", json={"spec": BASE, "change": "just this week"})
    body = r.json()
    assert r.status_code == 200 and body["understood"]
    assert seen["task"] is REFINE
    assert "just this week" in seen["text"]
    assert '"metric": "plays"' in seen["text"], "the fields to keep have to be in front of it"
    assert body["spec"]["range"] == "7d" and len(runs) == 1


def test_a_refused_change_is_not_drawn(monkeypatch):
    _, runs = _fake(monkeypatch, ChartSpec(understood=False))
    body = client.post("/api/chart/refine", json={"spec": BASE, "change": "someone else's"}).json()
    assert body["understood"] is False and runs == []


def test_a_change_that_cannot_be_drawn_is_refused_like_any_chart(monkeypatch):
    _, runs = _fake(monkeypatch, ChartSpec(dimension="artist", series="artist"))
    body = client.post("/api/chart/refine", json={"spec": BASE, "change": "split it"}).json()
    assert body["understood"] is False and body["problems"] and runs == []


def test_nothing_reaches_the_model_without_a_change_and_a_valid_chart(monkeypatch):
    seen, _ = _fake(monkeypatch, ChartSpec())
    assert client.post("/api/chart/refine", json={"spec": BASE, "change": ""}).status_code == 422
    assert client.post("/api/chart/refine", json={"spec": BASE, "change": "   "}).status_code == 422
    bad = {"spec": {"range": "fortnight"}, "change": "make it plays"}
    assert client.post("/api/chart/refine", json=bad).status_code == 422
    assert "text" not in seen


def test_the_prompt_leaves_understood_for_the_model_to_decide():
    text = refine_input(ChartSpec(**BASE), "  make it hours  ")
    shown = text.split("Current chart:\n", 1)[1].split("\n\nRequested change:", 1)[0]
    fields = json.loads(shown)
    assert "understood" not in fields and fields["metric"] == "plays"
    assert text.endswith("Requested change:\nmake it hours")
