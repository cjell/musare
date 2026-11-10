"""A chart pinned on someone's profile, refreshed for whoever is looking at it.

The executor and the profile store are replaced, so these pin down whose
history the chart runs over and where its spec comes from: the owner's saved
profile, never the request.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from musicshare import users
from musicshare.api import app as app_mod
from musicshare.spec.chartrun import ChartData

client = TestClient(app_mod.app)

SPEC = {"range": "7d", "dimension": "track", "metric": "plays", "limit": 20}


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    """Two accounts with history; bob has one chart pinned, alice none."""
    monkeypatch.setattr(users, "USERS_DIR", tmp_path / "users")
    for uid in ("alice", "bob"):
        users.save({"id": uid, "display_name": uid.title()})
    docs = {
        "alice": {"pins": {"charts": [None, None]}},
        "bob": {"pins": {"charts": [{"id": "c1", "spec": SPEC, "data": {}}, None]}},
    }
    monkeypatch.setattr(app_mod.history_mod, "has_any", lambda uid=None: True)
    monkeypatch.setattr(app_mod.media_mod, "configured", lambda: True)
    monkeypatch.setattr(
        app_mod.media_mod, "get_doc", lambda name, owner=None: docs[users.current()]
    )
    monkeypatch.setattr(app_mod.live_mod, "pull_if_stale", lambda: None)
    ran: list[tuple[str, dict]] = []

    def run(spec, back=0, scale_over=0):
        ran.append((users.current(), spec.model_dump()))
        return ChartData(["a"], [3.0], "track", "plays", "last 7 days", "bar")

    monkeypatch.setattr(app_mod, "run_chart", run)
    return docs, ran


def test_a_friends_pinned_chart_runs_over_their_history(profiles):
    _, ran = profiles
    r = client.get("/api/chart/pinned?slot=0&user=bob")
    assert r.status_code == 200 and r.json()["understood"] is True
    assert [who for who, _ in ran] == ["bob"]
    assert ran[0][1]["dimension"] == "track" and ran[0][1]["metric"] == "plays"


def test_the_spec_comes_from_the_profile_never_the_request(profiles):
    _, ran = profiles
    r = client.get(
        "/api/chart/pinned?slot=0&user=bob&dimension=artist&metric=hours",
    )
    assert r.status_code == 200
    assert ran[0][1]["dimension"] == "track", "query parameters cannot change the chart"


def test_an_empty_slot_is_not_found(profiles):
    assert client.get("/api/chart/pinned?slot=1&user=bob").status_code == 404
    assert client.get("/api/chart/pinned?slot=0&user=alice").status_code == 404


def test_a_slot_outside_the_pins_is_refused(profiles):
    assert client.get("/api/chart/pinned?slot=2&user=bob").status_code == 422
    assert client.get("/api/chart/pinned?slot=-1&user=bob").status_code == 422


def test_someone_who_does_not_exist_cannot_be_viewed(profiles):
    assert client.get("/api/chart/pinned?slot=0&user=nobody").status_code == 404


def test_a_pin_saved_against_an_older_schema_falls_back(profiles):
    docs, ran = profiles
    docs["bob"]["pins"]["charts"][0]["spec"] = {"range": "fortnight"}
    r = client.get("/api/chart/pinned?slot=0&user=bob")
    assert r.status_code == 200 and r.json()["understood"] is False
    assert ran == []


def test_the_older_profile_format_is_read_too(profiles):
    docs, ran = profiles
    docs["bob"] = {"0": docs["bob"]}
    assert client.get("/api/chart/pinned?slot=0&user=bob").status_code == 200
    assert [who for who, _ in ran] == ["bob"]
