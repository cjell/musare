"""Shows are searched around each account's own saved location.

Ticketmaster is replaced, so these pin down where the search is centred, that
nothing is searched before a location is saved, and that a location is stored
coarsely and per account.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from musicshare import shows, users
from musicshare.api import app as app_mod

client = TestClient(app_mod.app)


@pytest.fixture
def searched(one_account, monkeypatch):
    """Record where Ticketmaster was asked to look; answer with no events."""
    monkeypatch.setattr(app_mod.history_mod, "has_any", lambda uid=None: True)
    monkeypatch.setattr(shows, "_my_artists", lambda: {})
    monkeypatch.setattr(shows, "fan_counts", lambda names: {})
    calls: list[tuple[float, float]] = []

    def fetch(lat=None, lon=None, **kw):
        calls.append((lat, lon))
        return []

    monkeypatch.setattr(shows, "fetch_events", fetch)
    return calls


def test_no_location_means_no_search(searched):
    r = client.get("/api/shows")
    assert r.status_code == 409 and r.json()["detail"] == "no_location"
    assert searched == [], "never somewhere else's shows in place of yours"
    assert client.get("/api/location").json() == {"location": None}


def test_shows_are_searched_around_the_saved_location(searched):
    assert client.post("/api/location", json={"lat": 40.7128, "lon": -74.006}).status_code == 200
    assert client.get("/api/shows").status_code == 200
    assert searched == [(40.71, -74.01)]


def test_a_location_is_kept_to_about_a_kilometre(searched):
    body = client.post("/api/location", json={"lat": 35.913212, "lon": -79.055845}).json()
    assert body["location"] == {"lat": 35.91, "lon": -79.06}
    assert users.load("tester")["location"] == {"lat": 35.91, "lon": -79.06}


def test_moving_drops_the_show_list_for_the_old_place(searched):
    client.post("/api/location", json={"lat": 35.91, "lon": -79.06})
    client.get("/api/shows")
    assert users.cache(shows.CACHE_NAME, "tester").exists()
    client.post("/api/location", json={"lat": 40.71, "lon": -74.01})
    assert not users.cache(shows.CACHE_NAME, "tester").exists()
    client.get("/api/shows")
    assert searched[-1] == (40.71, -74.01)


@pytest.mark.parametrize(
    "bad",
    [{"lat": 91, "lon": 0}, {"lat": 0, "lon": -181}, {"lat": "here"}, {}],
)
def test_a_place_that_is_not_on_earth_is_refused(searched, bad):
    assert client.post("/api/location", json=bad).status_code == 422
    assert users.load("tester").get("location") is None


def test_a_location_belongs_to_one_account(searched):
    users.save({"id": "other", "display_name": "Other"})
    shows.set_location(51.5, -0.12, uid="other")
    assert shows.location("other") == (51.5, -0.12)
    assert shows.location("tester") is None
