"""The status endpoint: what the avatar is told, and when."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from musicshare.api import app as app_mod

client = TestClient(app_mod.app)


def playing(artist: str, is_playing: bool = True) -> dict:
    return {
        "track": "A Song",
        "artist": artist,
        "album": "An Album",
        "art": None,
        "is_playing": is_playing,
        "progress_ms": 1000,
        "duration_ms": 200000,
        "url": None,
        "uri": "spotify:track:x",
    }


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(app_mod.live_mod, "last_played", lambda: None)


def test_nothing_playing_is_sleeping(monkeypatch):
    monkeypatch.setattr(app_mod.live_mod, "now_playing", lambda: None)
    assert client.get("/api/now").json()["state"] == "sleeping"


def test_paused_is_sleeping(monkeypatch):
    """Spotify keeps returning the track with is_playing false, so without this
    the avatar sits on whatever was playing when you stopped."""
    monkeypatch.setattr(app_mod.live_mod, "now_playing", lambda: playing("Metallica", False))
    body = client.get("/api/now").json()
    assert body["state"] == "sleeping"
    # The row still says what is paused - only the avatar treats it as silence.
    assert body["now"]["artist"] == "Metallica"


def test_a_known_artist_resolves_on_a_cold_process(monkeypatch):
    """No map request has been made, so the fitted artefacts are not loaded."""
    monkeypatch.setattr(app_mod.live_mod, "now_playing", lambda: playing("Metallica"))
    monkeypatch.setattr(app_mod, "_atlas_cache", {})
    assert client.get("/api/now").json()["state"] == "angry"


def test_an_unplaceable_artist_is_unknown(monkeypatch):
    monkeypatch.setattr(
        app_mod.live_mod, "now_playing", lambda: playing("not a real artist at all")
    )
    assert client.get("/api/now").json()["state"] == "unknown"


def test_a_lookup_failure_does_not_break_the_status(monkeypatch):
    def boom(_):
        raise RuntimeError("storage gone")

    monkeypatch.setattr(app_mod.live_mod, "now_playing", lambda: playing("Metallica"))
    monkeypatch.setattr(app_mod.genrelib, "state_of", boom)
    assert client.get("/api/now").json()["state"] == "unknown"
