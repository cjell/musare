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


@pytest.mark.skipif(not app_mod.genrelib.TABLE.exists(), reason="no genre serving table built")
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


# ------------------------------------------------------- the profile document


def test_profile_rejects_things_that_are_not_objects(monkeypatch):
    """The page stores one object; an array or a bare string would round-trip
    and then fail to merge on the way back out."""
    monkeypatch.setattr(app_mod.media_mod, "configured", lambda: True)
    monkeypatch.setattr(app_mod.media_mod, "put_doc", lambda *a, **k: "x")
    assert client.put("/api/profile", content=b"[1,2]").status_code == 400
    assert client.put("/api/profile", content=b'"hello"').status_code == 400
    assert client.put("/api/profile", content=b"not json at all").status_code == 400
    assert client.put("/api/profile", json={"ok": True}).status_code == 200


def test_an_oversized_profile_is_refused(monkeypatch):
    monkeypatch.setattr(app_mod.media_mod, "configured", lambda: True)
    big = b'{"x":"' + b"a" * (app_mod.MAX_PROFILE_BYTES + 10) + b'"}'
    assert client.put("/api/profile", content=big).status_code == 413


def test_a_profile_that_cannot_be_read_is_not_an_error(monkeypatch):
    """A page that cannot reach storage falls back to what the browser holds,
    which is where all of this used to live - so this must not 500."""

    def boom(*a, **k):
        raise RuntimeError("storage gone")

    monkeypatch.setattr(app_mod.media_mod, "configured", lambda: True)
    monkeypatch.setattr(app_mod.media_mod, "get_doc", boom)
    r = client.get("/api/profile")
    assert r.status_code == 200
    assert r.json() == {"profile": None, "stored": False}
