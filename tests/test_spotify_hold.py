"""The site stops asking Spotify anything while Spotify has asked it to wait.

Its rate limit is on the app's client id, so a 429 for one account is a 429 for
all of them, and asking again straight away keeps the limit in place.
"""

from __future__ import annotations

import httpx
import pytest

from musicshare import live, playlists


@pytest.fixture
def spotify(monkeypatch):
    """A Spotify that answers each request with the next queued response."""
    asked: list[str] = []
    queue: list[httpx.Response] = []

    def get(url, **kw):
        asked.append(url)
        return queue.pop(0)

    monkeypatch.setattr(live.httpx, "get", get)
    monkeypatch.setattr(playlists.httpx, "get", get)
    monkeypatch.setattr(live, "access_token", lambda uid=None: "tok")
    monkeypatch.setattr(playlists, "access_token", lambda uid=None: "tok")
    monkeypatch.setattr(live, "_hold_until", 0.0)
    live._now_cache.clear()
    yield asked, queue
    live._now_cache.clear()


def limited(after: str | None = "120") -> httpx.Response:
    headers = {"Retry-After": after} if after is not None else {}
    return httpx.Response(429, headers=headers, text='{"error":"QUOTA_EXCEEDED"}')


def test_a_429_pauses_now_playing_for_as_long_as_spotify_says(spotify):
    asked, queue = spotify
    queue.append(limited("120"))
    assert live.now_playing("a", ttl=0) is None
    assert 115 < live.spotify_held() <= 120
    assert live.now_playing("a", ttl=0) is None
    assert live.now_playing("b", ttl=0) is None, "every account waits, not just the one refused"
    assert len(asked) == 1


def test_a_429_naming_no_wait_pauses_a_minute(spotify):
    _, queue = spotify
    queue.append(limited(None))
    live.now_playing("a", ttl=0)
    assert 55 < live.spotify_held() <= 60


def test_playlists_wait_too_and_a_pause_they_hit_holds_now_playing(spotify):
    asked, queue = spotify
    queue.append(limited("30"))
    assert playlists._get("/me/playlists", uid="a") is None
    assert playlists._get("/me/playlists", uid="a") is None
    assert live.now_playing("a", ttl=0) is None
    assert len(asked) == 1


def test_one_answer_per_account_is_shared_by_every_page_for_a_while(spotify):
    asked, queue = spotify
    queue.append(httpx.Response(204))
    for _ in range(5):
        live.now_playing("a")
    assert len(asked) == 1 and live.NOW_TTL >= 20
