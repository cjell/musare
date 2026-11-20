"""Hours and plays beside pinned songs and artists.

Over a small history built here, and with the stored profile replaced, so what
a visitor can and cannot ask about someone else's history is pinned down.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import pytest
from fastapi.testclient import TestClient

from musicshare import pinstats
from musicshare.api import app as app_mod
from musicshare.spec import chartrun

client = TestClient(app_mod.app)

# (track, artist, uri id, plays, ms each)
ROWS = [
    ("Fears", "MTNS", "a1", 4, 240_000),
    ("Fears - Remastered 2011", "MTNS", "a2", 2, 240_000),
    ("Fears", "Other Band", "a3", 3, 200_000),
    ("Numb & Getting Colder", "Flume, KUCKA", "a4", 2, 180_000),
    ("Skip", "Flume", "a5", 1, 10_000),
    ("Yonkers", "Tyler, The Creator", "a6", 2, 180_000),
]


@pytest.fixture
def plays(monkeypatch):
    con = duckdb.connect()
    con.execute("install icu; load icu;")
    con.execute(
        "create table plays (played_at timestamp, track_uri varchar, track_name varchar, "
        "artist_name varchar, album_name varchar, ms_played bigint, platform varchar, "
        "skipped boolean)"
    )
    rows, i = [], 0
    for track, artist, uid, n, ms in ROWS:
        for _ in range(n):
            when = datetime(2025, 1, 1) + timedelta(hours=i)
            rows.append((when, f"spotify:track:{uid}", track, artist, "LP", ms, "ios", False))
            i += 1
    con.executemany("insert into plays values (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    monkeypatch.setattr(chartrun, "_connect", lambda spec=None: con)
    return con


def test_a_song_counts_with_its_versions_and_not_namesakes(plays):
    got = pinstats.stats([{"id": "zz", "name": "fears", "artist_name": "MTNS"}], [])
    assert got["tracks"]["zz"] == {"hours": 0.4, "plays": 6}


def test_the_id_matches_on_its_own(plays):
    got = pinstats.stats([{"id": "a3", "name": "Something Else", "artist_name": "?"}], [])
    assert got["tracks"]["a3"]["plays"] == 3


def test_a_song_with_featured_names(plays):
    got = pinstats.stats([{"id": "q", "name": "Numb & Getting Colder", "artist_name": "Flume"}], [])
    assert got["tracks"]["q"]["plays"] == 2


def test_an_artist_counts_with_features_and_a_short_play_only_as_time(plays):
    got = pinstats.stats(
        [], [{"id": "f", "name": "flume"}, {"id": "t", "name": "Tyler, The Creator"}]
    )
    assert got["artists"]["f"]["plays"] == 2, "a ten-second play is not a listen"
    assert got["artists"]["f"]["hours"] == round((2 * 180_000 + 10_000) / 3_600_000, 2)
    assert got["artists"]["t"]["plays"] == 2, "a comma in a name keeps it whole"


def test_never_played_is_zero_and_junk_is_skipped(plays):
    got = pinstats.stats(
        [{"id": "n", "name": "Nothing", "artist_name": "Nobody"}, {"name": "no id"}, "junk"], None
    )
    assert got == {"tracks": {"n": {"hours": 0.0, "plays": 0}}, "artists": {}}


@pytest.fixture
def api(plays, monkeypatch):
    monkeypatch.setattr(app_mod.history_mod, "has_any", lambda uid=None: True)
    monkeypatch.setattr(app_mod.media_mod, "configured", lambda: True)
    saved = {"pins": {"tracks": [{"id": "a1", "name": "Fears", "artist_name": "MTNS"}]}}
    monkeypatch.setattr(app_mod.media_mod, "get_doc", lambda name, owner=None: saved)


def test_looking_at_a_profile_reads_its_saved_pins_only(api):
    """A visitor cannot point a song of their choosing at someone's history:
    the GET has no body, and the pins come from the saved profile."""
    r = client.get("/api/pins/stats")
    assert r.status_code == 200
    assert list(r.json()["tracks"]) == ["a1"] and r.json()["artists"] == {}


def test_your_own_pins_come_from_the_page(api):
    r = client.post("/api/pins/stats", json={"artists": [{"id": "x", "name": "MTNS"}]})
    assert r.json()["artists"]["x"]["plays"] == 6 and r.json()["tracks"] == {}


def test_too_many_pins_are_refused(api):
    many = [{"id": str(i), "name": "x"} for i in range(pinstats.MAX_ITEMS + 1)]
    assert client.post("/api/pins/stats", json={"artists": many}).status_code == 422


def test_the_numbers_are_viewable_like_a_profile():
    assert "/api/pins/stats" in app_mod.VIEWABLE
