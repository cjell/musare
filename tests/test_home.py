"""The Home screen's derived rows. Reads the local history, never a model."""

from __future__ import annotations

import pytest

from musicshare import home
from musicshare.taste import has_history

pytestmark = pytest.mark.skipif(not has_history(), reason="no play history ingested")


def test_discoveries_are_shaped_for_the_page():
    for d in home.discoveries(4):
        assert d["name"] and d["plays"] > 0
        assert d["days"] >= home.MIN_DISCOVER_DAYS
        assert d["plays"] >= home.MIN_DISCOVER_PLAYS


def test_discoveries_respect_the_limit():
    assert len(home.discoveries(2)) <= 2


def test_a_collaboration_with_a_known_act_is_not_a_discovery():
    """The failure this guards: "you discovered Lana Del Rey", to someone who
    has played her for years, because the credit string was new."""
    found = {d["name"].lower() for d in home.discoveries(50)}
    known = {
        row[0].lower()
        for row in home._q("""
            with bounds as (select max(played_at) as tip from plays)
            select distinct lower(trim(unnest(string_split(artist_name, ','))))
            from plays
            where ms_played >= 30000 and artist_name is not null
              and played_at <= (select tip from bounds) - interval 30 day
        """)
    }
    for name in found:
        parts = {p.strip() for p in name.split(",")}
        assert not (parts & known), f"{name} contains an act played before the window"


def test_every_discovery_was_first_heard_inside_the_window():
    for d in home.discoveries(10):
        rows = home._q(
            """
            with bounds as (select max(played_at) as tip from plays)
            select min(played_at) > (select tip from bounds) - interval 30 day
            from plays where artist_name = ? and ms_played >= 30000
            """,
            [d["name"]],
        )
        assert rows[0][0], f"{d['name']} was played before the window"


def test_a_discovery_must_still_be_in_rotation():
    """'Kept' is a claim about now, not about three weeks ago."""
    for d in home.discoveries(10):
        rows = home._q(
            """
            with bounds as (select max(played_at) as tip from plays)
            select max(played_at) > (select tip from bounds) - interval 7 day
            from plays where artist_name = ? and ms_played >= 30000
            """,
            [d["name"]],
        )
        assert rows[0][0], f"{d['name']} has not been played in the last week"


def test_a_renamed_artist_is_not_a_discovery():
    """Chris Stussy became CHRIS STASSY in Aug 2026 - a new name, a catalogue
    this listener had played for two years. A discovery has to bring new songs."""
    found = {d["name"].lower() for d in home.discoveries(50)}
    assert "chris stassy" not in found


def test_song_discoveries_are_shaped_for_the_page():
    for t in home.song_discoveries(5):
        assert t["name"] and t["artist"] and t["uri"]
        assert t["plays"] >= home.MIN_DISCOVER_PLAYS
        assert t["days"] >= home.MIN_DISCOVER_DAYS


def test_a_song_may_be_by_an_artist_you_already_know():
    """The artist rule must not leak into the song rule - finding a great track
    by someone played for years is exactly what finding a song means."""
    songs = home.song_discoveries(20)
    new_artists = {d["name"].lower() for d in home.discoveries(50)}
    assert songs, "no song discoveries to check"
    assert any(t["artist"].lower() not in new_artists for t in songs)


def test_a_rereleased_song_is_not_a_discovery():
    """A fresh track id whose title and artist are already in the history is a
    re-release, the same rename problem arriving through a different door."""
    for t in home.song_discoveries(20):
        rows = home._q(
            """
            with bounds as (select max(played_at) as tip from plays)
            select count(*) from plays
            where lower(trim(track_name)) = lower(trim(?))
              and lower(trim(artist_name)) = lower(trim(?))
              and ms_played >= 30000
              and played_at <= (select tip from bounds) - interval 30 day
            """,
            [t["name"], t["artist"]],
        )
        assert rows[0][0] == 0, f"{t['name']} by {t['artist']} predates the window"


def test_song_discoveries_are_still_in_rotation():
    for t in home.song_discoveries(10):
        rows = home._q(
            """
            with bounds as (select max(played_at) as tip from plays)
            select max(played_at) > (select tip from bounds) - interval 7 day
            from plays where track_uri = ? and ms_played >= 30000
            """,
            [t["uri"]],
        )
        assert rows[0][0], f"{t['name']} has not been played in the last week"


def test_song_movers_are_shaped_for_the_page():
    for d in home._song_movers("down", 5):
        assert d["name"] and d["artist"] and d["uri"]
        assert d["plays_prev"] >= home.MIN_PRIOR_PLAYS_SONG
        assert d["pct"] < 0


def test_a_cooling_song_is_still_being_played():
    """Cooling means falling, not stopped - the same floor the artist query uses."""
    for d in home._song_movers("down", 5):
        assert d["plays"] >= home.MIN_RECENT_PLAYS_SONG


def test_a_climbing_song_actually_climbed():
    for d in home._song_movers("up", 5):
        assert d["plays"] > d["plays_prev"] and d["pct"] > 0


def test_movers_are_counted_in_plays_not_milliseconds():
    """Milliseconds are not comparable across the export and the live endpoint:
    live rows carry track duration as a stand-in, which runs about double for a
    listener who skips 75% of what he plays. A play is a play in both."""
    for rows in (home._movers("down", 2), home._song_movers("down", 2)):
        for r in rows:
            assert "plays" in r and "plays_prev" in r
            assert "minutes" not in r and "minutes_prev" not in r
    # And nothing in the weekly feed reports time any more.
    assert "hours" not in home.week_stats()
    assert all("minutes" not in d for d in home.discoveries(4))


def test_a_skipped_play_does_not_move_anything():
    """At 250 plays a day, a track that wanders in and is dropped after ten
    seconds would otherwise dominate a week-over-week percentage."""
    counted = home._q(f"""
        with bounds as (select max(played_at) as tip from plays)
        select count(*) from plays
        where played_at > (select tip from bounds) - interval 14 day
          and ms_played < {home.MIN_MS}
    """)[0][0]
    assert counted > 0, "no short plays in the window; the guard is untested"
    for r in home._movers("down", 4) + home._song_movers("down", 4):
        assert r["plays_prev"] >= 1
