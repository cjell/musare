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


def test_found_artists_are_listed_most_played_first():
    """Sapian at 6 plays over 3 days sat above M-High at 11 over 2."""
    found = home.discoveries(10)
    keys = [(d["plays"], d["days"]) for d in found]
    assert keys == sorted(keys, reverse=True)


def test_back_in_rotation_is_what_climbing_and_found_cannot_hold():
    """One play last week and seven this week went nowhere on Home."""
    back = home.back_in_rotation(50)
    for a in back:
        assert a["plays_prev"] < home.MIN_PRIOR_PLAYS_ARTIST
        assert a["plays"] >= home.MIN_BACK_PLAYS_ARTIST
    names = {a["name"] for a in back}
    assert not names & {a["name"] for a in home._movers("up", 50)}, "one place or the other"
    assert not names & {a["name"] for a in home.discoveries(50)}, "someone new is found, not back"


def test_songs_back_in_rotation_follow_the_song_floors_most_played_first():
    rows = home.songs_back_in_rotation(10)
    for t in rows:
        assert t["plays_prev"] < home.MIN_PRIOR_PLAYS_SONG
        assert t["plays"] >= home.MIN_BACK_PLAYS_SONG
    plays = [t["plays"] for t in rows]
    assert plays == sorted(plays, reverse=True)


def test_on_repeat_only_lists_songs_played_again_and_again():
    """A thin week comes up short rather than padding with songs heard once."""
    rows = home.on_repeat(20)
    assert len(rows) <= 20
    assert all(r["plays"] >= home.MIN_REPEAT_PLAYS for r in rows)
    plays = [r["plays"] for r in rows]
    assert plays == sorted(plays, reverse=True)


def test_a_collaboration_with_a_known_act_is_not_a_discovery():
    """The failure this guards: "you discovered Lana Del Rey", to someone who
    has played her for years, because the credit string was new."""
    found = {d["name"].lower() for d in home.discoveries(50)}
    known = {
        row[0].lower()
        for row in home._q(f"""
            with bounds as (select max(played_at) as tip from plays)
            select distinct lower(trim(unnest(string_split(artist_name, ','))))
            from plays
            where ms_played >= 30000 and artist_name is not null
              and played_at <= (select tip from bounds) - interval {home.DISCOVER_DAYS} day
        """)
    }
    for name in found:
        parts = {p.strip() for p in name.split(",")}
        assert not (parts & known), f"{name} contains an act played before the window"


def test_every_discovery_was_first_heard_inside_the_window():
    for d in home.discoveries(10):
        rows = home._q(
            f"""
            with bounds as (select max(played_at) as tip from plays)
            select min(played_at) > (select tip from bounds) - interval {home.DISCOVER_DAYS} day
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


def test_a_song_may_be_by_an_artist_you_already_know(monkeypatch):
    """The artist rule must not leak into the song rule - finding a great track
    by someone played for years is exactly what finding a song means.

    Checked over a month rather than the live week. This proves a property of the
    rule by sampling real data, and a week can legitimately hold one song by one
    new act - which says nothing about the rule and fails the test anyway."""
    monkeypatch.setattr(home, "DISCOVER_DAYS", 30)
    songs = home.song_discoveries(20)
    new_artists = {d["name"].lower() for d in home.discoveries(50)}
    assert songs, "no song discoveries to check"
    assert any(t["artist"].lower() not in new_artists for t in songs)


def test_a_rereleased_song_is_not_a_discovery():
    """A fresh track id whose title and artist are already in the history is a
    re-release, the same rename problem arriving through a different door."""
    for t in home.song_discoveries(20):
        rows = home._q(
            f"""
            with bounds as (select max(played_at) as tip from plays)
            select count(*) from plays
            where lower(trim(track_name)) = lower(trim(?))
              and lower(trim(artist_name)) = lower(trim(?))
              and ms_played >= 30000
              and played_at <= (select tip from bounds) - interval {home.DISCOVER_DAYS} day
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
    assert all("minutes" not in d for d in home.discoveries(4))


def test_the_week_strip_reports_both_units():
    """Plays are what the movers compare on, because a count means the same in
    both sources. Hours are what a person has a feel for, so the strip carries
    them too - on the estimate live.py derives, not on full track durations."""
    w = home.week_stats()
    for k in ("plays", "plays_prev", "hours", "hours_prev", "tracks", "new_artists"):
        assert k in w, f"week strip lost {k}"
    assert w["hours"] >= 0 and w["plays"] >= 0


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


# ------------------------------------------------------------ feed plumbing


def test_the_discovery_window_is_a_week_like_the_rest_of_the_feed():
    assert home.DISCOVER_DAYS == 7


def test_a_cached_feed_is_served_only_while_the_newest_play_is_unchanged():
    """The old rule was a 30-minute clock, which could not see plays the
    scheduled poller wrote out of band."""
    tip = "2026-09-14T16:47:13"
    fresh = {"v": home.CACHE_VERSION, "through": tip}
    assert home._cache_usable(fresh, tip)
    assert not home._cache_usable(fresh, "2026-09-14T17:02:40"), "a newer play makes it stale"
    assert not home._cache_usable({"v": home.CACHE_VERSION - 1, "through": tip}, tip), (
        "an older cache version is rebuilt even with no new plays"
    )
    assert not home._cache_usable(None, tip)
    assert not home._cache_usable(fresh, None)


def test_the_picture_comes_from_the_act_credited_on_the_played_track():
    """Searching "My Friend" ranks Mark Lee first; the played track credits My
    Friend by id, which no name search can get wrong."""
    track = {
        "artists": [{"name": "My Friend", "id": "right"}, {"name": "Tommy Farrow", "id": "other"}]
    }
    assert home._credited(track, "My Friend") == "right"
    assert home._credited(track, "my friend") == "right"


def test_a_joined_live_credit_resolves_on_its_first_act():
    track = {
        "artists": [{"name": "My Friend", "id": "right"}, {"name": "Tommy Farrow", "id": "other"}]
    }
    assert home._credited(track, "My Friend, Tommy Farrow") == "right"


def test_an_act_not_on_the_track_is_not_guessed():
    track = {"artists": [{"name": "Mark Lee", "id": "wrong"}]}
    assert home._credited(track, "My Friend") is None
