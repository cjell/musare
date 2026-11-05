"""The watcher's decisions, run against the real functions in Supabase.

Marked integration because it needs the database, so it is deselected by
default. Every test runs inside a transaction that is rolled back, so nothing it
writes is ever visible to the jobs running against the same tables.

Each test first applies `supabase/capture.sql` inside that same transaction, so
what is tested is the file in the repository rather than whatever was last
installed, and a change can be tested before it is applied. Every real account
is paused inside the transaction too, so nothing here polls Spotify for one.

    ./mscs/python.exe -m pytest -m integration tests/test_capture_sql.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from musicshare.config import settings

pytestmark = pytest.mark.integration

SQL = Path(__file__).resolve().parents[1] / "supabase" / "capture.sql"
ME = "TESTlistener"
YOU = "TESTfriend"

T0 = datetime(2026, 9, 14, 12, 0, 0, tzinfo=UTC)
A = "spotify:track:TESTaaaaaaaaaaaaaaaaaa"
B = "spotify:track:TESTbbbbbbbbbbbbbbbbbb"


@pytest.fixture(scope="module")
def conn():
    if not settings().database_url:
        pytest.skip("DATABASE_URL is not set")
    import psycopg

    with psycopg.connect(settings().database_url, prepare_threshold=None) as c:
        yield c


@pytest.fixture
def db(conn):
    conn.execute("set local lock_timeout = '10s'")
    # Rows from before accounts, if this database still has any, go to a test
    # owner - and come back when the transaction rolls back.
    conn.execute("select set_config('listen.owner', 'TESTowner', true)")
    conn.execute(SQL.read_text(encoding="utf-8"))
    conn.execute("update listen.accounts set active = false")
    yield conn
    conn.rollback()


def snap(uri: str | None, prog: int = 0, playing: bool = True, dur: int = 200_000) -> dict | None:
    if uri is None:
        return None
    return {
        "is_playing": playing,
        "progress_ms": prog,
        "item": {
            "uri": uri,
            "name": uri[-4:],
            "duration_ms": dur,
            "artists": [{"name": "One"}, {"name": "Two"}],
            "album": {"name": "Al"},
        },
    }


def step(db, state, s, at):
    out = db.execute(
        "select listen.watch_step(%s::jsonb, %s::jsonb, %s)",
        [
            json.dumps(state) if state is not None else None,
            json.dumps(s) if s is not None else None,
            at,
        ],
    ).fetchone()[0]
    return out["state"], out["play"]


def run(db, *polls):
    """Feed (seconds after T0, snapshot) pairs through, collecting finished plays."""
    state, plays = None, []
    for secs, s in polls:
        state, play = step(db, state, s, T0 + timedelta(seconds=secs))
        if play:
            plays.append(play)
    return state, plays


def ended(play) -> datetime:
    return datetime.fromisoformat(play["played_at"])


# ------------------------------------------------------------------ plays


def test_a_track_that_keeps_playing_finishes_nothing(db):
    state, plays = run(db, (0, snap(A, 5_000)), (30, snap(A, 35_000)), (60, snap(A, 65_000)))
    assert plays == []
    assert state["heard_ms"] == 65_000


def test_the_next_track_finishes_the_last_one_when_it_started(db):
    """B is 5s in at the second poll, so A ended 5s before it - 25s after the
    first poll, at which point A had 30s left and 25s of it was heard."""
    _, plays = run(db, (0, snap(A, 170_000)), (30, snap(B, 5_000)))
    [a] = plays
    assert a["track_uri"] == A
    assert a["ms_played"] == 195_000
    assert ended(a) == T0 + timedelta(seconds=25)
    assert a["artist_name"] == "One, Two"


def test_a_skip_counts_what_was_heard_not_the_track_length(db):
    _, plays = run(db, (0, snap(A, 1_000)), (30, snap(A, 31_000)), (60, snap(B, 20_000)))
    assert plays[0]["ms_played"] == 41_000


def test_a_song_on_repeat_is_a_play_each_time(db):
    """The case recently-played lost: one song, ten times, reported twice."""
    polls = []
    t = 0
    for _ in range(3):
        for prog in (10_000, 40_000, 70_000, 100_000):  # a 110s track, polled every 30s
            polls.append((t, snap(A, prog, dur=110_000)))
            t += 30
    _, plays = run(db, *polls)
    assert len(plays) == 2, "two restarts seen; the third play is still going"
    assert all(p["track_uri"] == A and p["ms_played"] == 110_000 for p in plays)


def test_seeking_back_mid_track_is_not_a_repeat(db):
    state, plays = run(db, (0, snap(A, 150_000)), (30, snap(A, 60_000)))
    assert plays == []
    assert state["heard_ms"] == 150_000, "going backwards adds nothing"


def test_seeking_forward_counts_only_the_time_that_passed(db):
    state, _ = run(db, (0, snap(A, 50_000)), (30, snap(A, 150_000)))
    assert state["heard_ms"] == 50_000 + 32_000


def test_paused_time_is_not_listening(db):
    _, plays = run(
        db,
        (0, snap(A, 100_000)),
        (30, snap(A, 110_000, playing=False)),
        (600, snap(A, 110_000, playing=False)),
        (630, snap(B, 3_000)),
    )
    [a] = plays
    assert a["ms_played"] == 110_000
    assert ended(a) == T0 + timedelta(seconds=30), "dated when it last moved, not when B began"


def test_one_empty_answer_does_not_end_a_play(db):
    """Spotify blinks out between tracks and on device handoffs. Ending the play
    there and seeing it again next poll would count one listen twice."""
    state, plays = run(db, (0, snap(A, 10_000)), (30, None), (60, snap(A, 70_000)))
    assert plays == []
    assert "gone_at" not in state


def test_nothing_playing_for_a_while_ends_the_play(db):
    state, plays = run(db, (0, snap(A, 10_000)), (30, None), (60, None), (90, None))
    assert state is None
    [a] = plays
    assert a["ms_played"] == 40_000, "played on until the first empty poll, then stopped"


def test_a_flicker_under_a_second_is_not_a_play(db):
    _, plays = run(db, (0, snap(A, 100)), (1, snap(B, 500)))
    assert plays == []


def test_an_episode_reads_as_nothing_playing(db):
    episode = {
        "is_playing": True,
        "progress_ms": 1000,
        "item": {"uri": "spotify:episode:xyz", "name": "Pod"},
    }
    state, plays = run(db, (0, episode))
    assert state is None and plays == []


# ------------------------------------------------------------------- backup


def test_recently_played_rows_estimate_listening_from_the_gap(db):
    def item(when, ms):
        return {
            "played_at": when,
            "track": {
                "uri": A,
                "name": "S",
                "duration_ms": ms,
                "artists": [{"name": "X"}],
                "album": {"name": "Al"},
            },
        }

    body = {
        "items": [  # newest first, as Spotify sends them
            item("2026-09-14T12:04:05Z", 200_000),
            item("2026-09-14T12:00:45Z", 200_000),
            item("2026-09-14T12:00:00Z", 200_000),
        ]
    }
    rows = db.execute(
        "select ms_played from listen.recent_rows(%s::jsonb)", [json.dumps(body)]
    ).fetchall()
    assert [r[0] for r in rows] == [200_000, 45_000, 200_000]


# -------------------------------------------------------------------- dedupe


def add(db, uri, at, src, ms=100_000, name="A Song", artist="An Artist", uid=ME):
    return db.execute(
        "select listen.add_play(%s::jsonb, %s, %s)",
        [
            json.dumps(
                {
                    "played_at": at.isoformat(),
                    "track_uri": uri,
                    "ms_played": ms,
                    "track_name": name,
                    "artist_name": artist,
                }
            ),
            src,
            uid,
        ],
    ).fetchone()[0]


def rows_for(db, uri, uid=ME):
    return db.execute(
        "select source from listen.plays where track_uri = %s and user_id = %s order by source",
        [uri, uid],
    ).fetchall()


def test_the_watchers_row_replaces_the_backups_estimate(db):
    add(db, A, T0, "recent")
    add(db, A, T0 + timedelta(seconds=4), "watch")
    assert rows_for(db, A) == [("watch",)]


def test_the_backup_does_not_add_a_play_the_watcher_already_has(db):
    add(db, A, T0, "watch")
    assert add(db, A, T0 + timedelta(seconds=-6), "recent") == 0
    assert rows_for(db, A) == [("watch",)]


def test_the_same_backup_row_arriving_every_half_hour_is_kept_once(db):
    assert add(db, B, T0, "recent") == 1
    assert add(db, B, T0, "recent") == 0


def test_a_backup_play_the_watcher_missed_is_kept(db):
    add(db, A, T0, "watch")
    assert add(db, A, T0 + timedelta(minutes=10), "recent") == 1


def test_one_song_under_two_ids_is_one_play(db):
    """Power Trip came back under a different id from each endpoint, 0.2s apart,
    and was stored twice."""
    add(db, A, T0, "watch", name="Power Trip", artist="J. Cole, Miguel")
    assert (
        add(
            db,
            B,
            T0 + timedelta(milliseconds=200),
            "recent",
            name="power trip ",
            artist="J. Cole, Miguel",
        )
        == 0
    )
    assert rows_for(db, B) == []


def test_the_watcher_replaces_a_backup_row_filed_under_another_id(db):
    add(db, B, T0, "recent", name="Hello Juliet", artist="Clarion")
    add(db, A, T0 + timedelta(milliseconds=120), "watch", name="Hello Juliet", artist="Clarion")
    assert rows_for(db, B) == []
    assert rows_for(db, A) == [("watch",)]


def test_different_songs_ending_together_are_both_kept(db):
    add(db, A, T0, "watch", name="One", artist="X")
    assert add(db, B, T0, "recent", name="Two", artist="X") == 1


def test_a_repeat_past_the_tolerance_is_its_own_play(db):
    add(db, A, T0, "watch")
    assert add(db, A, T0 + timedelta(seconds=40), "recent") == 1


def test_two_people_playing_the_same_song_are_two_plays(db):
    """Dedupe is within one account. Across two it would delete a friend's play."""
    add(db, A, T0, "watch", uid=ME)
    assert add(db, A, T0 + timedelta(seconds=3), "recent", uid=YOU) == 1
    add(db, A, T0 + timedelta(seconds=5), "watch", uid=YOU)
    assert rows_for(db, A, ME) == [("watch",)]
    assert rows_for(db, A, YOU) == [("watch",)]


# ------------------------------------------------------------------- accounts


def secret(db, uid):
    row = db.execute(
        "select decrypted_secret from vault.decrypted_secrets where name = %s",
        [f"spotify_refresh_token:{uid}"],
    ).fetchone()
    return row and row[0]


def test_enrolling_adds_the_account_and_its_token(db):
    db.execute("select listen.enroll(%s, %s)", [YOU, "first-token"])
    assert db.execute(
        "select active from listen.accounts where user_id = %s", [YOU]
    ).fetchone() == (True,)
    assert secret(db, YOU) == "first-token"


def test_enrolling_again_replaces_the_token_and_forgets_the_old_access(db):
    db.execute("select listen.enroll(%s, %s)", [YOU, "first-token"])
    db.execute(
        "insert into listen.token (user_id, access_token, expires_at) values (%s, 'old', now() + interval '1 hour')",
        [YOU],
    )
    db.execute("update listen.accounts set active = false where user_id = %s", [YOU])
    db.execute("select listen.enroll(%s, %s)", [YOU, "second-token"])
    assert secret(db, YOU) == "second-token"
    assert (
        db.execute("select count(*) from listen.token where user_id = %s", [YOU]).fetchone()[0] == 0
    )
    assert db.execute(
        "select active from listen.accounts where user_id = %s", [YOU]
    ).fetchone() == (True,), "signing in again resumes a paused account"


def test_an_account_that_cannot_be_polled_is_recorded_against_itself(db):
    """No refresh token: the poll fails before any request, and says whose it was."""
    db.execute("insert into listen.accounts (user_id) values (%s)", [YOU])
    db.execute("select listen.watch_one(%s)", [YOU])
    err = db.execute(
        "select last_error from listen.status where job = 'watch' and user_id = %s", [YOU]
    ).fetchone()[0]
    assert "no refresh token" in err
    assert (
        db.execute(
            "select count(*) from listen.status where job = 'watch' and user_id = %s", [ME]
        ).fetchone()[0]
        == 0
    ), "nobody else's status moved"


def test_applying_the_file_twice_changes_nothing(db):
    before = db.execute("select count(*) from listen.plays").fetchone()[0]
    db.execute(SQL.read_text(encoding="utf-8"))
    assert db.execute("select count(*) from listen.plays").fetchone()[0] == before
