"""How live rows are written down and when they are allowed to be deleted.

No network: these drive `_write` and `prune` directly, which is where the two
bugs that mattered lived.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
import pytest

from musicshare import live


def row(when: datetime, uri: str = "spotify:track:a") -> dict:
    return {
        "played_at": when,
        "track_uri": uri,
        "track_name": "A Song",
        "artist_name": "An Artist",
        "album_name": "An Album",
        "ms_played": 210_000,
    }


@pytest.fixture
def store(tmp_path, monkeypatch):
    d = tmp_path / "live"
    d.mkdir()
    monkeypatch.setattr(live, "LIVE_DIR", d)
    return d


def rows_in(path) -> int:
    return duckdb.connect().execute(
        f"select count(*) from read_parquet('{path.as_posix()}')"
    ).fetchone()[0]


def test_a_days_plays_land_in_one_file(store):
    live._write([row(datetime(2026, 9, 14, 1, 0)), row(datetime(2026, 9, 14, 9, 0), "b")])
    files = list(store.glob("*.parquet"))
    assert [f.name for f in files] == ["live-20260914.parquet"]
    assert rows_in(files[0]) == 2


def test_a_second_sync_folds_in_rather_than_adding_a_file(store):
    """A file per sync is 48 a day once polling is on; this is the fix."""
    live._write([row(datetime(2026, 9, 14, 1, 0))])
    live._write([row(datetime(2026, 9, 14, 2, 0), "b")])
    files = list(store.glob("*.parquet"))
    assert len(files) == 1
    assert rows_in(files[0]) == 2


def test_the_same_play_arriving_twice_is_stored_once(store):
    """The fifty-play window overlaps itself on every sync by design, so the
    same rows arrive again and again."""
    again = row(datetime(2026, 9, 14, 1, 0))
    live._write([again])
    live._write([again])
    assert rows_in(next(store.glob("*.parquet"))) == 1


def test_plays_spanning_midnight_split_by_the_day_they_happened(store):
    live._write([row(datetime(2026, 9, 14, 23, 50)), row(datetime(2026, 9, 15, 0, 10), "b")])
    assert sorted(f.name for f in store.glob("*.parquet")) == [
        "live-20260914.parquet",
        "live-20260915.parquet",
    ]


def _export_ending(tmp_path, when: datetime) -> str:
    path = tmp_path / "export.parquet"
    con = duckdb.connect()
    con.execute(
        "create table e as select ?::timestamp as played_at, 'x' as track_uri", [when]
    )
    con.execute(f"copy e to '{path.as_posix()}' (format parquet)")
    return path.as_posix()


def test_prune_deletes_only_what_the_export_covers(store, tmp_path, monkeypatch):
    live._write([row(datetime(2026, 9, 10, 12, 0))])  # covered
    live._write([row(datetime(2026, 9, 20, 12, 0), "b")])  # not covered
    monkeypatch.setattr(live, "EXPORT_GLOB", _export_ending(tmp_path, datetime(2026, 9, 15)))
    monkeypatch.setattr(live, "has_export", lambda: True)

    assert live.prune() == 1
    assert [f.name for f in store.glob("*.parquet")] == ["live-20260920.parquet"]


def test_prune_keeps_everything_when_there_is_no_export(store, monkeypatch):
    """Nothing else holds these rows, so a missing export means keep them all."""
    live._write([row(datetime(2020, 1, 1, 12, 0))])
    monkeypatch.setattr(live, "has_export", lambda: False)
    assert live.prune() == 0
    assert len(list(store.glob("*.parquet"))) == 1


def test_prune_does_not_delete_by_file_count(store, tmp_path, monkeypatch):
    """The bug this replaces: a flat keep-the-newest-twelve rule would have
    thrown away a day's polling every time the app was opened."""
    for day in range(1, 21):
        live._write([row(datetime(2026, 9, day, 12, 0), f"t{day}")])
    assert len(list(store.glob("*.parquet"))) == 20
    monkeypatch.setattr(live, "EXPORT_GLOB", _export_ending(tmp_path, datetime(2026, 8, 1)))
    monkeypatch.setattr(live, "has_export", lambda: True)
    assert live.prune() == 0
    assert len(list(store.glob("*.parquet"))) == 20
