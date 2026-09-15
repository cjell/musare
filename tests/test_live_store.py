"""How captured plays are stored locally, and when they are allowed to be deleted.

No network: these drive `_store`, `prune` and the pull bookkeeping directly. The
capture decisions themselves live in SQL and are tested against the database in
test_capture_sql.py.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

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
    return (
        duckdb.connect()
        .execute(f"select count(*) from read_parquet('{path.as_posix()}')")
        .fetchone()[0]
    )


def names(store) -> list[str]:
    return sorted(f.name for f in store.glob("*.parquet"))


def test_a_days_plays_land_in_one_file(store):
    live._store([row(datetime(2026, 9, 14, 1, 0)), row(datetime(2026, 9, 14, 9, 0), "b")], None)
    assert names(store) == ["live-20260914.parquet"]
    assert rows_in(store / "live-20260914.parquet") == 2


def test_plays_spanning_midnight_split_by_the_day_they_happened(store):
    live._store([row(datetime(2026, 9, 14, 23, 50)), row(datetime(2026, 9, 15, 0, 10), "b")], None)
    assert names(store) == ["live-20260914.parquet", "live-20260915.parquet"]


def test_a_pull_replaces_a_day_rather_than_merging_into_it(store):
    """The database drops a backup estimate when the watcher's measurement of the
    same play arrives. Merging would keep the old copy and count the play twice."""
    estimate = row(datetime(2026, 9, 14, 1, 0, 0))
    measured = row(datetime(2026, 9, 14, 1, 0, 4))
    live._store([estimate], date(2026, 9, 14))
    live._store([measured], date(2026, 9, 14))
    got = (
        duckdb.connect()
        .execute(
            f"select played_at from read_parquet('{(store / 'live-20260914.parquet').as_posix()}')"
        )
        .fetchall()
    )
    assert got == [(measured["played_at"],)]


def test_a_day_the_database_no_longer_holds_is_removed(store):
    live._store([row(datetime(2026, 9, 14, 1, 0))], None)
    assert live._store([], date(2026, 9, 14)) is True
    assert names(store) == []


def test_days_before_the_reread_window_are_left_alone(store):
    live._store([row(datetime(2026, 9, 10, 12, 0))], None)
    live._store([row(datetime(2026, 9, 14, 12, 0), "b")], date(2026, 9, 13))
    assert names(store) == ["live-20260910.parquet", "live-20260914.parquet"]


def test_an_identical_pull_changes_nothing(store):
    """Every page load pulls; rewriting unchanged files each time would also
    throw away the feed cache for no reason."""
    rows = [row(datetime(2026, 9, 14, 1, 0))]
    assert live._store(rows, None) is True
    assert live._store(rows, None) is False


def test_a_day_with_no_album_names_still_reads_beside_the_others(store):
    """Inferred types made an all-null column type null in one file and string
    in the next, and a glob over both refuses to read."""
    bare = {**row(datetime(2026, 9, 14, 1, 0)), "album_name": None}
    live._store([bare, row(datetime(2026, 9, 15, 1, 0), "b")], None)
    n = (
        duckdb.connect()
        .execute(f"select count(*) from read_parquet('{(store / '*.parquet').as_posix()}')")
        .fetchone()[0]
    )
    assert n == 2


def _export_ending(tmp_path, when: datetime) -> str:
    path = tmp_path / "export.parquet"
    con = duckdb.connect()
    con.execute("create table e as select ?::timestamp as played_at, 'x' as track_uri", [when])
    con.execute(f"copy e to '{path.as_posix()}' (format parquet)")
    return path.as_posix()


def test_prune_deletes_only_what_the_export_covers(store, tmp_path, monkeypatch):
    live._store([row(datetime(2026, 9, 10, 12, 0)), row(datetime(2026, 9, 20, 12, 0), "b")], None)
    monkeypatch.setattr(live, "EXPORT_GLOB", _export_ending(tmp_path, datetime(2026, 9, 15)))
    monkeypatch.setattr(live, "has_export", lambda: True)

    assert live.prune() == 1
    assert names(store) == ["live-20260920.parquet"]


def test_prune_keeps_everything_when_there_is_no_export(store, monkeypatch):
    """Nothing else local holds these rows, so a missing export means keep them all."""
    live._store([row(datetime(2020, 1, 1, 12, 0))], None)
    monkeypatch.setattr(live, "has_export", lambda: False)
    assert live.prune() == 0
    assert len(names(store)) == 1


def test_prune_does_not_delete_by_file_count(store, tmp_path, monkeypatch):
    """The bug this replaced: a flat keep-the-newest-twelve rule would have
    thrown away a day's plays every time the app was opened."""
    live._store([row(datetime(2026, 9, day, 12, 0), f"t{day}") for day in range(1, 21)], None)
    assert len(names(store)) == 20
    monkeypatch.setattr(live, "EXPORT_GLOB", _export_ending(tmp_path, datetime(2026, 8, 1)))
    monkeypatch.setattr(live, "has_export", lambda: True)
    assert live.prune() == 0
    assert len(names(store)) == 20


# ---------------------------------------------------------------- pulling


def result(ok_ago: timedelta | None) -> live.PullResult:
    watch = None if ok_ago is None else {"last_ok_at": datetime.now(UTC) - ok_ago}
    return live.PullResult(pulled=0, changed=False, newest=None, watch=watch)


def test_a_watcher_that_checked_in_recently_is_not_stalled():
    assert not result(timedelta(seconds=40)).stalled


def test_a_watcher_silent_past_the_limit_is_stalled():
    """No plays and a stopped job look identical in the plays table. Only the
    job's own check-ins tell them apart."""
    assert result(live.STALE_AFTER + timedelta(seconds=1)).stalled


def test_a_watcher_that_never_ran_is_stalled():
    assert result(None).stalled


def test_an_unreachable_database_degrades_to_no_pull(monkeypatch):
    def boom():
        raise OSError("no route to host")

    monkeypatch.setattr(live, "_last_pull", None)
    monkeypatch.setattr(live, "pull", boom)
    assert live.pull_if_stale() is None


def test_pulls_are_spaced_and_the_repeat_reports_no_change(monkeypatch):
    calls = []

    def fake():
        calls.append(1)
        return live.PullResult(3, True, None, {"last_ok_at": datetime.now(UTC)})

    monkeypatch.setattr(live, "_last_pull", None)
    monkeypatch.setattr(live, "pull", fake)
    monkeypatch.setattr(live, "prune", lambda: 0)
    first = live.pull_if_stale()
    second = live.pull_if_stale()
    assert len(calls) == 1
    assert first.changed and not second.changed, "the page must not rebuild on a cached answer"


def test_ago_reads_naive_timestamps_as_utc():
    """The store holds naive UTC; mixing that with an aware now() is a
    TypeError, which is how this was found."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("capture", "scripts/capture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    naive = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=5)
    assert mod.ago(naive).endswith("m ago")
    assert mod.ago(datetime.now(UTC)) == "0s ago"


# ------------------------------------------------------------ cover picking


def _sizes(host):
    return [
        {"url": f"https://{host}/640/x", "width": 640, "height": 640},
        {"url": f"https://{host}/300/x", "width": 300, "height": 300},
        {"url": f"https://{host}/60/x", "width": 60, "height": 60},
    ]


def test_a_mosaic_is_asked_for_at_twice_the_size():
    """Four covers in one image means the useful resolution is the tile: a 300px
    mosaic is four 150px covers, and it is the only cover that upscales in a
    cell a single 300px cover fills comfortably."""
    assert "/640/" in live.pick_image(_sizes("mosaic.scdn.co"), 240)


def test_an_ordinary_cover_is_not_upsized():
    """The doubling is for tiling, not for everything - a single cover at 300
    already clears a 118px cell at 2x."""
    assert "/300/" in live.pick_image(_sizes("i.scdn.co"), 240)


def test_small_rows_still_get_small_images():
    assert "/60/" in live.pick_image(_sizes("i.scdn.co"), 60)


def test_a_cover_with_no_dimensions_is_used_as_is():
    """Uploaded playlist covers come back as one entry with null width."""
    only = [{"url": "https://i.scdn.co/image/only", "width": None, "height": None}]
    assert live.pick_image(only, 240) == "https://i.scdn.co/image/only"


def test_no_images_is_none_rather_than_an_error():
    assert live.pick_image([], 240) is None
    assert live.pick_image(None, 240) is None
