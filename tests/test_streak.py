"""The profile's streak badge: which days count and when a run is over."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from musicshare import history
from musicshare.streak import current_run, streak

TODAY = date(2026, 9, 16)


def run_of(n: int, ending: date) -> set[date]:
    return {ending - timedelta(days=i) for i in range(n)}


def test_run_ending_today_counts_today():
    assert current_run(run_of(5, TODAY), TODAY) == (5, TODAY - timedelta(days=4))


def test_no_play_yet_today_is_not_a_broken_streak():
    yesterday = TODAY - timedelta(days=1)
    assert current_run(run_of(5, yesterday), TODAY) == (5, yesterday - timedelta(days=4))


def test_a_full_day_without_a_play_ends_it():
    assert current_run(run_of(5, TODAY - timedelta(days=2)), TODAY) == (0, None)


def test_a_missed_day_splits_the_run():
    days = run_of(3, TODAY) | run_of(10, TODAY - timedelta(days=4))
    assert current_run(days, TODAY)[0] == 3


def test_days_nobody_could_see_are_bridged():
    """Sept 10 2026: after the export was generated, before capture started."""
    export_end, live_start = date(2026, 9, 9), date(2026, 9, 11)
    days = run_of(462, export_end) | run_of(6, TODAY)
    n, since = current_run(days, TODAY, (export_end, live_start))
    assert n == 469
    assert since == export_end - timedelta(days=461)


def test_the_bridge_does_not_cover_a_day_that_was_seen():
    """The export's own last day is observed - no play there is a real miss."""
    export_end, live_start = date(2026, 9, 9), date(2026, 9, 11)
    days = run_of(6, TODAY) | run_of(30, export_end - timedelta(days=1))
    assert current_run(days, TODAY, (export_end, live_start))[0] == 7


@pytest.mark.skipif(not history.has_export(), reason="no export ingested")
def test_real_history_gives_a_shaped_answer():
    got = streak()
    assert got["days"] >= 0
    assert (got["since"] is None) == (got["days"] == 0)
