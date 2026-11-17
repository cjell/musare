"""Answers that are one number or one date rather than a chart.

Over a small history built here, so every count is known in advance and the
tests do not move as the real history grows. No model.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import pytest

from musicshare.spec import chartrun
from musicshare.spec.chart import ChartSpec
from musicshare.spec.plot import plan
from musicshare.spec.validate import validate_chart

MIN = chartrun.MIN_MS

# (track, artist, album, uri, plays, ms each, skipped)
ROWS = [
    ("Fears", "MTNS", "Salvage", "spotify:track:aaaaaaaaaaaaaaaaaaaaa1", 12, 240_000, False),
    (
        "Fears - Remastered 2011",
        "MTNS",
        "Salvage",
        "spotify:track:aaaaaaaaaaaaaaaaaaaaa2",
        3,
        240_000,
        False,
    ),
    ("Fears", "Other Band", "Elsewhere", "spotify:track:aaaaaaaaaaaaaaaaaaaaa3", 2, 200_000, True),
    ("Height", "MexikoDro", "Up", "spotify:track:aaaaaaaaaaaaaaaaaaaaa4", 4, 180_000, None),
    (
        "Don" + chr(0x2019) + "t Stop",
        "MTNS",
        "Salvage",
        "spotify:track:aaaaaaaaaaaaaaaaaaaaa5",
        1,
        10_000,
        True,
    ),
]


@pytest.fixture
def history(monkeypatch):
    """A `plays` table the executor reads instead of the real one."""
    con = duckdb.connect()
    con.execute("install icu; load icu;")
    con.execute(
        "create table plays (played_at timestamp, track_uri varchar, track_name varchar, "
        "artist_name varchar, album_name varchar, ms_played bigint, platform varchar, "
        "skipped boolean)"
    )
    start = datetime(2024, 1, 15, 18)
    rows, i = [], 0
    for track, artist, album, uri, n, ms, skipped in ROWS:
        for _ in range(n):
            # Spread over months, so first, last and the busiest month differ.
            when = start + timedelta(days=9 * i)
            rows.append((when, uri, track, artist, album, ms, "ios", skipped))
            i += 1
    con.executemany("insert into plays values (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    monkeypatch.setattr(chartrun, "_connect", lambda spec=None: con)
    return con


def ask(**fields) -> chartrun.ChartData:
    spec = ChartSpec(**fields)
    assert validate_chart(spec) == []
    return chartrun.run(spec)


def test_a_song_is_counted_in_any_case_and_with_its_versions(history):
    d = ask(answer="number", metric="plays", tracks=["fears"])
    assert d.chart == "number" and not d.empty
    assert d.stat["value"] == 15, "12 plays and 3 of the remaster, the other band's left out"
    assert d.stat["sentence"] == "You've played Fears by MTNS 15 times."
    assert d.stat["uri"] == "spotify:track:aaaaaaaaaaaaaaaaaaaaa1"


def test_a_shared_title_names_the_others(history):
    d = ask(answer="number", metric="plays", tracks=["Fears"])
    assert "Other Band" in (d.note or "")


def test_naming_the_artist_counts_theirs(history):
    d = ask(answer="number", metric="plays", tracks=["Fears"], artists=["other band"])
    assert d.stat["value"] == 2
    assert d.stat["subject"] == "Fears by Other Band"
    assert not d.note


def test_curly_and_straight_apostrophes_are_the_same_title(history):
    d = ask(answer="number", metric="hours", tracks=["don't stop"])
    assert d.stat["found"] is True


def test_a_short_play_is_not_a_listen_but_its_time_counts(history):
    plays = ask(answer="number", metric="plays", tracks=["Don't Stop"])
    hours = ask(answer="number", metric="hours", tracks=["Don't Stop"])
    assert plays.stat["value"] == 0
    assert hours.stat["big"] == "0" and hours.stat["unit"] == "minutes"


def test_an_album_counts_every_song_on_it(history):
    d = ask(answer="number", metric="plays", albums=["salvage"])
    assert d.stat["value"] == 15
    assert d.stat["subject"] == "Salvage by MTNS"


def test_hours_read_as_minutes_under_an_hour(history):
    d = ask(answer="number", metric="hours", tracks=["Height"])
    assert (d.stat["big"], d.stat["unit"]) == ("12", "minutes")


def test_a_total_over_everything_has_no_subject(history):
    d = ask(answer="number", metric="plays")
    assert d.stat["value"] == 21
    assert d.stat["sentence"] == "You've played 21 songs."
    assert d.stat["uri"] == ""
    labels = [f[0] for f in d.stat["facts"]]
    assert "Artists" in labels and "First played" not in labels


def test_distinct_counts(history):
    assert ask(answer="number", metric="distinct_artists").stat["value"] == 3
    d = ask(answer="number", metric="distinct_tracks", artists=["MTNS"])
    assert d.stat["value"] == 2, "two versions of Fears; the short play is not a listen"
    assert " by MTNS" in d.stat["sentence"]


def test_first_and_last_played(history):
    first = ask(answer="first_played", tracks=["fears"])
    last = ask(answer="last_played", tracks=["fears"])
    assert first.stat["big"] == "Jan 15, 2024"
    assert first.stat["sentence"].startswith("You first played Fears by MTNS on ")
    assert last.stat["big"] != first.stat["big"]
    assert "ago" in first.stat["unit"]


def test_a_skip_rate_without_skip_data_says_so(history):
    d = ask(answer="number", metric="skip_rate", tracks=["Height"])
    assert d.stat["value"] is None
    assert "No skip data" in d.stat["sentence"]


def test_a_skip_rate(history):
    d = ask(answer="number", metric="skip_rate", tracks=["fears"], artists=["Other Band"])
    assert d.stat["big"] == "100%"


def test_nothing_by_that_name(history):
    d = ask(answer="number", metric="plays", tracks=["no such song"])
    assert d.stat["found"] is False
    assert d.stat["sentence"] == 'No song called "no such song" in your listening.'
    assert not d.empty, "saying so is the answer"


def test_none_in_the_window_is_a_zero_not_a_miss(history):
    d = ask(answer="number", metric="plays", tracks=["fears"], range="today")
    assert d.stat["found"] is True and d.stat["value"] == 0
    assert d.stat["sentence"] == "You haven't played Fears by MTNS today."


def test_the_busiest_month_is_a_fact(history):
    facts = dict(ask(answer="number", metric="plays", tracks=["fears"]).stat["facts"])
    assert "Busiest month" in facts and "Time" in facts


def test_the_plan_lays_it_out_rather_than_drawing_it(history):
    d = ask(answer="number", metric="plays", tracks=["fears"])
    p = plan({"chart": d.chart, "stat": d.stat, "labels": [], "values": []})
    assert p == {"kind": "number", "stat": d.stat}


def test_the_track_filter_narrows_a_chart_too(history):
    d = ask(dimension="date", grain="month", metric="plays", tracks=["fears"])
    assert d.chart == "line" and sum(d.values) == 15


def test_a_number_ignores_the_shape_fields():
    """A grain left beside 'number' is not a chart that cannot be drawn."""
    spec = ChartSpec(answer="number", dimension="date", grain=None, series="artist")
    assert validate_chart(spec) == []


def test_old_specs_still_parse_as_charts():
    spec = ChartSpec.model_validate({"dimension": "artist", "metric": "hours"})
    assert spec.answer == "chart" and spec.tracks == [] and spec.albums == []
