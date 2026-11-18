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


# ---------------------------------------------------------------- one winner


def test_the_top_song_names_the_winner_and_the_runner_up(history):
    d = ask(answer="top", dimension="track", limit=1, metric="plays")
    assert d.chart == "number" and d.stat["answer"] == "top"
    assert (d.stat["big"], d.stat["unit"]) == ("Fears", "by MTNS")
    assert d.stat["sentence"] == "Your top song is Fears by MTNS - 12 plays."
    facts = dict(d.stat["facts"])
    assert facts["Next"] == "Height · 4 plays"
    assert facts["Plays"] == "15", "the winner's own count takes in its other versions"
    assert d.stat["uri"] == "spotify:track:aaaaaaaaaaaaaaaaaaaaa1"


def test_narrowed_to_one_artist_it_reads_as_theirs(history):
    d = ask(answer="top", dimension="album", artists=["mtns"], limit=1, metric="plays")
    assert d.stat["sentence"] == "Your top MTNS album is Salvage - 15 plays."


def test_narrowed_to_one_album_it_says_so(history):
    d = ask(answer="top", dimension="track", albums=["salvage"], limit=1, metric="plays")
    assert d.stat["sentence"].startswith("Your top song off Salvage is Fears by MTNS")


def test_a_time_of_day_is_said_the_way_people_say_it(history):
    d = ask(answer="top", dimension="hour_of_day", limit=1, metric="plays")
    assert d.stat["big"].endswith(("am", "pm"))


def test_a_winner_among_nobody(history):
    d = ask(answer="top", dimension="album", artists=["nobody at all"], limit=1)
    assert d.stat["found"] is False
    assert d.stat["sentence"] == 'No artist called "nobody at all" in your listening.'


def test_a_top_answer_over_dates_still_needs_a_bucket():
    spec = ChartSpec(answer="top", dimension="date")
    assert [p.field for p in validate_chart(spec)] == ["grain"]


# ------------------------------------------------------------- reading names

NAMED = [
    ("Lover, You Should've Come Over", "Jeff Buckley", "Grace", 30),
    ("Lover, You Should've Come Over - Live", "Jeff Buckley", "Live", 5),
    ("Thinking About You", "Faye Webster", "Atlanta Millionaires Club", 20),
    ("Thinking About You", "Radiohead", "Pablo Honey", 2),
    ("good kid", "Kendrick Lamar", "good kid, m.A.A.d city", 40),
    ("The Thing", "Kendrick Lamar Tribute Band", "Covers", 3),
    ("Kyoto", "Phoebe Bridgers", "Punisher", 12),
    ("Mrs. Robinson", "Simon & Garfunkel", "Bookends", 6),
]


@pytest.fixture
def named(monkeypatch):
    con = duckdb.connect()
    con.execute("install icu; load icu;")
    con.execute(
        "create table plays (played_at timestamp, track_uri varchar, track_name varchar, "
        "artist_name varchar, album_name varchar, ms_played bigint, platform varchar, "
        "skipped boolean)"
    )
    rows, i = [], 0
    for track, artist, album, n in NAMED:
        for _ in range(n):
            rows.append(
                (
                    datetime(2024, 1, 1) + timedelta(days=i),
                    f"spotify:track:{i:022d}",
                    track,
                    artist,
                    album,
                    200_000,
                    "ios",
                    False,
                )
            )
            i += 1
    con.executemany("insert into plays values (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    monkeypatch.setattr(chartrun, "_connect", lambda spec=None: con)
    return con


def test_punctuation_does_not_decide_a_title(named):
    d = ask(answer="number", metric="plays", tracks=["lover you should've come over"])
    assert d.stat["value"] == 35, "with its live version, and nothing to report"
    assert not d.note


@pytest.mark.parametrize(
    "typed, artist",
    [
        ("kendrik lamar", "Kendrick Lamar"),
        ("phobe bridgers", "Phoebe Bridgers"),
        ("kendrick", "Kendrick Lamar"),
        ("simon and garfunkel", "Simon & Garfunkel"),
    ],
)
def test_a_name_is_read_as_the_one_played(named, typed, artist):
    d = ask(answer="number", metric="plays", artists=[typed])
    assert d.stat["subject"] == artist
    exact = typed == "simon and garfunkel"
    assert bool(d.note) is not exact, "anything but an exact match says what it was read as"


def test_a_partial_name_goes_to_the_one_played_most(named):
    """'kendrick' is a whole word of two acts here; the one played more wins."""
    d = ask(answer="number", metric="plays", artists=["kendrick"])
    assert d.stat["value"] == 40


def test_a_partial_title_with_its_artist(named):
    d = ask(answer="number", metric="plays", tracks=["thinking about"], artists=["radiohead"])
    assert d.stat["value"] == 2
    assert d.note == 'Read "thinking about" as Thinking About You.'


@pytest.mark.parametrize("typed", ["the", "kyo", "zzqx", "bridgers phoebe"])
def test_some_names_are_not_guessed_at(named, typed):
    """Too short to read as part of a name, or too far from any spelling."""
    d = ask(answer="number", metric="plays", artists=[typed])
    assert d.stat["found"] is False


def test_charts_read_names_too(named):
    d = ask(
        dimension="date", grain="month", metric="plays", tracks=["lover you shouldve come over"]
    )
    assert sum(d.values) == 35


# -------------------------------------------------------------- recent plays

# Oldest first: (track, artist, seconds played). Played a minute apart.
SEQUENCE = [
    ("Ivy", "Frank Ocean", 200),
    ("Fears", "MTNS", 200),
    ("Fears", "MTNS", 200),
    ("Nights", "Frank Ocean", 200),
    ("Fears", "MTNS", 200),
    ("Skipped", "Someone", 5),
    ("Fears", "MTNS", 200),
    ("Height", "MexikoDro", 200),
]


@pytest.fixture
def sequence(monkeypatch):
    con = duckdb.connect()
    con.execute("install icu; load icu;")
    con.execute(
        "create table plays (played_at timestamp, track_uri varchar, track_name varchar, "
        "artist_name varchar, album_name varchar, ms_played bigint, platform varchar, "
        "skipped boolean)"
    )
    start = datetime(2024, 3, 1, 12)
    con.executemany(
        "insert into plays values (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                start + timedelta(minutes=i),
                f"spotify:track:{track.lower():a<22}",
                track,
                artist,
                "Album",
                secs * 1000,
                "ios",
                False,
            )
            for i, (track, artist, secs) in enumerate(SEQUENCE)
        ],
    )
    monkeypatch.setattr(chartrun, "_connect", lambda spec=None: con)
    return con


def rows(d):
    return [(r["name"], r["times"]) for r in d.stat["rows"]]


def test_recent_plays_newest_first_with_runs_folded(sequence):
    d = ask(answer="recent")
    assert d.chart == "recent"
    # The skip between the third and fourth Fears is not a listen, so those two
    # are one run; the first pair is a run of its own.
    assert rows(d) == [("Height", 1), ("Fears", 2), ("Nights", 1), ("Fears", 2), ("Ivy", 1)]
    assert d.stat["rows"][0]["at"] == "2024-03-01T12:07:00Z"
    assert d.stat["rows"][0]["uri"].startswith("spotify:track:")


def test_recent_plays_stop_at_the_number_asked_for(sequence):
    d = ask(answer="recent", limit=2)
    assert rows(d) == [("Height", 1), ("Fears", 2)]
    assert d.title == "Last 2 songs"


def test_runs_are_runs_in_the_listening_not_in_the_filter(sequence):
    """Filtered to Fears, the plays either side of Nights are still two runs."""
    d = ask(answer="recent", tracks=["fears"])
    assert rows(d) == [("Fears", 2), ("Fears", 2)]
    assert d.title == "Recent plays of Fears by MTNS"


def test_recent_plays_by_one_artist(sequence):
    d = ask(answer="recent", artists=["frank ocean"])
    assert rows(d) == [("Nights", 1), ("Ivy", 1)]
    assert d.title == "Recent Frank Ocean songs", "ten were asked for and two exist"


def test_recent_plays_of_nobody(sequence):
    d = ask(answer="recent", artists=["nobody at all"])
    assert d.stat["rows"] == [] and not d.empty
    assert d.stat["sentence"] == 'No artist called "nobody at all" in your listening.'


def test_the_plan_hands_the_list_over(sequence):
    d = ask(answer="recent")
    p = plan({"chart": d.chart, "stat": d.stat, "labels": [], "values": []})
    assert p["kind"] == "recent" and p["stat"]["rows"]


# ------------------------------------------------------- windows and titles

DASH = chr(0x2013)


@pytest.mark.parametrize(
    "fields, title",
    [
        ({"dimension": "track"}, "Top songs"),
        ({"dimension": "track", "artists": ["Drake"], "limit": 5}, "Top 5 Drake songs"),
        ({"dimension": "track", "albums": ["Blonde"]}, "Top songs off Blonde"),
        ({"dimension": "track", "metric": "skip_rate"}, "Most skipped songs"),
        ({"dimension": "track", "sort": "asc"}, "Least played songs"),
        ({"dimension": "date", "grain": "month"}, "Listening by month"),
        ({"dimension": "date", "grain": "year", "artists": ["Drake"]}, "Drake by year"),
        ({"dimension": "hour_of_day", "range": "today"}, "Today by hour"),
        (
            {"dimension": "date", "grain": "year", "series": "artist", "artists": ["A", "B"]},
            "A vs B by year",
        ),
        (
            {
                "dimension": "date",
                "grain": "year",
                "series": "artist",
                "per_period": True,
                "limit": 3,
            },
            "Top 3 artists each year",
        ),
        ({"dimension": "date", "grain": "month", "cumulative": True}, "Listening, running total"),
        ({"answer": "top", "dimension": "album", "artists": ["Drake"]}, "Top Drake album"),
        ({"answer": "number", "metric": "plays", "tracks": ["Fears"]}, "Plays · Fears"),
        ({"answer": "first_played", "artists": ["Sade"]}, "First play of Sade"),
        ({"answer": "recent"}, "Recent plays"),
    ],
)
def test_plain_titles(fields, title):
    assert chartrun.plain_title(ChartSpec(**fields)) == title


def test_a_window_of_the_last_few_years(history):
    """The plays run from Jan 2024, nine days apart; the newest is the anchor."""
    every = ask(answer="number", metric="plays").stat["value"]
    recent = ask(answer="number", metric="plays", last=30, last_unit="day")
    assert 0 < recent.stat["value"] < every
    assert recent.stat["period"].startswith("Last 30 days (")


def test_a_span_of_years(history):
    d = ask(answer="number", metric="plays", from_year=2024, to_year=2024)
    assert d.stat["value"] == ask(answer="number", metric="plays", year=2024).stat["value"]
    assert d.stat["period"] == f"2024 {DASH} 2024"
    assert ask(answer="number", metric="plays", from_year=2030).stat["value"] == 0


def test_every_answer_says_its_window_in_dates(history):
    d = ask(dimension="track")
    assert d.span.startswith(f"All time (Jan 2024 {DASH} ")


@pytest.mark.parametrize(
    "fields, field",
    [
        ({"last": 2}, "last"),
        ({"last_unit": "year"}, "last"),
        ({"to_year": 2022}, "to_year"),
        ({"from_year": 2023, "to_year": 2021}, "to_year"),
    ],
)
def test_a_window_has_to_make_sense(fields, field):
    assert field in [p.field for p in validate_chart(ChartSpec(**fields))]


def test_windows_that_are_not_the_menu_do_not_step(history):
    assert not chartrun.steppable(ChartSpec(range="7d", last=2, last_unit="week"))
