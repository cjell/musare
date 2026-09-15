"""Fast tests for the deterministic half. No API, no cost, runs in normal CI."""

from __future__ import annotations

from datetime import date

import pytest

from musicshare import shows as shows_mod
from musicshare.spec import ShowFilter, apply, validate

TODAY = date(2026, 9, 10)


def show(**kw):
    base = {
        "artist": "Slow Pulp",
        "date": "2026-09-20",
        "distance_mi": 10.0,
        "fans": 40_000,
        "yours": True,
        "familiarity": "regular",
    }
    base.update(kw)
    return base


def test_defaults_keep_everything():
    shows = [show(), show(artist="Deftones", yours=False, familiarity=None)]
    assert len(apply(ShowFilter(), shows, TODAY)) == 2


@pytest.mark.parametrize(
    "want,kept",
    [
        ("any", ["Slow Pulp", "Deftones", "Wednesday", "Horsegirl"]),
        ("heard", ["Slow Pulp", "Deftones", "Wednesday"]),
        ("regular", ["Slow Pulp", "Wednesday"]),
        ("favorite", ["Wednesday"]),
        ("new", ["Horsegirl"]),
    ],
)
def test_each_level_includes_the_ones_above_it(want, kept):
    """Asking for artists you listen to regularly should not hide your favourites."""
    shows = [
        show(),
        show(artist="Deftones", familiarity="heard"),
        show(artist="Wednesday", familiarity="favorite"),
        show(artist="Horsegirl", familiarity=None, yours=False),
    ]
    out = apply(ShowFilter(familiarity=want), shows, TODAY)
    assert [s["artist"] for s in out] == kept


def test_a_row_cached_before_levels_existed_is_not_new():
    """It has a play on record and no level; calling that act new would be wrong."""
    old = show(artist="Deftones")
    del old["familiarity"]
    assert apply(ShowFilter(familiarity="new"), [old], TODAY) == []


@pytest.mark.parametrize(
    "days,level",
    [(0, None), (1, "heard"), (14, "heard"), (15, "regular"), (49, "regular"), (50, "favorite")],
)
def test_levels_are_counted_in_separate_days_not_plays(days, level):
    assert shows_mod.familiarity_of(days) == level


@pytest.mark.parametrize("radius,kept", [(5, 0), (10, 1), (500, 1)])
def test_radius_is_inclusive(radius, kept):
    assert len(apply(ShowFilter(radius_mi=radius), [show()], TODAY)) == kept


def test_null_radius_uses_the_apps_default_not_the_models():
    # 60 miles is inside the 50-mile default? No - so a null radius must still
    # apply the default rather than meaning "unlimited".
    assert apply(ShowFilter(radius_mi=None), [show(distance_mi=60)], TODAY) == []
    assert len(apply(ShowFilter(radius_mi=None), [show(distance_mi=40)], TODAY)) == 1


def test_missing_distance_is_kept():
    """Absent is not the same as far."""
    assert len(apply(ShowFilter(radius_mi=5), [show(distance_mi=None)], TODAY)) == 1


def test_window_excludes_past_and_far_future():
    shows = [show(date="2026-09-01"), show(date="2026-09-15"), show(date="2027-06-01")]
    out = apply(ShowFilter(within_days=30), shows, TODAY)
    assert [s["date"] for s in out] == ["2026-09-15"]


def test_fan_bounds():
    shows = [show(fans=1_000), show(fans=500_000)]
    assert len(apply(ShowFilter(max_fans=50_000), shows, TODAY)) == 1
    assert len(apply(ShowFilter(min_fans=50_000), shows, TODAY)) == 1


def test_unknown_fans_excluded_only_when_a_fan_filter_is_asked_for():
    s = [show(fans=None)]
    assert len(apply(ShowFilter(), s, TODAY)) == 1
    assert apply(ShowFilter(max_fans=50_000), s, TODAY) == []


def test_named_artists_are_case_insensitive():
    shows = [show(), show(artist="Deftones")]
    assert len(apply(ShowFilter(artists=["deftones"]), shows, TODAY)) == 1


def test_not_understood_returns_nothing():
    assert apply(ShowFilter(understood=False, radius_mi=500), [show()], TODAY) == []


def test_malformed_date_does_not_crash():
    assert len(apply(ShowFilter(), [show(date="not-a-date")], TODAY)) == 1


def test_validator_accepts_a_sane_spec():
    assert validate(ShowFilter(max_fans=50_000, radius_mi=200)) == []


def test_validator_catches_inverted_bounds():
    problems = validate(ShowFilter(min_fans=900_000, max_fans=1_000))
    assert [p.field for p in problems] == ["min_fans"]


def test_validator_catches_an_artist_dump():
    problems = validate(ShowFilter(artists=[f"artist {i}" for i in range(40)]))
    assert [p.field for p in problems] == ["artists"]


def test_schema_exposes_no_identity_or_destination_fields():
    """The capability ceiling: an injection can only ask for what this can say."""
    forbidden = {"user", "user_id", "account", "url", "path", "table", "query", "sql", "email"}
    assert not (set(ShowFilter.model_json_schema()["properties"]) & forbidden)


# ----------------------------------------------------- genre and comparison rules


def test_too_many_genres_is_a_misparse():
    from musicshare.spec.vocab import GENRES, MAX_GENRES

    spec = ShowFilter(genres=list(GENRES[: MAX_GENRES + 1]))
    assert [p.field for p in validate(spec)] == ["genres"]


def test_a_genre_outside_the_vocabulary_cannot_be_built():
    """The capability ceiling: an injection cannot ask for a word that is not here."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ShowFilter(genres=["drop table shows"])


def test_a_comparison_needs_an_ordered_axis():
    from musicshare.spec import ChartSpec, validate_chart

    assert [p.field for p in validate_chart(ChartSpec(dimension="artist", series="artist"))] == [
        "series"
    ]
    assert validate_chart(ChartSpec(dimension="date", grain="year", series="artist")) == []


def test_a_refusal_is_not_judged_on_its_leftovers():
    from musicshare.spec import ChartSpec, validate_chart

    spec = ChartSpec(understood=False, dimension="artist", series="genre")
    assert validate_chart(spec) == []


def test_today_over_time_is_drawn_by_hour_not_refused_for_a_grain():
    """A date axis over one day has a single bucket, so it is derived to hours."""
    from musicshare.spec import ChartSpec, validate_chart
    from musicshare.spec.chartrun import effective

    spec = ChartSpec(range="today", dimension="date")
    assert validate_chart(spec) == []
    drawn = effective(spec)
    assert drawn.dimension == "hour_of_day" and drawn.grain is None


def test_a_date_chart_still_needs_a_grain_on_any_other_range():
    from musicshare.spec import ChartSpec, validate_chart

    problems = validate_chart(ChartSpec(range="7d", dimension="date"))
    assert [p.field for p in problems] == ["grain"]


@pytest.mark.parametrize(
    "kw,ok",
    [
        ({"dimension": "hour_of_day", "range": "today", "metric": "hours"}, True),
        ({"dimension": "date", "grain": "month", "metric": "plays"}, True),
        ({"dimension": "artist", "metric": "hours"}, False),
        ({"dimension": "date", "grain": "day", "metric": "distinct_artists"}, False),
        ({"dimension": "date", "grain": "day", "metric": "skip_rate"}, False),
    ],
)
def test_a_running_total_needs_an_order_and_a_quantity_that_adds(kw, ok):
    """Distinct artists would be counted again every hour; a summed rate is not one."""
    from musicshare.spec import ChartSpec, validate_chart

    assert (validate_chart(ChartSpec(cumulative=True, **kw)) == []) is ok


def test_a_running_total_is_always_a_line():
    from musicshare.spec.chartrun import chart_for

    assert chart_for("hour_of_day", cumulative=True) == "line"
    assert chart_for("hour_of_day") == "bar"
