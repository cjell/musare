"""Chart spec and executor. Hits the local Parquet, never a model."""

from __future__ import annotations

import pytest

from musicshare.spec.chart import MAX_BARS, MAX_LINES, ChartSpec
from musicshare.spec.chartrun import (
    DIMENSIONS,
    MAX_TABLE_ROWS,
    METRICS,
    MIN_SAMPLE,
    chart_for,
    run,
)
from musicshare.taste import has_history

pytestmark = pytest.mark.skipif(not has_history(), reason="no play history ingested")


def test_default_spec_produces_a_chart():
    d = run(ChartSpec())
    assert d.labels and len(d.labels) == len(d.values)


@pytest.mark.parametrize("metric", list(METRICS))
def test_every_metric_runs(metric):
    d = run(ChartSpec(metric=metric, dimension="artist", limit=5))
    assert not d.empty
    assert all(v is not None for v in d.values)


@pytest.mark.parametrize("dimension", [*DIMENSIONS, "date"])
def test_every_dimension_runs(dimension):
    spec = ChartSpec(dimension=dimension, grain="year" if dimension == "date" else None)
    assert not run(spec).empty


def test_limit_is_respected():
    assert len(run(ChartSpec(dimension="artist", limit=3)).labels) == 3


def test_cyclical_dimensions_keep_their_own_order():
    """A day-of-week chart sorted by size is unreadable."""
    d = run(ChartSpec(dimension="day_of_week", metric="hours"))
    assert d.labels == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert len(run(ChartSpec(dimension="hour_of_day", metric="plays")).labels) == 24


def test_hours_are_local_not_utc():
    """The export is UTC; a naive timestamp has to be marked before converting.

    Without that the peak lands at midnight UTC instead of a plausible evening.
    """
    d = run(ChartSpec(dimension="hour_of_day", metric="plays"))
    peak = max(zip(d.labels, d.values, strict=True), key=lambda kv: kv[1])[0]
    assert 17 <= int(peak) <= 23, f"peak hour {peak} looks like unconverted UTC"


def test_sort_direction_changes_the_answer():
    top = run(ChartSpec(metric="skip_rate", dimension="artist", limit=5, sort="desc"))
    bottom = run(ChartSpec(metric="skip_rate", dimension="artist", limit=5, sort="asc"))
    assert top.labels != bottom.labels
    assert top.values[0] > bottom.values[0]


def test_rate_metrics_exclude_tiny_samples():
    """A 100% skip rate over two plays is noise, not a finding."""
    d = run(ChartSpec(metric="skip_rate", dimension="artist", limit=10))
    assert all(v < 100 for v in d.values)
    assert d.note and str(MIN_SAMPLE) in d.note


def test_year_filter_narrows():
    all_time = run(ChartSpec(metric="plays", dimension="artist", limit=1))
    one_year = run(ChartSpec(metric="plays", dimension="artist", limit=1, year=2019))
    assert one_year.values[0] < all_time.values[0]


def test_artist_filter_restricts_to_that_artist():
    d = run(ChartSpec(dimension="artist", artists=["Juice WRLD"], limit=10))
    assert d.labels == ["Juice WRLD"]


def test_artist_filter_is_parameterised_not_interpolated():
    """A quote in an artist name must not be able to reach the query text."""
    d = run(ChartSpec(dimension="artist", artists=["'; drop table plays; --"]))
    assert d.empty


def test_not_understood_runs_nothing():
    d = run(ChartSpec(understood=False, dimension="artist"))
    assert d.empty and d.note


def test_date_dimension_is_chronological():
    d = run(ChartSpec(dimension="date", grain="year", metric="hours"))
    assert d.labels == sorted(d.labels)


def test_title_falls_back_to_something_descriptive():
    assert run(ChartSpec(title="", metric="hours", dimension="artist")).title == "hours by artist"


def test_limit_cannot_exceed_the_schema_cap():
    with pytest.raises(ValueError):
        ChartSpec(limit=MAX_BARS + 1)


def test_chart_type_is_derived_not_chosen():
    """Removing it from the schema removed a whole class of undrawable specs."""
    assert chart_for("date") == "line"
    assert all(chart_for(d) == "bar" for d in DIMENSIONS)
    assert "chart" not in ChartSpec.model_json_schema()["properties"]


# ------------------------------------------------------- genre and comparison


def test_genre_dimension_runs():
    d = run(ChartSpec(dimension="genre", limit=5))
    assert not d.empty
    assert all(label for label in d.labels), "a bar per genre needs a name on every bar"


def test_genre_filter_narrows():
    whole = run(ChartSpec(dimension="artist", limit=40))
    metal = run(ChartSpec(dimension="artist", genres=["metal"], limit=40))
    assert sum(metal.values) < sum(whole.values)


def test_a_genre_with_no_region_comes_back_empty_and_says_so():
    d = run(ChartSpec(dimension="date", grain="year", genres=["vaporwave"]))
    assert d.empty
    assert d.note and "vaporwave" in d.note


def test_series_produces_one_line_each():
    d = run(ChartSpec(dimension="date", grain="year", series="artist", artists=["drake", "future"]))
    assert len(d.series) == 2
    assert not d.values, "the data is in series, not values - never both"
    assert all(len(ln.values) == len(d.labels) for ln in d.series)


def test_series_is_drawn_as_a_line_even_on_a_cyclical_axis():
    assert chart_for("hour_of_day") == "bar"
    assert chart_for("hour_of_day", "artist") == "line"


def test_series_caps_the_number_of_lines():
    d = run(ChartSpec(dimension="date", grain="year", series="genre"))
    assert 0 < len(d.series) <= MAX_LINES


def test_a_gap_in_a_count_is_zero_and_a_gap_in_a_rate_is_unknown():
    """Zero plays really is zero hours; it is not a 0% skip rate."""
    counted = run(ChartSpec(dimension="date", grain="year", series="genre", metric="plays"))
    assert all(v is not None for ln in counted.series for v in ln.values)
    rated = run(ChartSpec(dimension="date", grain="year", series="genre", metric="skip_rate"))
    assert any(v is None for ln in rated.series for v in ln.values) or rated.empty


def test_named_genres_are_the_lines_not_the_regions_underneath():
    """'rap vs rock' is two lines, not the nine regions those words cover."""
    d = run(ChartSpec(dimension="date", grain="year", series="genre", genres=["rap", "rock"]))
    assert [ln.name for ln in d.series] == ["rap", "rock"]


def test_unnamed_genre_series_means_the_biggest_ones():
    d = run(ChartSpec(dimension="date", grain="year", series="genre"))
    assert all(ln.name not in ("rap", "rock") for ln in d.series), "region names, not words"


def test_overlapping_genres_land_on_one_line_each():
    """A region covered by two named genres goes to whichever was asked first."""
    d = run(ChartSpec(dimension="date", grain="year", series="genre", genres=["rap", "hip hop"]))
    totals = {ln.name: sum(v for v in ln.values if v) for ln in d.series}
    assert set(totals) <= {"rap", "hip hop"}


def test_named_genres_are_the_bars_too():
    """Same rule as the lines: 'jazz or soul' is two bars, not nine regions."""
    d = run(ChartSpec(dimension="genre", genres=["jazz", "soul"]))
    assert sorted(d.labels) == ["jazz", "soul"]


def test_limit_governs_the_number_of_lines():
    """'top 3 artists every year' asked for 3 and was drawn 6."""
    d = run(ChartSpec(dimension="date", grain="year", series="artist", limit=3))
    assert len(d.series) == 3


def test_the_line_ceiling_still_applies():
    d = run(ChartSpec(dimension="date", grain="year", series="artist", limit=MAX_BARS))
    assert len(d.series) <= MAX_LINES


# ------------------------------------------------------- per-period rankings


def test_per_period_ranks_inside_each_bucket():
    """The failure this replaces: 9 of the 14 artists who actually led a year
    never appeared, because the chart showed the all-time top N per year."""
    d = run(ChartSpec(dimension="date", grain="year", series="artist", per_period=True, limit=3))
    assert d.table and not d.values and not d.series
    by_bucket: dict[str, list[str]] = {}
    for c in d.table:
        by_bucket.setdefault(c.bucket, []).append(c.name)
    assert all(len(v) <= 3 for v in by_bucket.values())
    # Different periods must be allowed to disagree, or this is just a comparison.
    assert len({tuple(v) for v in by_bucket.values()}) > 1


def test_per_period_is_a_table_not_a_chart():
    assert chart_for("date", "artist", True) == "table"
    assert chart_for("date", "artist", False) == "line"


def test_ranks_are_ordered_and_start_at_one():
    d = run(ChartSpec(dimension="date", grain="year", series="artist", per_period=True, limit=3))
    seen: dict[str, list[int]] = {}
    for c in d.table:
        seen.setdefault(c.bucket, []).append(c.rank)
    for ranks in seen.values():
        assert ranks == sorted(ranks) and ranks[0] == 1


def test_a_table_is_capped_and_says_so():
    d = run(ChartSpec(dimension="date", grain="month", series="artist", per_period=True, limit=5))
    assert len(d.table) <= MAX_TABLE_ROWS
    if len(d.table) == MAX_TABLE_ROWS:
        assert d.note and str(MAX_TABLE_ROWS) in d.note


def test_newest_period_first_so_truncation_drops_the_oldest():
    d = run(ChartSpec(dimension="date", grain="year", series="artist", per_period=True, limit=1))
    buckets = [c.bucket for c in d.table]
    assert buckets == sorted(buckets, reverse=True)


@pytest.mark.parametrize("series", ["artist", "track", "album", "genre"])
def test_every_rankable_thing_runs(series):
    d = run(ChartSpec(dimension="date", grain="year", series=series, per_period=True, limit=2))
    assert d.table, f"{series} produced nothing"
