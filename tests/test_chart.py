"""Chart spec and executor. Hits the local Parquet, never a model."""

from __future__ import annotations

import pytest

from musicshare.spec.chart import MAX_SERIES, ChartSpec
from musicshare.spec.chartrun import DIMENSIONS, METRICS, MIN_SAMPLE, chart_for, run
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
        ChartSpec(limit=MAX_SERIES + 1)


def test_chart_type_is_derived_not_chosen():
    """Removing it from the schema removed a whole class of undrawable specs."""
    assert chart_for("date") == "line"
    assert all(chart_for(d) == "bar" for d in DIMENSIONS)
    assert "chart" not in ChartSpec.model_json_schema()["properties"]
