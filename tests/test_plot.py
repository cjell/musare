"""The drawing plan: every layout decision the page used to make untested.

No history and no model - plans are built from payloads written out here, so
each chart shape that once arrived with its own drawing bug has a case.
"""

from __future__ import annotations

import pytest

from musicshare.spec.plot import MAX_X_TICKS, nice_top, plan, tick_label, value_label, y_scale


def payload(**kw):
    base = {
        "labels": [],
        "values": [],
        "series": [],
        "table": [],
        "x_label": "",
        "y_label": "hours",
    }
    base.update(kw)
    return base


# ------------------------------------------------------------------- scale


@pytest.mark.parametrize(
    "peak,labels",
    [
        (3.19, ["0", "1", "2", "3", "4"]),
        (0.9, ["0", "0.2", "0.4", "0.6", "0.8", "1"]),
        (63.5, ["0", "20", "40", "60", "80"]),
        (105.1, ["0", "25", "50", "75", "100", "125"]),
        (13941, ["0", "5k", "10k", "15k", "20k"]),
        (4, ["0", "1", "2", "3", "4", "5"]),
        (11, ["0", "2.5", "5", "7.5", "10", "12.5"]),
    ],
)
def test_gridlines_land_on_round_steps_just_above_the_data(peak, labels):
    """The chart that peaked at 3.19 read 0, 2.5, 5 and left a third of itself empty."""
    scale = y_scale(peak)
    assert [t["label"] for t in scale["ticks"]] == labels
    assert scale["max"] >= peak and scale["max"] == scale["ticks"][-1]["at"]


def test_labels_carry_the_precision_of_their_step_and_no_more():
    assert tick_label(0.25, 0.25) == "0.25"
    assert tick_label(2500, 500) == "2500"
    assert tick_label(5000, 5000) == "5k"


def test_an_empty_scale_still_has_somewhere_to_draw():
    assert nice_top(0) == 1.0


def test_values_beside_bars_are_rounded_for_reading():
    assert [value_label(v) for v in (0.43, 12.0, 105.4, 2400)] == ["0.4", "12", "105", "2k"]


# ------------------------------------------------------------------- kinds


def test_names_lie_on_their_side_with_values_at_the_ends():
    p = plan(payload(labels=["Phoebe Bridgers", "J. Cole"], values=[15.0, 105.1], x_label="artist"))
    assert p["kind"] == "hbar"
    assert p["value_labels"] == ["15", "105"]
    assert p["height"] >= 120


def test_a_long_name_is_cut_for_the_axis_and_kept_whole_for_the_tooltip():
    name = "Da Fonk (feat. Joni) - Club Mix - Mochakk"
    p = plan(payload(labels=[name], values=[1.0], x_label="track"))
    assert p["labels"] == [name]
    assert len(p["tick_labels"][0]) <= 22 and p["tick_labels"][0].endswith("…")


def test_hours_of_the_day_stand_up():
    p = plan(payload(labels=["00", "01", "02"], values=[0.0, 1.0, 0.5], x_label="hour of day"))
    assert p["kind"] == "bar" and p["value_labels"] == []


def test_a_table_and_an_empty_chart_say_so():
    assert (
        plan(payload(table=[{"bucket": "2024", "rank": 1, "name": "x", "value": 1}]))["kind"]
        == "table"
    )
    assert plan(payload(labels=[], values=[]))["kind"] == "empty"


# ------------------------------------------------------------------- lines


def test_a_line_puts_each_point_in_the_middle_of_its_bucket_under_its_label():
    """Year labels used to sit at the start of a bucket and points in the middle."""
    p = plan(
        payload(
            labels=["2024-01-01", "2025-01-01", "2026-01-01"],
            values=[3.0, 2.0, 1.0],
            x_label="year",
            chart="line",
        )
    )
    xs = [pt[0] for pt in p["series"][0]["points"]]
    assert xs == [0.5, 1.5, 2.5]
    assert [t["at"] for t in p["x"]["ticks"]] == xs
    assert [t["label"] for t in p["x"]["ticks"]] == ["2024", "2025", "2026"]


def test_months_across_years_carry_the_year():
    p = plan(
        payload(
            labels=["2025-12-01", "2026-01-01"], values=[1.0, 2.0], x_label="month", chart="line"
        )
    )
    assert [t["label"] for t in p["x"]["ticks"]] == ["Dec '25", "Jan '26"]


def test_a_running_total_of_today_starts_at_zero_and_stops_at_now():
    """It used to run flat past the current time."""
    p = plan(
        payload(
            labels=["00", "01", "02"],
            values=[0.0, 0.4, 1.3],
            x_label="hour of day",
            chart="line",
            cumulative=True,
            partial_last=0.25,
        )
    )
    pts = p["series"][0]["points"]
    assert pts[0] == [0.0, 0.0]
    assert [pt[0] for pt in pts[1:]] == [1.0, 2.0, 2.25], "each hour's total where the hour ends"
    ticks = p["x"]["ticks"]
    assert [t["at"] for t in ticks] == [0.0, 1.0, 2.0, 3.0], "labels where hours begin"
    assert ticks[-1]["label"] == "03" and p["x"]["max"] == 4, "an hour past now is on the axis"
    assert p["now"] is True and p["series"][0]["labels"][0] == "start"


def test_a_running_total_over_complete_buckets_has_no_now_marker():
    p = plan(payload(labels=["a", "b"], values=[1.0, 2.0], chart="line", cumulative=True))
    assert p["now"] is False and p["series"][0]["points"][-1][0] == 2.0


def test_every_compared_line_is_named_and_gaps_stay_gaps():
    """The legend cut off two of six names when the page drew it."""
    names = ["trap", "gangsta rap", "alternative rnb", "cloud rap", "modern country", "indie rock"]
    p = plan(
        payload(
            labels=["2024-01-01", "2025-01-01"],
            series=[{"name": n, "values": [1.0, None]} for n in names],
            x_label="year",
            chart="line",
        )
    )
    assert [s["name"] for s in p["series"]] == names
    assert p["legend"] is True and p["fill"] is False
    assert p["series"][0]["points"][1][1] is None


def test_a_long_axis_is_thinned_to_labels_that_fit():
    labels = [f"2026-01-{d:02d}" for d in range(1, 31)]
    p = plan(payload(labels=labels, values=[1.0] * 30, x_label="day", chart="line"))
    assert len(p["x"]["ticks"]) <= MAX_X_TICKS


def test_a_total_closing_in_on_a_gridline_gets_room_above_it():
    """A live total at 3.93 sat flush against a top of 4 until it crossed it."""
    assert y_scale(3.93)["max"] == 5


def test_a_running_total_of_today_follows_each_play_and_lies_flat_through_breaks():
    timeline = [[0.0, 0.0], [10.0, 0.0], [10.1, 0.1], [12.2, 0.1], [12.3, 0.2], [16.2, 0.2]]
    p = plan(
        payload(
            labels=[f"{h:02d}" for h in range(17)],
            values=[0.0] * 17,
            x_label="hour of day",
            chart="line",
            cumulative=True,
            partial_last=0.2,
            timeline=timeline,
        )
    )
    line = p["series"][0]
    assert line["points"] == timeline
    assert line["labels"][2] == "10:06" and p["now"] is True
    assert p["y"]["max"] > 0.2


def test_a_day_in_progress_shows_hours_ahead_so_the_line_is_not_against_the_edge():
    """4pm used to appear only once it was 4pm."""
    p = plan(
        payload(
            labels=[f"{h:02d}" for h in range(17)],
            values=[0.0] * 17,
            x_label="hour of day",
            chart="line",
            cumulative=True,
            partial_last=0.2,
            timeline=[[0.0, 0.0], [16.2, 1.0]],
        )
    )
    assert p["x"]["max"] == 19
    assert p["series"][0]["points"][-1][0] < 17, "the line still stops at now"
    assert any(t["label"] == "18" for t in p["x"]["ticks"]) or p["x"]["max"] > 18


def test_a_finished_period_spans_exactly_its_buckets():
    labels = ["2024-01-01", "2025-01-01"]
    p = plan(payload(labels=labels, values=[1.0, 2.0], x_label="year", chart="line"))
    assert p["x"]["max"] == 2


def test_every_point_fits_under_the_top_gridline():
    p = plan(payload(labels=["a", "b"], values=[3.19, 0.2], chart="line"))
    assert all(pt[1] <= p["y"]["max"] for pt in p["series"][0]["points"])
