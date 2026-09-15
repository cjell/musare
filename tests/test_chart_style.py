"""Chart styles: what a person can change about a chart's look, and what they cannot.

No model and no history. A style never changes a number, so these build plans from
payloads written out here and check the drawing each style produces, the options
each chart shape offers, and the fallbacks when a stored style no longer fits.
"""

from __future__ import annotations

import pydantic
import pytest
from fastapi.testclient import TestClient

from musicshare.api import app as app_mod
from musicshare.spec.plot import ChartStyle, plan, y_scale

client = TestClient(app_mod.app)


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


HOURS = payload(labels=["00", "01", "02"], values=[0.2, 1.0, 0.5], x_label="hour of day")
ARTISTS = payload(labels=["Phoebe Bridgers", "J. Cole"], values=[15.0, 105.1], x_label="artist")
YEARS = payload(
    labels=["2024-01-01", "2025-01-01"], values=[3.0, 2.0], x_label="year", chart="line"
)
COMPARED = payload(
    labels=["2024-01-01", "2025-01-01"],
    series=[{"name": "rap", "values": [1.0, 2.0]}, {"name": "rock", "values": [2.0, 1.0]}],
    x_label="year",
    chart="line",
)
RUNNING = payload(labels=["a", "b"], values=[1.0, 2.0], chart="line", cumulative=True)


# ------------------------------------------------------------------- the menu


def test_a_style_is_a_menu_and_nothing_else():
    with pytest.raises(pydantic.ValidationError):
        ChartStyle(type="pie")
    with pytest.raises(pydantic.ValidationError):
        ChartStyle(color="#ff0000")
    with pytest.raises(pydantic.ValidationError):
        ChartStyle(sql="drop table plays")


@pytest.mark.parametrize(
    "chart,types",
    [
        (ARTISTS, ["bar"]),
        (HOURS, ["bar", "line", "area"]),
        (YEARS, ["bar", "line", "area"]),
        (COMPARED, ["line"]),
        (RUNNING, ["line", "area"]),
    ],
)
def test_each_shape_offers_only_the_types_that_make_sense_for_it(chart, types):
    """A line across a list of names implies an order that is not there."""
    assert plan(chart)["styles"]["types"] == types


def test_the_default_look_is_unchanged_by_having_styles():
    assert plan(HOURS)["kind"] == "bar"
    assert plan(ARTISTS)["kind"] == "hbar"
    assert plan(YEARS)["fill"] is True
    assert plan(COMPARED)["fill"] is False


# ------------------------------------------------------------------- type


def test_hour_bars_can_be_a_line_or_an_area():
    line = plan(HOURS, {"type": "line"})
    area = plan(HOURS, {"type": "area"})
    assert line["kind"] == "line" and line["fill"] is False
    assert area["kind"] == "line" and area["fill"] is True
    assert [pt[1] for pt in line["series"][0]["points"]] == [0.2, 1.0, 0.5], "same numbers"


def test_a_trend_can_be_bars():
    p = plan(YEARS, {"type": "bar"})
    assert p["kind"] == "bar" and p["labels"] == ["2024", "2025"]


def test_a_stored_type_that_does_not_fit_falls_back_rather_than_breaking():
    """A comparison asked to be bars is still drawn, as the line it has to be."""
    assert plan(COMPARED, {"type": "bar"})["kind"] == "line"
    assert plan(ARTISTS, {"type": "area"})["kind"] == "hbar"
    assert plan(RUNNING, {"type": "bar"})["kind"] == "line"


# ------------------------------------------------------------ gridlines, values


def test_gridlines_setting_changes_how_many_there_are_not_the_numbers():
    fewer, normal, more = (len(y_scale(3.19, d)["ticks"]) for d in (3, 5, 10))
    assert fewer < normal < more
    assert len(plan(HOURS, {"gridlines": "more"})["y"]["ticks"]) > len(plan(HOURS)["y"]["ticks"])


def test_values_can_be_shown_on_upright_bars_and_hidden_on_sideways_ones():
    assert plan(HOURS)["value_labels"] == []
    assert plan(HOURS, {"values": "show"})["value_labels"] == ["0.2", "1", "0.5"]
    assert plan(ARTISTS, {"values": "hide"})["value_labels"] == []


def test_values_are_only_offered_where_there_are_bars_to_put_them_on():
    assert plan(HOURS)["styles"]["values"] is True
    assert plan(HOURS, {"type": "line"})["styles"]["values"] is False


# -------------------------------------------------------------------- colour


def test_one_series_takes_a_colour_and_a_comparison_keeps_its_palette():
    assert plan(HOURS, {"color": "teal"})["color"] == "teal"
    compared = plan(COMPARED, {"color": "teal"})
    assert compared["color"] is None and compared["styles"]["color"] is False


# ----------------------------------------------------------------------- api


def test_the_plan_endpoint_draws_a_style():
    r = client.post("/api/chart/plan", json={"data": HOURS, "style": {"type": "line"}})
    assert r.status_code == 200 and r.json()["plan"]["kind"] == "line"


def test_the_plan_endpoint_still_takes_bare_numbers_from_old_saved_charts():
    r = client.post("/api/chart/plan", json=HOURS)
    assert r.status_code == 200 and r.json()["plan"]["kind"] == "bar"


def test_a_style_off_the_menu_is_refused():
    r = client.post("/api/chart/plan", json={"data": HOURS, "style": {"color": "chartreuse"}})
    assert r.status_code == 422
