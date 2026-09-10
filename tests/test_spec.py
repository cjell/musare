"""Fast tests for the deterministic half. No API, no cost, runs in normal CI."""

from __future__ import annotations

from datetime import date

import pytest

from musicshare.spec import ShowFilter, apply, validate

TODAY = date(2026, 9, 10)


def show(**kw):
    base = {
        "artist": "Slow Pulp", "date": "2026-09-20", "distance_mi": 10.0,
        "fans": 40_000, "yours": True,
    }
    base.update(kw)
    return base


def test_defaults_keep_everything():
    shows = [show(), show(artist="Deftones", yours=False)]
    assert len(apply(ShowFilter(), shows, TODAY)) == 2


def test_only_mine():
    shows = [show(), show(artist="Deftones", yours=False)]
    out = apply(ShowFilter(only_mine=True), shows, TODAY)
    assert [s["artist"] for s in out] == ["Slow Pulp"]


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
