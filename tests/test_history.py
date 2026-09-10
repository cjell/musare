"""The union of export and live rows, and the rule that governs it."""

from __future__ import annotations

import pytest

from musicshare import history
from musicshare.history import COLUMNS, LIVE_ONLY_NULL, connect

pytestmark = pytest.mark.skipif(not history.has_export(), reason="no export ingested")


def test_view_has_every_export_column_plus_source():
    con = connect()
    cols = [r[0] for r in con.execute("describe plays").fetchall()]
    assert set(COLUMNS).issubset(cols)
    assert "source" in cols


def test_every_row_declares_where_it_came_from():
    con = connect()
    sources = {r[0] for r in con.execute("select distinct source from plays").fetchall()}
    assert sources and sources <= {"export", "api"}


def test_live_rows_never_predate_the_export_watermark():
    """The export wins for every period it covers - that is the whole rule."""
    con = connect()
    if not history.has_live():
        pytest.skip("no live rows yet")
    overlap = con.execute("""
        select count(*) from plays
        where source = 'api'
          and played_at <= (select max(played_at) from plays where source = 'export')
    """).fetchone()[0]
    assert overlap == 0


def test_live_rows_null_the_fields_they_cannot_know():
    con = connect()
    if not history.has_live():
        pytest.skip("no live rows yet")
    for col in sorted(LIVE_ONLY_NULL):
        n = con.execute(
            f"select count(*) from plays where source = 'api' and {col} is not null"
        ).fetchone()[0]
        assert n == 0, f"{col} should be null on api rows, {n} are not"


def test_skip_rate_ignores_unknown_rather_than_scoring_it_zero():
    """A null `skipped` must not drag a rate down as if it were a completion."""
    con = connect()
    got = con.execute("""
        select avg(case when skipped then 1.0 when not skipped then 0.0 end) * 100
        from (select true as skipped union all select false union all select null)
    """).fetchone()[0]
    assert got == pytest.approx(50.0)
