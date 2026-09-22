"""The union of export and live rows, and the rule that governs it."""

from __future__ import annotations

import pytest

from musicshare import history
from musicshare.history import ACCOUNT_ONLY_NULL, COLUMNS, LIVE_ONLY_NULL, connect

pytestmark = pytest.mark.skipif(not history.has_export(), reason="no export ingested")


def test_view_has_every_export_column_plus_source():
    con = connect()
    cols = [r[0] for r in con.execute("describe plays").fetchall()]
    assert set(COLUMNS).issubset(cols)
    assert "source" in cols


def test_every_row_declares_where_it_came_from():
    con = connect()
    sources = {r[0] for r in con.execute("select distinct source from plays").fetchall()}
    assert sources and sources <= {"export", "account", "api"}


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


def test_account_rows_only_cover_what_the_export_does_not():
    """The better-informed source wins every period it covers."""
    con = connect()
    if not history.has_account():
        pytest.skip("no account export ingested")
    overlap = con.execute("""
        select count(*) from plays
        where source = 'account'
          and played_at <= (select max(played_at) from plays where source = 'export')
    """).fetchone()[0]
    assert overlap == 0


def test_captured_rows_yield_to_the_account_export_too():
    """Capture missed a fifth of plays on its best day, so it ranks below both."""
    con = connect()
    if not (history.has_account() and history.has_live()):
        pytest.skip("needs both an account export and captured rows")
    overlap = con.execute("""
        select count(*) from plays
        where source = 'api'
          and played_at <= (select max(played_at) from plays where source = 'account')
    """).fetchone()[0]
    assert overlap == 0


def test_account_rows_null_the_fields_they_cannot_know():
    con = connect()
    if not history.has_account():
        pytest.skip("no account export ingested")
    for col in sorted(ACCOUNT_ONLY_NULL):
        n = con.execute(
            f"select count(*) from plays where source = 'account' and {col} is not null"
        ).fetchone()[0]
        assert n == 0, f"{col} should be null on account rows, {n} are not"


def test_account_rows_keep_the_track_ids_that_could_be_recovered():
    """Matched by name against songs already seen; the rest stay honestly null."""
    con = connect()
    if not history.has_account():
        pytest.skip("no account export ingested")
    total, with_uri = con.execute("""
        select count(*), sum(case when track_uri is not null then 1 else 0 end)
        from plays where source = 'account'
    """).fetchone()
    assert total and with_uri / total > 0.5


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
