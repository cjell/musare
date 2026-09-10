"""Run a ChartSpec against the local play history.

The model picks from menus; this turns the menu choices into SQL. No value the
model produced is ever interpolated into the query text - dimensions, metrics
and grains map through lookup tables keyed by their enum, and artist names are
bound as parameters. A spec cannot express a table, a column or a file, and
this cannot be talked into running one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import duckdb

from musicshare.config import settings
from musicshare.spec.chart import ChartSpec

# The export stores UTC. A naive timestamp has to be marked as UTC before it can
# be converted, or DuckDB reads it as already-local and the hours do not move -
# which is how "listening by hour" peaked at midnight instead of 10pm.
LOCAL = "America/New_York"
LOCAL_TS = f"((played_at AT TIME ZONE 'UTC') AT TIME ZONE '{LOCAL}')"

# A play is a listen once it clears half a minute; anything shorter is a skip or
# a mis-tap. Time listened counts every millisecond, and skip rate needs the
# skips themselves, so neither filters.
MIN_MS = 30_000

# A rate over three plays is not a rate. Without a floor, "what I skip the most"
# returns four artists at exactly 100% - each one a single skipped play - which
# is noise wearing the clothes of a finding.
MIN_SAMPLE = 25
RATE_METRICS = {"skip_rate"}

METRICS: dict[str, tuple[str, bool, str]] = {
    # name -> (sql, needs_min_ms, axis label)
    "hours": ("sum(ms_played) / 3600000.0", False, "hours"),
    "plays": ("count(*)", True, "plays"),
    "tracks": ("count(distinct track_uri)", True, "tracks"),
    "artists": ("count(distinct artist_name)", True, "artists"),
    "skip_rate": ("avg(case when skipped then 1.0 else 0 end) * 100", False, "% skipped"),
}

# Raw platform strings are build identifiers - "Windows 10 (10.0.19041; x64)",
# "iOS 14.8.1 (iPhone11,6)". Nobody wants a bar per OS build.
PLATFORM = """case
    when lower(platform) like '%ios%' or lower(platform) like '%iphone%' then 'iOS'
    when lower(platform) like '%android%' then 'Android'
    when lower(platform) like '%windows%' then 'Windows'
    when lower(platform) like '%mac%' or lower(platform) like '%os x%' then 'Mac'
    when lower(platform) like '%web%' then 'Web'
    else 'Other' end"""

DIMENSIONS: dict[str, tuple[str, str]] = {
    "artist": ("artist_name", "artist"),
    "album": ("album_name", "album"),
    "track": ("track_name", "track"),
    "platform": (PLATFORM, "platform"),
    "hour_of_day": (f"hour({LOCAL_TS})", "hour of day"),
    "day_of_week": (f"isodow({LOCAL_TS})", "day of week"),
}

GRAINS = {"day": "day", "week": "week", "month": "month", "year": "year"}

RANGES = {
    "7d": "interval 7 day",
    "30d": "interval 30 day",
    "90d": "interval 90 day",
    "6mo": "interval 6 month",
    "12mo": "interval 12 month",
}

DOW = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}

# Cyclical dimensions read in their own order, not by size - a day-of-week chart
# sorted by volume is unreadable.
NATURAL_ORDER = {"hour_of_day", "day_of_week", "date"}


@dataclass
class ChartData:
    labels: list[str]
    values: list[float]
    x_label: str
    y_label: str
    title: str
    chart: str
    total_rows: int = 0
    note: str | None = None
    sql: str = field(default="", repr=False)

    @property
    def empty(self) -> bool:
        return not self.values


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("install icu; load icu;")  # timezone conversion lives in icu
    con.execute(f"create view plays as select * from read_parquet('{settings().plays_glob}')")
    return con


def _where(spec: ChartSpec, needs_min: bool) -> tuple[str, list[Any]]:
    clauses, params = ["artist_name is not null"], []
    if needs_min:
        clauses.append(f"ms_played >= {MIN_MS}")
    if spec.year is not None:
        clauses.append(f"year({LOCAL_TS}) = {int(spec.year)}")
    elif spec.range == "this_year":
        clauses.append(f"year({LOCAL_TS}) = year(current_date)")
    elif spec.range in RANGES:
        clauses.append(f"played_at >= (select max(played_at) from plays) - {RANGES[spec.range]}")
    if spec.artists:
        # The only user-supplied strings that reach the query, and they are bound.
        marks = ", ".join("?" for _ in spec.artists)
        clauses.append(f"lower(artist_name) in ({marks})")
        params += [a.strip().lower() for a in spec.artists]
    return " and ".join(clauses), params


def run(spec: ChartSpec) -> ChartData:
    if not spec.understood:
        return ChartData(
            [], [], "", "", spec.title, spec.chart, note="not a question about listening"
        )

    metric_sql, needs_min, y_label = METRICS[spec.metric]
    where, params = _where(spec, needs_min)

    if spec.dimension == "date":
        grain = GRAINS[spec.grain or "month"]
        dim_sql, x_label = f"date_trunc('{grain}', {LOCAL_TS})", grain
        order, limit = "order by 1", ""
    else:
        dim_sql, x_label = DIMENSIONS[spec.dimension]
        if spec.dimension in NATURAL_ORDER:
            order, limit = "order by 1", ""
        else:
            order = f"order by 2 {'asc' if spec.sort == 'asc' else 'desc'}"
            limit = f"limit {int(spec.limit)}"

    having = "v is not null"
    if spec.metric in RATE_METRICS and spec.dimension not in NATURAL_ORDER:
        having += f" and count(*) >= {MIN_SAMPLE}"

    sql = f"""
        select {dim_sql} as k, {metric_sql} as v
        from plays
        where {where}
        group by 1
        having {having}
        {order} {limit}
    """
    con = _connect()
    rows = con.execute(sql, params).fetchall()
    total = con.execute(f"select count(*) from plays where {where}", params).fetchone()[0]

    labels, values = [], []
    for k, v in rows:
        if k is None:
            continue
        if spec.dimension == "day_of_week":
            labels.append(DOW.get(int(k), str(k)))
        elif spec.dimension == "hour_of_day":
            labels.append(f"{int(k):02d}")
        elif spec.dimension == "date":
            labels.append(str(k)[:10])
        else:
            labels.append(str(k))
        values.append(round(float(v), 2))

    return ChartData(
        labels=labels,
        values=values,
        x_label=x_label,
        y_label=y_label,
        title=spec.title or f"{y_label} by {x_label}",
        chart=spec.chart,
        total_rows=total,
        note=(
            f"artists with fewer than {MIN_SAMPLE} plays excluded"
            if spec.metric in RATE_METRICS and spec.dimension not in NATURAL_ORDER
            else None
        ),
        sql=sql.strip(),
    )
