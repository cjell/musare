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

from musicshare import genres as genrelib
from musicshare.history import connect
from musicshare.spec.chart import MAX_LINES, ORDERED, ChartSpec

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
    "distinct_tracks": ("count(distinct track_uri)", True, "tracks"),
    "distinct_artists": ("count(distinct artist_name)", True, "artists"),
    "skip_rate": (
        "avg(case when skipped then 1.0 when not skipped then 0.0 end) * 100",
        False,
        "% skipped",
    ),
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

# Genre arrives by join rather than from a column, so it is named here and
# resolved in `_from`. Everything else is an expression over `plays` alone.
GENRE_SQL = "g.genre"

# What `series` names, as SQL. A track keyed on its title alone would merge four
# different songs called "Don't Stop", so the key carries the artist - and since
# it is also what gets printed, the separator is the one a reader expects.
SPLITS: dict[str, str] = {
    "artist": "artist_name",
    "track": "track_name || ' - ' || artist_name",
    "album": "album_name || ' - ' || artist_name",
    "genre": GENRE_SQL,
}

# Rows a table may carry. A chart caps itself - forty bars, six lines - and a
# table has no such shape to stop it: "top 3 artists every day" is three
# thousand rows that nobody asked to scroll. Truncation is fine; silent
# truncation is not, so the caller says so in the note.
MAX_TABLE_ROWS = 60

DIMENSIONS: dict[str, tuple[str, str]] = {
    "artist": ("artist_name", "artist"),
    "genre": (GENRE_SQL, "genre"),
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
# sorted by volume is unreadable. Same set the schema uses to decide whether a
# comparison has an axis to run along, imported rather than repeated.
NATURAL_ORDER = ORDERED


@dataclass
class Cell:
    """One row of a table: which period, what placed there, and how much."""

    bucket: str
    rank: int
    name: str
    value: float


@dataclass
class Line:
    """One series on a split chart, aligned to the shared labels."""

    name: str
    values: list[float | None]


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
    # Empty for an ordinary chart, which keeps `values` the single source for
    # every caller that predates splitting. Populated only when the spec asked
    # for a comparison, and then `values` stays empty - one of the two is the
    # data and there is never a question which.
    series: list[Line] = field(default_factory=list)
    # Populated instead of the other two when the answer exists but cannot be
    # drawn. Showing the numbers beats drawing a chart of a different question,
    # which is what this used to do.
    table: list[Cell] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.values and not self.table and not any(ln.values for ln in self.series)


def chart_for(dimension: str, series: str = "none", per_period: bool = False) -> str:
    """Dates read as a line; everything else is a comparison, so bars.

    A split chart is always a line too. Grouped bars at 24 hours by 4 artists is
    96 rectangles and no legible trend, and the split is only ever allowed on a
    dimension that has an order to run along - which is exactly the thing a line
    draws well.

    A per-period ranking is a table, and that is decided here rather than asked
    for. Every period holds different names, so there is no line to follow and no
    shared axis to put bars on; nine years by three artists is twenty-seven
    labelled bars on a phone, which is not a chart, it is a table drawn badly.
    Deriving it keeps the model out of a decision it would get wrong the moment
    it was unsure - the same reason chart type is not a field.
    """
    if per_period:
        return "table"
    return "line" if dimension == "date" or series != "none" else "bar"


def _connect(spec: ChartSpec | None = None) -> duckdb.DuckDBPyConnection:
    # icu carries the timezone conversion; history unions the export with any
    # live rows past its high-water mark.
    con = connect(icu=True)
    # Only when the spec asks for it. A chart of top artists should not fail
    # because a build step that has nothing to do with it has not been run.
    if spec is not None and needs_genre(spec):
        genrelib.attach(con)
    return con


def needs_genre(spec: ChartSpec) -> bool:
    """Whether this spec has to reach the atlas at all."""
    return spec.dimension == "genre" or spec.series == "genre" or bool(spec.genres)


def _from(spec: ChartSpec) -> str:
    """`plays`, with the genre of each play's artist when the spec needs one.

    A left join, not an inner one: an artist the corpus has not caught up with
    still counts towards an hours total, and silently dropping their plays would
    make every unrelated chart quietly wrong the moment genre was involved. The
    predicates below decide what an unmatched row means for the chart at hand.
    """
    if not needs_genre(spec):
        return "plays p"
    return "plays p left join artist_genre g on lower(p.artist_name) = g.artist"


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
    if spec.genres:
        # The model chose from an enum and the widening ran over a file this
        # build wrote, so none of these strings came from a user - they are bound
        # anyway, because "it cannot be hostile" is an argument that stops being
        # true one refactor later.
        covered = genrelib.expand(spec.genres)
        if not covered:
            # Asked for a real genre this corpus has no region for. An impossible
            # predicate is the honest answer: the chart comes back empty and the
            # note says which word did it.
            clauses.append("false")
        else:
            marks = ", ".join("?" for _ in covered)
            clauses.append(f"g.genre in ({marks})")
            params += covered
    # One bar per genre cannot have a bar for "no genre". A filtered chart has
    # already excluded them; this is for the unfiltered "what do I listen to".
    if spec.dimension == "genre" or spec.series == "genre":
        clauses.append("g.genre is not null")
    return " and ".join(clauses), params


def _label(spec: ChartSpec, k: Any) -> str:
    """One bucket's label, in the form its dimension reads best."""
    if spec.dimension == "day_of_week":
        return DOW.get(int(k), str(k))
    if spec.dimension == "hour_of_day":
        return f"{int(k):02d}"
    if spec.dimension == "date":
        return str(k)[:10]
    return str(k)


def _bucket_label(spec: ChartSpec, k: Any) -> str:
    """A period's name in a table.

    Separate from `_label`, which feeds a chart axis the page then formats for
    itself. A table row has no axis to be formatted against, so the label has to
    arrive finished - and "2026-01-01" is not what a year is called.
    """
    if spec.dimension != "date":
        return _label(spec, k)
    iso = str(k)[:10]
    return {"year": iso[:4], "month": iso[:7]}.get(spec.grain or "month", iso)


def _axis(spec: ChartSpec) -> tuple[str, str, str, str, list[Any]]:
    """The dimension's SQL, its label, how the rows come back, and its parameters."""
    if spec.dimension == "date":
        grain = GRAINS[spec.grain or "month"]
        return f"date_trunc('{grain}', {LOCAL_TS})", grain, "order by 1", "", []
    dim_sql, x_label = DIMENSIONS[spec.dimension]
    params: list[Any] = []
    if spec.dimension == "genre":
        dim_sql, params = _named_genres(spec)
    if spec.dimension in NATURAL_ORDER:
        return dim_sql, x_label, "order by 1", "", params
    order = f"order by 2 {'asc' if spec.sort == 'asc' else 'desc'}"
    return dim_sql, x_label, order, f"limit {int(spec.limit)}", params


def _note(spec: ChartSpec, rated: bool, truncated: bool = False) -> str | None:
    """Whatever the reader needs in order not to misread the chart."""
    bits = []
    if truncated:
        bits.append(f"most recent {MAX_TABLE_ROWS} rows")
    if spec.genres:
        missing = genrelib.unmatched(spec.genres)
        if missing:
            bits.append(f"no {' or '.join(missing)} in the corpus")
    if rated:
        bits.append(f"fewer than {MIN_SAMPLE} plays excluded")
    return "; ".join(bits) or None


def run(spec: ChartSpec) -> ChartData:
    if not spec.understood:
        return ChartData(
            [],
            [],
            "",
            "",
            spec.title,
            chart_for(spec.dimension, spec.series, spec.per_period),
            note="not a question about listening",
        )

    metric_sql, needs_min, y_label = METRICS[spec.metric]
    where, params = _where(spec, needs_min)
    frm = _from(spec)
    dim_sql, x_label, order, limit, dim_params = _axis(spec)

    # A rate needs a floor under it, but a floor on a ranked chart and a floor on
    # a series are different bets: one drops a bar, the other puts a hole in a
    # line. Both beat drawing 100% from a single play.
    rated = spec.metric in RATE_METRICS and spec.dimension not in NATURAL_ORDER
    having = "v is not null"
    if rated:
        having += f" and count(*) >= {MIN_SAMPLE}"

    con = _connect(spec)
    chart = chart_for(spec.dimension, spec.series, spec.per_period)

    if spec.per_period:
        data = _per_period(spec, con, frm, where, params, dim_sql, dim_params, metric_sql)
    elif spec.series != "none":
        data = _split(spec, con, frm, where, params, dim_sql, dim_params, metric_sql, having)
    else:
        sql = f"""
            select {dim_sql} as k, {metric_sql} as v
            from {frm}
            where {where}
            group by 1
            having {having}
            {order} {limit}
        """
        rows = con.execute(sql, dim_params + params).fetchall()
        labels, values = [], []
        for k, v in rows:
            if k is None:
                continue
            labels.append(_label(spec, k))
            values.append(round(float(v), 2))
        data = (labels, values, [], sql.strip(), [])

    labels, values, series, sql, table = data
    total = con.execute(f"select count(*) from {frm} where {where}", params).fetchone()[0]

    return ChartData(
        labels=labels,
        values=values,
        x_label=x_label,
        y_label=y_label,
        title=spec.title or f"{y_label} by {x_label}",
        chart=chart,
        total_rows=total,
        note=_note(spec, rated, truncated=len(table) >= MAX_TABLE_ROWS),
        sql=sql,
        series=series,
        table=table,
    )


def _named_genres(spec: ChartSpec) -> tuple[str, list[Any]]:
    """Buckets for the genres the user named, or the plain column if they named none.

    "rap versus rock" has to come back as two buckets called rap and rock, not
    as the nine regions underneath them - splitting on the region drew the top
    six genres over time, which is a fine chart and not the one that was asked
    for. The same holds for a bar chart: "is it jazz or soul" wants two bars.

    So one rule covers both - naming genres makes those words the buckets,
    whether they are bars or lines. Overlaps go to whichever was asked for
    first, because a region can only be in one bucket and the order the user
    said them in is the only ranking available.
    """
    if not spec.genres:
        return GENRE_SQL, []

    whens, params, claimed = [], [], set()
    for g in spec.genres:
        regions = [r for r in genrelib.expand([g]) if r not in claimed]
        if not regions:
            continue
        claimed.update(regions)
        marks = ", ".join("?" for _ in regions)
        whens.append(f"when {GENRE_SQL} in ({marks}) then ?")
        params += [*regions, str(g)]
    if not whens:
        return GENRE_SQL, []
    return "case " + " ".join(whens) + " end", params


def _split_by(spec: ChartSpec) -> tuple[str, list[Any]]:
    """What each line on a split chart is."""
    if spec.series == "genre":
        return _named_genres(spec)
    return SPLITS[spec.series], []


def _per_period(
    spec: ChartSpec,
    con: duckdb.DuckDBPyConnection,
    frm: str,
    where: str,
    params: list[Any],
    dim_sql: str,
    dim_params: list[Any],
    metric_sql: str,
) -> tuple[list[str], list[float], list[Line], str, list[Cell]]:
    """The top few inside every period, ranked separately in each.

    One window function does the whole thing. The alternative - rank overall,
    then hope the same names lead every period - is what this replaces, and it
    was wrong in a way that looked right: nine of the fourteen artists who
    actually led a year of this listener's history never appeared on that chart.

    Periods come back newest first. A table long enough to be cut should lose
    2018 rather than this month, and a list of what led each period reads
    perfectly well downwards.
    """
    split_sql, split_params = _split_by(spec)
    sql = f"""
        select bucket, name, v, rk from (
          select {dim_sql} as bucket, {split_sql} as name, {metric_sql} as v,
                 row_number() over (
                   partition by {dim_sql} order by {metric_sql} desc, {split_sql}
                 ) as rk
          from {frm}
          where {where}
          group by 1, 2
        )
        where rk <= {int(spec.limit)} and name is not null and v is not null
        order by bucket desc, rk
        limit {MAX_TABLE_ROWS}
    """
    # dim_sql and split_sql each appear twice - once in the select list, once in
    # the window - so their parameters are bound twice, in that order.
    bound = dim_params + split_params + dim_params + split_params + params
    rows = con.execute(sql, bound).fetchall()
    table = [
        Cell(bucket=_bucket_label(spec, b), rank=int(rk), name=str(n), value=round(float(v), 2))
        for b, n, v, rk in rows
    ]
    return [], [], [], sql.strip(), table


def _split(
    spec: ChartSpec,
    con: duckdb.DuckDBPyConnection,
    frm: str,
    where: str,
    params: list[Any],
    dim_sql: str,
    dim_params: list[Any],
    metric_sql: str,
    having: str,
) -> tuple[list[str], list[float], list[Line], str]:
    """One line per artist or per genre, over a shared axis.

    Two passes would be tidier - pick the lines, then fetch them - but the data
    for both is the same grouping, so this fetches once and chooses in Python.
    """
    split_sql, split_params = _split_by(spec)
    sql = f"""
        select {dim_sql} as k, {split_sql} as s, {metric_sql} as v
        from {frm}
        where {where}
        group by 1, 2
        having {having}
        order by 1
    """
    # Both expressions sit in the select list, in this order, so their
    # parameters are bound before the where clause's.
    rows = [
        r
        for r in con.execute(sql, dim_params + split_params + params).fetchall()
        if r[0] is not None and r[1] is not None
    ]
    if not rows:
        return [], [], [], sql.strip(), []

    totals: dict[str, float] = {}
    for _, name, v in rows:
        totals[str(name)] = totals.get(str(name), 0.0) + float(v)

    # `limit` is how many bars a ranked chart shows, and with a split it is how
    # many lines - "top 3 artists every year" said 3 and got six of them, because
    # this reached for the ceiling instead of the request. The ceiling still
    # applies: six lines is where a legend stops being readable.
    lines = min(int(spec.limit), MAX_LINES)

    if spec.series == "artist" and spec.artists:
        # The user named them, so they are the lines - in the order they said
        # them, and including one with no plays, because "you have never played
        # this" is the answer to a comparison rather than a reason to hide it.
        wanted = [a.strip() for a in spec.artists if a.strip()]
        chosen, seen = [], set()
        lower = {k.lower(): k for k in totals}
        for a in wanted[:lines]:
            key = lower.get(a.lower(), a)
            if key.lower() not in seen:
                seen.add(key.lower())
                chosen.append(key)
    else:
        chosen = sorted(totals, key=lambda n: -totals[n])[:lines]

    # A gap means no plays in that bucket. For hours and counts that really is
    # zero; for a rate it is unknown, and drawing 0% would invent a fact - so
    # those come back null and the line breaks instead.
    gap: float | None = None if spec.metric in RATE_METRICS else 0.0

    order_key: dict[Any, Any] = {}
    cells: dict[tuple[Any, str], float] = {}
    for k, name, v in rows:
        order_key.setdefault(k, k)
        cells[(k, str(name))] = round(float(v), 2)

    keys = sorted(order_key)
    labels = [_label(spec, k) for k in keys]
    series = [Line(name=c, values=[cells.get((k, c), gap) for k in keys]) for c in chosen]
    return labels, [], series, sql.strip(), []
