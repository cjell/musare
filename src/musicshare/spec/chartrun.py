"""Run a ChartSpec against the local play history.

The model picks from menus; this turns the menu choices into SQL. No value the
model produced is ever interpolated into the query text - dimensions, metrics
and grains map through lookup tables keyed by their enum, and artist names are
bound as parameters. A spec cannot express a table, a column or a file, and
this cannot be talked into running one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import duckdb

from musicshare import genres as genrelib
from musicshare.history import connect
from musicshare.spec.chart import MAX_LINES, ORDERED, ChartSpec

# Month names for a window's label. Shared with the drawing plan rather than
# written out twice - the same three letters mean the same thing on both.
from musicshare.spec.plot import MONTHS

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


# Today is a calendar day where the listener is, not the last 24 hours: at 9am
# "today" means since midnight. Unlike the relative ranges it is anchored on the
# clock rather than on the newest play, because a live chart of today that stops
# at yesterday's last play is a chart of yesterday.
def _day(back: int = 0) -> str:
    """One calendar day: today, or `back` days before it.

    A clause rather than a constant because a live chart of today is also the
    answer for yesterday - the same spec with the day moved. Nothing is stored to
    make that work: the plays are on disk, so any past day can be recomputed, and
    a day recomputed after an export replaces its estimates reads *better* than a
    snapshot taken at the time would.
    """
    day = f"cast((current_timestamp AT TIME ZONE '{LOCAL}') as date)"
    if back:
        day = f"({day} - interval {int(back)} day)"
    return f"cast({LOCAL_TS} as date) = {day}"


TODAY = _day()

DOW = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}

# Windows that can be stepped back through, and how long a step is. A rolling
# window moves by its own length - the seven days before the seven you are
# looking at - and "today" moves by a day. The month-shaped ranges are left out:
# a step of "6 months" is not a fixed number of days, and nothing asks for it.
STEP_DAYS = {"7d": 7, "30d": 30, "90d": 90}

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
    # A running total is drawn on a time axis, where each point is the total by the
    # end of its bucket. For a chart of today the page also needs how far into the
    # current hour "now" is, so the line stops there instead of at the end of an
    # hour that has not finished. None whenever the last bucket is not that hour.
    cumulative: bool = False
    partial_last: float | None = None
    # Which window this is - "Today", "Wed, Sep 16", "Sep 3 - Sep 10" - and
    # whether anything was played before it, so a caller stepping back through
    # earlier windows knows when to stop offering.
    window: str = ""
    earlier: bool = False
    # A running total of today as it happened, as [hours since midnight, total]
    # pairs, one ramp per play. Empty for every other chart.
    timeline: list[list[float]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.values and not self.table and not any(ln.values for ln in self.series)


def chart_for(
    dimension: str, series: str = "none", per_period: bool = False, cumulative: bool = False
) -> str:
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
    # A running total is a line whatever its axis. It only rises, and the rise is
    # the thing being shown - bars stood side by side do not draw it.
    return "line" if dimension == "date" or series != "none" or cumulative else "bar"


def effective(spec: ChartSpec) -> ChartSpec:
    """The spec as it will be drawn.

    A date axis over a single day has one bucket, so today over time is today by
    hour. Derived rather than asked for, for the reason chart type is not a field:
    a combination the model can get half-right is better removed than described.
    """
    if spec.range == "today" and spec.dimension == "date":
        return spec.model_copy(update={"dimension": "hour_of_day", "grain": None})
    return spec


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


def _where(spec: ChartSpec, needs_min: bool, back: int = 0) -> tuple[str, list[Any]]:
    clauses, params = ["artist_name is not null"], []
    if needs_min:
        clauses.append(f"ms_played >= {MIN_MS}")
    if spec.year is not None:
        clauses.append(f"year({LOCAL_TS}) = {int(spec.year)}")
    elif spec.range == "today":
        clauses.append(_day(back))
    elif spec.range == "this_year":
        clauses.append(f"year({LOCAL_TS}) = year(current_date)")
    elif spec.range in RANGES:
        anchor = "(select max(played_at) from plays)"
        step = STEP_DAYS.get(spec.range)
        if back and step:
            # The window before this one: one length further back, and stopping
            # where the window being stepped away from starts.
            clauses.append(f"played_at >= {anchor} - interval {(back + 1) * step} day")
            clauses.append(f"played_at < {anchor} - interval {back * step} day")
        else:
            clauses.append(f"played_at >= {anchor} - {RANGES[spec.range]}")
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


def _pretty(d: date) -> str:
    return f"{MONTHS[d.month - 1]} {d.day}"


def _window(con: duckdb.DuckDBPyConnection, spec: ChartSpec, back: int) -> tuple[str, bool]:
    """What this chart covers, in dates, and whether anything was played before it.

    Said here because this is where the listener's timezone lives, and worked out
    from the history rather than from a cap: an arrow that steps into a window
    with nothing in it is an arrow that lies. The caller decides how far back it
    is willing to go; this decides how far back there is anything to see.
    """
    if not steppable(spec):
        return "", False
    today, newest = con.execute(
        f"select cast((current_timestamp AT TIME ZONE '{LOCAL}') as date), "
        f"cast(max({LOCAL_TS}) as date) from plays"
    ).fetchone()
    if newest is None:
        return "", False

    if spec.range == "today":
        day = today - timedelta(days=back)
        label = "Today" if back == 0 else f"{DOW[day.isoweekday()]}, {_pretty(day)}"
        start = day
    else:
        step = STEP_DAYS[spec.range]
        end = newest - timedelta(days=back * step)
        start = end - timedelta(days=step)
        label = "Last " + spec.range[:-1] + " days" if back == 0 else ""
        if back:
            label = f"{_pretty(start)} - {_pretty(end)}"
    earlier = con.execute(
        f"select 1 from plays where cast({LOCAL_TS} as date) < ? limit 1", [start]
    ).fetchone()
    return label, earlier is not None


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


def steppable(spec: ChartSpec) -> bool:
    """Whether this chart has earlier windows to show.

    A named year and the open-ended ranges do not: "all time" has no previous
    all time, and a year is already a fixed window the spec can name itself.
    """
    return spec.year is None and (spec.range == "today" or spec.range in STEP_DAYS)


def run(spec: ChartSpec, back: int = 0) -> ChartData:
    """The numbers for a chart. `back` steps to an earlier window - the day
    before, the seven days before those - for the ranges that have one."""
    spec = effective(spec)
    back = max(0, int(back)) if steppable(spec) else 0
    if not spec.understood:
        return ChartData(
            [],
            [],
            "",
            "",
            spec.title,
            chart_for(spec.dimension, spec.series, spec.per_period, spec.cumulative),
            note="not a question about listening",
        )

    metric_sql, needs_min, y_label = METRICS[spec.metric]
    where, params = _where(spec, needs_min, back)
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
    chart = chart_for(spec.dimension, spec.series, spec.per_period, spec.cumulative)
    partial: float | None = None
    timeline: list[list[float]] = []

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
        if spec.dimension == "hour_of_day" and spec.metric not in RATE_METRICS:
            labels, values = _every_hour(spec, con, labels, values, back)
            if spec.range == "today" and spec.year is None:
                # A finished day has no "now": it runs to midnight, and the
                # marker and the short axis both belong to a day in progress.
                if back == 0:
                    partial = _into_this_hour(con)
                if spec.cumulative and spec.metric in ("hours", "plays"):
                    timeline = _timeline(spec, con, frm, where, params, back)
        data = (labels, values, [], sql.strip(), [])

    labels, values, series, sql, table = data
    if spec.cumulative:
        values = _running(values)
        series = [Line(ln.name, _running(ln.values)) for ln in series]
        y_label = f"total {y_label}"
    total = con.execute(f"select count(*) from {frm} where {where}", params).fetchone()[0]
    window, earlier = _window(con, spec, back)

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
        cumulative=spec.cumulative,
        partial_last=partial,
        timeline=timeline,
        window=window,
        earlier=earlier,
    )


def _timeline(
    spec: ChartSpec,
    con: duckdb.DuckDBPyConnection,
    frm: str,
    where: str,
    params: list[Any],
    back: int = 0,
) -> list[list[float]]:
    """Today's running total play by play, instead of hour by hour.

    Hourly points joined by straight lines hid how a day was listened to: a break
    from 12:16 to 13:12 drew as a slow climb, and a total that stopped at 15:30
    kept rising to the next hour's mark. Here each play adds its time across the
    minutes it played - it ends at `played_at` and began its own length earlier -
    so the line climbs while music is on and lies flat while it is not. A count
    has no duration, so for plays each one is a step where it ended. The line
    stops at the current minute.
    """
    clock = f"hour({LOCAL_TS}) + minute({LOCAL_TS}) / 60.0 + second({LOCAL_TS}) / 3600.0"
    rows = con.execute(
        f"select {clock} as h, ms_played from {frm} where {where} order by 1", params
    ).fetchall()
    now = con.execute(
        f"select hour(t) + minute(t) / 60.0 + second(t) / 3600.0 "
        f"from (select current_timestamp AT TIME ZONE '{LOCAL}' as t)"
    ).fetchone()[0]

    points, total, cursor = [[0.0, 0.0]], 0.0, 0.0
    for end_h, ms in rows:
        end = float(end_h)
        if spec.metric == "plays":
            points += [[end, total], [end, total + 1]]
            total += 1
        else:
            heard = (ms or 0) / 3_600_000
            # Plays can overlap by a second or two; a line cannot go back in time.
            start = max(cursor, end - heard)
            points += [[start, total], [end, total + heard]]
            total += heard
        cursor = end
    # Today stops at this minute; a day that is over runs flat to midnight,
    # because the hours after the last play are hours that happened.
    points.append([24.0 if back else max(float(now), cursor), total])
    return [[round(x, 4), round(y, 3)] for x, y in points]


def _into_this_hour(con: duckdb.DuckDBPyConnection) -> float:
    """How much of the current local hour has passed, from 0 to 1."""
    row = con.execute(
        f"select minute(ts), second(ts) from (select current_timestamp AT TIME ZONE '{LOCAL}' as ts)"
    ).fetchone()
    return round((int(row[0]) + int(row[1]) / 60) / 60, 3)


def _running(values: list[float] | list[float | None]) -> list[float]:
    """Each point as everything up to it.

    Applied after the query and after empty hours are filled, so the line keeps
    its level through a quiet stretch instead of stepping over it. A gap adds
    nothing, because a bucket with no plays in it holds no hours or plays - the
    validator has already refused the metrics for which that would not be true.
    """
    total, out = 0.0, []
    for v in values:
        total += v or 0.0
        out.append(round(total, 2))
    return out


def _every_hour(
    spec: ChartSpec,
    con: duckdb.DuckDBPyConnection,
    labels: list[str],
    values: list[float],
    back: int = 0,
) -> tuple[list[str], list[float]]:
    """Hours with nothing played as zeros, rather than as missing bars.

    Without this 10am sits beside 2pm as if they were neighbours. A chart of today
    stops at the current hour: the hours still to come have not happened, which
    is not the same as nothing having been played in them. A rate is left alone,
    because a rate over no plays is unknown rather than zero.
    """
    have = dict(zip(labels, values, strict=True))
    last = 23
    if spec.range == "today" and spec.year is None and back == 0:
        now = con.execute(f"select hour(current_timestamp AT TIME ZONE '{LOCAL}')").fetchone()
        last = int(now[0])
    hours = [f"{h:02d}" for h in range(last + 1)]
    return hours, [have.get(h, 0.0) for h in hours]


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
