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

# Songs and albums are counted per title AND artist. Keyed on the title alone,
# "Stay" by Rihanna and "Stay" by The Kid LAROI were one bar with both their
# plays in it, and the artist was gone before anything could show it. The key
# carries both, split apart again after the query: the title is the label, short
# enough for an axis, and the artist rides along for the hover.
BY_ARTIST = frozenset({"track", "album"})
# Bars that play when tapped, and what they play: the track this listener has
# played most inside each bar - the song itself, the album's most-played song,
# the artist's most-played song. A song rather than the album or the artist's
# top tracks because the history has track uris and nothing else, so it costs no
# Spotify request, and because "your song of theirs" is the more personal answer.
PLAYABLE_BARS = frozenset({"track", "album", "artist"})
KEY_SEP = ""  # a control character, so no real title can contain it

DIMENSIONS: dict[str, tuple[str, str]] = {
    "artist": ("artist_name", "artist"),
    "genre": (GENRE_SQL, "genre"),
    "album": ("album_name || chr(31) || coalesce(artist_name, '')", "album"),
    "track": ("track_name || chr(31) || coalesce(artist_name, '')", "track"),
    "platform": (PLATFORM, "platform"),
    "hour_of_day": (f"hour({LOCAL_TS})", "hour of day"),
    "day_of_week": (f"isodow({LOCAL_TS})", "day of week"),
}

GRAINS = {"day": "day", "week": "week", "month": "month", "year": "year"}
# What `last` counts, mapped rather than interpolated, like everything else here.
UNITS = {"day": "day", "week": "week", "month": "month", "year": "year"}

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
    # The height to draw to when several windows are meant to be compared - the
    # tallest any of them reaches. None when nobody asked for a shared scale.
    scale_peak: float | None = None
    # A running total of today as it happened, as [hours since midnight, total]
    # pairs, one ramp per play. Empty for every other chart.
    timeline: list[list[float]] = field(default_factory=list)
    # The artist under each label, for charts of songs or albums - the hover's
    # second line. Empty for every other chart.
    subs: list[str] = field(default_factory=list)
    # What tapping each bar plays - see PLAYABLE_BARS. Aligned to the labels,
    # with "" for a bar that has nothing to play. Empty for every other chart.
    uris: list[str] = field(default_factory=list)
    # The answer to a question with one number - or one date - for an answer,
    # in place of everything above: see `_answer`. None for every chart.
    stat: dict[str, Any] | None = None
    # What the window covers, in words and dates - "Last 2 years (Nov 2024 -
    # Nov 2026)". Empty for a chart whose `window` already says it.
    span: str = ""

    @property
    def empty(self) -> bool:
        # A count of nothing is still an answer: "you have not played it this
        # week" is what was asked.
        if self.stat is not None:
            return False
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


def _where(
    spec: ChartSpec, needs_min: bool, back: int = 0, windowed: bool = True
) -> tuple[str, list[Any]]:
    """The filters for one chart. `windowed` false leaves out the range itself,
    for a caller that means to ask about several windows at once."""
    clauses, params = ["artist_name is not null"], []
    if needs_min:
        clauses.append(f"ms_played >= {MIN_MS}")
    if not windowed:
        pass
    elif spec.year is not None:
        clauses.append(f"year({LOCAL_TS}) = {int(spec.year)}")
    elif spec.from_year is not None:
        clauses.append(f"year({LOCAL_TS}) >= {int(spec.from_year)}")
        if spec.to_year is not None:
            clauses.append(f"year({LOCAL_TS}) <= {int(spec.to_year)}")
    elif spec.last is not None and spec.last_unit in UNITS:
        # Anchored on the newest play, like the menu's relative windows.
        clauses.append(
            f"played_at >= (select max(played_at) from plays) - "
            f"interval {int(spec.last)} {UNITS[spec.last_unit]}"
        )
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
    for kind, names in (("track", spec.tracks), ("album", spec.albums)):
        if names:
            clause, bound = _titled(kind, names, bool(spec.artists))
            clauses.append(clause)
            params += bound
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


# A name as typed against a name as stored. Case, punctuation, "&" for "and",
# and the tails the catalogue adds - " - Remastered 2011", "(feat. X)", "[Live]"
# - do not decide whether "lover you shouldve come over" means Lover, You
# Should've Come Over, so all of them are dropped from both sides by the same
# expression. Versions of one song are counted together. A title that starts
# with a bracket keeps it: "(What's the Story) Morning Glory?" is not empty.
# Punctuation is the ASCII class plus curly quotes, so a title in another
# script keeps its letters.
CURLY = chr(0x2018) + chr(0x2019) + chr(0x201C) + chr(0x201D)


def _norm(col: str) -> str:
    x = f"replace(lower({col}), '&', ' and ')"
    x = rf"regexp_replace({x}, '^(.+?)\s*(\(|\[|\s-\s).*$', '\1')"
    x = f"regexp_replace({x}, '[[:punct:]{CURLY}]', '', 'g')"
    return rf"trim(regexp_replace({x}, '\s+', ' ', 'g'))"


TITLE_COLS = {"track": "track_name", "album": "album_name"}


def _titled(kind: str, names: list[str], has_artist: bool) -> tuple[str, list[Any]]:
    """Plays of the named songs or records, as one clause with its parameters.

    A title is not unique - several acts have a song called Fears - and without
    an artist named alongside it the question means the one this person plays.
    So each title is pinned to the artist with the most plays under it; the
    answer says which, and names the others (`_namesakes`). With an artist named,
    the artist clause already does the narrowing.
    """
    col = TITLE_COLS[kind]
    parts, params = [], []
    for name in names:
        name = name.strip()
        if has_artist:
            parts.append(f"{_norm(col)} = {_norm('?')}")
            params.append(name)
        else:
            parts.append(
                f"({_norm(col)} = {_norm('?')} and artist_name = ("
                f"select artist_name from plays where {_norm(col)} = {_norm('?')} "
                "group by 1 order by count(*) desc, 1 limit 1))"
            )
            params += [name, name]
    return "(" + " or ".join(parts) + ")", params


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


def _scale_peak(
    spec: ChartSpec,
    con: duckdb.DuckDBPyConnection,
    frm: str,
    metric_sql: str,
    windows: int,
) -> float | None:
    """The tallest this chart gets across the windows a reader can step through.

    A chart drawn to its own peak is redrawn to the same height every day, so a
    quiet Tuesday and a loud Monday look identical and only the axis labels
    differ - which is the opposite of what stepping through days is for. Holding
    one scale over the whole range makes the comparison the picture.

    Only for charts of the hours of a day, which is what Home pins and what can
    be stepped. A running total's height is the window's total; anything else is
    its tallest hour.
    """
    if windows < 1 or spec.dimension != "hour_of_day" or spec.series != "none":
        return None
    if spec.per_period or spec.metric in RATE_METRICS:
        return None

    where, params = _where(spec, METRICS[spec.metric][1], windowed=False)
    if spec.range == "today":
        bucket = f"cast({LOCAL_TS} as date)"
        span = f"{bucket} > cast((current_timestamp AT TIME ZONE '{LOCAL}') as date) - {windows}"
    elif spec.range in STEP_DAYS:
        step = STEP_DAYS[spec.range]
        newest = "(select max(played_at) from plays)"
        bucket = f"cast(date_diff('day', played_at, {newest}) / {step} as integer)"
        span = f"played_at > {newest} - interval {windows * step} day"
    else:
        return None

    # Per bucket per hour first, because a chart's height is an hour's worth -
    # or, for a running total, everything up to the end of the window. Each hour
    # is rounded the way run() rounds it before a running total adds them up, so
    # the shared height is measured in the same units as the line drawn under
    # it; summing the exact values put it a hundredth below a day it had to fit.
    inner = "sum(round(v, 2))" if spec.cumulative else "max(round(v, 2))"
    row = con.execute(
        f"""
        select max(tall) from (
          select {inner} as tall from (
            select {bucket} as w, hour({LOCAL_TS}) as h, {metric_sql} as v
            from {frm}
            where {where} and {span}
            group by 1, 2
          ) group by w
        )
        """,
        params,
    ).fetchone()
    return round(float(row[0]), 3) if row and row[0] is not None else None


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
    if spec.year is not None or spec.from_year is not None or spec.last is not None:
        return False
    return spec.range == "today" or spec.range in STEP_DAYS


def run(spec: ChartSpec, back: int = 0, scale_over: int = 0) -> ChartData:
    """The numbers for a chart - after reading the names in it as the ones in
    this history (`resolve`), and saying so wherever a name was read as another.
    """
    spec, fixes = resolve(spec)
    d = _run(spec, back, scale_over)
    if fixes:
        d.note = " ".join([*fixes, *([d.note] if d.note else [])])
    # The window in dates, so an answer over the wrong one is visible on it.
    # A list of recent plays is newest first already; "all time" on it would
    # only say that nothing narrowed it.
    unwindowed = spec.range == "all" and spec.year is None and spec.from_year is None
    if (
        spec.understood
        and not d.window
        and not (spec.answer == "recent" and unwindowed and spec.last is None)
    ):
        d.span = _span(spec)
        if d.stat is not None and d.span:
            d.stat["period"] = d.span
    return d


def _run(spec: ChartSpec, back: int = 0, scale_over: int = 0) -> ChartData:
    """The numbers for a chart.

    `back` steps to an earlier window - the day before, the seven days before
    those - for the ranges that have one. `scale_over` says how many windows the
    caller can step through, so every one of them can be drawn to the same
    height instead of each to its own.
    """
    spec = effective(spec)
    back = max(0, int(back)) if steppable(spec) else 0
    scale_over = max(0, int(scale_over)) if steppable(spec) else 0
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

    if spec.answer == "top":
        return _top(spec)
    if spec.answer == "recent":
        return _recent(spec)
    if spec.answer != "chart":
        return _answer(spec)

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
    uris: list[str] = []
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
        labels, values, keys = [], [], []
        for k, v in rows:
            if k is None:
                continue
            keys.append(k)
            labels.append(_label(spec, k))
            values.append(round(float(v), 2))
        if spec.dimension in PLAYABLE_BARS and keys:
            uris = _most_played(con, frm, where, params, dim_sql, dim_params, keys)
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
    subs: list[str] = []
    if spec.dimension in BY_ARTIST and labels:
        parts = [lb.split(KEY_SEP, 1) for lb in labels]
        labels = [pt[0] for pt in parts]
        subs = [pt[1] if len(pt) > 1 else "" for pt in parts]
    if spec.cumulative:
        values = _running(values)
        series = [Line(ln.name, _running(ln.values)) for ln in series]
        y_label = f"total {y_label}"
    total = con.execute(f"select count(*) from {frm} where {where}", params).fetchone()[0]
    window, earlier = _window(con, spec, back)
    peak = (
        _scale_peak(spec, con, frm, metric_sql, max(scale_over, back) + 1) if scale_over else None
    )

    return ChartData(
        labels=labels,
        values=values,
        x_label=x_label,
        y_label=y_label,
        title=plain_title(spec),
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
        scale_peak=peak,
        subs=subs,
        uris=uris,
    )


def _most_played(
    con: duckdb.DuckDBPyConnection,
    frm: str,
    where: str,
    params: list[Any],
    dim_sql: str,
    dim_params: list[Any],
    keys: list[Any],
) -> list[str]:
    """The most-played track uri inside each of these bars, in their order.

    Over the same plays the chart counted, so a chart of this week plays the
    song that put the artist on it this week."""
    rows = con.execute(
        f"""
        select k, arg_max(track_uri, n) from (
          select {dim_sql} as k, track_uri, count(*) as n
          from {frm}
          where {where} and track_uri like 'spotify:track:%'
          group by 1, 2
        )
        group by 1
        """,
        dim_params + params,
    ).fetchall()
    best = dict(rows)
    return [best.get(k) or "" for k in keys]


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


# ---------------------------------------------------------------- one number
#
# "How many times have I played Fears" has one number for an answer, and a bar
# chart of one bar is a chart of the wrong thing. The model only says which
# question this is - answer, metric, filters - and everything a reader sees is
# counted here and worded by template, so no number on the card came from the
# model.

PERIODS = {
    "all": ("", "All time"),
    "today": (" today", "Today"),
    "7d": (" in the last 7 days", "Last 7 days"),
    "30d": (" in the last 30 days", "Last 30 days"),
    "90d": (" in the last 90 days", "Last 90 days"),
    "6mo": (" in the last 6 months", "Last 6 months"),
    "12mo": (" in the last 12 months", "Last 12 months"),
    "this_year": (" this year", "This year"),
}

KIND_WORD = {"track": "song", "album": "album"}


def _period(spec: ChartSpec) -> tuple[str, str]:
    """The window as part of a sentence, and as a label on its own."""
    if spec.year is not None:
        return f" in {spec.year}", str(spec.year)
    if spec.from_year is not None:
        if spec.to_year is None:
            return f" since {spec.from_year}", f"Since {spec.from_year}"
        return (
            f" from {spec.from_year} to {spec.to_year}",
            f"{spec.from_year} \u2013 {spec.to_year}",
        )
    if spec.last is not None and spec.last_unit:
        n, unit = int(spec.last), spec.last_unit
        words = f"{unit}" if n == 1 else f"{n} {unit}s"
        return f" in the last {words}", f"Last {words}"
    return PERIODS.get(spec.range, ("", ""))


def _day_text(d: date) -> str:
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def _ago(d: date, today: date) -> str:
    days = (today - d).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 45:
        return f"{days} days ago"
    months = round(days / 30.44)
    if months < 18:
        return f"{months} months ago"
    return f"{round(days / 365.25)} years ago"


def _hours_text(h: float) -> tuple[str, str]:
    """Hours as a number and its unit: minutes under an hour, so nothing reads 0.3."""
    if h < 1:
        m = round(h * 60)
        return str(m), "minute" if m == 1 else "minutes"
    if h < 10:
        return f"{h:.1f}", "hours"
    return f"{round(h):,}", "hours"


def _count(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def _subject(spec: ChartSpec, con: duckdb.DuckDBPyConnection) -> tuple[str, list[str], list[str]]:
    """What the question is about, in its catalogue spelling; the names asked
    for that were never played; and what else goes by a title that was."""
    found: list[str] = []
    missing: list[str] = []
    also: list[str] = []
    has_artist = bool(spec.artists)
    for kind, names in (("track", spec.tracks), ("album", spec.albums)):
        col = TITLE_COLS[kind]
        for name in names:
            clause, params = _titled(kind, [name], has_artist)
            if has_artist:
                marks = ", ".join("?" for _ in spec.artists)
                clause += f" and lower(artist_name) in ({marks})"
                params += [a.strip().lower() for a in spec.artists]
            row = con.execute(
                f"select {col}, artist_name from plays where {clause} "
                "group by 1, 2 order by count(*) desc limit 1",
                params,
            ).fetchone()
            if not row:
                missing.append(f'{KIND_WORD[kind]} called "{name.strip()}"')
                continue
            found.append(f"{row[0]} by {row[1]}")
            if has_artist:
                continue
            rows = con.execute(
                f"select artist_name from plays where {_norm(col)} = {_norm('?')} "
                "and artist_name <> ? group by 1 order by count(*) desc limit 8",
                [name.strip(), row[1]],
            ).fetchall()
            # The catalogue spells some acts more than one way - a stray space,
            # a different dash - so the same name can come back twice.
            others: list[str] = []
            for (o,) in rows:
                if o.strip().lower() not in {x.lower() for x in [row[1], *others]}:
                    others.append(o.strip())
            if others:
                also.append(
                    f"Other {KIND_WORD[kind]}s called {row[0]}: by "
                    + "; ".join(others[:3])
                    + ". Name the artist to count theirs."
                )
    if not (spec.tracks or spec.albums):
        for name in spec.artists:
            row = con.execute(
                "select artist_name from plays where lower(artist_name) = ? "
                "group by 1 order by count(*) desc limit 1",
                [name.strip().lower()],
            ).fetchone()
            if row:
                found.append(row[0])
            else:
                missing.append(f'artist called "{name.strip()}"')
    if spec.genres:
        kinds = " and ".join(str(g) for g in spec.genres)
        found = [f"{' and '.join(found)} ({kinds} only)"] if found else [kinds]
    return " and ".join(found), missing, also


def _answer(spec: ChartSpec) -> ChartData:
    """One number, or one date, about whatever the filters leave."""
    con = _connect(spec)
    frm = _from(spec)
    where, params = _where(spec, False)
    listened = f"ms_played >= {MIN_MS}"
    row = con.execute(
        f"""
        select
          count(*) filter (where {listened}),
          coalesce(sum(ms_played), 0) / 3600000.0,
          count(distinct track_uri) filter (where {listened}),
          count(distinct artist_name) filter (where {listened}),
          avg(case when skipped then 1.0 when not skipped then 0.0 end) * 100,
          count(*) filter (where skipped is not null),
          cast(min({LOCAL_TS}) filter (where {listened}) as date),
          cast(max({LOCAL_TS}) filter (where {listened}) as date)
        from {frm}
        where {where}
        """,
        params,
    ).fetchone()
    plays, hours, n_tracks, n_artists, skip, rated, first, last = row
    plays, rated = int(plays or 0), int(rated or 0)
    today = con.execute(
        f"select cast((current_timestamp AT TIME ZONE '{LOCAL}') as date)"
    ).fetchone()[0]
    subject, missing, also = _subject(spec, con)
    in_period, period_label = _period(spec)
    it = subject or "anything"
    hrs, hrs_unit = _hours_text(float(hours or 0))

    stat: dict[str, Any] = {
        "answer": spec.answer,
        "metric": spec.metric,
        "period": period_label,
        "subject": subject,
        "found": not missing,
        "value": None,
        "big": "",
        "unit": "",
        "sentence": "",
        "facts": [],
        "uri": "",
    }

    if missing:
        stat["sentence"] = f"No {' or '.join(missing)} in your listening."
    elif spec.answer in ("first_played", "last_played"):
        when = first if spec.answer == "first_played" else last
        verb = "first" if spec.answer == "first_played" else "last"
        if when is None:
            stat["sentence"] = f"You haven't played {it}{in_period}."
        else:
            stat.update(big=_day_text(when), unit=_ago(when, today))
            stat["sentence"] = (
                f"You {verb} played {subject}{in_period} on {_day_text(when)}."
                if subject
                else f"Your listening{in_period} {'starts' if verb == 'first' else 'ends'} "
                f"on {_day_text(when)}."
            )
    # A play under half a minute is a mis-tap, not a listen: it counts towards
    # time, and nothing else.
    elif not plays and (spec.metric != "hours" or not hours):
        stat.update(value=0, big="0", unit=METRICS[spec.metric][2])
        stat["sentence"] = f"You haven't played {it}{in_period}."
    elif spec.metric == "plays":
        stat.update(value=plays, big=f"{plays:,}", unit="play" if plays == 1 else "plays")
        stat["sentence"] = (
            f"You've played {subject} {_count(plays, 'time', 'times')}{in_period}."
            if subject
            else f"You've played {_count(plays, 'song', 'songs')}{in_period}."
        )
    elif spec.metric == "hours":
        stat.update(value=round(float(hours or 0), 2), big=hrs, unit=hrs_unit)
        stat["sentence"] = (
            f"You've listened to {subject} for {hrs} {hrs_unit}{in_period}."
            if subject
            else f"You've listened for {hrs} {hrs_unit}{in_period}."
        )
    elif spec.metric == "distinct_tracks":
        n = int(n_tracks or 0)
        stat.update(value=n, big=f"{n:,}", unit="song" if n == 1 else "different songs")
        by = " by " if spec.artists and not (spec.tracks or spec.albums) else " from "
        of = f"{by}{subject}" if subject else ""
        stat["sentence"] = f"You've played {_count(n, 'song', 'different songs')}{of}{in_period}."
    elif spec.metric == "distinct_artists":
        n = int(n_artists or 0)
        stat.update(value=n, big=f"{n:,}", unit="artist" if n == 1 else "different artists")
        of = f" in {subject}" if subject else ""
        stat["sentence"] = (
            f"You've listened to {_count(n, 'artist', 'different artists')}{of}{in_period}."
        )
    elif not rated:
        # Whether a play was skipped comes only from the Spotify data download;
        # plays captured since then do not say. 0% would be a guess.
        stat["sentence"] = (
            f"No skip data for {it}{in_period} - skips come from your Spotify data download."
        )
    else:  # skip_rate
        pct = round(float(skip or 0))
        stat.update(value=pct, big=f"{pct}%", unit="skipped")
        stat["sentence"] = (
            f"You skip {subject} {pct}% of the time{in_period}."
            if subject
            else f"You skip {pct}% of what you play{in_period}."
        )

    if not missing and (plays or hours):
        stat["facts"] = _facts(
            spec, con, frm, where, params, plays, hrs, hrs_unit, first, last, rated
        )
        if spec.tracks or spec.albums or spec.artists:
            top = con.execute(
                f"select track_uri from {frm} where {where} and track_uri like 'spotify:track:%' "
                "group by 1 order by count(*) desc limit 1",
                params,
            ).fetchone()
            stat["uri"] = top[0] if top else ""

    notes = list(also)
    if spec.metric == "skip_rate" and spec.answer == "number" and 0 < rated < MIN_SAMPLE:
        notes.append(f"Only {_count(rated, 'play', 'plays')} to go on, so read it loosely.")
    unit = METRICS[spec.metric][2]
    return ChartData(
        labels=[],
        values=[],
        x_label="",
        y_label=unit,
        title=plain_title(spec),
        chart="number",
        total_rows=plays,
        note=" ".join(notes) or None,
        stat=stat,
    )


def _facts(
    spec: ChartSpec,
    con: duckdb.DuckDBPyConnection,
    frm: str,
    where: str,
    params: list[Any],
    plays: int,
    hrs: str,
    hrs_unit: str,
    first: date | None,
    last: date | None,
    rated: int,
) -> list[list[str]]:
    """The few lines under the number: whatever the headline did not already say."""
    out: list[list[str]] = []
    about = bool(spec.tracks or spec.albums or spec.artists or spec.genres)
    if not (spec.answer == "number" and spec.metric == "plays"):
        out.append(["Plays", f"{plays:,}"])
    if not (spec.answer == "number" and spec.metric == "hours"):
        out.append(["Time", f"{hrs} {hrs_unit}"])
    # When something was first and last heard says something about a song or an
    # artist. About everything, over a window, it only repeats the window's ends,
    # so a total gets its variety instead.
    if not about:
        n_artists, n_tracks = con.execute(
            f"select count(distinct artist_name), count(distinct track_uri) "
            f"from {frm} where {where} and ms_played >= {MIN_MS}",
            params,
        ).fetchone()
        if spec.metric != "distinct_artists":
            out.append(["Artists", f"{n_artists:,}"])
        if spec.metric != "distinct_tracks":
            out.append(["Songs", f"{n_tracks:,}"])
    if about and first is not None and spec.answer != "first_played":
        out.append(["First played", _day_text(first)])
    if about and last is not None and spec.answer != "last_played" and last != first:
        out.append(["Last played", _day_text(last)])
    if spec.metric == "skip_rate" and spec.answer == "number":
        out.append(["Out of", _count(rated, "play", "plays")])
    # The busiest month, when the plays span enough months for one to stand out.
    if about and first is not None and last is not None and (last - first).days > 62:
        top = con.execute(
            f"""
            select date_trunc('month', {LOCAL_TS}) as m, count(*) as n
            from {frm}
            where {where} and ms_played >= {MIN_MS}
            group by 1 order by 2 desc, 1 desc limit 1
            """,
            params,
        ).fetchone()
        if top:
            m = top[0]
            out.append(
                [
                    "Busiest month",
                    f"{MONTHS[m.month - 1]} {m.year} · {_count(top[1], 'play', 'plays')}",
                ]
            )
    return out[:4]


# ----------------------------------------------------------------- one winner
#
# "My favourite Drake album" names one album, and a chart of one bar is a
# chart of the wrong thing. The ranking is the chart's own query - the same
# filters, the same metric, the same floor under a rate - asked for two rows,
# so the winner and the runner-up behind it. The card then reads the way the
# number cards do, with the winner's own facts underneath.

THING = {
    "artist": "artist",
    "album": "album",
    "track": "song",
    "genre": "genre",
    "platform": "device",
    "hour_of_day": "time of day",
    "day_of_week": "day of the week",
}
GRAIN_THING = {"day": "day", "week": "week", "month": "month", "year": "year"}
DAY_NAMES = {
    "Mon": "Monday",
    "Tue": "Tuesday",
    "Wed": "Wednesday",
    "Thu": "Thursday",
    "Fri": "Friday",
    "Sat": "Saturday",
    "Sun": "Sunday",
}
ENTITY = frozenset({"artist", "album", "track"})


def _winner_name(spec: ChartSpec, label: str) -> str:
    """A bucket as a person would say it, not as an axis prints it."""
    if spec.dimension == "hour_of_day":
        h = int(label)
        return f"{(h % 12) or 12}{'am' if h < 12 else 'pm'}"
    if spec.dimension == "day_of_week":
        return DAY_NAMES.get(label, label)
    if spec.dimension == "date":
        d = date.fromisoformat(label[:10])
        grain = spec.grain or "month"
        if grain == "year":
            return str(d.year)
        if grain == "month":
            return f"{MONTHS[d.month - 1]} {d.year}"
        if grain == "week":
            return f"Week of {_day_text(d)}"
        return _day_text(d)
    return label


def _amount(spec: ChartSpec, v: float) -> str:
    """The winning figure, as words that follow its name."""
    if spec.metric == "hours":
        n, unit = _hours_text(float(v))
        return f"{n} {unit}"
    if spec.metric == "plays":
        return _count(int(v), "play", "plays")
    if spec.metric == "distinct_tracks":
        return _count(int(v), "song", "different songs")
    if spec.metric == "distinct_artists":
        return _count(int(v), "artist", "different artists")
    return f"skipped {round(v)}% of the time"


def _top(spec: ChartSpec) -> ChartData:
    """The single winner, the one behind it, and the winner's own facts."""
    ranked = spec.dimension not in ORDERED
    asked = spec.model_copy(
        update={
            "answer": "chart",
            "series": "none",
            "per_period": False,
            "cumulative": False,
            # An ordered axis comes back in its own order, every bucket, and the
            # winner is picked below; a ranking is asked for its top two.
            "limit": 2 if ranked else spec.limit,
        }
    )
    d = run(asked)
    order = sorted(range(len(d.values)), key=lambda i: d.values[i], reverse=spec.sort != "asc")
    order = [
        i for i in order if d.values[i] is not None and (d.values[i] > 0 or spec.sort == "asc")
    ]

    _, period_label = _period(spec)
    in_period = _period(spec)[0]
    thing = (
        GRAIN_THING.get(spec.grain or "month", "period")
        if spec.dimension == "date"
        else THING.get(spec.dimension, spec.dimension)
    )
    lead = "least played" if spec.sort == "asc" else "top"
    if spec.metric == "skip_rate":
        lead = "least skipped" if spec.sort == "asc" else "most skipped"

    stat: dict[str, Any] = {
        "answer": "top",
        "metric": spec.metric,
        "period": period_label,
        "subject": "",
        "found": bool(order),
        "value": None,
        "big": "",
        "unit": "",
        "sentence": "",
        "facts": [],
        "uri": "",
    }
    note = d.note
    if not order:
        _, missing, _ = _subject(spec, _connect(spec))
        stat["found"] = False
        stat["sentence"] = (
            f"No {' or '.join(missing)} in your listening."
            if missing
            else f"Nothing to rank{in_period}."
        )
    else:
        i = order[0]
        name = _winner_name(spec, d.labels[i])
        by = d.subs[i] if d.subs and i < len(d.subs) else ""
        # Narrowed to one artist, it reads "your top Drake album is Views" -
        # in the catalogue's spelling of Drake, not the question's.
        whose = f"{by} " if by and len(spec.artists) == 1 else ""
        named = f"{name} by {by}" if by and not whose else name
        stat.update(
            value=d.values[i],
            big=name,
            unit=f"by {by}" if by else "",
            subject=named,
            uri=d.uris[i] if d.uris and i < len(d.uris) else "",
        )
        # Narrowed to one record, it reads "your top song off Blonde".
        off = ""
        if spec.dimension == "track" and len(spec.albums) == 1:
            off = f" off {_album_title(spec, spec.albums[0])}"
        stat["sentence"] = (
            f"Your {lead} {whose}{thing}{off}{in_period} is {named} - {_amount(spec, d.values[i])}."
        )
        facts: list[list[str]] = []
        if spec.dimension in ENTITY:
            facts = _winner_facts(spec, name, by)
        if len(order) > 1:
            j = order[1]
            nxt = _winner_name(spec, d.labels[j])
            facts.append(["Next", f"{nxt} · {_amount(spec, d.values[j])}"])
        stat["facts"] = facts

    return ChartData(
        labels=[],
        values=[],
        x_label="",
        y_label=d.y_label,
        title=plain_title(spec),
        chart="number",
        total_rows=d.total_rows,
        note=note,
        stat=stat,
    )


def _album_title(spec: ChartSpec, typed: str) -> str:
    """A record's name as the catalogue spells it, rather than as it was typed."""
    row = (
        _connect(spec)
        .execute(
            f"select album_name from plays where {_norm('album_name')} = {_norm('?')} "
            "group by 1 order by count(*) desc limit 1",
            [typed.strip()],
        )
        .fetchone()
    )
    return row[0] if row else typed.strip()


def _winner_facts(spec: ChartSpec, name: str, by: str) -> list[list[str]]:
    """The winner's own plays, time and dates, worked out the way a number
    question about it would be - over the same window and filters."""
    update: dict[str, Any] = {"answer": "number", "metric": "plays"}
    if spec.dimension == "artist":
        update["artists"] = [name]
    elif spec.dimension == "album":
        update.update(albums=[name], artists=[by] if by else spec.artists)
    else:
        update.update(tracks=[name], artists=[by] if by else spec.artists)
    st = _answer(spec.model_copy(update=update)).stat or {}
    facts = [["Plays", f"{st['value']:,}"]] if st.get("value") is not None else []
    # Time and the dates, without the busiest month: the card already has its
    # runner-up, and four lines is what fits.
    keep = {"Time", "First played", "Last played"}
    return facts + [f for f in st.get("facts", []) if f[0] in keep][:2]


# ------------------------------------------------------------ reading names
#
# A name has to be found in this history before anything can be counted for
# it, and people do not type names the way the catalogue spells them: "kendrick"
# for Kendrick Lamar, "faye webstr", a title without its comma. The model cannot
# fix that - it has never seen this library - so it is fixed here, against what
# was actually played, in order of how sure the match is:
#
#   exact     the same once case, punctuation and version tails are gone
#   partial   every word typed is a whole word of the name, in order -
#             "kendrick" in Kendrick Lamar, "thinking about" in Thinking About You
#   close     a spelling a few letters off, for names of four letters or more
#
# Within a tier the one played most wins. Anything matched other than exactly
# is reported on the answer, so a wrong guess is visible rather than silently
# counted.

MIN_PARTIAL = 4
MIN_CLOSE = 4
CLOSE_ENOUGH = 0.9

NAME_COLS = {"artist": "artist_name", "track": "track_name", "album": "album_name"}


def _closest(
    con: duckdb.DuckDBPyConnection, kind: str, typed: str, artists: list[str]
) -> tuple[str, bool] | None:
    """The played name `typed` most likely means, and whether it was exact."""
    col = NAME_COLS[kind]
    scope, params = f"{col} is not null", []
    if artists and kind != "artist":
        marks = ", ".join("?" for _ in artists)
        scope += f" and lower(artist_name) in ({marks})"
        params += [a.lower() for a in artists]
    row = con.execute(
        f"""
        with names as (
          select {col} as name, {_norm(col)} as k, count(*) as n
          from plays where {scope} group by 1
        ), q as (select {_norm("?")} as q)
        select name, k = q as exact,
          case
            when k = q then 3
            when length(q) >= {MIN_PARTIAL}
                 and (' ' || k || ' ') like ('% ' || q || ' %') then 2
            else 1
          end as tier,
          jaro_winkler_similarity(k, q) as near,
          n
        from names, q
        where q <> '' and (
          k = q
          or (length(q) >= {MIN_PARTIAL} and (' ' || k || ' ') like ('% ' || q || ' %'))
          or (length(q) >= {MIN_CLOSE} and jaro_winkler_similarity(k, q) >= {CLOSE_ENOUGH})
        )
        order by tier desc, case when tier = 1 then near else 0 end desc, n desc, name
        limit 1
        """,
        [*params, typed.strip()],
    ).fetchone()
    return (row[0], bool(row[1])) if row else None


def resolve(spec: ChartSpec) -> tuple[ChartSpec, list[str]]:
    """The spec with each name as this history spells it, and a line for every
    name that was read as something other than itself.

    Artists first, because a song named with its artist is looked for among
    that artist's songs. A name with no match at all is left as typed, and the
    answer then says nothing by that name was played.
    """
    if not (spec.artists or spec.tracks or spec.albums):
        return spec, []
    con = _connect(spec)
    fixes: list[str] = []
    update: dict[str, list[str]] = {}

    def read(kind: str, names: list[str], artists: list[str]) -> list[str]:
        out = []
        for typed in names:
            hit = _closest(con, kind, typed, artists)
            if not hit:
                out.append(typed)
                continue
            name, exact = hit
            if not exact:
                fixes.append(f'Read "{typed.strip()}" as {name}.')
            out.append(name)
        return out

    if spec.artists:
        update["artists"] = read("artist", spec.artists, [])
    artists = update.get("artists", [])
    if spec.tracks:
        update["tracks"] = read("track", spec.tracks, artists)
    if spec.albums:
        update["albums"] = read("album", spec.albums, artists)
    return spec.model_copy(update=update), fixes


# ------------------------------------------------------------- recent plays
#
# "The last 10 songs I played" is not a total of anything: it is the plays
# themselves, newest first. Same filters and window as any chart, the same
# half-minute line between a listen and a mis-tap, and a song played several
# times running is one row with its count, so a loop does not fill the list.

MAX_RECENT = 40


def _recent(spec: ChartSpec) -> ChartData:
    """The latest plays, newest first, repeats in a row folded into one."""
    con = _connect(spec)
    frm = _from(spec)
    where, params = _where(spec, True)
    want = max(1, min(int(spec.limit), MAX_RECENT))
    # A run is one song played several times in a row in the listening itself,
    # so runs are found over every play first and the filters applied after:
    # filtered first, a list of one song's plays would be one endless run. The
    # count is each run's in full, however long the loop.
    runs = f"""
        with seq as (
          select *,
                 coalesce(track_uri, track_name || chr(31) || coalesce(artist_name, '')) as k
          from plays
          where ms_played >= {MIN_MS} and track_name is not null
        ), marked as (
          select *, case when k = lag(k) over (order by played_at desc) then 0 else 1 end as fresh
          from seq
        )
        select *, sum(fresh) over (order by played_at desc rows unbounded preceding) as run
        from marked
    """
    rows = con.execute(
        f"""
        with runs as ({runs})
        select arg_max(track_name, played_at), arg_max(artist_name, played_at),
               arg_max(coalesce(track_uri, ''), played_at),
               strftime(max(played_at), '%Y-%m-%dT%H:%M:%SZ'), count(*)
        from {frm.replace("plays p", "runs p", 1)}
        where {where}
        group by run
        order by run
        limit {want}
        """,
        params,
    ).fetchall()
    out = [
        {
            "name": name,
            "artist": artist or "",
            "uri": uri if uri.startswith("spotify:track:") else "",
            "at": at,
            "times": int(times),
        }
        for name, artist, uri, at, times in rows
    ]

    in_period, period_label = _period(spec)
    who, missing = "", []
    if spec.artists or spec.albums or spec.tracks or spec.genres:
        who, missing, _ = _subject(spec, con)
    things = "songs"
    if len(spec.albums) == 1:
        things = f"songs off {_album_title(spec, spec.albums[0])}"
    elif spec.artists and who and not spec.tracks:
        things = f"{who} songs"
    # "Last 10 songs" when there are ten. A list of one song's plays, or one
    # cut short by the window, is recent rather than a count it does not reach.
    title = f"Last {len(out)} {things}" if len(out) == want > 1 else f"Recent {things}"
    if spec.tracks and who:
        title = f"Recent plays of {who}"
    if out:
        sentence = ""
    elif missing:
        sentence = f"No {' or '.join(missing)} in your listening."
    elif who:
        sentence = f"You haven't played {who}{in_period}."
    else:
        sentence = f"Nothing played{in_period}."
    stat: dict[str, Any] = {
        "answer": "recent",
        "metric": "plays",
        "period": period_label if spec.range != "all" or spec.year else "Recent plays",
        "subject": who,
        "found": bool(out),
        "rows": out,
        "sentence": sentence,
        "value": len(out),
        "big": "",
        "unit": "",
        "facts": [],
        "uri": "",
    }
    return ChartData(
        labels=[],
        values=[],
        x_label="",
        y_label="plays",
        title=title,
        chart="recent",
        total_rows=len(rows),
        stat=stat,
    )


# ------------------------------------------------------------- the window
#
# Every answer says what it covers in dates as well as words. The bug this came
# from was a window the menu could not say quietly becoming all time; a window
# printed on the answer cannot be wrong without it showing.


def _span(spec: ChartSpec) -> str:
    """'Last 2 years (Nov 2024 - Nov 2026)', 'All time (Aug 2018 - Nov 2026)'."""
    con = _connect(spec)
    _, label = _period(spec)
    newest, oldest = con.execute(
        f"select cast(max({LOCAL_TS}) as date), cast(min({LOCAL_TS}) as date) from plays"
    ).fetchone()
    if newest is None:
        return label
    start, end = None, newest
    if spec.year is not None:
        return label
    if spec.from_year is not None:
        # "2020 - 2022" already says it; "Since 2023" gets where it runs to.
        if spec.to_year is not None:
            return label
        start = max(date(int(spec.from_year), 1, 1), oldest)
    elif spec.last is not None and spec.last_unit in UNITS:
        start = con.execute(
            f"select cast(max({LOCAL_TS}) - interval {int(spec.last)} "
            f"{UNITS[spec.last_unit]} as date) from plays"
        ).fetchone()[0]
    elif spec.range in RANGES:
        start = con.execute(
            f"select cast(max({LOCAL_TS}) - {RANGES[spec.range]} as date) from plays"
        ).fetchone()[0]
    elif spec.range == "this_year":
        start = date(newest.year, 1, 1)
    elif spec.range == "all":
        start = oldest
    else:
        return label
    start = max(start, oldest)
    return f"{label} ({_dates(start, end)})"


def _dates(start: date, end: date) -> str:
    """Two dates as briefly as reads clearly: months over long spans, days over short."""
    if (end - start).days >= 60:
        a, b = f"{MONTHS[start.month - 1]} {start.year}", f"{MONTHS[end.month - 1]} {end.year}"
    elif start.year != end.year:
        a, b = _day_text(start), _day_text(end)
    else:
        a, b = _pretty(start), _pretty(end)
    return a if a == b else f"{a} \u2013 {b}"


# ------------------------------------------------------------- plain titles
#
# A title says what the answer is, built from what was asked: "Top Drake
# songs", "Listening by month", "Plays of Fears by MTNS". Nothing clever and
# nothing the model wrote, so it is never odd and never wrong, and the window
# lives on the line under it rather than in it. A person can rename any chart.

PLURAL = {
    "artist": "artists",
    "album": "albums",
    "track": "songs",
    "genre": "genres",
    "platform": "devices",
}
METRIC_WORD = {
    "hours": "Hours",
    "plays": "Plays",
    "distinct_tracks": "Different songs",
    "distinct_artists": "Different artists",
    "skip_rate": "Skip rate",
}


def _named(spec: ChartSpec) -> str:
    """The thing a question is narrowed to, as a short phrase - or nothing."""
    if spec.tracks:
        who = f" by {spec.artists[0]}" if len(spec.artists) == 1 else ""
        return " and ".join(spec.tracks) + who
    if spec.albums:
        return " and ".join(spec.albums)
    if spec.artists:
        return " and ".join(spec.artists)
    if spec.genres:
        return " and ".join(str(g) for g in spec.genres)
    return ""


def plain_title(spec: ChartSpec) -> str:
    """What the answer is, in a few plain words."""
    spec = effective(spec)
    named = _named(spec)
    if spec.answer in ("first_played", "last_played"):
        which = "First" if spec.answer == "first_played" else "Last"
        return f"{which} play of {named}" if named else f"{which} play"
    if spec.answer == "number":
        return f"{METRIC_WORD[spec.metric]} \u00b7 {named}" if named else METRIC_WORD[spec.metric]
    if spec.answer == "recent":
        return f"Recent plays \u00b7 {named}" if named else "Recent plays"

    ranked = spec.dimension in PLURAL
    if spec.answer == "top" or (ranked and spec.limit == 1 and not spec.per_period):
        thing = (
            THING.get(spec.dimension, "") if spec.dimension != "date" else (spec.grain or "month")
        )
        return _ranked_title(spec, thing, single=True)
    if spec.per_period:
        many = (
            PLURAL.get(spec.series, "") if spec.limit > 1 else THING.get(spec.series, spec.series)
        )
        lead = f"Top {spec.limit} {many}" if spec.limit > 1 else f"Top {many}"
        return f"{lead} each {spec.grain or 'month'}"
    if spec.series != "none":
        if spec.series == "artist" and spec.artists:
            names = " vs ".join(spec.artists)
        elif spec.series == "genre" and spec.genres:
            names = " vs ".join(str(g) for g in spec.genres)
        else:
            names = f"Top {PLURAL.get(spec.series, spec.series)}"
        return f"{names} {_along(spec)}"
    if ranked:
        return _ranked_title(spec, PLURAL[spec.dimension], single=False)
    lead = named or ("Today" if spec.range == "today" else "Listening")
    if spec.cumulative:
        return f"{lead}, running total"
    return f"{lead} {_along(spec)}"


def _along(spec: ChartSpec) -> str:
    """'by month', 'by hour', 'by day of week' - what an ordered axis runs along."""
    if spec.dimension == "date":
        return f"by {spec.grain or 'month'}"
    return {"hour_of_day": "by hour", "day_of_week": "by day of week"}.get(spec.dimension, "")


def _ranked_title(spec: ChartSpec, thing: str, single: bool) -> str:
    """'Top Drake songs', 'Most skipped songs', 'Top song off Blonde'."""
    if spec.metric == "skip_rate":
        lead = "Least skipped" if spec.sort == "asc" else "Most skipped"
    elif spec.metric in ("distinct_tracks", "distinct_artists"):
        lead = "Most varied"
    else:
        lead = "Least played" if spec.sort == "asc" else "Top"
    whose = ""
    if spec.dimension in ("track", "album") and len(spec.artists) == 1:
        whose = f"{spec.artists[0]} "
    tail = ""
    if spec.dimension == "track" and len(spec.albums) == 1:
        tail = f" off {spec.albums[0]}"
    elif spec.dimension == "artist" and spec.genres:
        whose = " and ".join(str(g) for g in spec.genres) + " "
    if not single and spec.limit and spec.limit not in (1, 10):
        lead = f"{lead} {spec.limit}"
    return f"{lead} {whose}{thing}{tail}"
