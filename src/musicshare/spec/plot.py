"""How a chart is drawn, decided in Python where it can be tested.

The page used to work this out for itself in hand-written canvas code - where the
gridlines go, whether labels sit under bars or beside them, where a running
total's points land - and every chart shape nobody had looked at yet arrived with
its own drawing bug: a line running past the current time, a legend that cut off
two of six names, year labels sitting between their points. None of it was wrong
data. All of it was layout, decided in the one place with no tests.

So the decisions live here. `plan` turns a chart's numbers into an exact
description of the drawing - which kind of chart, where every point and tick sits,
what each label says - and the page hands that to Chart.js without deciding
anything itself. Colours and personal preferences are a layer on top of this, not
a reason to change it.

The input is the chart payload the API already sends (`labels`, `values`,
`series`, `table`, axis labels, `cumulative`, `partial_last`), so a chart saved
before plans existed can be planned from what it kept.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import Any

# Bars that name things rather than positions on a scale. Their labels are words,
# which do not fit under vertical bars at phone width, so the bars lie on their side.
RANKED = frozenset({"artist", "album", "track", "genre", "platform"})
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
ISO_DAY = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

# Labels along the bottom of a line. Past this they collide at phone width.
MAX_X_TICKS = 8
# A bar label longer than this is cut, with the whole name kept for the tooltip.
MAX_BAR_LABEL = 22

HEIGHT = 150
# Room above the highest value before the top gridline. Without it a live total
# at 3.93 sat flush against a top of 4, now-marker and all, until it crossed it
# and the scale jumped. A tenth leaves space to grow into and still rounds to the
# same top for most charts.
HEADROOM = 1.1
LEGEND = 22
ROW = 20


# ------------------------------------------------------------------- scale


def nice_step(top: float) -> float:
    """A round gap between gridlines - 1, 2, 2.5 or 5 of a power of ten - that gives
    about five of them.

    Three fixed lines at zero, half and top read 0, 2.5, 5 on a chart peaking at
    3.2, which left most values between two labels and a third of the plot empty.
    """
    if not top > 0:
        return 1.0
    raw = top / 5
    mag = 10 ** math.floor(math.log10(raw))
    n = raw / mag
    for s in (1, 2, 2.5, 5):
        if n <= s + 1e-9:
            return s * mag
    return 10 * mag


def nice_top(peak: float) -> float:
    """The first gridline at or above the data, so the tallest point sits near the top."""
    if not peak > 0:
        return 1.0
    step = nice_step(peak)
    return round(math.ceil(peak / step - 1e-9) * step, 10)


def tick_label(v: float, step: float) -> str:
    """A gridline's label, with exactly the precision its step needs.

    A value formatter rounds to one decimal and to whole thousands, which is right
    beside a bar and wrong on a scale: steps of 0.25 would read 0.3 and 0.5, and
    2,500 would read 3k. Trailing zeros go - 0.2 ... 1, not 0.0 ... 1.0.
    """
    if not v:
        return "0"
    thousands = step >= 1000
    exponent = Decimal(str(round(step / 1000 if thousands else step, 6))).normalize().as_tuple()
    decimals = max(0, -int(exponent.exponent))
    out = f"{(v / 1000 if thousands else v):.{decimals}f}"
    if decimals:
        out = out.rstrip("0").rstrip(".")
    return out + ("k" if thousands else "")


def y_scale(peak: float) -> dict[str, Any]:
    top = nice_top(peak * HEADROOM)
    step = nice_step(top)
    count = max(1, round(top / step))
    ticks = []
    for i in range(count + 1):
        at = round(step * i, 10)
        ticks.append({"at": at, "label": tick_label(at, step)})
    return {"max": top, "ticks": ticks}


def value_label(v: float | None) -> str:
    """A number printed at the end of a bar."""
    if v is None:
        return ""
    if v >= 1000:
        return f"{round(v / 1000)}k"
    if v >= 100:
        return str(round(v))
    return f"{round(v, 1):g}"


# ------------------------------------------------------------------ labels


def x_label(raw: str, axis: str, many_years: bool) -> str:
    """A bucket's name, at the grain it was bucketed at.

    Date buckets arrive as ISO days. Months spanning more than one year carry the
    year, or twelve months from last September read as two Septembers.
    """
    m = ISO_DAY.fullmatch(raw)
    if not m:
        return raw
    year, month, day = m.groups()
    if axis == "year":
        return year
    if axis == "month":
        return MONTHS[int(month) - 1] + (f" '{year[2:]}" if many_years else "")
    return f"{month}/{day}"


def _clock(hours: float) -> str:
    """Hours since midnight as a time of day, for a point's tooltip."""
    minutes = round(hours * 60)
    return f"{minutes // 60 % 24:02d}:{minutes % 60:02d}"


def _span(n: int, axis: str, partial: float | None) -> int:
    """How many hours a chart of today reaches along the bottom.

    A day still in progress stopped at the current hour, so 4pm did not appear
    until it was 4pm and the line ran into the right edge. A tenth more, at
    least an hour, gives it somewhere to go - the same idea as the headroom above
    the top gridline. Every other chart spans exactly its buckets.
    """
    if partial is None or axis != "hour of day":
        return n
    return min(24, n + max(1, math.ceil(n * (HEADROOM - 1))))


def _hour_ticks(n: int, span: int, labels: list[str], at: list[float]) -> list[dict[str, Any]]:
    """Ticks along a span that may run past the data: hours ahead are labelled too."""
    every = max(1, math.ceil(span / MAX_X_TICKS))
    return [
        {"at": at[i] if i < n else float(i), "label": labels[i] if i < n else f"{i:02d}"}
        for i in range(0, span, every)
    ]


def _short(text: str) -> str:
    return text if len(text) <= MAX_BAR_LABEL else text[: MAX_BAR_LABEL - 1] + "…"


# -------------------------------------------------------------------- plan


def plan(p: dict[str, Any]) -> dict[str, Any]:
    """The drawing, exactly, for one chart's numbers."""
    if p.get("table"):
        return {"kind": "table"}

    raw_labels = [str(x) for x in (p.get("labels") or [])]
    values = list(p.get("values") or [])
    series = [s for s in (p.get("series") or []) if s.get("values")]
    if not raw_labels or (not values and not series):
        return {"kind": "empty"}

    axis = str(p.get("x_label") or "")
    unit = str(p.get("y_label") or "")
    cumulative = bool(p.get("cumulative"))
    partial = p.get("partial_last")

    years = {m.group(1) for m in (ISO_DAY.fullmatch(x) for x in raw_labels) if m}
    labels = [x_label(x, axis, len(years) > 1) for x in raw_labels]
    lines = [{"name": str(s["name"]), "values": list(s["values"])} for s in series] or [
        {"name": unit, "values": values}
    ]
    peak = max((v for ln in lines for v in ln["values"] if v is not None), default=0)
    scale = {**y_scale(peak), "unit": unit}

    if not (p.get("chart") == "line" or series or cumulative):
        # Words lie on their side; short positions stand up. Hours and weekdays are
        # two or three characters, names are not.
        sideways = axis in RANKED or max(len(x) for x in labels) > 4
        return {
            "kind": "hbar" if sideways else "bar",
            "labels": labels,
            "tick_labels": [_short(x) for x in labels] if sideways else labels,
            "values": values,
            "value_labels": [value_label(v) for v in values] if sideways else [],
            "y": scale,
            "height": max(120, len(values) * ROW + 8) if sideways else HEIGHT,
            "legend": False,
            "fill": False,
            "now": False,
        }

    n = len(labels)
    span = _span(n, axis, partial)
    timeline = p.get("timeline") or []
    if cumulative and timeline and not series:
        # Today, play by play: the points are the listening itself, on the same
        # hour axis as the labels, ending at the current minute.
        pts = [[float(x), float(y)] for x, y in timeline]
        return {
            "kind": "line",
            "x": {
                "min": 0,
                "max": span,
                "ticks": _hour_ticks(n, span, labels, [float(i) for i in range(n)]),
            },
            "series": [{"name": unit, "points": pts, "labels": [_clock(x) for x, _ in pts]}],
            "y": {**y_scale(max(y for _, y in pts)), "unit": unit},
            "legend": False,
            "fill": True,
            "now": True,
            "height": HEIGHT,
        }
    if cumulative:
        # An amount at a moment, not over a bucket: zero at the left edge, each
        # bucket's total where that bucket ends, and today's last point at the
        # current minute rather than at the end of an hour that has not finished.
        xs = [float(i + 1) for i in range(n)]
        if partial is not None:
            xs[-1] = n - 1 + float(partial)
        ticks_at = [float(i) for i in range(n)]
    else:
        xs = [i + 0.5 for i in range(n)]
        ticks_at = xs

    def points(vals: list[float | None]) -> tuple[list[list[Any]], list[str]]:
        pts = [[x, v] for x, v in zip(xs, vals, strict=True)]
        names = list(labels)
        if cumulative:
            return [[0.0, 0.0], *pts], ["start", *names]
        return pts, names

    drawn = []
    for ln in lines:
        pts, names = points(ln["values"])
        drawn.append({"name": ln["name"], "points": pts, "labels": names})

    return {
        "kind": "line",
        "x": {"min": 0, "max": span, "ticks": _hour_ticks(n, span, labels, ticks_at)},
        "series": drawn,
        "y": scale,
        "legend": len(drawn) > 1,
        "fill": len(drawn) == 1,
        "now": cumulative and partial is not None,
        "height": HEIGHT + (LEGEND if len(drawn) > 1 else 0),
    }
