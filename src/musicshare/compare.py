"""Comparing a chart with someone else: the request, the answer, and the result.

One person asks a question and sends it to another. Nothing of the other
person's is run until they accept - a pinned chart is something they chose to
show, but this is a new question over their history, so it waits for a yes.
Once accepted it is live for both of them, and either can remove it, which
ends it for both: consent that can be given can be taken back.

The store is one JSON file on the app's own disk, beside the user records -
the same volume, the same single machine - so there is nothing to migrate and
a local server writes only to its own copy. Small by design: a request is a
spec and two ids, never anybody's numbers. The numbers are worked out when
someone looks, from each person's own history.
"""

from __future__ import annotations

import json
import secrets
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from musicshare import users
from musicshare.config import ROOT
from musicshare.spec.chart import ChartSpec

STORE = ROOT / "data" / "social" / "compares.json"

# Outstanding requests one person can have sent at once. Enough for anyone
# actually asking; a ceiling on someone filling another's inbox.
MAX_PENDING = 20
MAX_NAME = 60

_lock = threading.Lock()


class CompareError(ValueError):
    """A request that cannot be made or changed - said to the person as is."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read(path: Path | None) -> list[dict[str, Any]]:
    path = path or STORE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return []
    return [r for r in data.get("compares", []) if isinstance(r, dict)]


def _write(path: Path | None, items: list[dict[str, Any]]) -> None:
    path = path or STORE
    # Written beside and swapped in, so a crash mid-write leaves the old file.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"compares": items}, indent=1), encoding="utf-8")
    tmp.replace(path)


def involving(uid: str, path: Path | None = None) -> list[dict[str, Any]]:
    """Every comparison this person sent or was sent, newest first - except
    the ones turned down, which only the sender is told about."""
    with _lock:
        items = _read(path)
    mine = [r for r in items if r["from"] == uid or (r["to"] == uid and r["status"] != "declined")]
    return sorted(mine, key=lambda r: r["at"], reverse=True)


def get(cid: str, path: Path | None = None) -> dict[str, Any] | None:
    with _lock:
        return next((r for r in _read(path) if r["id"] == cid), None)


def ask(
    sender: str,
    to: str,
    spec: ChartSpec,
    q: str = "",
    name: str | None = None,
    title: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    """Send a comparison request. The same question to the same person while
    one is still waiting is the request already sent, not a second one."""
    if to == sender:
        raise CompareError("that's you")
    if not users.exists(to):
        raise CompareError("no such person")
    spec_json = spec.model_dump(mode="json")
    with _lock:
        items = _read(path)
        for r in items:
            if (
                r["from"] == sender
                and r["to"] == to
                and r["spec"] == spec_json
                and r["status"] != "declined"
            ):
                return r
        waiting = sum(1 for r in items if r["from"] == sender and r["status"] == "pending")
        if waiting >= MAX_PENDING:
            raise CompareError(f"you have {waiting} requests waiting already")
        rec = {
            "id": secrets.token_urlsafe(9),
            "from": sender,
            "to": to,
            "spec": spec_json,
            "q": (q or "")[:400],
            "name": ((name or "").strip()[:MAX_NAME]) or None,
            "title": (title or "")[:120],
            "at": _now(),
            "status": "pending",
            "decided_at": None,
        }
        items.append(rec)
        _write(path, items)
    return rec


def decide(cid: str, uid: str, accept: bool, path: Path | None = None) -> dict[str, Any]:
    """Accept or decline - only the person it was sent to, and only once."""
    with _lock:
        items = _read(path)
        rec = next((r for r in items if r["id"] == cid), None)
        if rec is None or rec["to"] != uid:
            raise CompareError("no such request")
        if rec["status"] != "pending":
            raise CompareError(f"already {rec['status']}")
        rec["status"] = "accepted" if accept else "declined"
        rec["decided_at"] = _now()
        _write(path, items)
    return rec


def remove(cid: str, uid: str, path: Path | None = None) -> bool:
    """Either person ends it, for both. Withdrawing a request is the same act."""
    with _lock:
        items = _read(path)
        keep = [r for r in items if not (r["id"] == cid and uid in (r["from"], r["to"]))]
        if len(keep) == len(items):
            return False
        _write(path, keep)
    return True


def rename(cid: str, uid: str, name: str | None, path: Path | None = None) -> dict[str, Any]:
    """A comparison is named like any chart; either person can."""
    with _lock:
        items = _read(path)
        rec = next((r for r in items if r["id"] == cid and uid in (r["from"], r["to"])), None)
        if rec is None:
            raise CompareError("no such comparison")
        rec["name"] = ((name or "").strip()[:MAX_NAME]) or None
        _write(path, items)
    return rec


def person(uid: str) -> dict[str, Any]:
    pub = users.public(uid) if users.exists(uid) else {"id": uid}
    return {"id": uid, "name": pub.get("name") or uid, "image": pub.get("image")}


def view(rec: dict[str, Any], me: str) -> dict[str, Any]:
    """A comparison as one of the two people sees it."""
    return {
        "id": rec["id"],
        "from": person(rec["from"]),
        "to": person(rec["to"]),
        "mine": rec["from"] == me,
        "status": rec["status"],
        "spec": rec["spec"],
        "q": rec["q"],
        "name": rec.get("name"),
        "title": rec.get("title") or "",
        "at": rec["at"],
        "decided_at": rec.get("decided_at"),
    }


# ------------------------------------------------------------ side by side
#
# How two answers to one question are set beside each other depends on what
# the answer is. A number or a winner is two cards; a ranking is two lists with
# what they share marked; anything along an ordered axis - months, hours,
# weekdays - is one chart with a line or a bar each. A chart that already
# compares several things, or ranks inside every period, would be a comparison
# of comparisons drawn on one axis, so those are shown one above the other.

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def kind_of(spec: ChartSpec, payloads: list[dict[str, Any] | None]) -> str:
    from musicshare.spec.chart import ORDERED
    from musicshare.spec.chartrun import effective

    if spec.answer in ("number", "first_played", "last_played", "top"):
        return "numbers"
    if spec.answer == "recent":
        return "recent"
    eff = effective(spec)
    if eff.per_period or eff.series != "none" or any(p and p.get("table") for p in payloads):
        return "stacked"
    if eff.dimension in ORDERED:
        return "lines"
    return "ranked"


def _order(spec: ChartSpec, label: str) -> tuple[int, str]:
    from musicshare.spec.chartrun import effective

    dim = effective(spec).dimension
    if dim == "hour_of_day":
        return (int(label) if label.isdigit() else 99, label)
    if dim == "day_of_week":
        return (DAY_ORDER.index(label) if label in DAY_ORDER else 99, label)
    return (0, label)  # ISO dates sort as text


def merge_lines(
    spec: ChartSpec, payloads: list[dict[str, Any]], names: list[str]
) -> dict[str, Any]:
    """Two answers along one ordered axis, as one chart with a series each.

    The two histories need not cover the same buckets - one person's first
    month may be the other's fortieth - so the axis is every bucket either has.
    A bucket one person has no plays in is nothing for them, which for a count
    is zero, for a running total is where they had got to, and for a rate is a
    gap rather than an invented 0%.
    """
    from musicshare.spec import plot

    labels = sorted(
        {lb for p in payloads for lb in (p.get("labels") or [])}, key=lambda x: _order(spec, x)
    )
    rate = spec.metric == "skip_rate"
    series = []
    for p, name in zip(payloads, names, strict=True):
        got = dict(zip(p.get("labels") or [], p.get("values") or [], strict=False))
        values: list[float | None] = []
        last: float | None = None
        for lb in labels:
            if lb in got:
                last = got[lb]
                values.append(got[lb])
            elif rate:
                values.append(None)
            elif p.get("cumulative"):
                values.append(last if last is not None else 0)
            else:
                values.append(0)
        series.append({"name": name, "values": values})
    first = payloads[0]
    out: dict[str, Any] = {
        "labels": labels,
        "values": [],
        "series": series,
        "title": first.get("title"),
        "x_label": first.get("x_label"),
        "y_label": first.get("y_label"),
        "chart": "line" if first.get("chart") == "line" else "bar",
        "cumulative": bool(first.get("cumulative")),
        "note": None,
        "span": first.get("span", ""),
    }
    out["plan"] = plot.plan(out)
    return out
