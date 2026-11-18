"""Comparing a chart with others: the request, the answers, and the result.

One person asks a question and sends it to one or two others. Nothing of
anyone else's is run until they accept - a pinned chart is something they chose
to show, but this is a new question over their history, so it waits for a yes.
Each person answers for themselves: the comparison is live as soon as one of
them accepts, holds whoever has accepted, and says who it is still waiting on.
Anyone asked can leave it, which takes only them out; the one who asked can
remove it, which ends it for everyone. Consent that can be given can be taken
back.

The store is one JSON file on the app's own disk, beside the user records -
the same volume, the same single machine - so there is nothing to migrate and
a local server writes only to its own copy. Small by design: a request is a
spec and some ids, never anybody's numbers. The numbers are worked out when
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
# People asked in one comparison, not counting the one asking: three in all.
MAX_OTHERS = 2

_lock = threading.Lock()


class CompareError(ValueError):
    """A request that cannot be made or changed - said to the person as is."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _shape(rec: dict[str, Any]) -> dict[str, Any]:
    """A record in its current form. The first comparisons were one-to-one, with
    `to` a single id and one `status`; those read as a group of one."""
    if isinstance(rec.get("to"), str):
        uid = rec["to"]
        rec["to"] = [uid]
        rec["answers"] = {
            uid: {"status": rec.pop("status", "pending"), "at": rec.pop("decided_at", None)}
        }
    rec.setdefault("answers", {})
    for uid in rec["to"]:
        rec["answers"].setdefault(uid, {"status": "pending", "at": None})
    return rec


def _read(path: Path | None) -> list[dict[str, Any]]:
    path = path or STORE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return []
    return [_shape(r) for r in data.get("compares", []) if isinstance(r, dict)]


def _write(path: Path | None, items: list[dict[str, Any]]) -> None:
    path = path or STORE
    # Written beside and swapped in, so a crash mid-write leaves the old file.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"compares": items}, indent=1), encoding="utf-8")
    tmp.replace(path)


def answer_of(rec: dict[str, Any], uid: str) -> str | None:
    a = rec["answers"].get(uid)
    return a["status"] if a else None


def accepted(rec: dict[str, Any]) -> list[str]:
    return [u for u in rec["to"] if answer_of(rec, u) == "accepted"]


def waiting_on(rec: dict[str, Any]) -> list[str]:
    return [u for u in rec["to"] if answer_of(rec, u) == "pending"]


def status(rec: dict[str, Any]) -> str:
    """Live once anyone has accepted; waiting while nobody has and someone
    still might; declined when everyone asked has said no."""
    if accepted(rec):
        return "accepted"
    if waiting_on(rec):
        return "pending"
    return "declined"


def participants(rec: dict[str, Any]) -> list[str]:
    """Whose answers are shown: the one who asked, and everyone who said yes."""
    return [rec["from"], *accepted(rec)]


def can_see(rec: dict[str, Any], uid: str) -> bool:
    """The one who asked always; anyone asked unless they said no."""
    return uid == rec["from"] or answer_of(rec, uid) in ("pending", "accepted")


def involving(uid: str, path: Path | None = None) -> list[dict[str, Any]]:
    """Every comparison this person sent or was sent, newest first - except the
    ones they turned down. Declines are told only to the one who asked."""
    with _lock:
        items = _read(path)
    mine = [r for r in items if can_see(r, uid)]
    return sorted(mine, key=lambda r: r["at"], reverse=True)


def get(cid: str, path: Path | None = None) -> dict[str, Any] | None:
    with _lock:
        return next((r for r in _read(path) if r["id"] == cid), None)


def ask(
    sender: str,
    to: list[str] | str,
    spec: ChartSpec,
    q: str = "",
    name: str | None = None,
    title: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    """Send a comparison request to one or two people. The same question to
    the same people while it is still open is the request already sent."""
    others = [to] if isinstance(to, str) else list(dict.fromkeys(to))
    if not others:
        raise CompareError("pick someone to compare with")
    if len(others) > MAX_OTHERS:
        raise CompareError(f"up to {MAX_OTHERS} people at once")
    if sender in others:
        raise CompareError("that's you")
    for uid in others:
        if not users.exists(uid):
            raise CompareError("no such person")
    spec_json = spec.model_dump(mode="json")
    with _lock:
        items = _read(path)
        for r in items:
            if (
                r["from"] == sender
                and sorted(r["to"]) == sorted(others)
                and r["spec"] == spec_json
                and status(r) != "declined"
            ):
                return r
        open_ = sum(1 for r in items if r["from"] == sender and waiting_on(r))
        if open_ >= MAX_PENDING:
            raise CompareError(f"you have {open_} requests waiting already")
        rec = {
            "id": secrets.token_urlsafe(9),
            "from": sender,
            "to": others,
            "answers": {u: {"status": "pending", "at": None} for u in others},
            "spec": spec_json,
            "q": (q or "")[:400],
            "name": ((name or "").strip()[:MAX_NAME]) or None,
            "title": (title or "")[:120],
            "at": _now(),
        }
        items.append(rec)
        _write(path, items)
    return rec


def decide(cid: str, uid: str, accept: bool, path: Path | None = None) -> dict[str, Any]:
    """Accept or decline - each person asked, for themselves, and only once."""
    with _lock:
        items = _read(path)
        rec = next((r for r in items if r["id"] == cid), None)
        if rec is None or uid not in rec["to"]:
            raise CompareError("no such request")
        if answer_of(rec, uid) != "pending":
            raise CompareError(f"already {answer_of(rec, uid)}")
        rec["answers"][uid] = {"status": "accepted" if accept else "declined", "at": _now()}
        _write(path, items)
    return rec


def remove(cid: str, uid: str, path: Path | None = None) -> bool:
    """The one who asked ends it for everyone - withdrawing a request is the
    same act. Anyone else leaves, which takes only them out of it."""
    with _lock:
        items = _read(path)
        rec = next((r for r in items if r["id"] == cid), None)
        if rec is None or not (uid == rec["from"] or uid in rec["to"]):
            return False
        if uid == rec["from"]:
            items = [r for r in items if r["id"] != cid]
        else:
            rec["to"] = [u for u in rec["to"] if u != uid]
            rec["answers"].pop(uid, None)
            if not rec["to"]:
                items = [r for r in items if r["id"] != cid]
        _write(path, items)
    return True


def rename(cid: str, uid: str, name: str | None, path: Path | None = None) -> dict[str, Any]:
    """A comparison is named like any chart; anyone in it can."""
    with _lock:
        items = _read(path)
        rec = next((r for r in items if r["id"] == cid and can_see(r, uid)), None)
        if rec is None:
            raise CompareError("no such comparison")
        rec["name"] = ((name or "").strip()[:MAX_NAME]) or None
        _write(path, items)
    return rec


def person(uid: str) -> dict[str, Any]:
    pub = users.public(uid) if users.exists(uid) else {"id": uid}
    return {"id": uid, "name": pub.get("name") or uid, "image": pub.get("image")}


def view(rec: dict[str, Any], me: str) -> dict[str, Any]:
    """A comparison as one of the people in it sees it."""
    mine = rec["from"] == me
    return {
        "id": rec["id"],
        "from": person(rec["from"]),
        "to": [person(u) for u in rec["to"]],
        "with": [person(u) for u in participants(rec) if u != me],
        "waiting": [person(u) for u in waiting_on(rec)],
        "declined": [person(u) for u in rec["to"] if answer_of(rec, u) == "declined"],
        "mine": mine,
        # The comparison as a whole, and this person's own answer to it.
        "status": status(rec),
        "my_answer": None if mine else answer_of(rec, me),
        "spec": rec["spec"],
        "q": rec["q"],
        "name": rec.get("name"),
        "title": rec.get("title") or "",
        "at": rec["at"],
    }


# ------------------------------------------------------------ side by side
#
# How two or three answers to one question are set beside each other depends
# on what the answer is. A number or a winner is a card each; a ranking is a
# list each with what they share marked; anything along an ordered axis - months, hours,
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
