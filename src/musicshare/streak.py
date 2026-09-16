"""The listening streak on the profile badge: consecutive local days with a play.

Two rules make this more than a gaps-and-islands query.

A day nobody could see is not a day missed. The export stops when it was
generated and live capture began a day or so later, and the recently-played
endpoint only reaches back fifty plays - so anything played in between was out
of reach before syncing started. A day inside that window has no evidence either
way, and at a couple of hundred plays a day the honest reading is that the run
carried on through it. The next export fills the window and the rule goes quiet.

A streak stays alive until a whole local day passes without a play. Opening the
profile at 9am, before the first song, should not report zero.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from musicshare.history import connect
from musicshare.spec.chartrun import LOCAL, LOCAL_TS


def current_run(
    days: set[date],
    today: date,
    unobserved: tuple[date, date] | None = None,
) -> tuple[int, date | None]:
    """Length and first day of the run reaching today or yesterday, else (0, None).

    `unobserved` is the export's last day and live capture's first; the days
    strictly between them count as played.
    """

    def covered(d: date) -> bool:
        if d in days:
            return True
        return unobserved is not None and unobserved[0] < d < unobserved[1]

    d = today if covered(today) else today - timedelta(days=1)
    if not covered(d):
        return 0, None

    n = 0
    while covered(d):
        n += 1
        d -= timedelta(days=1)
    return n, d + timedelta(days=1)


def streak(today: date | None = None) -> dict[str, Any]:
    con = connect(icu=True)
    rows = con.execute(f"""
        select cast({LOCAL_TS} as date) as day,
               bool_or(source = 'export'),
               bool_or(source = 'api')
        from plays group by 1
    """).fetchall()

    export_days = [d for d, from_export, _ in rows if from_export]
    live_days = [d for d, _, from_live in rows if from_live]
    gap = (max(export_days), min(live_days)) if export_days and live_days else None

    if today is None:
        # asked of DuckDB so the clock and the plays agree on what "local" means
        today = con.execute(
            f"select cast(current_timestamp at time zone '{LOCAL}' as date)"
        ).fetchone()[0]

    n, since = current_run({r[0] for r in rows}, today, gap)
    return {"days": n, "since": since.isoformat() if since else None}
