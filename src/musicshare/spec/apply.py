"""Execute a spec against shows. No model is involved here, deliberately.

The model describes a query; this runs it. That split is what keeps a wrong
guess to a wrong list of concerts rather than a wrong database call, and it is
what makes the whole path testable without spending a token.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from musicshare import genres
from musicshare.spec.filter import DEFAULT_DAYS, DEFAULT_RADIUS_MI, ShowFilter


def apply(spec: ShowFilter, shows: list[dict[str, Any]], today: date | None = None) -> list[dict]:
    """Shows matching the spec, in the order they were given (chronological)."""
    if not spec.understood:
        return []

    today = today or date.today()
    # Null means the user never raised it, so the app decides, not the model.
    cutoff = today + timedelta(days=spec.within_days or DEFAULT_DAYS)
    wanted = {a.strip().lower() for a in spec.artists if a.strip()}
    # Widened once, outside the loop: a request for "metal" covers eight regions
    # and recomputing that per show would run the regex 152 times a row.
    covered = set(genres.expand(spec.genres)) if spec.genres else set()

    out = []
    for s in shows:
        if spec.only_mine and not s.get("yours"):
            continue

        # Missing is not the same as far, or as unknown-ly popular: a field the
        # data does not carry must not silently exclude a show.
        mi = s.get("distance_mi")
        if mi is not None and mi > (spec.radius_mi or DEFAULT_RADIUS_MI):
            continue

        d = s.get("date")
        if d:
            try:
                if not (today <= date.fromisoformat(d) <= cutoff):
                    continue
            except ValueError:
                pass

        fans = s.get("fans")
        if fans is not None:
            if spec.max_fans is not None and fans > spec.max_fans:
                continue
            if spec.min_fans is not None and fans < spec.min_fans:
                continue
        elif spec.max_fans is not None or spec.min_fans is not None:
            # A fan filter was asked for and this artist has no count. Excluding
            # is the honest choice: the user asked for a property we cannot
            # confirm this show has.
            continue

        if wanted and (s.get("artist") or "").strip().lower() not in wanted:
            continue

        # Same rule as the fan bounds above, for the same reason: a genre was
        # asked for, and an act the corpus cannot place is an act we cannot
        # confirm plays it. Listing them anyway would quietly turn "metal shows"
        # back into "shows".
        if spec.genres and s.get("genre") not in covered:
            continue

        out.append(s)
    return out
