"""Pull recently-played from Spotify and keep it beside the export.

The endpoint returns the last fifty plays and nothing else - roughly a day for
a heavy listener - so this is a rolling window, not a history. Whatever falls
out of it before a sync runs is gone unless a later export covers it. Polling
is a commitment rather than a convenience, and the export stays the safety net.

What comes back is also thinner than the export: a timestamp, a track, and its
duration. There is no ms_played, no skipped, no platform. Duration stands in
for ms_played as a documented upper bound - it assumes every play finished,
which is wrong for the roughly 30% that are skips - and everything else is
null, so metrics can tell "unknown" from "zero".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb
import httpx

from musicshare.config import ROOT, settings
from musicshare.history import LIVE_DIR, connect

log = logging.getLogger(__name__)

TOKENS = ROOT / "data" / "cache" / "spotify_user_token.json"
TOKEN_URL = "https://accounts.spotify.com/api/token"
RECENT = "https://api.spotify.com/v1/me/player/recently-played"


class NoToken(RuntimeError):
    pass


@dataclass
class SyncResult:
    fetched: int
    added: int
    oldest: datetime | None
    newest: datetime | None
    watermark: datetime | None

    @property
    def missed(self) -> bool:
        """True when the window itself starts after our last known play.

        Only meaningful because sync() fetches the whole window rather than
        asking for rows after a cursor: if the oldest play Spotify still holds
        is newer than the newest play we have, everything between them is gone.
        """
        return bool(self.watermark and self.oldest and self.oldest > self.watermark)


def access_token() -> str:
    """Refresh the stored token. Spotify's access tokens last an hour."""
    if not TOKENS.exists():
        raise NoToken(f"no token at {TOKENS} - run scripts/oauth_probe.py --login")
    tok = json.loads(TOKENS.read_text(encoding="utf-8"))
    if "refresh_token" not in tok:
        raise NoToken("stored token has no refresh_token - sign in again")

    s = settings()
    r = httpx.post(
        TOKEN_URL,
        data={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"]},
        auth=(s.spotify_client_id, s.spotify_client_secret),
        timeout=25,
    )
    if r.status_code != 200:
        raise NoToken(f"refresh failed: HTTP {r.status_code} {r.text[:160]}")

    fresh = r.json()
    # A refresh response may omit refresh_token; keep the one that still works.
    tok.update({k: v for k, v in fresh.items() if v})
    TOKENS.write_text(json.dumps(tok, indent=2), encoding="utf-8")
    return tok["access_token"]


def fetch(after: datetime | None = None, limit: int = 50) -> list[dict]:
    params: dict[str, object] = {"limit": min(limit, 50)}
    if after:
        # `after` is a unix millisecond cursor; asking for only what is new keeps
        # the response small, but the fifty-row cap still applies.
        params["after"] = int(after.timestamp() * 1000)
    r = httpx.get(
        RECENT, params=params, headers={"Authorization": f"Bearer {access_token()}"}, timeout=25
    )
    if r.status_code != 200:
        raise RuntimeError(f"recently-played: HTTP {r.status_code} {r.text[:160]}")
    return r.json().get("items", [])


def _rows(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        t = it.get("track") or {}
        if not t.get("uri"):
            continue  # local files and podcasts have no track uri
        album = t.get("album") or {}
        out.append(
            {
                "played_at": datetime.fromisoformat(it["played_at"].replace("Z", "+00:00"))
                .astimezone(UTC)
                .replace(tzinfo=None),
                "track_uri": t["uri"],
                "track_name": t.get("name"),
                "artist_name": ", ".join(a["name"] for a in t.get("artists", []) if a.get("name")),
                "album_name": album.get("name"),
                # An upper bound, not a measurement. See the module docstring.
                "ms_played": t.get("duration_ms"),
            }
        )
    return out


def watermark() -> datetime | None:
    """The latest play we already know about, from either source."""
    con = connect()
    return con.execute("select max(played_at) from plays").fetchone()[0]


def sync() -> SyncResult:
    mark = watermark()
    # Deliberately no `after` cursor. With one, the oldest row returned is just
    # the first play past the cursor, which says nothing about whether anything
    # fell out of the window - it cannot tell "plays were lost" from "you were
    # asleep". Fetching the whole window makes its start meaningful, and costs
    # the same single request.
    items = fetch()
    rows = _rows(items)

    oldest = min((r["played_at"] for r in rows), default=None)
    newest = max((r["played_at"] for r in rows), default=None)
    new = [r for r in rows if mark is None or r["played_at"] > mark]

    if new:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        con.register("incoming", _as_arrow(new))
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        path = LIVE_DIR / f"live-{stamp}.parquet"
        con.execute(
            f"copy (select * from incoming) to '{path.as_posix()}' (format parquet, compression zstd)"
        )
        log.info("wrote %d rows to %s", len(new), path.name)

    return SyncResult(len(rows), len(new), oldest, newest, mark)


def _as_arrow(rows: list[dict]):
    import pyarrow as pa

    return pa.Table.from_pylist(rows)


def coverage() -> dict[str, object]:
    con = connect()
    by_source = dict(con.execute("select source, count(*) from plays group by 1").fetchall())
    n, lo, hi = con.execute("select count(*), min(played_at), max(played_at) from plays").fetchone()
    return {"total": n, "first": lo, "last": hi, "by_source": by_source}


def prune(keep: int = 12) -> int:
    """Keep the newest sync files; the rest are already covered by an export."""
    files = sorted(LIVE_DIR.glob("live-*.parquet"))
    stale = files[:-keep] if len(files) > keep else []
    for f in stale:
        f.unlink()
    return len(stale)
