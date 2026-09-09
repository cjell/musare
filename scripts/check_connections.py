"""Verify every external service in .env is reachable.

Prints pass/fail per service and never prints a secret - only a masked prefix,
so the output is safe to paste into a chat or an issue.

    python scripts/check_connections.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"

OK, BAD, SKIP = "PASS", "FAIL", "SKIP"


def load_env(path: Path) -> dict[str, str]:
    """Minimal .env reader - no dependency, and it tolerates '=' inside values."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def mask(v: str) -> str:
    if not v:
        return "(empty)"
    return f"{v[:6]}...({len(v)} chars)"


results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str) -> None:
    results.append((name, status, detail))


def check_postgres(env: dict[str, str]) -> None:
    dsn = env.get("DATABASE_URL", "")
    if not dsn or "PASSWORD" in dsn or "xxxx" in dsn:
        record("Postgres", SKIP, "DATABASE_URL not filled in")
        return
    try:
        import psycopg
    except ImportError:
        record("Postgres", SKIP, "psycopg not installed")
        return
    try:
        with psycopg.connect(dsn, connect_timeout=15) as conn, conn.cursor() as cur:
            cur.execute("select current_database(), current_user, version()")
            db, user, ver = cur.fetchone()
            short = ver.split(",")[0]
            record("Postgres", OK, f"{short} | db={db} user={user}")

            cur.execute("select extname from pg_extension order by extname")
            exts = [r[0] for r in cur.fetchall()]
            has_vec = "vector" in exts
            record(
                "pgvector",
                OK if has_vec else BAD,
                "enabled" if has_vec else f"NOT enabled (found: {', '.join(exts)})",
            )

            cur.execute("""
                select table_name from information_schema.tables
                where table_schema = 'public' order by table_name
            """)
            tables = [r[0] for r in cur.fetchall()]
            record(
                "Tables",
                OK,
                f"{len(tables)} in public"
                + (f": {', '.join(tables)}" if tables else " (empty, as expected)"),
            )
    except Exception as e:  # report any failure, never crash the run
        record("Postgres", BAD, f"{type(e).__name__}: {str(e).strip()[:160]}")


def check_supabase_rest(env: dict[str, str]) -> None:
    url, key = env.get("SUPABASE_URL", ""), env.get("SUPABASE_ANON_KEY", "")
    if not url or not key or "xxxx" in url:
        record("Supabase REST", SKIP, "URL or anon key not filled in")
        return
    # The root /rest/v1/ serves the OpenAPI spec and Supabase restricts that to
    # secret keys, so hitting it with the publishable key is a 401 even when the
    # key is fine. Ask for a table that does not exist instead: a valid key gets
    # routed and returns PostgREST's 404, an invalid one is rejected at 401.
    try:
        r = httpx.get(
            f"{url.rstrip('/')}/rest/v1/__connectivity_probe",
            headers={"apikey": key},
            params={"select": "*"},
            timeout=20,
        )
        if r.status_code == 401:
            record("Supabase REST", BAD, f"401 rejected | key {mask(key)}")
        else:
            record("Supabase REST", OK, f"key accepted (HTTP {r.status_code}) | {mask(key)}")
    except Exception as e:  # report it, keep the other checks running
        record("Supabase REST", BAD, f"{type(e).__name__}: {str(e)[:120]}")


def check_ticketmaster(env: dict[str, str]) -> None:
    key = env.get("TICKETMASTER_API_KEY", "")
    if not key:
        record("Ticketmaster", SKIP, "no key")
        return
    try:
        r = httpx.get(
            "https://app.ticketmaster.com/discovery/v2/events.json",
            params={"apikey": key, "size": 1, "city": "Raleigh", "classificationName": "music"},
            timeout=25,
        )
        if r.status_code == 200:
            total = r.json().get("page", {}).get("totalElements", "?")
            record("Ticketmaster", OK, f"HTTP 200 | {total} music events near Raleigh")
        else:
            record("Ticketmaster", BAD, f"HTTP {r.status_code} | {r.text[:110]}")
    except Exception as e:  # report it, keep the other checks running
        record("Ticketmaster", BAD, f"{type(e).__name__}: {str(e)[:120]}")


def check_spotify(env: dict[str, str]) -> None:
    cid, secret = env.get("SPOTIFY_CLIENT_ID", ""), env.get("SPOTIFY_CLIENT_SECRET", "")
    if not cid or not secret:
        record("Spotify", SKIP, "client id/secret not filled in")
        return
    try:
        # Client-credentials proves the app's identity without a user login.
        r = httpx.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(cid, secret),
            timeout=25,
        )
        if r.status_code == 200:
            record("Spotify", OK, f"token issued, expires in {r.json().get('expires_in')}s")
        else:
            record("Spotify", BAD, f"HTTP {r.status_code} | {r.text[:110]}")
    except Exception as e:  # report it, keep the other checks running
        record("Spotify", BAD, f"{type(e).__name__}: {str(e)[:120]}")


def check_openai(env: dict[str, str]) -> None:
    key = env.get("OPENAI_API_KEY", "")
    if not key:
        record("OpenAI", SKIP, "no key")
        return
    try:
        r = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=25,
        )
        if r.status_code == 200:
            ids = sorted(m["id"] for m in r.json().get("data", []))
            record("OpenAI", OK, f"{len(ids)} models available")
            print("\n  models your account can see:")
            for i in ids:
                print(f"    {i}")
            print()
        else:
            record("OpenAI", BAD, f"HTTP {r.status_code} | {r.text[:110]}")
    except Exception as e:  # report it, keep the other checks running
        record("OpenAI", BAD, f"{type(e).__name__}: {str(e)[:120]}")


def main() -> int:
    env = load_env(ENV)
    if not env:
        print(f"No .env found at {ENV}")
        return 1

    print(f"Reading {ENV}\n")
    print("Variables set:")
    for k in [
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "DATABASE_URL",
        "TICKETMASTER_API_KEY",
        "SPOTIFY_CLIENT_ID",
        "SPOTIFY_CLIENT_SECRET",
        "SPOTIFY_REDIRECT_URI",
        "OPENAI_API_KEY",
    ]:
        v = env.get(k, "")
        flag = "set  " if v and "xxxx" not in v and "PASSWORD" not in v else "MISSING"
        shown = v if k == "SPOTIFY_REDIRECT_URI" else mask(v)
        print(f"  [{flag}] {k:<28} {shown}")
    print()

    check_postgres(env)
    check_supabase_rest(env)
    check_ticketmaster(env)
    check_spotify(env)
    check_openai(env)

    print("Results:")
    width = max(len(n) for n, _, _ in results)
    for name, status, detail in results:
        print(f"  {status:<4} {name:<{width}}  {detail}")

    failed = [n for n, s, _ in results if s == BAD]
    print()
    print(f"{len(failed)} failing" if failed else "All checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
