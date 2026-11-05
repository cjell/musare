"""The facts at the top of a profile, computed from that person's history.

These were written by hand once, for one listener - "14,000+ hrs", "61 genres
this year", six genre chips - after measuring that one history. With more than
one account a hand-written card is either one person's numbers on everyone's
profile or a blank on theirs, so the same measurements are made here, for any
account, from its own plays.

Only plays that cleared 30 seconds count, the rule every other screen uses, so
a profile's hours agree with its own Home screen.
"""

from __future__ import annotations

from typing import Any

from musicshare import genres as genrelib
from musicshare.history import connect

MIN_MS = 30_000
# A genre counts toward "genres this year" past this many hours of it - five
# was the bar the hand-written badge was measured at, and it keeps a single
# stray album from counting as a genre someone listens to.
GENRE_HOURS = 5
# Chips on the profile, most listened first over the last year.
TOP_GENRES = 6


def card(uid: str | None = None) -> dict[str, Any]:
    con = connect(uid=uid)
    hours, plays, artists, first = con.execute(f"""
        select coalesce(sum(ms_played), 0) / 3600000.0, count(*),
               count(distinct artist_name), min(played_at)
        from plays where ms_played >= {MIN_MS}
    """).fetchone()

    top: list[str] = []
    genres_year = 0
    if genrelib.ready():
        genrelib.attach(con)
        rows = con.execute(f"""
            select g.genre, sum(p.ms_played) / 3600000.0 as h
            from plays p join artist_genre g on lower(p.artist_name) = g.artist
            where p.ms_played >= {MIN_MS}
              and p.played_at > (select max(played_at) from plays) - interval 365 day
            group by 1 order by 2 desc
        """).fetchall()
        top = [g for g, _ in rows[:TOP_GENRES]]
        genres_year = sum(1 for _, h in rows if h >= GENRE_HOURS)

    return {
        "hours": round(float(hours), 1),
        "plays": int(plays),
        "artists": int(artists),
        "since": first.year if first else None,
        "genres": top,
        "genres_year": genres_year,
    }
