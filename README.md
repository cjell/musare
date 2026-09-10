# MusicShare

A social network built around musical taste as identity. Not a streaming app and
not a recommender - the question it answers is *whose taste should I trust*.

Status: early. The interaction design exists as a working mockup; the data
layer and the LLM pipelines are being built underneath it.

## Why it looks like this

Two products already occupy this space. **Airbuds** is friend-gated - you see
nothing until you add people - and paywalls its features. **Equals** is artist
fandom chatrooms with messaging behind $4.99/week. MusicShare is open and
discoverable by default, ranks people rather than hosting fan clubs, and does
not paywall seeing people, messaging, or participating.

Nothing in the app is authored. There are no posts. Every profile is generated
from listening behaviour, so nothing is written into a void and nobody performs
for a feed.

## Architecture

```
Spotify export  ->  Parquet (raw)  ->  DuckDB (transform)  ->  Postgres (serve)
   503k plays        8.5 MB            ~0.1s queries          ~2 MB per user
```

Raw play events stay local. Postgres stores answers, not events - a profile
needs to know you have played an artist for 532 hours, not that you played a
particular song at 11:47pm on a Tuesday in 2019.

Data separates into four domains with almost nothing in common:

| Domain | Example | Source | Mutability |
|---|---|---|---|
| Catalog | artist, track, artwork, popularity | Spotify API | rarely changes, shared |
| Behavior | 503,236 play events | data export | append-only |
| Derived | top artists, badges, taste vector | computed | recomputed on a schedule |
| Authored | pins, follows, bio, accent | the user | constantly |

### The Spotify constraint

Spotify closed extended API quota to individuals in May 2025 - it now requires a
registered business with 250k+ MAU. Development mode (25 whitelisted users) is
therefore permanent. That is the full API, so nothing here is degraded; it just
cannot scale, which is a deliberate accepted limit rather than an oversight.

The consequence worth designing for: the data export needs Spotify exactly once,
when the file is generated. If API access disappeared entirely, every profile
would keep its map position, badges, matching and visualizations. Only freshness
and pinning depend on the live API.

## The AI features

Two features, one architecture:

```
natural language -> structured spec -> validator -> your query -> your components
```

The model never emits SQL, never produces numbers, and never writes the answer.
It fills out a schema-constrained form; application code executes it. A spec
that references a metric the semantic layer does not expose fails closed.

1. **Pinnable visualizations** - describe a chart of your listening, pin the
   result to your profile. This is the app's only authoring primitive, and it is
   made of evidence rather than opinion.
2. **Natural-language show discovery** - *"shows for artists I like with small
   fanbases, I don't mind driving a few hours"* becomes a filter over live
   Ticketmaster data, saved as a named filter others can subscribe to.

Correctness here is measured, not asserted. `evals/cases/shows.yaml` holds 54
hand-written inputs with the spec each should produce, scored per field with
tolerances - "a few hours" is 150 miles or 250 depending who you ask, and
failing a good answer for being differently good teaches nothing.

    python evals/run.py                        # report + failure taxonomy
    python evals/run.py --model gpt-5.4-mini   # compare
    python evals/run.py --compare a.json b.json
    pytest -m eval                             # the same, as a CI gate

Measured on 2026-09-10, 54 cases:

| | gpt-5.4-nano | gpt-5.4-mini |
|---|---|---|
| exact match | **96%** | 93% |
| field accuracy | **97%** | 93% |
| avg latency | 1660ms | 1684ms |

The cheaper model wins. `mini` misses every inverted request - it reads "a big
name" and "someone famous" as no constraint at all, scoring 40% on `min_fans`
where `nano` scores 100%. Benchmarks would not have told you that about this
task.

Two findings from the first run worth keeping:

**Prose could not fix defaults.** Requests that mention no distance should
leave it unset; the first run scored 25% there, and adding an instruction
saying so made it *worse* - 0% - because the model became consistently wrong
instead of randomly wrong. Models fill fields; they do not decline to. Making
`radius_mi` and `within_days` nullable, so "not mentioned" is a value the model
can actually emit, took both to 100% and the suite from 89% to 96%.

**One failure was ours.** An empty request returned a provider 400 rather than
an answer. The eval found it; the fix is four lines and no round trip.

## Layout

    src/musicshare/
      spec/filter.py     the form the model fills; its descriptions are the prompt
      spec/apply.py      executes a spec against shows - no model involved
      spec/validate.py   semantic checks the schema cannot express
      spec/generate.py   text -> ShowFilter, the only LLM call
      shows.py           Ticketmaster events joined to local play history
      config.py          settings from .env
      spotify/client.py  catalog client (client-credentials; search, tracks, artists)
      taste.py           personal top lists from local history
      web/render.py      wraps the Artifact-format mockup into a real document
      api/app.py         dev server: search proxy + mockup
    evals/
      cases/shows.yaml   the golden set
      run.py             runner, report, model comparison
    scripts/
      check_connections.py         verify every service, printing no secrets
      oauth_probe.py               what a Spotify user token can actually reach
      ingest_streaming_history.py  Spotify export -> partitioned Parquet
    profile.html         the interaction design, single file, no build step

## Running it

    conda env create -f environment.yml -p ./mscs
    ./mscs/python.exe -m pip install -e ".[dev]"
    cp .env.example .env        # then fill it in
    ./mscs/python.exe scripts/check_connections.py

Ingest a Spotify extended-streaming-history export:

    ./mscs/python.exe scripts/ingest_streaming_history.py "<export dir>"

Serve the mockup with live catalog search:

    ./mscs/python.exe -m uvicorn musicshare.api.app:app --reload --port 8000
