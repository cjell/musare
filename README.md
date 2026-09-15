# MusicShare

A social network built around musical taste as identity. Not a streaming app and
not a recommender - the question it answers is *whose taste should I trust*.

Everything on a profile is computed from listening history. Nothing is authored:
there are no posts, so nothing is written into a void and nobody performs for a
feed.

## What is real here

This runs on one person's actual Spotify export - **509,459 plays from August
2018 to September 2026** - against a corpus of **130,896 artists**. The
following are computed, not mocked:

- the taste map and its 152 named genre regions
- top artists, tracks and albums; the weekly feed; discoveries
- the live status row and the avatar state
- genre-filtered show search, and every chart
- playlists, pinning, and the artist/album/track pickers

The app ships with **two sample profiles** (Devin, Jules) so the social shape is
legible with one real account. They are labelled *Sample profile* in the app and
every number on them is invented. Anything social on the real profile - match
percentages, "9 going", "11 listen here" - is invented for the same reason:
there is one user. **The single-listener features are real; the between-people
features are not yet measurable.**

## Why it looks like this

Two products already occupy this space. **Airbuds** is friend-gated - you see
nothing until you add people - and paywalls its features. **Equals** is artist
fandom chatrooms with messaging behind $4.99/week. MusicShare is open and
discoverable by default, ranks people rather than hosting fan clubs, and does
not paywall seeing people, messaging, or participating.

## Architecture

```
Spotify export     ->  Parquet (raw)   ->  DuckDB (transform)  ->  the app
   509k plays           8.6 MB              ~0.1s queries
Supabase capture   ->  Parquet (live)  ^
```

Two sources, one view. The export is complete and exact up to the day it was
generated. Everything since is captured inside Supabase, and the two union at
read time under a single rule: **the export wins for every period it covers**,
so live rows only survive past its high-water mark and a future export silently
upgrades them to exact ones.

Capture started as a poller on recently-played and was rebuilt after measuring
it. The endpoint omitted most plays - fifteen on a day with hours of listening,
one song played ten times reported twice - while currently-playing was right
every time it was asked. So `supabase/capture.sql` asks what is playing every 30
seconds and writes a play when the track changes, restarts, or stops, with
listening time measured from playback progress rather than assumed from the
track's length. Recently-played stays on every 30 minutes as a backup, and a
watcher row for the same play replaces its estimate.

It runs in the database - pg_cron, the `http` extension, secrets in Vault -
because a laptop is not online when its owner is walking around with headphones
in. The decision logic is one pure function, so every transition (a skip, a
repeat, a pause, a one-poll dropout) is tested by handing it two snapshots.

    ./mscs/python.exe scripts/capture.py install   # secrets, schema and jobs into Supabase
    ./mscs/python.exe scripts/capture.py status    # is it checking in, what has it recorded
    ./mscs/python.exe scripts/capture.py pull      # copy plays into the local store now

Opening the app pulls too. Local day files are replaced from the database rather
than merged, so a superseded estimate cannot survive as a second copy.

Personalization - pins, bio, colours, photos, avatar art - is stored server-side
as one document plus uploaded images. It used to live in `localStorage`, which
is scoped to a browser *and* an origin, so a profile built in one browser did
not exist in another.

## The map

The largest single piece of work, and the one nothing else here could fake.

Every artist in the corpus is embedded from its tags into **200 dimensions**,
then clustered twice for two different questions. *Modes* describe a person -
six of them, from one listener's hours. *Regions* describe the music - 152 of
them, from the corpus, identical for everyone.

Two rules hold it together:

**Similarity is measured in 200 dimensions, never in the 2D projection.** UMAP
is a picture, not a metric; distance on it is flat past about three units.

**One fixed basemap, fitted on the whole corpus.** Separate fits produce
unrelated coordinate systems, so a map you could compare two people on has to
come from one fitting. The fitted artefacts are 534MB, and only drawing the map
loads them, once per process. Everything else - genre filters, the live status,
the avatar - reads a 1.5MB table produced from them at build time.

Region names come from a **frozen 385-word vocabulary** derived from the corpus
itself. Naming was the one place a model wrote free text and the one thing that
drifted - 42 of 80 names changed in a single rebuild, "melodic trap" becoming
"trap". A schema enum cannot return a word that is not on the list.

That change also closed a measurement leak. Exemplars were ranked by fan count
from a service whose audience is not evenly spread, so French rap won inside a
large mixed rap region and the model named the whole thing "french rap" - over
Kanye, J. Cole and Lil Wayne. A word absent from the list cannot be chosen.

## The AI features

One architecture, four jobs:

```
natural language -> structured spec -> validator -> your query -> your components
```

The model never emits SQL, never produces numbers, and never writes the answer.
It fills out a schema-constrained form; application code executes it. **The
schema is the capability ceiling** - it names no table, no path, no user, so the
worst a hostile input achieves is the wrong list of concerts.

1. **Charts** - "top 3 artists every year", "rap vs rock over the years". Every
   field is a menu, so the whole grammar is renderable by construction.
2. **Show discovery** - "small metal bands near me this weekend" becomes a
   filter over live Ticketmaster data.
3. **Region naming** - 152 regions named from the frozen vocabulary.
4. **Avatar states** - each region sorted once into one of six states, so the
   live status row is a dictionary lookup rather than a model call.

Some answers cannot be drawn. "Top 3 artists every year" wants a different top 3
in each column, which no chart shape expresses, so it comes back as a table -
and **that is derived in code, not offered to the model**, because a table it
could ask for would become the answer to everything it was unsure about.
Refusal is separate and intact: a request for data this app does not hold still
returns nothing.

### Measurement

Correctness is measured. Cases are scored per field with tolerances - "a few
hours" is 150 miles or 250 depending who you ask, and failing a good answer for
being differently good teaches nothing.

    ./mscs/python.exe evals/run.py --suite shows     # report + failure taxonomy
    ./mscs/python.exe evals/run.py --suite charts
    pytest -m eval                                   # the same, as a CI gate

**Held-out sets, each written before the feature existed and run once:**

| suite | result |
|---|---|
| shows + genre | 16/17 (94%) |
| charts + series | 17/18 (94%) |
| charts + per-period ranking | 14/14 (100%) |
| shows + familiarity scale | 22/22 (100%) |
| charts + today | 14/17 (82%) |
| charts + running totals, and the three today misses | 15/15 (100%) |
| changing an existing chart, fields kept | 17/17 (100%) |

A held-out set is single-use. Reading its failures and fixing against them makes
it a training set, so these are quoted once and the earlier two files in
`evals/cases/` are training data now, whatever their filenames say.

**Model choice is measured per task, not assumed:**

| suite | gpt-5.4 | gpt-5.4-nano |
|---|---|---|
| shows | 99% | 54% |
| charts | 99% | 89% |

That gap was found the hard way. The endpoints defaulted to nano while every
number in the reports came from gpt-5.4, so "top 3 artists every year" failed
eight times out of eight in the browser while the held-out set read 100%.
Endpoints now pin the measured model.

### What measurement changed

**Prose could not fix defaults.** Requests mentioning no distance should leave
it unset; the first run scored 25%, and adding an instruction saying so made it
*worse* - 0% - because the model became consistently wrong instead of randomly
wrong. Models fill fields; they do not decline to. Making the field nullable, so
"not mentioned" is a value it can emit, took it to 100%.

**A field that can express an invalid state will express it.** `ChartSpec` had a
`chart` field, and the model would read "over time", pick a line, and leave
`dimension` on artist - a pair that cannot be drawn. Four rounds of rewording
moved the score without fixing it. Chart type is a function of dimension, so it
was deleted and derived in code; the failure class went with it.

**A single run is not a measurement.** The same code scored 100% then 98% on
consecutive runs with nothing changed. Figures here come from repeated runs, and
the CI floors sit well below the observed range on purpose.

**Mood is not in this data.** The avatar was going to infer mood from listening.
Measured: 12% of corpus artists carry any mood tag, 3% carry one usable at the
artist level, and behaviour signals spanned 1.74-2.35 across a whole day. So it
reacts instead of inferring - the current track's region has a state, decided
once. That is a claim about the music, not about the listener, and it is the
honest version of the feature.

## Layout

    src/musicshare/
      spec/filter.py     the show form; its field descriptions are the prompt
      spec/chart.py      the chart form - metrics, dimensions, series, ranking
      spec/vocab.py      385 frozen genre labels, and the enum built from them
      spec/name.py       naming schema for modes and regions
      spec/mood.py       the six avatar states, and why mood-from-listening was rejected
      spec/validate.py   semantic checks the schemas cannot express
      spec/generate.py   text -> spec; the only LLM call
      spec/apply.py      executes a show filter - no model involved
      spec/chartrun.py   executes a chart spec, in SQL, from menus only
      spec/plot.py       how a chart is drawn - kind, points, ticks, labels - decided in tested code
      embed.py           artist tags -> 200-dimension space
      project.py         the fitted basemap and 2D positions, for drawing only
      regions.py         152 genre regions, their names and states
      modes.py           one listener as weighted clusters
      genres.py          artist -> region -> state, the 1.5MB serving table
      home.py            the weekly feed: movers, discoveries, on repeat
      live.py            pulls captured plays down; now-playing; the live store
      history.py         one view over the export and the live rows
      media.py           uploads and the profile document
      shows.py           Ticketmaster events joined to local play history
      api/app.py         the server
    supabase/capture.sql listening capture: the 30-second watcher, the backup, dedupe
    evals/cases/         golden sets and held-out sets
    profile.html         the interaction design, single file, no build step

## Running it

    conda env create -f environment.yml -p ./mscs
    ./mscs/python.exe -m pip install -e ".[dev]"
    cp .env.example .env        # then fill it in
    ./mscs/python.exe scripts/check_connections.py

Ingest an extended-streaming-history export, then build the serving table:

    ./mscs/python.exe scripts/ingest_streaming_history.py "<export dir>"
    ./mscs/python.exe -m musicshare.genres

Serve it:

    ./mscs/python.exe -m uvicorn musicshare.api.app:app --reload --port 8000

Tests are free and offline by default; the ones that cost money or reach a third
party are marked and deselected:

    ./mscs/python.exe -m pytest                  # 244 tests, no network, ~20s
    ./mscs/python.exe -m pytest -m eval          # spends real requests

CI runs lint and that same default suite on every push. The listening history
and corpus are gitignored, so the 93 tests that read them skip there and 151
run - the local run with data is the full one.

### The Spotify constraint

Spotify closed extended API quota to individuals in May 2025 - it now requires a
registered business with 250k+ MAU. Development mode (25 whitelisted users) is
therefore permanent. That is the full API, so nothing here is degraded; it just
cannot scale, which is a deliberate accepted limit rather than an oversight.

The consequence worth designing for: the export needs Spotify exactly once, when
the file is generated. If API access disappeared entirely, every profile would
keep its map position, badges, matching and charts. Only freshness and pinning
depend on the live API.
