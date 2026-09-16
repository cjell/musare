# MusicShare

A social network built around musical taste as identity. Not a streaming app and
not a recommender - the question I want it to answer is *whose taste should I
trust*.

Everything on a profile is computed from listening history. Nothing is authored:
there are no posts, so nothing is written into a void and nobody performs for a
feed.

## What it runs on

My own Spotify history - **509,811 plays and 14,264 hours, from August 2018 to
today** - against a corpus of **130,896 artists**. That length is the reason the
project works at all: a taste map, a weekly feed and eight years of charts need
a history that already exists, and mine does.

The taste map, the feed, the charts, the show search, the live status row and
every ranking are computed from that history. What is *not* real is anything
between people - match percentages, "9 going", "11 listen here" - because there
is one user. Two sample profiles (Devin, Jules) exist so the social shape is
legible, and they are labelled *Sample profile* in the app with every number on
them invented. **The single-listener features are real; the between-people
features are not yet measurable.**

---

## The taste map

https://github.com/user-attachments/assets/959a974e-b17c-4e05-acab-24695f64dbff

The largest piece of work here, and the one nothing else could fake. Every
artist in the corpus is embedded from its tags into **200 dimensions**, then
clustered twice for two different questions: *modes* describe a person - six of
them, from one listener's hours - and *regions* describe the music, 152 of them,
from the corpus, identical for everyone.

Two rules hold it together.

**Similarity is measured in 200 dimensions, never in the 2D projection.** UMAP
is a picture, not a metric; distance on it goes flat past about three units.

**One fixed basemap, fitted on the whole corpus.** Separate fits produce
unrelated coordinate systems, so a map you can compare two people on has to come
from one fitting. The fitted artefacts are 534MB and only drawing the map loads
them, once per process. Everything else - genre filters, the live status, the
avatar - reads a 1.5MB table built from them.

Region names come from a **frozen 385-word vocabulary** derived from the corpus
itself. Naming was the one place I let a model write free text and the one thing
that drifted: 42 of 80 names changed in a single rebuild, "melodic trap" becoming
"trap". A schema enum cannot return a word that is not on the list.

That change also closed a measurement leak. Exemplars were ranked by fan count
from a service whose audience is not evenly spread, so French rap won inside a
large mixed rap region and the model named the whole thing "french rap" - over
Kanye, J. Cole and Lil Wayne. A word absent from the list cannot be chosen.

---

## Charts you ask for in words

https://github.com/user-attachments/assets/9be70e99-cf69-4811-b83d-3e41a4ebce27

"Top 3 artists every year", "rap vs rock over the years", "my favourite genres
by day of the week". The pipeline is the same one every language feature here
uses:

```
natural language -> structured spec -> validator -> my query -> my components
```

The model never emits SQL, never produces numbers and never writes the answer.
It fills in `ChartSpec`, where every field is a menu - metric, dimension, grain,
range, series, ranking - so the whole grammar is a few hundred charts and each
one is renderable by construction. **The schema is the capability ceiling:** it
names no table, no path and no user.

There is deliberately no `chart` field. It was one, and the model would read
"over time", pick a line, and leave `dimension` on artist - a pair that cannot be
drawn. Chart type is a function of dimension, so I deleted the field and derived
it in code, which deleted the whole failure class with it.

How a chart is *drawn* is decided in Python too, in `spec/plot.py`, and tested
there: which shape, where every tick and point sits, what each label says. Bars
for names, lines for time, grouped or stacked bars or a heatmap when a comparison
runs across named buckets like weekdays, and a running total that stops at the
current minute rather than running past it. The page hands that plan to Chart.js
and decides nothing itself.

Two things sit on top. A chart can be **changed in place** - "make it monthly",
"only 2024" - which sends the current spec and the request back through the same
validate-run-draw path. And a saved chart is either **frozen**, keeping the
numbers it had, or **live**, re-running its spec every time it is shown, which is
what makes a chart pinned to my home screen keep up with what I played an hour
ago.

---

## Shows worth going to

https://github.com/user-attachments/assets/26d0aab9-85a0-4d4b-8e5b-afe474fa098d

Live Ticketmaster events joined to my own play history, so the list is not
"concerts near you" but concerts by people I actually listen to. The same
spec pipeline reads the request: "small metal bands near me this weekend"
becomes a `ShowFilter` - distance, date window, fan-size ceiling, genre, and how
well I have to know an artist for it to count.

That last one is a scale rather than a switch: **Anybody, Heard before, Listen
regularly, Favorites only, New to me.** It started as "only artists I listen to",
which collapsed a real spectrum into a yes or no.

Refusal is part of the design. A request for someone else's shows, or for data
this app does not hold, returns nothing rather than a guess - and because the
schema cannot express those things, that is a property of the grammar rather
than a rule the model is asked to follow.

The funnel icon lights when a filter is narrowing the list, derived from the
filter itself rather than a hand-written list of fields, so a filter added later
cannot quietly fail to show up.

---

## The weekly feed

Home is one week of listening, measured against the week before it: artists and
songs **climbing** or **cooling off**, artists and songs **back in rotation**,
what I **found and kept**, and what was **on repeat**.

Each of those is a rule with a floor under it, because a week of one listener is
a small sample. Movers count plays that cleared 30 seconds - a track wandering in
off a radio and being dropped should move nothing - and need at least four plays
in the prior week for a percentage to mean anything.

That floor is also why *back in rotation* exists. An artist I played once last
week and seven times this week had no prior week to take a percentage of, was not
new, and did not have a single song played enough to chart - so Michael Jackson
fell through every section on the screen. Back in rotation catches exactly that:
played before, quiet last week, heavy this week. The floors come from 26 fully
recorded weeks.

*Found and kept* is the fussiest rule. A new *name* is not a new *act* - Spotify
credits a collaboration as one comma-joined string, so "Flume, KUCKA" looks new
while Flume is played constantly - so a credit counts only when every act in it
is new. An artist who renames themselves is not a discovery either, which is why
a discovery has to bring unfamiliar song titles too.

---

## Capture

The export is complete and exact up to the day it was generated; everything since
is captured live, and the two union at read time under one rule - **the export
wins for every period it covers** - so live rows only survive past its high-water
mark and a future export silently upgrades them.

```
Spotify export     ->  Parquet (raw)   ->  DuckDB (transform)  ->  the app
   509k plays           8.6 MB              ~0.1s queries
Supabase capture   ->  Parquet (live)  ^
```

I started with a poller on recently-played and rebuilt it after measuring what it
returned: fifteen plays on a day I had listened for hours, one song played ten
times reported twice. Currently-playing was right every time I asked it. So
`supabase/capture.sql` asks what is playing every 30 seconds and writes a play
when the track changes, restarts or stops, with listening time measured from
playback progress rather than assumed from the track's length. Recently-played
stays on every 30 minutes as a backup.

It runs in the database - pg_cron, the `http` extension, secrets in Vault -
because my laptop is not online when I am walking around campus with headphones
in. The decision logic is one pure function, so every transition (a skip, a
repeat, a pause, a one-poll dropout) is tested by handing it two snapshots.

---

## How I measure the language features

Correctness is scored per field, with tolerances - "a few hours" is 150 miles or
250 depending who you ask, and failing a good answer for being differently good
teaches me nothing.

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

A held-out set is single-use. Reading its failures and fixing against them turns
it into a training set, so I quote these once and treat the earlier two files in
`evals/cases/` as training data now, whatever their filenames say.

**Model choice is measured per task, not assumed:**

| suite | gpt-5.4 | gpt-5.4-nano |
|---|---|---|
| shows | 99% | 54% |
| charts | 99% | 89% |

I found that gap the hard way. The endpoints defaulted to nano while every number
in my reports came from gpt-5.4, so "top 3 artists every year" failed eight times
out of eight in the browser while the held-out set read 100%. Endpoints now pin
the measured model.

Alongside that, 332 tests run offline in about 20 seconds and CI runs them plus
lint on every push. The history and corpus are gitignored, so the tests that read
them skip in CI - the local run with data is the full one.

### What measuring changed

**Prose could not fix defaults.** Requests mentioning no distance should leave it
unset; the first run scored 25%, and adding an instruction saying so made it
*worse* - 0% - because the model became consistently wrong instead of randomly
wrong. Models fill fields; they do not decline to. Making the field nullable, so
"not mentioned" is a value it can emit, took it to 100%.

**A single run is not a measurement.** The same code scored 100% then 98% on
consecutive runs with nothing changed. The figures here come from repeated runs,
and the CI floors sit well below the observed range on purpose.

**Mood is not in this data.** The avatar was going to infer mood from listening.
Measured: 12% of corpus artists carry any mood tag, 3% carry one usable at the
artist level, and behaviour signals spanned 1.74-2.35 across a whole day. So it
reacts instead of inferring - the current track's region has a state, decided
once. That is a claim about the music, not about me, and it is the honest version
of the feature.

---

## What is where

    src/musicshare/
      spec/filter.py     the show form; its field descriptions are the prompt
      spec/chart.py      the chart form - metrics, dimensions, series, ranking
      spec/vocab.py      385 frozen genre labels, and the enum built from them
      spec/name.py       naming schema for modes and regions
      spec/mood.py       the six avatar states, and why mood-from-listening was rejected
      spec/validate.py   semantic checks the schemas cannot express
      spec/generate.py   text -> spec; the only model call
      spec/apply.py      executes a show filter - no model involved
      spec/chartrun.py   executes a chart spec, in SQL, from menus only
      spec/plot.py       how a chart is drawn - shape, points, ticks, labels
      embed.py           artist tags -> 200-dimension space
      project.py         the fitted basemap and 2D positions, for drawing only
      regions.py         152 genre regions, their names and states
      modes.py           one listener as weighted clusters
      genres.py          artist -> region -> state, the 1.5MB serving table
      home.py            the weekly feed: movers, discoveries, back in rotation
      live.py            pulls captured plays down; now-playing; the live store
      history.py         one view over the export and the live rows
      media.py           uploads and the profile document
      shows.py           Ticketmaster events joined to local play history
      api/app.py         the server
    supabase/capture.sql the 30-second watcher, the backup, dedupe
    evals/cases/         golden sets and held-out sets
    profile.html         the interaction design, single file, no build step

Python 3.13, FastAPI, DuckDB over Parquet, Pydantic for every spec, Supabase
(Postgres, pg_cron, Storage) for capture and profile data, and one HTML file for
the front end - no build step, because the design is the thing I iterate on
fastest and a toolchain between me and it costs more than it gives.

## The Spotify constraint

Spotify closed extended API quota to individuals in May 2025 - it now requires a
registered business with 250k+ MAU. Development mode (25 whitelisted users) is
therefore permanent for me. That is the full API, so nothing here is degraded; it
just cannot scale, which is a limit I accepted deliberately rather than an
oversight.

The consequence worth designing for: the export needs Spotify exactly once, when
the file is generated. If API access disappeared entirely, every profile would
keep its map position, badges, matching and charts. Only freshness and pinning
depend on the live API.
