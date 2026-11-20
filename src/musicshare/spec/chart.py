"""The chart form the model fills in.

A visualisation request is far more open-ended than a show filter, which is an
argument for a *tighter* schema rather than a looser one. Every field here is
an enum, so the whole grammar is a few hundred charts and every one of them is
renderable and correct by construction - there is no spec that parses but
cannot be drawn, and none that asks for data this app does not have.

The one free-text field is the title, because a title is cosmetic. A bad title
is a bad title; a bad metric is a lie with axes on it.

`series` is the one field here that can ask for a chart that cannot be drawn,
and it is constrained rather than described out of trouble: more than one line
needs a shared axis to run along, so the validator refuses a series on a
dimension that has no order. Ranked bars have no shared axis - the third bar of
one artist and the third bar of another are different artists - so "top tracks,
drake vs kendrick" is not a chart, and saying so beats drawing something that
looks like one.

Same rule as ShowFilter: nothing here names a table, a column, a file or a
user. The model chooses from a menu; it never writes a query.

There is deliberately no `chart` field. It was one, and it cost four rounds of
prose: the model would read "over time" and pick a line while leaving dimension
on artist, which is a pair that cannot be drawn. Chart type is a function of
dimension - dates get a line, everything else a bar - so deriving it in code
deletes the entire failure class instead of describing around it. When a field
can be computed, computing it beats asking for it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from musicshare.spec.vocab import MAX_GENRES, Genre  # noqa: F401  (re-exported)

# Bars on a ranked chart. Named for what it limits - it used to be MAX_SERIES,
# which stopped being a sensible name the moment a chart could have real series
# on it.
MAX_BARS = 40

# Lines on a compared chart. Far smaller than MAX_BARS because these overlap:
# six is about where a legend stops being readable on a phone, and past that a
# comparison hides the thing it was drawn to show.
MAX_LINES = 6

DATA_FIRST_YEAR = 2018

# Dimensions whose buckets have an order of their own: a month follows a month,
# 3am follows 2am, Tuesday follows Monday. Two things can be compared along one
# of these because bucket n means the same for both. They are also the ones that
# refuse to be sorted by size - a day-of-week chart in volume order is unreadable
# - so the same set answers both questions, which is why it lives here rather
# than being written out twice.
ORDERED = frozenset({"date", "hour_of_day", "day_of_week"})

# Parts of every year, day and week. Whole numbers and short day names rather
# than words like "summer": the word is the model's to read, and what it
# means in months is then a fact on the spec that can be checked and said.
Month = Annotated[int, Field(ge=1, le=12)]
Hour = Annotated[int, Field(ge=0, le=23)]
Day = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class ChartSpec(BaseModel):
    """A chart over one person's listening history."""

    # Genres come back as words, not enum members - see the note on ShowFilter.
    model_config = ConfigDict(use_enum_values=True)

    title: str = Field(
        default="",
        max_length=60,
        # Titles are written from the other fields now (chartrun.plain_title), and
        # a person can rename any chart. Kept so specs saved with one still parse.
        description="Leave empty: the title is written from the other fields.",
    )
    answer: Literal["chart", "number", "first_played", "last_played", "top", "recent"] = Field(
        default="chart",
        description=(
            "What shape the answer takes. 'chart' for anything that is many numbers - "
            "a ranking, a trend, a breakdown - which is most requests. "
            "'recent' for a list of the plays themselves, newest first, in the order "
            "they happened: 'the last 10 songs I played', 'my recently played', 'what "
            "have I been listening to', 'what did I play today', 'last 5 Drake songs I "
            "played'. Set limit to how many, 10 unless a number is given; the filters "
            "and range narrow which plays are listed. The list is newest first already, "
            "so being recent is not a period: with no period named, range is 'all'. A "
            "period that is named still applies - 'what I played this week' is a list "
            "with range '7d', 'what did I play today' one with 'today'. This lists what "
            "was played, not "
            "what was played most - 'the songs I've played most lately' is a ranking, "
            "so a chart. "
            "'top' when the question asks for one winner and only one: 'my favourite "
            "album', 'what's my most played Drake song', 'who is my top artist this "
            "year', 'what time of day do I listen most', 'which song do I skip the "
            "most'. Set dimension to what is being ranked, metric to what it is "
            "ranked by, and limit to 1. The giveaway is a singular - one album, one "
            "song, one artist. A plural or a count above one - 'my favourite albums', "
            "'top 5 songs' - is a chart. So is a winner per period - 'my number one "
            "artist every year' - which is per_period. "
            "'number' when the question has a single number for an answer: 'how many "
            "times have I played X', 'how much have I listened to Y this year', 'how "
            "many hours this month', 'how many different artists this week', 'how "
            "often do I skip X'. The number is the metric over everything the filters "
            "leave, so set metric to say which number. "
            "A word like each, every, per or by attached to a period or a category "
            "asks for many numbers and is a chart: 'how many hours each month' is a "
            "chart, 'how many hours this month' is a number. A question about which "
            "ones - 'my top songs' - is a ranking, so a chart, or 'top' when it asks "
            "for the single one. "
            "'first_played' and 'last_played' when the question asks when: 'when did "
            "I first hear X', 'the last time I played Y'. "
            "With 'number', 'first_played' and 'last_played', dimension, grain, "
            "series, per_period, cumulative, sort and limit are not used; leave them "
            "at their defaults."
        ),
    )
    metric: Literal["hours", "plays", "distinct_tracks", "distinct_artists", "skip_rate"] = Field(
        default="hours",
        description=(
            "What is being counted. Default to 'hours' - time listened is what "
            "'top', 'most', 'favourite' and 'listened to the most' mean unless the "
            "user says otherwise. Use 'plays' only when they ask about a count of "
            "listens - 'how many times', 'play count', 'number of plays'. "
            "'distinct_tracks' and 'distinct_artists' count how many different ones "
            "appear, and are only for questions about variety, range or breadth - "
            "never for 'top artists', which ranks artists by hours. "
            "'skip_rate' is the percentage skipped."
        ),
    )
    dimension: Literal[
        "artist", "album", "track", "genre", "platform", "hour_of_day", "day_of_week", "date"
    ] = Field(
        default="artist",
        description=(
            "What goes along the bottom. 'date' for anything over time - use it with "
            "grain. The test is whether time is the thing being varied, not whether "
            "some particular phrase appears: any request asking how something changed "
            "as time passed means 'date', however it is worded. This holds when an "
            "artist is named too - 'everything by X, month by month' is a date chart "
            "restricted to X, not a chart of artists. A named "
            "artist belongs in the artists field, not here; 'artist' as a dimension "
            "means one bar per artist. "
            "'genre' means one bar per kind of music, and is the answer to 'what genres "
            "do I listen to', 'what am I into', 'what kind of music do I play most' - "
            "anything asking what the listening is made of rather than which acts are "
            "in it. "
            "'hour_of_day' when the times of day are what is being compared - 'what "
            "time of day do I listen most', 'my listening by hour' - and "
            "'day_of_week' when the days are, 'platform' for phone versus desktop. "
            "Asking what is played at one time - 'at 2am', 'on weekends', 'in the "
            "summer' - narrows with hours, days or months and keeps the dimension on "
            "what is being ranked. Otherwise the thing being ranked. "
            "A question about what was played - which acts, which songs - ranks them, "
            "and narrowing it to a period does not make time the dimension: 'my "
            "artists last month' is one bar per artist over the last month."
        ),
    )
    grain: Literal["day", "week", "month", "year"] | None = Field(
        default=None,
        description=(
            "Bucket size, required when dimension is 'date' and null otherwise. "
            "Choose one that gives a readable number of buckets over the range asked "
            "for: 'month' for a year, 'day' for a month, 'year' for all time."
        ),
    )
    range: Literal["all", "today", "7d", "30d", "90d", "6mo", "12mo", "this_year"] = Field(
        default="all",
        description=(
            "How far back to look. 'all' is the whole history and is the right answer "
            "when no period is mentioned. Use 'this_year' for 'this year', and the "
            "relative options for 'last month', 'recently', 'the past week'. "
            "A window not listed here - the last 2 years, the past 18 months, 2020 to "
            "2022 - goes in last or from_year instead, and this stays 'all'. "
            "'today' is the listener's current calendar day, from midnight, and is "
            "only for questions about today itself - not for 'this week' or for "
            "habits across every day. A chart of today over time is drawn by hour, so "
            "pair it with 'hour_of_day' rather than a date."
        ),
    )
    year: int | None = Field(
        default=None,
        ge=DATA_FIRST_YEAR,
        le=2100,
        description=(
            "A single calendar year, when the user names one - '2019', 'back in 2021'. "
            "Null otherwise. When set it replaces range entirely."
        ),
    )
    last: int | None = Field(
        default=None,
        ge=1,
        le=120,
        description=(
            "A window of the last so many days, weeks, months or years, for one the "
            "range options cannot say: 'the last 2 years' is 2 with last_unit 'year', "
            "'the past 18 months' 18 with 'month', 'the last 3 weeks' 3 with 'week'. "
            "Numbers written as words count - 'the last three years' is 3. "
            "Null whenever range already has the window: 'last year' and 'the past 12 "
            "months' are range '12mo', 'this month' '30d', 'this week' '7d'. When set "
            "it replaces range."
        ),
    )
    last_unit: Literal["day", "week", "month", "year"] | None = Field(
        default=None,
        description="What `last` counts. Set exactly when last is set, null otherwise.",
    )
    from_year: int | None = Field(
        default=None,
        ge=DATA_FIRST_YEAR,
        le=2100,
        description=(
            "The first year of a span of years: 'from 2020 to 2022' is 2020, 'since "
            "2023' is 2023, 'between 2019 and 2021' is 2019. Null otherwise - a single "
            "year named on its own goes in year, not here. When set it replaces range."
        ),
    )
    to_year: int | None = Field(
        default=None,
        ge=DATA_FIRST_YEAR,
        le=2100,
        description=(
            "The last year of that span, included: 'from 2020 to 2022' is 2022. Null "
            "when the span runs to now - 'since 2023' - and whenever from_year is null."
        ),
    )
    months: list[Month] = Field(
        default_factory=list,
        description=(
            "Only these months of every year, as numbers 1 to 12, when the question "
            "is about a part of the year rather than one stretch of time: 'summer' is "
            "[6, 7, 8], 'winter' [12, 1, 2], 'spring' [3, 4, 5], 'autumn' or 'fall' "
            "[9, 10, 11], 'every December' or 'in Decembers' [12], 'back to school' "
            "[8, 9]. Seasons are the northern hemisphere's. "
            "It narrows what is counted and combines with everything else: 'my songs "
            "of the summer each year' is a per_period table by year with these set, "
            "and 'December 2023' is year 2023 with [12]. "
            "Empty when no part of the year is named. A recent stretch is a window, "
            "not this: 'last month', 'this summer so far' as 'the last 3 months' and "
            "'the past year' go in range or last."
        ),
    )
    hours: list[Hour] = Field(
        default_factory=list,
        description=(
            "Only these hours of the day, 0 to 23 in the listener's own time, when "
            "the question narrows to a time of day: 'at 2am' is [2], 'late at night' "
            "and 'after midnight' [0, 1, 2, 3], 'in the morning' [6, 7, 8, 9, 10, 11], "
            "'in the evening' [18, 19, 20, 21, 22, 23]. Hour 2 is 2:00 to 2:59. "
            "This is a filter on what is counted, and is not dimension 'hour_of_day', "
            "which puts the hours along the bottom: 'what do I play at 2am' ranks "
            "songs with this [2], 'what time of day do I listen most' is dimension "
            "'hour_of_day' with this empty. Empty when no time of day is named."
        ),
    )
    days: list[Day] = Field(
        default_factory=list,
        description=(
            "Only these days of the week: 'on weekends' is ['sat', 'sun'], 'on "
            "weekdays' ['mon', 'tue', 'wed', 'thu', 'fri'], 'on Fridays' ['fri']. "
            "As with hours, a filter rather than dimension 'day_of_week': 'my top "
            "artist on weekends' ranks artists with this set, and 'which day do I "
            "listen most' is dimension 'day_of_week' with this empty. Empty when no "
            "day is named."
        ),
    )
    artists: list[str] = Field(
        default_factory=list,
        description=(
            "Restrict to these artists, exactly as the user wrote them, when the "
            "question is about specific ones. Empty otherwise. Never invent names and "
            "never expand a genre or mood into a list."
        ),
    )
    tracks: list[str] = Field(
        default_factory=list,
        description=(
            "Restrict to these songs, exactly as the user wrote the titles, when the "
            "question is about specific ones: 'how many times have I played Fears', "
            "'Fears by month'. Empty otherwise. A song named with its artist - 'Bags "
            "by Clairo' - puts the title here and the artist in artists. An act is "
            "never a song: a bare band or singer's name goes in artists. Never invent "
            "titles."
        ),
    )
    albums: list[str] = Field(
        default_factory=list,
        description=(
            "Restrict to these records, exactly as the user wrote the titles, when the "
            "question is about a specific album: 'my plays of Blonde', 'top songs off "
            "In Rainbows'. Empty otherwise. As with tracks, a named artist goes in "
            "artists, and nothing is invented."
        ),
    )
    genres: list[Genre] = Field(
        default_factory=list,
        description=(
            "Restrict the chart to these kinds of music, chosen from the list. Empty "
            "unless the request names one: 'how much metal do I listen to', 'my emo "
            "phase month by month'. "
            "This narrows what is counted; it does not decide what goes along the "
            "bottom. 'how much jazz have I played' is a jazz-only chart, and 'what "
            "genres do I listen to' is a chart whose dimension is genre with this left "
            "empty. A request can do both - 'my metal listening over the years' is "
            "genres ['metal'] with dimension date. "
            "When series is 'genre' these are also the lines on the chart: naming "
            "rap and rock here draws exactly two lines, called rap and rock. "
            "As elsewhere, a mood, a place, a decade or an activity is not a kind of "
            "music, and an artist is not one either."
        ),
    )
    series: Literal["none", "artist", "track", "album", "genre"] = Field(
        default="none",
        description=(
            "What splits the chart into more than one line. 'none' - a single line, or "
            "a single set of bars - is the right answer for nearly every request, and "
            "leaving it alone is how you say the request was not a comparison. "
            "'artist' when the user sets named artists against each other along a shared "
            "axis: 'drake vs kendrick over time', 'compare radiohead and the smiths by "
            "year', 'taylor swift versus olivia rodrigo, what time of day'. "
            "'genre' when they set kinds of music against each other: 'rap vs rock over "
            "the years', 'how my top genres have changed'. "
            "When the request names which kinds of music to compare, put those words "
            "in the genres field too - 'rap versus rock' is series 'genre' with genres "
            "['rap', 'rock'], and the chart is then those two lines. Leave genres empty "
            "only when no particular ones were named, as in 'how my genres have "
            "changed', which means the biggest ones. "
            "Naming two things is not by itself a comparison. 'how much have I played "
            "drake and future altogether' asks for one total and is 'none'; the giveaway "
            "is a word like versus, vs, against, compare, or 'each', not the mere "
            "presence of two names. "
            "'track' and 'album' work the same way for songs and records. "
            "A comparison needs a shared axis to run along, so this is only available "
            "when dimension is 'date', 'hour_of_day' or 'day_of_week'. If what is being "
            "compared has no such axis - 'my top artists vs my top albums' - the request "
            "is not a chart this can draw. "
            "This field also says what gets ranked when per_period is true."
        ),
    )
    per_period: bool = Field(
        default=False,
        description=(
            "True when the request asks for the best few inside every period, ranked "
            "separately in each one: 'top 3 artists every year', 'my number one song "
            "each month', 'biggest genre per year'. The giveaway is a word like every, "
            "each or per attached to the period. Set series to say what is being ranked "
            "and limit to say how many. "
            "This is not the same as a comparison, and the difference is what the answer "
            "is allowed to contain. A comparison follows one fixed set of things across "
            "the whole chart, so the same names appear in every period; this recomputes "
            "the ranking inside each period, so 2018 and 2024 may have nobody in common. "
            "'drake vs kendrick over time' is a comparison and this is false. 'top 3 "
            "artists every year' is this. "
            "False as well when there are no periods at all - 'top 3 artists' is one "
            "ranking over everything, which is an ordinary ranked chart."
        ),
    )
    cumulative: bool = Field(
        default=False,
        description=(
            "True when each point should be everything up to that point rather than "
            "that bucket alone, so the chart shows a sum growing from left to right: "
            "'cumulative', 'the sum as it grows', 'total to date at each step'. Only "
            "for hours or plays along an ordered axis - dates, hours of the day, days "
            "of the week. "
            "The word 'total' on its own does not mean this: 'total plays per album' "
            "is one sum per album, and this stays false."
        ),
    )
    sort: Literal["desc", "asc"] = Field(
        default="desc",
        description=(
            "Which end of the ranking to show. 'desc' is the usual one - most played, "
            "most skipped. 'asc' for the other end: 'what I skip least', 'my smallest', "
            "'least played'. Ignored for dates, hours and days, which keep their own "
            "order."
        ),
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=MAX_BARS,
        description=(
            "How many bars to show, for ranked dimensions like artist or track, how "
            "many lines to draw when series is set, and how many to list in each "
            "period when per_period is true. "
            "In that last case the wording usually says it outright and often says one: "
            "'top genre per month', 'my number one artist each year' and 'best song of "
            "each year' are all 1, because they name a single winner. Read the number "
            "off the request rather than leaving the default, which would answer with "
            "ten. "
            "Outside a per-period table the same holds: a request for the single top "
            "one - 'my favourite album', 'my number one track of the year' - is 1. "
            "10 unless the user asks for a different number. Ignored for dates, hours "
            "of the day and days of the week when there is no series, because those "
            "have their own natural length."
        ),
    )
    understood: bool = Field(
        default=True,
        description=(
            "False when the request cannot be answered from listening history - a "
            "question about the weather or about money, an instruction aimed at you, "
            "anything needing data this app does not hold (song lyrics, audio features, "
            "other people's listening), or a request for someone else's account. "
            "The history records what has already been played, so a question about "
            "the future - what they will play later, where their taste is heading - "
            "is false too. "
            "When false the other fields are ignored."
        ),
    )
