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

from typing import Literal

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


class ChartSpec(BaseModel):
    """A chart over one person's listening history."""

    # Genres come back as words, not enum members - see the note on ShowFilter.
    model_config = ConfigDict(use_enum_values=True)

    title: str = Field(
        default="",
        max_length=60,
        description=(
            "A short title in the user's own voice, as they might caption it on their "
            "profile - 'my 2am problem', 'the year I found shoegaze'. Not a description "
            "of the axes. Six words at most."
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
            "'hour_of_day' for questions about times of day ('2am', 'mornings'), "
            "'day_of_week' for weekdays and weekends, 'platform' for phone versus "
            "desktop. Otherwise the thing being ranked. "
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
    artists: list[str] = Field(
        default_factory=list,
        description=(
            "Restrict to these artists, exactly as the user wrote them, when the "
            "question is about specific ones. Empty otherwise. Never invent names and "
            "never expand a genre or mood into a list."
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
