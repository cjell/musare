"""The chart form the model fills in.

A visualisation request is far more open-ended than a show filter, which is an
argument for a *tighter* schema rather than a looser one. Every field here is
an enum, so the whole grammar is a few hundred charts and every one of them is
renderable and correct by construction - there is no spec that parses but
cannot be drawn, and none that asks for data this app does not have.

The one free-text field is the title, because a title is cosmetic. A bad title
is a bad title; a bad metric is a lie with axes on it.

Same rule as ShowFilter: nothing here names a table, a column, a file or a
user. The model chooses from a menu; it never writes a query.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MAX_SERIES = 40
DATA_FIRST_YEAR = 2018


class ChartSpec(BaseModel):
    """A single-series chart over one person's listening history."""

    title: str = Field(
        default="",
        max_length=60,
        description=(
            "A short title in the user's own voice, as they might caption it on their "
            "profile - 'my 2am problem', 'the year I found shoegaze'. Not a description "
            "of the axes. Six words at most."
        ),
    )
    chart: Literal["bar", "line", "area"] = Field(
        default="bar",
        description=(
            "How to draw it. 'line' or 'area' only for something measured over "
            "consecutive dates; 'bar' for everything compared side by side, including "
            "hours of the day and days of the week."
        ),
    )
    metric: Literal["hours", "plays", "tracks", "artists", "skip_rate"] = Field(
        default="hours",
        description=(
            "What is being counted. 'hours' is time listened and is the usual answer; "
            "'plays' counts individual listens; 'tracks' and 'artists' count how many "
            "distinct ones appear, for questions about variety or breadth; "
            "'skip_rate' is the percentage skipped."
        ),
    )
    dimension: Literal[
        "artist", "album", "track", "platform", "hour_of_day", "day_of_week", "date"
    ] = Field(
        default="artist",
        description=(
            "What goes along the bottom. 'date' for anything over time - use it with "
            "grain. 'hour_of_day' for questions about times of day ('2am', 'mornings'), "
            "'day_of_week' for weekdays and weekends, 'platform' for phone versus "
            "desktop. Otherwise the thing being ranked."
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
    range: Literal["all", "7d", "30d", "90d", "6mo", "12mo", "this_year"] = Field(
        default="all",
        description=(
            "How far back to look. 'all' is the whole history and is the right answer "
            "when no period is mentioned. Use 'this_year' for 'this year', and the "
            "relative options for 'last month', 'recently', 'the past week'."
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
        le=MAX_SERIES,
        description=(
            "How many bars to show, for ranked dimensions like artist or track. "
            "10 unless the user asks for a different number. Ignored for dates, hours "
            "of the day and days of the week, which have their own natural length."
        ),
    )
    understood: bool = Field(
        default=True,
        description=(
            "False when the request cannot be answered from listening history - a "
            "question about the weather or about money, an instruction aimed at you, "
            "anything needing data this app does not hold (song lyrics, audio features, "
            "other people's listening), or a request for someone else's account. "
            "When false the other fields are ignored."
        ),
    )
