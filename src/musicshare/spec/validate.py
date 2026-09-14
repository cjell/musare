"""Checks the schema cannot express.

Structured outputs guarantee a ShowFilter parses and that its numbers sit in
range. They cannot guarantee the spec makes sense - 400 miles for "walking
distance" is perfectly valid and perfectly wrong. Anything caught here fails
closed rather than running a query nobody asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

from musicshare.spec.chart import DATA_FIRST_YEAR, ORDERED, ChartSpec
from musicshare.spec.filter import ShowFilter
from musicshare.spec.mood import Moods
from musicshare.spec.name import Naming
from musicshare.spec.vocab import MAX_GENRES

MAX_ARTISTS = 12

# A name is letters, and the handful of marks real genre names use: "hip-hop",
# "r&b", "rock 'n' roll", "post-punk/new wave", "2000s pop". Allowlisting by
# character class rather than by byte keeps accented names workable while
# refusing brackets, braces, pipes, backticks and everything else that only
# appears when something is trying to be read as code rather than as a label.
NAME_MARKS = set(" -'&/.+")


@dataclass(frozen=True)
class SpecProblem:
    field: str
    message: str


def validate(spec: ShowFilter) -> list[SpecProblem]:
    """Empty list means the spec is safe to execute."""
    # A refusal carries whatever defaults the model left behind, and the schema
    # says those are ignored. Judging them turns a correct refusal into an error.
    if not spec.understood:
        return []
    problems: list[SpecProblem] = []

    if spec.max_fans is not None and spec.max_fans < 1:
        problems.append(SpecProblem("max_fans", "must be at least 1"))
    if spec.min_fans is not None and spec.min_fans < 1:
        problems.append(SpecProblem("min_fans", "must be at least 1"))
    if spec.max_fans is not None and spec.min_fans is not None and spec.min_fans > spec.max_fans:
        problems.append(SpecProblem("min_fans", f"min {spec.min_fans} exceeds max {spec.max_fans}"))

    # A model asked for "shows by every artist in my library" would happily
    # produce a list of hundreds; that is a mis-parse, not a request.
    if len(spec.artists) > MAX_ARTISTS:
        problems.append(SpecProblem("artists", f"{len(spec.artists)} named, limit {MAX_ARTISTS}"))
    if any(len(a) > 120 for a in spec.artists):
        problems.append(SpecProblem("artists", "an artist name is implausibly long"))

    # The schema guarantees every genre is one of the 385 words. It says nothing
    # about how many, and a model that has read a sentence as a genre list is
    # asking for something the user did not.
    if len(spec.genres) > MAX_GENRES:
        problems.append(SpecProblem("genres", f"{len(spec.genres)} named, limit {MAX_GENRES}"))

    return problems


def validate_chart(spec: ChartSpec) -> list[SpecProblem]:
    """Charts that would parse and draw, but say nothing true."""
    if not spec.understood:
        return []
    problems: list[SpecProblem] = []

    if spec.dimension == "date" and spec.grain is None:
        problems.append(SpecProblem("grain", "a chart over time needs a bucket size"))
    if spec.year is not None and spec.year < DATA_FIRST_YEAR:
        problems.append(SpecProblem("year", f"history starts in {DATA_FIRST_YEAR}"))
    if len(spec.artists) > MAX_ARTISTS:
        problems.append(SpecProblem("artists", f"{len(spec.artists)} named, limit {MAX_ARTISTS}"))
    if len(spec.genres) > MAX_GENRES:
        problems.append(SpecProblem("genres", f"{len(spec.genres)} named, limit {MAX_GENRES}"))

    # Two lines need a shared axis to run along. On a ranked dimension there is
    # none - the third bar of one artist and the third bar of another are
    # different tracks - so this is not a chart that can be drawn wrong, it is a
    # chart that cannot be drawn. Failing closed says so; drawing it would hand
    # back something that looks like an answer.
    if spec.series != "none" and spec.dimension not in ORDERED:
        problems.append(
            SpecProblem("series", f"cannot compare along '{spec.dimension}', which has no order")
        )
    # Ranking inside a period needs a period to rank inside, and something to
    # rank. Both come from fields the model fills in separately, so this is the
    # one combination it can get half-right.
    if spec.per_period:
        if spec.series == "none":
            problems.append(SpecProblem("per_period", "nothing named to rank in each period"))
        if spec.dimension not in ORDERED:
            problems.append(
                SpecProblem("per_period", f"'{spec.dimension}' is not a period to rank inside")
            )

    # There is deliberately no "genre cannot be both the axis and the split"
    # check. It reads like a real rule and is unreachable: 'genre' is not an
    # ordered dimension, so the clause above has already rejected any spec it
    # would have caught.

    return problems


def validate_naming(naming: Naming, k: int, artists: set[str] | None = None) -> list[SpecProblem]:
    """Names that parse but do not name k distinct things.

    The schema can hold the length of a name and the range of an index. It
    cannot say that every mode got exactly one, that no two are the same, or
    that what came back is a name rather than a sentence - and those are the
    failures that matter, because a duplicate name is the defect this whole step
    exists to remove.

    `artists` is the set of names appearing in the profile. A smaller model asked
    to name the Drake mode answered "drake rap", which is a real failure the
    schema cannot see: naming a mode after one of its members describes nothing
    and reads as a mistake to the one person who knows the mode is not just him.
    """
    if not naming.understood:
        return []
    problems: list[SpecProblem] = []

    got = [n.index for n in naming.names]
    if len(got) != k:
        problems.append(SpecProblem("names", f"{len(got)} names for {k} modes"))
    if len(set(got)) != len(got):
        problems.append(SpecProblem("index", "a mode was named more than once"))
    if any(i >= k for i in got):
        problems.append(SpecProblem("index", f"names a mode above {k - 1}"))
    # There is deliberately no separate "covers every mode" check. k distinct
    # indices that are all below k are necessarily exactly 0..k-1, so the three
    # checks above already imply coverage; a fourth one was unreachable.

    seen: set[str] = set()
    for n in naming.names:
        # An enum member, so the shape checks below are belt and braces now -
        # the schema already guarantees the word is one of the 446. Distinctness
        # is the part it cannot enforce.
        name = str(getattr(n.name, "value", n.name)).strip()
        if not name:
            problems.append(SpecProblem("name", f"mode {n.index} came back empty"))
            continue
        if name.lower() in seen:
            problems.append(SpecProblem("name", f"'{name}' is used twice"))
        seen.add(name.lower())
        # No word-count or character rules any more: a name is an enum member,
        # so its shape is guaranteed by the schema. Enforcing "three words at
        # most" here rejected "east coast hip hop", which is a real label from
        # the vocabulary - the rule outlived the free text it was written for.
        if artists and name.lower() in artists:
            problems.append(SpecProblem("name", f"'{name}' is an artist, not a genre"))

    return problems


def validate_moods(moods: Moods, k: int) -> list[SpecProblem]:
    """Every region placed exactly once.

    Deliberately no distinctness check, which is the opposite of validate_naming.
    Two regions cannot share a name because they are two places; dozens of regions
    share a state because a state is a grouping. Reusing the naming validator here
    would have rejected every correct answer.
    """
    if not moods.understood:
        return []
    problems: list[SpecProblem] = []

    got = [m.index for m in moods.moods]
    if len(got) != k:
        problems.append(SpecProblem("moods", f"{len(got)} states for {k} regions"))
    if len(set(got)) != len(got):
        problems.append(SpecProblem("index", "a region was given two states"))
    if any(i >= k for i in got):
        problems.append(SpecProblem("index", f"names a region above {k - 1}"))
    return problems
