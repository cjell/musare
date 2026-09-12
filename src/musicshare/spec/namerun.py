"""Turn a fitted profile into named modes.

Same split as everywhere else: the model describes, this executes. Here
"executing" is only writing a string onto a Mode, which is why the risk lives in
the input rather than the output.

The input is not user text. It is Last.fm tags and artist names, and anybody can
write a Last.fm tag - so the block handed to the model is assembled rather than
concatenated: control characters and newlines stripped, every field length-capped,
a fixed number of tags and artists per mode. A tag cannot open a new "Mode 7"
section or append a line that reads like an instruction, because it cannot
contain a newline by the time it gets there. The schema and `validate_naming`
cover what comes back; this covers what goes in.

Names never replace the tag-derived labels, they sit beside them. A model that
returns "emo rap" for the trap mode has made the profile readable; a model that
returns something useless has not destroyed the evidence of what the mode
actually is.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from musicshare.modes import TasteProfile
from musicshare.spec.generate import NAMES, REGIONS, Generated, generate
from musicshare.spec.name import Naming
from musicshare.spec.validate import SpecProblem, validate_naming

log = logging.getLogger(__name__)

# Shows and charts answer a person who is waiting, so they run on the small fast
# model. Naming runs once when a profile is fitted, which buys a better one for
# nothing. Measured on a real six-mode profile: gpt-5.4-nano failed validation
# 3/3, concatenating tags into "indie pop rnb singer pop"; mini passed 3/3 but
# named the Drake mode "drake rap" once; gpt-5.4 passed 6/6.
#
# It is not deterministic, and the way it varies is worth knowing. Over six runs
# the three tight modes came back identical every time (shoegaze, emo rap,
# bedroom pop) while the broad 26% mode - Frank Ocean next to Taylor Swift next
# to Lil Yachty - drew four different names. Naming stability tracks how coherent
# the cluster is, so disagreement here is a reading on the mode rather than noise
# in the model. It is also why a name, once given, is kept: see `name`.
NAME_MODEL = "gpt-5.4"

TAGS_SHOWN = 8
ARTISTS_SHOWN = 8
MAX_TAG_CHARS = 40
MAX_ARTIST_CHARS = 60

# Anything that is not printable text on one line. Stripping these is what makes
# the assembled block structurally honest - a tag cannot become a new section.
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
_WS = re.compile(r"\s+")


def _flat(text: str, limit: int) -> str:
    return _WS.sub(" ", _CONTROL.sub(" ", text or "")).strip()[:limit]


def as_input(p: TasteProfile) -> str:
    """The profile as a block of text for the model. Data, never instructions."""
    lines = []
    for i, m in enumerate(p.modes):
        # Each value is quoted. Stripping newlines already stops a tag opening a
        # new section, but "Mode 9 - 99% of listening" sitting bare in a list of
        # tags still reads like structure; inside quotes it reads like the string
        # it is.
        tags = ", ".join(f'"{_flat(t, MAX_TAG_CHARS)}"' for t in m.tags[:TAGS_SHOWN] if t.strip())
        artists = ", ".join(
            f'"{_flat(a, MAX_ARTIST_CHARS)}"' for a in m.members[:ARTISTS_SHOWN] if a.strip()
        )
        lines.append(
            f"Mode {i} - {m.share:.0%} of listening\n  tags: {tags}\n  most played: {artists}"
        )
    return "\n\n".join(lines)


def _artist_names(p: TasteProfile) -> set[str]:
    return {a.strip().lower() for m in p.modes for a in m.members if a.strip()}


def name_modes(
    p: TasteProfile, model: str = NAME_MODEL
) -> tuple[Naming, list[SpecProblem], Generated]:
    """Ask for names. Returns what came back and what is wrong with it."""
    g = generate(as_input(p), task=NAMES, model=model)
    naming = g.spec
    assert isinstance(naming, Naming)
    problems = validate_naming(naming, len(p.modes), artists=_artist_names(p))
    if problems:
        log.warning("naming rejected: %s", "; ".join(f"{x.field}: {x.message}" for x in problems))
    return naming, problems, g


def apply_names(p: TasteProfile, naming: Naming) -> TasteProfile:
    """Write validated names onto the profile. Call only on a clean validation."""
    by_index = {n.index: str(n.name.value).strip() for n in naming.names}
    for i, m in enumerate(p.modes):
        if i in by_index:
            m.name = by_index[i]
    return p


def name(
    p: TasteProfile, model: str = NAME_MODEL, force: bool = False
) -> tuple[TasteProfile, list[SpecProblem]]:
    """Name the modes if the result holds up, and leave them alone if it does not.

    Failing closed matters more here than it looks. The tag labels are already
    serviceable; a half-applied naming where two modes share a name would be
    strictly worse than no naming at all.

    An already-named profile is returned untouched, because the call is not
    deterministic and a mode that is "pop soul" today and "sad girl pop" tomorrow
    is worse than one with no name at all - it is supposed to be a thing the
    listener recognises about themselves. A refit produces fresh modes with no
    names and so gets named again; `force` is for deliberately re-rolling.
    """
    if not force and p.modes and all(m.name for m in p.modes):
        return p, []
    naming, problems, _ = name_modes(p, model=model)
    if problems or not naming.understood:
        return p, problems
    return apply_names(p, naming), []


# ----------------------------------------------------------------- the atlas


def as_regions_input(regions: list[Any]) -> str:
    """The whole atlas as one block. Eighty regions, one call.

    Naming them together is not a saving, it is the requirement: two places
    cannot share a name, and a model naming one region at a time has no way to
    know what it already used. The same reason modes are named together.
    """
    lines = []
    for i, r in enumerate(regions):
        tags = ", ".join(f'"{_flat(t, MAX_TAG_CHARS)}"' for t in r.tags[:TAGS_SHOWN] if t.strip())
        who = ", ".join(
            f'"{_flat(a, MAX_ARTIST_CHARS)}"' for a in r.exemplars[:ARTISTS_SHOWN] if a.strip()
        )
        lines.append(f"Region {i} - {r.n_artists} artists\n  tags: {tags}\n  best known: {who}")
    return "\n\n".join(lines)


def name_regions(
    regions: list[Any], model: str = NAME_MODEL
) -> tuple[Naming, list[SpecProblem], Generated]:
    g = generate(as_regions_input(regions), task=REGIONS, model=model)
    naming = g.spec
    assert isinstance(naming, Naming)
    problems = validate_naming(naming, len(regions))
    if problems:
        log.warning(
            "region naming rejected: %s",
            "; ".join(f"{x.field}: {x.message}" for x in problems[:6]),
        )
    return naming, problems, g


def _resolve_duplicates(atlas: Any, chosen: dict[int, str]) -> dict[int, str]:
    """Give a repeated label to one region and something true to the other.

    With free text a duplicate meant the model was confused and failing closed
    was right. With a closed vocabulary it is arithmetic: 152 regions drawing
    from 385 labels will collide, and refusing to name anything because two
    regions both looked like "broadway" would mean never naming anything.

    The first region to claim a label keeps it; the other falls back to its own
    highest-ranked tag that is still free, and to its tag label if none is. Index
    order, so the outcome is the same every run.
    """
    from musicshare.spec.vocab import GENRES

    allowed = set(GENRES)
    taken: set[str] = set()
    out: dict[int, str] = {}
    for i, r in enumerate(atlas.regions):
        want = chosen.get(i)
        if want and want not in taken:
            out[i] = want
            taken.add(want)
            continue
        alt = next((t for t in r.tags if t in allowed and t not in taken), None)
        if alt:
            log.info("region %d: %r was taken, using %r", i, want, alt)
            out[i] = alt
            taken.add(alt)
        elif want:
            # Nothing left that fits. A duplicate label beats no label.
            log.warning("region %d: keeping duplicate %r", i, want)
            out[i] = want
    return out


def name_atlas(
    atlas: Any, model: str = NAME_MODEL, force: bool = False
) -> tuple[Any, list[SpecProblem]]:
    """Name every region, or none of them.

    Fails closed for the same reason the modes do, and harder: the atlas is shared
    by every listener, so a half-named one is a map where some places have names
    and others do not, for everybody, until someone notices.
    """
    if not force and atlas.regions and all(r.name for r in atlas.regions):
        return atlas, []
    naming, problems, _ = name_regions(atlas.regions, model=model)
    if not naming.understood:
        return atlas, problems
    # A duplicate is the one problem that is resolved rather than refused.
    blocking = [p for p in problems if "used twice" not in p.message]
    if blocking:
        return atlas, blocking
    by_index = _resolve_duplicates(
        atlas, {n.index: str(n.name.value).strip() for n in naming.names}
    )
    for i, r in enumerate(atlas.regions):
        if i in by_index:
            r.name = by_index[i]
    return atlas, []


# ------------------------------------------------------------------- moods


def as_moods_input(regions: list[Any]) -> str:
    """The atlas again, for sorting into states rather than naming."""
    lines = []
    for i, r in enumerate(regions):
        tags = ", ".join(f'"{_flat(t, MAX_TAG_CHARS)}"' for t in r.tags[:TAGS_SHOWN] if t.strip())
        who = ", ".join(
            f'"{_flat(a, MAX_ARTIST_CHARS)}"' for a in r.exemplars[:ARTISTS_SHOWN] if a.strip()
        )
        name = _flat(getattr(r, "display", "") or r.label, MAX_TAG_CHARS)
        lines.append(f'Region {i} - "{name}"\n  tags: {tags}\n  best known: {who}')
    return "\n\n".join(lines)


def mood_atlas(atlas: Any, model: str = NAME_MODEL, force: bool = False) -> tuple[Any, list[Any]]:
    """Give every region a state, or leave them all alone.

    Fails closed like the naming does, and for a sharper reason: the avatar is
    the one thing on the profile that strangers watch change. A half-assigned
    atlas would leave some music silently falling through to the neutral state,
    which looks like the feature being broken rather than the listener being
    between songs.
    """
    from musicshare.spec.generate import MOODS
    from musicshare.spec.validate import validate_moods

    if not force and atlas.regions and all(getattr(r, "state", "") for r in atlas.regions):
        return atlas, []

    g = generate(as_moods_input(atlas.regions), task=MOODS, model=model)
    moods = g.spec
    problems = validate_moods(moods, len(atlas.regions))
    if problems or not moods.understood:
        log.warning("mood sorting rejected: %s", problems[:4])
        return atlas, problems

    by_index = {m.index: str(m.state.value) for m in moods.moods}
    for i, r in enumerate(atlas.regions):
        if i in by_index:
            r.state = by_index[i]
    return atlas, []
