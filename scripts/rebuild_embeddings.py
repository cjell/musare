"""Rebuild every fitted artefact, in the order they depend on each other.

    python scripts/rebuild_embeddings.py --corpus <tags.csv>

    space  ->  basemap  ->  atlas  ->  region names
                +->  every saved listener profile

Each step is fitted from the one before it, so they cannot be rebuilt
independently: new vectors mean new 2D positions, new positions mean the saved
map coordinates are stale, and new clusters mean the region names describe
regions that no longer exist. Doing this by hand in the wrong order leaves a
basemap indexed against a corpus that has changed size, which fails silently by
returning the wrong artist's position.

The expensive part is the basemap (a fitted UMAP index, a few minutes and a large
pickle). Everything else is under a minute.

Prominence is read from the Deezer cache rather than fetched - run
scripts/fetch_prominence.py first if it is empty, or the region exemplars fall
back to tag count and the names come out poor.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from musicshare import modes, regions
from musicshare.embed import build_space, save_space
from musicshare.project import fit_basemap, save_basemap
from musicshare.shows import FANS_CACHE
from musicshare.spec import namerun

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("rebuild")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="the Last.fm tag export: a CSV of artist and semicolon-separated tags",
    )
    ap.add_argument(
        "--keep-soft",
        action="store_true",
        help="keep geography and demographic tags (american, canadian, female vocalists). "
        "They form regions of their own - 'american pop', 'canadian indie' - which are "
        "not genres, which is why dropping them is the default here.",
    )
    ap.add_argument("--no-names", action="store_true", help="skip the naming call")
    args = ap.parse_args()

    t0 = time.time()

    log.info("1/4 fitting the space (drop_soft=%s)", not args.keep_soft)
    space = build_space(args.corpus, drop_soft=not args.keep_soft)
    save_space(space)
    log.info("    %d artists, %dd", len(space), space.dims)

    log.info("2/4 fitting the basemap - the slow one")
    b = fit_basemap(space)
    save_basemap(b)

    log.info("3/4 fitting the atlas")
    prom = {}
    if FANS_CACHE.exists():
        prom = {
            k: v for k, v in json.loads(FANS_CACHE.read_text(encoding="utf-8")).items() if v > 0
        }
    if not prom:
        log.warning("    no prominence cache; exemplars will be poor and so will the names")
    atlas = regions.fit(space, prominence=prom)
    regions.save(atlas)

    if args.no_names:
        log.info("4/4 skipped")
    else:
        log.info("4/4 naming %d regions", len(atlas))
        atlas, problems = namerun.name_atlas(atlas)
        if problems:
            # Failing closed leaves tag labels in place, which are serviceable.
            log.error("    naming rejected, tag labels kept: %s", problems[:4])
        else:
            regions.save(atlas)
            log.info("    %s ...", ", ".join(r.name for r in atlas.regions[:8]))

    # A profile's centres are vectors in the old space. Nothing downstream checks
    # them hard enough to fail, so a forgotten refit shows up as a map that is
    # subtly wrong rather than as an error - which is exactly what happened the
    # first time this script was run.
    stale = sorted(regions.OUT_DIR.glob("modes_*.json"))
    for path in stale:
        user = path.stem.removeprefix("modes_")
        log.info("5/5 refitting profile %r against the new space", user)
        p = modes.profile(space=space)
        # A refit produces fresh modes with no names, so they need naming again -
        # forgetting this leaves a profile showing tag labels like "rnb, pop"
        # where it used to say "alternative r&b".
        if not args.no_names:
            p, mode_problems = namerun.name(p)
            if mode_problems:
                log.error("    mode naming rejected, tag labels kept: %s", mode_problems[:3])
        modes.save(p, user=user)
        log.info("    k=%d, %s", p.k, ", ".join(m.display for m in p.modes))

    log.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
