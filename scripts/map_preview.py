"""Draw the taste map to a standalone, zoomable HTML file.

Not part of the app. The point is to judge the projection and the label behaviour
with eyes before any of it reaches profile.html, because the last round of
frontend work showed CSS regressions are cheap to cause and expensive to find.

The map is named territory - corpus regions - with the listener's own artists on
top. Labels appear by zoom rather than by a fixed count: each region carries the
smallest zoom at which its label fits without colliding, computed in
project.label_levels, and the page just filters on that number. Drawing the top
sixteen by hours instead does not work, because two heavy neighbouring regions
are exactly the pair most likely to overlap each other.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import numpy as np

from musicshare import regions
from musicshare.config import ROOT
from musicshare.embed import load_space
from musicshare.history import connect
from musicshare.project import atlas_map, load_basemap

logging.basicConfig(level=logging.WARNING)

OUT = ROOT / "data" / "embed" / "map_preview.html"
VIEW = 900


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write somewhere other than the default")
    args = ap.parse_args()
    out = pathlib.Path(args.out) if args.out else OUT

    b, space, con = load_basemap(), load_space(), connect()
    atlas = regions.load()
    m = atlas_map(atlas, con=con, b=b, space=space, view_px=VIEW)

    pts = np.array([[a["x"], a["y"]] for a in m["artists"]])
    back = np.array(m["backdrop"])
    lo = np.minimum(pts.min(0), back.min(0))
    hi = np.maximum(pts.max(0), back.max(0))
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))

    def to_view(x: float, y: float) -> tuple[float, float]:
        # Square frame and a flipped y, so the picture matches the numbers.
        return (x - lo[0]) / span * VIEW, VIEW - (y - lo[1]) / span * VIEW

    hours = np.array([a["hours"] for a in m["artists"]])
    # Hours span four orders of magnitude; a linear radius would draw the top
    # artist as a disc covering half the map.
    rad = 1.2 + 5.0 * (np.log1p(hours) / np.log1p(hours.max()))

    dots = [
        {
            "x": round(to_view(a["x"], a["y"])[0], 1),
            "y": round(to_view(a["x"], a["y"])[1], 1),
            "r": round(float(rad[j]), 1),
            "n": a["name"],
            "h": a["hours"],
        }
        for j, a in enumerate(m["artists"])
    ]
    grey = [[round(to_view(x, y)[0], 1), round(to_view(x, y)[1], 1)] for x, y in back]
    labels = [
        {
            "x": round(to_view(r["x"], r["y"])[0], 1),
            "y": round(to_view(r["x"], r["y"])[1], 1),
            "n": r["name"],
            "z": r["min_zoom"],
            "h": r["hours"],
        }
        for r in m["regions"]
        if r["yours"] and r["x"] is not None and r["min_zoom"] is not None
    ]

    rows = "".join(
        f'<tr><td>{r["name"]}</td><td class="n">{r["hours"]:,.0f}h</td>'
        f'<td class="n">{r["share"]:.1%}</td>'
        f'<td class="n">{r["your_artists"]:,} / {r["artists"]:,}</td>'
        f'<td class="n">{r["min_zoom"] or "-"}</td><td class="t">{r["label"]}</td></tr>'
        for r in sorted(m["regions"], key=lambda r: -r["hours"])
        if r["yours"]
    )
    occupied = sum(1 for r in m["regions"] if r["yours"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"""<!doctype html><meta charset="utf-8"><title>taste map preview</title>
<style>
  body{{margin:0;background:#0E0E12;color:#E9E9F0;font:14px system-ui,sans-serif}}
  main{{max-width:940px;margin:0 auto;padding:20px}}
  h1{{font-size:15px;font-weight:600;margin:0 0 4px}}
  p.sub{{color:#8A8A99;margin:0 0 14px;font-size:12px;line-height:1.55}}
  #wrap{{position:relative;border:1px solid #26262F;border-radius:12px;overflow:hidden;
         background:#0B0B0F;touch-action:none;cursor:grab}}
  #wrap.drag{{cursor:grabbing}}
  svg{{display:block;width:100%;height:auto}}
  #hud{{position:absolute;left:10px;top:10px;font:11px ui-monospace,monospace;
        color:#8A8A99;background:rgba(14,14,18,.82);padding:5px 8px;border-radius:7px}}
  #tip{{position:absolute;pointer-events:none;font:11px system-ui;background:#16161D;
        border:1px solid #2E2E39;padding:4px 7px;border-radius:6px;display:none;
        white-space:nowrap}}
  .ctl{{margin-top:10px;display:flex;gap:6px;align-items:center;font-size:12px;color:#8A8A99}}
  button{{background:#1A1A21;color:#E9E9F0;border:1px solid #2E2E39;border-radius:7px;
          padding:4px 10px;font:12px system-ui;cursor:pointer}}
  table{{width:100%;border-collapse:collapse;margin-top:18px;font-size:12px}}
  th{{text-align:left;color:#8A8A99;font-weight:500;padding:6px 8px;border-bottom:1px solid #26262F}}
  td{{padding:5px 8px;border-bottom:1px solid #1A1A21}}
  td.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
  td.t{{color:#6E6E7C}}
</style>
<main>
  <h1>{len(m["artists"]):,} artists you play, across {occupied} of {len(atlas)} regions</h1>
  <p class="sub">
    Scroll or pinch to zoom, drag to pan. Labels appear as you zoom in &mdash;
    {sum(1 for x in labels if x["z"] == 1)} at the start, all {len(labels)} by the end.
    Grey is the {len(b):,}-artist corpus; pink is you, sized by hours.<br>
    Positions are for drawing only &mdash; distance here tracks real similarity within
    about 3 units and means nothing beyond that, and the centre means nothing at all.
  </p>
  <div id="wrap">
    <svg id="map" viewBox="0 0 {VIEW} {VIEW}">
      <g id="grey" fill="#23232B"></g>
      <g id="you" fill="#E74094" fill-opacity="0.72"></g>
      <g id="lab" font-size="11.5" fill="#fff"></g>
    </svg>
    <div id="hud">zoom 1.0&times;</div>
    <div id="tip"></div>
  </div>
  <div class="ctl">
    <button id="out">&minus;</button><button id="in">+</button><button id="reset">reset</button>
    <span id="count"></span>
  </div>
  <table>
    <tr><th>region</th><th class="n">hours</th><th class="n">share</th>
        <th class="n">your artists / all</th><th class="n">zoom</th><th>tag label</th></tr>
    {rows}
  </table>
</main>
<script>
var V = {VIEW};
var GREY = {json.dumps(grey)};
var DOTS = {json.dumps(dots)};
var LABELS = {json.dumps(labels)};

var svg = document.getElementById("map"), wrap = document.getElementById("wrap");
var z = 1, cx = V / 2, cy = V / 2;

document.getElementById("grey").innerHTML =
  GREY.map(function(p){{ return '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="1"/>'; }}).join("");
document.getElementById("you").innerHTML =
  DOTS.map(function(d, i){{
    return '<circle data-i="' + i + '" cx="' + d.x + '" cy="' + d.y + '" r="' + d.r + '"/>';
  }}).join("");

function view(){{
  var w = V / z;
  /* Keep the frame inside the picture, so panning cannot lose the map. */
  cx = Math.min(V - w / 2, Math.max(w / 2, cx));
  cy = Math.min(V - w / 2, Math.max(w / 2, cy));
  svg.setAttribute("viewBox", (cx - w / 2) + " " + (cy - w / 2) + " " + w + " " + w);

  /* A label's threshold is the zoom at which it was measured to fit, so the test
     is a comparison rather than a collision pass on every frame. Text and dots
     are counter-scaled: at 4x the viewBox is a quarter the size, so an unscaled
     11px label would render as 44px. */
  var k = 1 / z, shown = 0;
  document.getElementById("lab").innerHTML = LABELS.map(function(l){{
    if(l.z > z) return "";
    shown++;
    return '<text x="' + l.x + '" y="' + l.y + '" text-anchor="middle" ' +
      'style="font-size:' + (11.5 * k).toFixed(2) + 'px" ' +
      'stroke="#0E0E12" stroke-width="' + (3.2 * k).toFixed(2) +
      '" paint-order="stroke">' + l.n + '</text>';
  }}).join("");
  document.getElementById("you").setAttribute("transform", "");
  document.querySelectorAll("#you circle").forEach(function(c, i){{
    c.setAttribute("r", (DOTS[i].r * Math.max(k, 0.34)).toFixed(2));
  }});
  document.getElementById("grey").querySelectorAll("circle").forEach(function(c){{
    c.setAttribute("r", Math.max(k, 0.3).toFixed(2));
  }});
  document.getElementById("hud").textContent = "zoom " + z.toFixed(1) + "\\u00D7";
  document.getElementById("count").textContent = shown + " of " + LABELS.length + " labels";
}}

wrap.addEventListener("wheel", function(e){{
  e.preventDefault();
  var r = wrap.getBoundingClientRect(), w = V / z;
  /* Zoom about the pointer, not the centre - anything else feels broken. */
  var px = cx - w / 2 + (e.clientX - r.left) / r.width * w;
  var py = cy - w / 2 + (e.clientY - r.top) / r.height * w;
  var nz = Math.min(16, Math.max(1, z * (e.deltaY < 0 ? 1.18 : 1 / 1.18)));
  cx = px + (cx - px) * (z / nz); cy = py + (cy - py) * (z / nz);
  z = nz; view();
}}, {{passive: false}});

var drag = null;
wrap.addEventListener("pointerdown", function(e){{
  drag = {{x: e.clientX, y: e.clientY}}; wrap.classList.add("drag");
  wrap.setPointerCapture(e.pointerId);
}});
wrap.addEventListener("pointerup", function(){{ drag = null; wrap.classList.remove("drag"); }});
wrap.addEventListener("pointermove", function(e){{
  var tip = document.getElementById("tip");
  if(drag){{
    var r = wrap.getBoundingClientRect(), w = V / z;
    cx -= (e.clientX - drag.x) / r.width * w;
    cy -= (e.clientY - drag.y) / r.height * w;
    drag = {{x: e.clientX, y: e.clientY}};
    tip.style.display = "none";
    view();
    return;
  }}
  var el = document.elementFromPoint(e.clientX, e.clientY);
  var i = el && el.dataset ? el.dataset.i : null;
  if(i == null){{ tip.style.display = "none"; return; }}
  var d = DOTS[i], r2 = wrap.getBoundingClientRect();
  tip.textContent = d.n + " \\u00B7 " + d.h.toFixed(1) + "h";
  tip.style.display = "block";
  tip.style.left = (e.clientX - r2.left + 10) + "px";
  tip.style.top = (e.clientY - r2.top + 10) + "px";
}});

document.getElementById("in").onclick = function(){{ z = Math.min(16, z * 1.6); view(); }};
document.getElementById("out").onclick = function(){{ z = Math.max(1, z / 1.6); view(); }};
document.getElementById("reset").onclick = function(){{ z = 1; cx = cy = V / 2; view(); }};
view();
</script>
""",
        encoding="utf-8",
    )
    at_one = sum(1 for x in labels if x["z"] == 1)
    print(f"wrote {out}  ({occupied} regions, {len(labels)} labels, {at_one} at zoom 1)")


if __name__ == "__main__":
    main()
