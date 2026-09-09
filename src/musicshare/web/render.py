"""Wrap the Artifact-format mockup into a real HTML document.

profile.html is authored as Artifact page content - no doctype, no <head>,
because the Artifact runtime supplies them. Served directly it lands in quirks
mode with no charset. This adds the shell, in memory, on every request, so
there is no build step between editing and refreshing.
"""

from __future__ import annotations

from pathlib import Path

from musicshare.config import ROOT

SRC = ROOT / "profile.html"

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="MusicShare - taste as identity.">
<meta name="theme-color" content="#0B0A0F">
<meta name="color-scheme" content="dark">
<style>html,body{background:#0B0A0F}</style>
"""


def render(src: Path = SRC) -> str:
    html = src.read_text(encoding="utf-8")
    marker = "</style>"
    cut = html.find(marker)
    if cut == -1:
        raise RuntimeError(f"no </style> in {src} - has the file structure changed?")
    cut += len(marker)
    return HEAD + html[:cut] + "\n</head>\n<body>\n" + html[cut:].strip() + "\n</body>\n</html>\n"
