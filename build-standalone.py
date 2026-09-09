"""Wrap profile.html into a standalone, hostable site in dist/.

profile.html is authored as Artifact page content: no doctype, no <head>, because
the Artifact runtime supplies those at publish time. That same file opened
directly (or dropped on a static host) lands in quirks mode and has no charset
or viewport, so this adds a real document shell around it.

One source of truth: edit profile.html, re-run this, redeploy dist/.

    python build-standalone.py
"""

import os
import shutil

SRC = "profile.html"
OUT_DIR = "dist"

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="MusicShare - taste as identity.">
<meta name="theme-color" content="#0B0A0F">
<meta name="color-scheme" content="dark">
<!-- lets "Add to Home Screen" launch without browser chrome on iOS -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="MusicShare">
<meta name="mobile-web-app-capable" content="yes">
<link rel="manifest" href="manifest.webmanifest">
<style>html,body{background:#0B0A0F}</style>
"""

MANIFEST = """{
  "name": "MusicShare",
  "short_name": "MusicShare",
  "start_url": ".",
  "display": "standalone",
  "background_color": "#0B0A0F",
  "theme_color": "#0B0A0F",
  "orientation": "portrait"
}
"""


def main():
    with open(SRC, encoding="utf-8") as f:
        src = f.read()

    # everything up to the end of the first <style> block belongs in <head>;
    # the markup and scripts after it belong in <body>
    marker = "</style>"
    cut = src.find(marker)
    if cut == -1:
        raise SystemExit("no </style> found in %s - has the file structure changed?" % SRC)
    cut += len(marker)
    head_part, body_part = src[:cut], src[cut:]

    doc = HEAD + head_part + "\n</head>\n<body>\n" + body_part.strip() + "\n</body>\n</html>\n"

    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8", newline="") as f:
        f.write(doc)
    with open(os.path.join(OUT_DIR, "manifest.webmanifest"), "w", encoding="utf-8", newline="") as f:
        f.write(MANIFEST)

    size = os.path.getsize(os.path.join(OUT_DIR, "index.html"))
    print("wrote %s/index.html (%.1f KB) and manifest.webmanifest" % (OUT_DIR, size / 1024.0))
    shutil.rmtree.__doc__  # noqa - keep shutil imported for future asset copying


if __name__ == "__main__":
    main()
