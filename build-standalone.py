"""Wrap profile.html into a standalone, hostable site in dist/.

profile.html is authored as Artifact page content: no doctype, no <head>, because
the Artifact runtime supplies those at publish time. That same file opened
directly (or dropped on a static host) lands in quirks mode and has no charset
or viewport, so this adds a real document shell around it.

One source of truth: edit profile.html, re-run this, redeploy dist/.

    python build-standalone.py
"""

from pathlib import Path

SRC = Path("profile.html")
OUT_DIR = Path("dist")

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
    src = SRC.read_text(encoding="utf-8")

    # everything up to the end of the first <style> block belongs in <head>;
    # the markup and scripts after it belong in <body>
    marker = "</style>"
    cut = src.find(marker)
    if cut == -1:
        raise SystemExit(f"no </style> found in {SRC} - has the file structure changed?")
    cut += len(marker)
    head_part, body_part = src[:cut], src[cut:]

    doc = HEAD + head_part + "\n</head>\n<body>\n" + body_part.strip() + "\n</body>\n</html>\n"

    OUT_DIR.mkdir(exist_ok=True)
    index = OUT_DIR / "index.html"
    index.write_text(doc, encoding="utf-8", newline="")
    (OUT_DIR / "manifest.webmanifest").write_text(MANIFEST, encoding="utf-8", newline="")

    size = index.stat().st_size
    print(f"wrote {OUT_DIR}/index.html ({size / 1024.0:.1f} KB) and manifest.webmanifest")


if __name__ == "__main__":
    main()
