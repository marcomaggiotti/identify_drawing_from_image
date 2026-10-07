"""Download free (SIL OFL / Apache) handwriting fonts from Google Fonts for the synthetic generator.

``drawid fonts`` / ``python scripts/download_fonts.py``. Fonts are not committed to the repository;
see each family's page on fonts.google.com for its licence.
"""

from __future__ import annotations

import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

FAMILIES = [
    "Homemade Apple",
    "Caveat",
    "Dawning of a New Day",
    "La Belle Aurore",
    "Cedarville Cursive",
    "Reenie Beanie",
    "Nothing You Could Do",
    "Zeyada",
    "Over the Rainbow",
    "Waiting for the Sunrise",
    "Kristi",
    "Mrs Saint Delafield",
    "Herr Von Muellerhoff",
    "Just Me Again Down Here",
    "Loved by the King",
    "Indie Flower",
    "Shadows Into Light",
    "Gochi Hand",
    "Kalam",
    "Nanum Pen Script",
]

CSS_API = "https://fonts.googleapis.com/css2?"


def download_fonts(out: str | Path = "assets/fonts", families: list[str] | None = None) -> int:
    """Fetch the TrueType files of ``families`` into ``out``; returns a process exit code."""
    families = families or FAMILIES
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    query = "&".join("family=" + urllib.parse.quote_plus(f) for f in families)
    # without a browser user agent the API serves plain TrueType files
    req = urllib.request.Request(CSS_API + query, headers={"User-Agent": "curl/8"})
    css = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    blocks = re.findall(r"font-family: '([^']+)';.*?src: url\((https://[^)]+\.ttf)\)", css, re.S)
    if not blocks:
        print("no fonts found in the Google Fonts response", file=sys.stderr)
        return 1
    seen = set()
    for family, url in blocks:
        if family in seen:
            continue
        seen.add(family)
        dest = out / (family.replace(" ", "") + ".ttf")
        if dest.exists():
            print(f"exists  {dest}")
            continue
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "curl/8"}), timeout=60).read()
        dest.write_bytes(data)
        print(f"saved   {dest} ({len(data) // 1024} KB)")
    print(f"{len(seen)} font(s) in {out}")
    return 0
