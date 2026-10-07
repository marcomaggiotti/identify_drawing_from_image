#!/usr/bin/env python3
"""Download free handwriting fonts for the synthetic generator (same as ``drawid fonts``).

    python scripts/download_fonts.py [--out assets/fonts]
"""

import argparse

from drawing_identifier.synthetic.fonts import FAMILIES, download_fonts

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="assets/fonts")
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    args = ap.parse_args()
    raise SystemExit(download_fonts(args.out, args.families))
