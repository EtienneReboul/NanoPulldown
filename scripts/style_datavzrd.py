#!/usr/bin/env python3
"""scripts/style_datavzrd.py <datavzrd-output-dir> [css-file]

datavzrd (Bootstrap 4) has no dark mode. Once its bundle is embedded in the
Snakemake .zip report we can no longer reach it from the parent stylesheet,
so we inline a dark skin into every HTML page of a freshly rendered bundle.

The pages have no <head> and mount Vue into <body>, so the style is anchored
in the implicit head (right after <meta charset>) where it survives the
render. Idempotent: re-running replaces the previously injected block.
"""
from __future__ import annotations

import sys
from pathlib import Path

MARK_OPEN = "<style data-ab-dark>"
MARK_CLOSE = "</style>"


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit("usage: style_datavzrd.py <dir> [css-file]")
    root = Path(sys.argv[1])
    css_path = (Path(sys.argv[2]) if len(sys.argv) > 2
                else Path(__file__).with_name("datavzrd-dark.css"))
    block = f"{MARK_OPEN}\n{css_path.read_text(encoding='utf-8')}\n{MARK_CLOSE}"

    pages = sorted(root.rglob("*.html"))
    if not pages:
        print(f"[style_datavzrd] no HTML under {root}", file=sys.stderr)
        return 0
    for page in pages:
        html = page.read_text(encoding="utf-8")
        if MARK_OPEN in html:
            pre, _, rest = html.partition(MARK_OPEN)
            _, _, post = rest.partition(MARK_CLOSE)
            html = pre + post
        low = html.lower()
        meta = low.find("<meta")
        if meta != -1:
            idx = html.find(">", meta) + 1
        else:
            htmltag = low.find("<html")
            idx = html.find(">", htmltag) + 1 if htmltag != -1 else 0
        page.write_text(html[:idx] + block + html[idx:], encoding="utf-8")
    print(f"[style_datavzrd] skinned {len(pages)} page(s) under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
