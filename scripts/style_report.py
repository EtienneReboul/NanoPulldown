#!/usr/bin/env python3
"""scripts/style_report.py <report.zip> [figure_theme]

Dark-theme fixes Snakemake's report can't take from a stylesheet, applied to
report.html inside the built zip (in place, idempotent):

* Workflow graph / runtimes / timeline are Vega charts drawn on a <canvas>,
  so report/custom.css can't recolour them: the rule graph's labels are
  hard-coded `"fill": {"value": "black"}` and its links `#ccc`. Rewrite those,
  and give all three specs a `config` with light axis / legend / title ink
  and no background.
* Figure cards: report/custom.css frames figures on `--fig-card`; set it to
  the figures' own surface (scripts/plot_theme.py) so a dark figure sits on a
  dark card and a light one on white.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_theme import theme  # noqa: E402

MARK = "/* nanopulldown:style_report */"
INK, INK2, RULE = "#e8eef5", "#93a4b6", "#2c3e52"
VEGA_CONFIG = {
    "background": None,
    "axis": {"labelColor": INK2, "titleColor": INK, "domainColor": INK2, "tickColor": INK2,
             "gridColor": RULE},
    "legend": {"labelColor": INK, "titleColor": INK},
    "title": {"color": INK},
    "view": {"stroke": RULE},
    "text": {"fill": INK},
}


def patch_html(html: str, card: str) -> str:
    if MARK in html:
        return html
    # rule graph: node labels + links
    html = re.sub(r'(rulegraph_spec = \{.*?)("fill": \{"value": )"black"',
                  lambda m: m.group(1) + m.group(2) + f'"{INK}"', html, count=1, flags=re.S)
    html = re.sub(r'(rulegraph_spec = \{.*?\};)',
                  lambda m: m.group(1).replace('"stroke": {"value": "#ccc"}', f'"stroke": {{"value": "{INK2}"}}'),
                  html, count=1, flags=re.S)
    # runtimes points: brighter than Bootstrap blue on the dark page
    html = html.replace('"color": {"value": "#007bff"}', '"color": {"value": "#3987e5"}')
    cfg = json.dumps(VEGA_CONFIG)
    for name in ("rulegraph_spec", "runtimes_spec", "timeline_spec"):
        html = html.replace(f"var {name} = {{", f'var {name} = {{"config": {cfg}, ', 1)
    style = f"<style>{MARK} :root {{ --fig-card: {card}; }}</style>"
    return html.replace("</head>", style + "</head>", 1) if "</head>" in html else style + html


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit("usage: style_report.py <report.zip> [figure_theme]")
    zpath = Path(sys.argv[1])
    th = theme()
    if len(sys.argv) > 2:
        th.mode = sys.argv[2]
        th.__post_init__()
    card = th.surface
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / zpath.name
        n = 0
        with zipfile.ZipFile(zpath) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.endswith("report.html"):
                    data = patch_html(data.decode("utf-8"), card).encode("utf-8")
                    n += 1
                zout.writestr(item, data)
        shutil.move(out, zpath)
    print(f"[style_report] patched {n} page(s) in {zpath} (figure card {card})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
