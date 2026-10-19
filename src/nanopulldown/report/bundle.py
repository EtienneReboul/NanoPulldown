"""
nanopulldown.report.bundle
==========================
The HTML report: the datavzrd table bundle, the Snakemake report zip, and the
dark-theme fixes neither takes from a stylesheet.

  report tables  datavzrd over the run's CSV tables -> <reports>/folding/tables/,
                 then a dark skin inlined into every page (style_datavzrd)
  report build   `snakemake --report <zip>` + style_report (the Snakefile's
                 onsuccess hook; never fails the pipeline)
  report style   style_report alone, on an existing zip

style_datavzrd: datavzrd (Bootstrap 4) has no dark mode. Once its bundle is
embedded in the Snakemake .zip report we can no longer reach it from the
parent stylesheet, so we inline a dark skin (assets/datavzrd-dark.css) into
every HTML page of a freshly rendered bundle. The pages have no <head> and
mount Vue into <body>, so the style is anchored in the implicit head (right
after <meta charset>) where it survives the render. Idempotent: re-running
replaces the previously injected block.

The report is a .zip (not a bare .html) because `snakemake --report` only
embeds interactive HTML items -- the datavzrd table bundle -- for the zip
form. Unzip it and open report.html; the dark theme lives in
report/custom.css, injected via --report-stylesheet.

style_report: dark-theme fixes Snakemake's report can't take from a stylesheet, applied to
report.html inside the built zip (in place, idempotent):

* Workflow graph / runtimes / timeline are Vega charts drawn on a <canvas>,
  so report/custom.css can't recolour them: the rule graph's labels are
  hard-coded `"fill": {"value": "black"}` and its links `#ccc`. Rewrite those,
  and give all three specs a `config` with light axis / legend / title ink
  and no background.
* Figure cards: report/custom.css frames figures on `--fig-card`; set it to
  the figures' own surface (nanopulldown.report.theme) so a dark figure sits on a
  dark card and a light one on white.

Stdlib only at import (style_report needs the theme, i.e. pyyaml).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
DATAVZRD_CSS = ASSETS / "datavzrd-dark.css"
DATAVZRD_MARK_OPEN = "<style data-ab-dark>"
DATAVZRD_MARK_CLOSE = "</style>"
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


def style_datavzrd(root: Path, css_path: Path = DATAVZRD_CSS) -> int:
    """Inline the dark skin into every HTML page under `root`; returns the page count."""
    block = f"{DATAVZRD_MARK_OPEN}\n{Path(css_path).read_text(encoding='utf-8')}\n{DATAVZRD_MARK_CLOSE}"
    pages = sorted(Path(root).rglob("*.html"))
    if not pages:
        print(f"[style_datavzrd] no HTML under {root}", file=sys.stderr)
        return 0
    for page in pages:
        html = page.read_text(encoding="utf-8")
        if DATAVZRD_MARK_OPEN in html:
            pre, _, rest = html.partition(DATAVZRD_MARK_OPEN)
            _, _, post = rest.partition(DATAVZRD_MARK_CLOSE)
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
    return len(pages)


def style_report(zpath: Path, mode: str | None = None) -> int:
    """Patch report.html inside the zip in place; returns the patched page count."""
    from nanopulldown.report.theme import theme

    zpath = Path(zpath)
    th = theme()
    if mode:
        th.mode = mode
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
    return n


def tables_main() -> int:
    ap = argparse.ArgumentParser(description="render the datavzrd table bundle and skin it dark")
    ap.add_argument("--config", type=Path, required=True, help="report/datavzrd/nanopulldown.datavzrd.yaml")
    ap.add_argument("--outdir", type=Path, required=True,
                    help="bundle dir; the config's relative table paths resolve against its parent")
    ap.add_argument("--css", type=Path, default=DATAVZRD_CSS)
    a = ap.parse_args()
    subprocess.run(["datavzrd", str(a.config.resolve()), "--overwrite-output", "--output", a.outdir.name],
                   cwd=a.outdir.parent, check=True)
    style_datavzrd(a.outdir, a.css)
    return 0


def style_main() -> int:
    ap = argparse.ArgumentParser(description="dark-theme fixes inside a built report zip")
    ap.add_argument("zip", type=Path)
    ap.add_argument("figure_theme", nargs="?", default=None)
    a = ap.parse_args()
    style_report(a.zip, a.figure_theme)
    return 0


def build_main() -> int:
    """Rebuild the report zip from the just-completed Snakemake run. Invoked
    from the Snakefile's onsuccess hook, so it must never fail the pipeline --
    a broken report build is logged and swallowed (always exits 0)."""
    from nanopulldown.core.config import REPO_ROOT

    ap = argparse.ArgumentParser(description="snakemake --report + dark-theme fixes (never fails)")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--out", type=Path, default=None, help="default: <repo>/reports/report.zip")
    a = ap.parse_args()

    repo = a.repo.resolve()
    out = a.out or Path("reports") / "report.zip"
    out = out if out.is_absolute() else repo / out
    css = repo / "report" / "custom.css"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[build_report] -> {out}", flush=True)

    if shutil.which("snakemake") is None:
        print("[build_report] WARNING: snakemake not on PATH — skipping report build", file=sys.stderr)
        return 0
    cmd = ["snakemake", "-s", str(repo / "Snakefile"), "--report", str(out)]
    if css.is_file():
        cmd += ["--report-stylesheet", str(css)]
    if subprocess.run(cmd, cwd=repo).returncode != 0:
        print("[build_report] WARNING: report build failed (non-fatal)", file=sys.stderr)
        return 0
    print(f"[build_report] ok: {out}", flush=True)
    try:
        style_report(out)
    except Exception as e:                                  # noqa: BLE001
        print(f"[build_report] WARNING: style_report failed (report left unstyled): {e}", file=sys.stderr)
    return 0
