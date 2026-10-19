#!/usr/bin/env python3
"""
nanopulldown.report.plots.violins — Stage 2g report figure (interactive)
============================================================
Plotly twin of plot_metric_violins.py: one violin panel per metric with every
point hoverable (bait × prey names, pair, sample, chain pair, value). Written as a directory with
a self-contained index.html (plotly.js inlined) so `snakemake --report` can
embed it via report(directory(...), htmlindex="index.html") and it works
offline.

--metrics / --title / --labels make it reusable for other column sets of the
results table (the Snakefile's PLIP interaction-type section uses it with the
plip_* count columns).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRICS = ["ptm", "iptm", "pair_iptm", "mean_plddt", "ipsae", "lis", "clis",
           "ilis", "pinc", "pdockq", "pdockq2", "plip_contacts"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--metrics", nargs="+", default=METRICS, help="results-table columns, one panel each")
    ap.add_argument("--labels", nargs="+", default=None, help="panel titles (same order as --metrics)")
    ap.add_argument("--title", default="Metric distributions (all pairs, samples, chain pairs)")
    ap.add_argument("--skip-all-zero", action="store_true", help="drop columns that are 0 everywhere")
    a = ap.parse_args()

    df = pd.read_csv(a.table)
    label = dict(zip(a.metrics, a.labels or a.metrics))
    present = [m for m in a.metrics if m in df.columns and df[m].notna().any()
               and not (a.skip_all_zero and not df[m].fillna(0).astype(bool).any())]
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not present:
        Path(outdir / "index.html").write_text("<p>no metric values</p>")
        return 0

    ncols = min(a.ncols, len(present))
    nrows = math.ceil(len(present) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[label[m] for m in present],
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    chain = (df["chain_i"].astype(str) + "-" + df["chain_j"].astype(str)
             if "chain_i" in df else pd.Series("", index=df.index))
    # 'BPM6 × MYC2' when the results table carries display names
    name = (df["bait"].astype(str) + " × " + df["prey"].astype(str)
            if {"bait", "prey"} <= set(df.columns) else df["pair"].astype(str))
    for k, m in enumerate(present):
        sub = df[df[m].notna()]
        fig.add_trace(go.Violin(
            y=sub[m], name=label[m], box_visible=True, meanline_visible=True,
            points="all", jitter=0.4, pointpos=0, marker_size=3, showlegend=False,
            customdata=list(zip(name[sub.index], sub["pair"], sub["sample_index"], chain[sub.index])),
            hovertemplate=(f"<b>%{{customdata[0]}}</b><br>{label[m]}=%{{y:.4g}}<br>pair=%{{customdata[1]}}"
                           "<br>sample=%{customdata[2]}<br>chains=%{customdata[3]}<extra></extra>"),
        ), row=k // ncols + 1, col=k % ncols + 1)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(height=340 * nrows, template="plotly_dark",
                      title=a.title)
    fig.write_html(outdir / "index.html", include_plotlyjs=True, full_html=True)
    print(f"[plot_metric_violins_interactive] {len(present)} metric(s) -> {outdir}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
