#!/usr/bin/env python3
"""
scripts/plot_metric_violins_interactive.py — Stage 2g report figure (interactive)
============================================================
Plotly twin of plot_metric_violins.py: one violin panel per metric with every
point hoverable (pair, sample, chain pair, value). Written as a directory with
a self-contained index.html (plotly.js inlined) so `snakemake --report` can
embed it via report(directory(...), htmlindex="index.html") and it works
offline.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRICS = ["ptm", "iptm", "pair_iptm", "mean_plddt", "ipsae", "lis", "clis",
           "ilis", "pinc", "pdockq", "pdockq2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ncols", type=int, default=4)
    a = ap.parse_args()

    df = pd.read_csv(a.table)
    present = [m for m in METRICS if m in df.columns and df[m].notna().any()]
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not present:
        Path(outdir / "index.html").write_text("<p>no metric values</p>")
        return 0

    ncols = min(a.ncols, len(present))
    nrows = math.ceil(len(present) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=present,
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    chain = (df["chain_i"].astype(str) + "-" + df["chain_j"].astype(str)
             if "chain_i" in df else pd.Series("", index=df.index))
    for k, m in enumerate(present):
        sub = df[df[m].notna()]
        fig.add_trace(go.Violin(
            y=sub[m], name=m, box_visible=True, meanline_visible=True,
            points="all", jitter=0.4, pointpos=0, marker_size=3, showlegend=False,
            customdata=list(zip(sub["pair"], sub["sample_index"], chain[sub.index])),
            hovertemplate=(f"{m}=%{{y:.4g}}<br>pair=%{{customdata[0]}}"
                           "<br>sample=%{customdata[1]}<br>chains=%{customdata[2]}<extra></extra>"),
        ), row=k // ncols + 1, col=k % ncols + 1)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(height=340 * nrows, template="plotly_dark",
                      title="Metric distributions (all pairs, samples, chain pairs)")
    fig.write_html(outdir / "index.html", include_plotlyjs=True, full_html=True)
    print(f"[plot_metric_violins_interactive] {len(present)} metric(s) -> {outdir}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
