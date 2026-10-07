#!/usr/bin/env python3
"""
scripts/plot_clash_test.py — Stage 2g report figure
=======================================================
Does OpenMM minimization reduce steric clashes? Takes the per-pair
clashes.csv files (scripts/count_clashes.py), keeps the (pair, sample) units
that have BOTH a raw and a minimized structure, and draws an interactive
paired violin (raw vs minimized, one hoverable line per unit) for the total
clash count and the inter-chain (bait<->prey) clash count. Written as a
directory with a self-contained index.html (plotly.js inlined) so
`snakemake --report` can embed it offline. Also writes the per-unit paired
counts (--out-counts) for the datavzrd table.

Test: paired Wilcoxon signed-rank (non-parametric), one-sided by default
(H1: raw > minimized). Zero differences are dropped (Wilcoxon's convention);
the matched-pairs rank-biserial correlation is reported as the effect size.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import wilcoxon

METRICS = {"n_clashes": "all clashes", "n_clashes_interchain": "inter-chain clashes"}


def paired_test(raw: np.ndarray, mini: np.ndarray, alternative: str) -> dict:
    diff = raw - mini
    nz = diff[diff != 0]
    res = {"n_pairs": len(diff), "n_nonzero": len(nz),
           "median_raw": float(np.median(raw)), "median_minimized": float(np.median(mini)),
           "median_diff": float(np.median(diff)),
           "n_improved": int((diff > 0).sum()), "n_worse": int((diff < 0).sum()),
           "statistic": np.nan, "p_value": np.nan, "rank_biserial": np.nan}
    if len(nz) == 0:
        return res
    stat, p = wilcoxon(nz, alternative=alternative)
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    res.update(statistic=float(stat), p_value=float(p),
               rank_biserial=float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out-stats", required=True)
    ap.add_argument("--out-counts", required=True)
    ap.add_argument("--alternative", default="greater",
                    choices=["greater", "two-sided", "less"],
                    help="H1 on (raw - minimized); 'greater' = minimization reduces clashes")
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(t) for t in a.tables], ignore_index=True)
    stats_rows, count_frames = [], []
    fig = make_subplots(rows=1, cols=len(METRICS), subplot_titles=list(METRICS.values()),
                        horizontal_spacing=0.08)
    for k, (col, label) in enumerate(METRICS.items(), start=1):
        wide = (df.pivot_table(index=["pair", "sample"], columns="state", values=col)
                .dropna(subset=["raw", "minimized"]))
        if len(wide) == 0:
            fig.layout.annotations[k - 1].text = f"{label}: no paired samples"
            continue
        raw, mini = wide["raw"].to_numpy(float), wide["minimized"].to_numpy(float)
        res = paired_test(raw, mini, a.alternative)
        stats_rows.append({"metric": col, "alternative": a.alternative, **res})
        count_frames.append(wide.reset_index().assign(metric=col)
                            [["metric", "pair", "sample", "raw", "minimized"]])

        units = [f"{p}, sample {s}" for p, s in wide.index]
        for name, vals in (("raw", raw), ("minimized", mini)):
            fig.add_trace(go.Violin(
                y=vals, x=[name] * len(vals), name=name, legendgroup=name,
                showlegend=False, box_visible=True, meanline_visible=True,
                points="all", jitter=0.3, pointpos=0, marker_size=4, opacity=0.8,
                text=units, hovertemplate=f"{name}: %{{y}}<br>%{{text}}<extra></extra>",
            ), row=1, col=k)
        for u, r, m in zip(units, raw, mini):
            fig.add_trace(go.Scatter(
                x=["raw", "minimized"], y=[r, m], mode="lines", showlegend=False,
                line=dict(color="rgba(150,150,150,0.25)", width=1),
                text=[u, u], hovertemplate=f"%{{text}}<br>%{{y}}<extra></extra>",
            ), row=1, col=k)
        p = res["p_value"]
        ptxt = "p = n/a (no differences)" if np.isnan(p) else f"Wilcoxon p = {p:.2g}"
        fig.layout.annotations[k - 1].text = f"{label} (n={len(wide)})<br>{ptxt}"
        print(f"[plot_clash_test] {col}: n={len(wide)} median {res['median_raw']:.0f} -> "
              f"{res['median_minimized']:.0f}, {ptxt}, rank-biserial={res['rank_biserial']:.2f}")
    fig.update_yaxes(title_text="clash count", rangemode="tozero")
    fig.update_layout(height=480, template="plotly_dark",
                      title="Clashes: raw vs minimized (paired)")

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.write_html(outdir / "index.html", include_plotlyjs=True, full_html=True)
    pd.DataFrame(stats_rows).to_csv(a.out_stats, index=False)
    counts_cols = ["metric", "pair", "sample", "raw", "minimized"]
    (pd.concat(count_frames, ignore_index=True) if count_frames
     else pd.DataFrame(columns=counts_cols)).to_csv(a.out_counts, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
