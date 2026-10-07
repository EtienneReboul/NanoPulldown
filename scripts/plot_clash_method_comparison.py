#!/usr/bin/env python3
"""
scripts/plot_clash_method_comparison.py
=======================================
Current Python clash counter (scripts/count_clashes.py) vs ChimeraX's own
`clashes` (scripts/count_clashes_chimerax.py), scored on the same structures.
For each state (raw, minimized) and metric (all / inter-chain clashes) the
(pair, sample) units are paired and tested with a two-sided paired Wilcoxon signed-rank (neither method is a priori larger),
with the matched-pairs rank-biserial as effect size (sign: current - chimerax).
Also reports Spearman rho and the exact-agreement fraction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from plot_clash_test import METRICS, paired_test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", nargs="+", required=True, help="clashes.csv files (count_clashes.py)")
    ap.add_argument("--chimerax", required=True, help="CSV from count_clashes_chimerax.py")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out-stats", required=True)
    ap.add_argument("--alternative", default="two-sided", choices=["two-sided", "greater", "less"])
    a = ap.parse_args()

    cur = pd.concat([pd.read_csv(t) for t in a.current], ignore_index=True)
    cx = pd.read_csv(a.chimerax)
    keys = ["pair", "sample", "state"]
    df = cur.merge(cx, on=keys, suffixes=("_cur", "_cx"))
    states = ["raw", "minimized"]
    fig = make_subplots(rows=len(METRICS), cols=len(states), horizontal_spacing=0.08,
                        vertical_spacing=0.14, subplot_titles=["-"] * (len(METRICS) * len(states)))
    rows, ann = [], 0
    for r, (col, label) in enumerate(METRICS.items(), start=1):
        for c, state in enumerate(states, start=1):
            d = df[df.state == state]
            cu, ch = d[f"{col}_cur"].to_numpy(float), d[f"{col}_cx"].to_numpy(float)
            res = paired_test(cu, ch, a.alternative)
            # paired_test names its columns raw/minimized: here they mean current/chimerax
            res = {("median_current" if k == "median_raw" else "median_chimerax" if k == "median_minimized" else k): v
                   for k, v in res.items()}
            rho = spearmanr(cu, ch)[0] if len(d) > 2 and cu.std() > 0 and ch.std() > 0 else np.nan
            res.update(metric=col, state=state, alternative=a.alternative, spearman_rho=rho,
                       frac_identical=float((cu == ch).mean()) if len(d) else np.nan,
                       mean_current=float(cu.mean()), mean_chimerax=float(ch.mean()))
            rows.append(res)
            p = res["p_value"]
            ptxt = "p = n/a (no differences)" if np.isnan(p) else f"Wilcoxon p = {p:.2g}"
            fig.layout.annotations[ann].text = f"{label}, {state} (n={len(d)})<br>{ptxt}"
            ann += 1
            hi = max(cu.max(initial=0), ch.max(initial=0)) + 1
            fig.add_trace(go.Scattergl(
                x=ch, y=cu, mode="markers", showlegend=False, marker=dict(size=4, opacity=0.5),
                text=[f"{p_}, sample {s}" for p_, s in zip(d.pair, d["sample"])],
                hovertemplate="ChimeraX %{x}<br>current %{y}<br>%{text}<extra></extra>"), row=r, col=c)
            fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", showlegend=False,
                                     line=dict(color="grey", dash="dash")), row=r, col=c)
            fig.update_xaxes(title_text="ChimeraX", row=r, col=c)
            fig.update_yaxes(title_text="current", row=r, col=c)
            print(f"[compare] {col} {state}: n={len(d)} median cur={res['median_current']:.0f} "
                  f"cx={res['median_chimerax']:.0f} mean cur={cu.mean():.2f} cx={ch.mean():.2f} "
                  f"{ptxt} rbc={res['rank_biserial']:.2f} rho={rho:.3f} identical={res['frac_identical']:.2f}")
    fig.update_layout(height=820, template="plotly_dark",
                      title="Clash counting: current implementation vs ChimeraX (paired)")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig.write_html(out / "index.html", include_plotlyjs=True, full_html=True)
    pd.DataFrame(rows).to_csv(a.out_stats, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
