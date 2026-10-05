#!/usr/bin/env python3
"""
scripts/plot_metric_violins.py — Stage 2g report figure
============================================================
One violin per confidence / interface metric, drawn from the flat
results_table.csv (every screened pair x diffusion sample x bait-prey chain
pair). Metrics live on different scales (pTM in 0..1, pLDDT in 0..100,
iLIS ~0..1, ...), so each metric gets its own panel with its own y-axis
rather than sharing one.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METRICS = ["ptm", "iptm", "pair_iptm", "mean_plddt", "ipsae", "lis", "clis",
           "ilis", "pinc", "pdockq", "pdockq2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="svg")
    ap.add_argument("--ncols", type=int, default=4)
    a = ap.parse_args()

    df = pd.read_csv(a.table)
    present = [m for m in METRICS if m in df.columns and df[m].notna().any()]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not present:
        fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
        ax.text(0.5, 0.5, "no metric values", ha="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        ncols = min(a.ncols, len(present))
        nrows = math.ceil(len(present) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 3.2 * nrows),
                                 constrained_layout=True, squeeze=False)
        for ax, m in zip(axes.flat, present):
            vals = pd.to_numeric(df[m], errors="coerce").dropna().to_numpy()
            if len(vals) > 1 and np.ptp(vals) > 0:
                parts = ax.violinplot(vals, showmedians=True, showextrema=False)
                for body in parts["bodies"]:
                    body.set_alpha(0.6)
            # raw points on top: small n is common (few pairs x few samples)
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(vals))
            ax.scatter(1 + jitter, vals, s=6, color="black", alpha=0.5, zorder=3)
            ax.set_xticks([])
            ax.set_title(f"{m} (n={len(vals)})", fontsize=9)
        for ax in list(axes.flat)[len(present):]:
            ax.axis("off")

    for fmt in a.formats.split(","):
        fig.savefig(f"{out.with_suffix('')}.{fmt.strip()}")
    plt.close(fig)
    print(f"[plot_metric_violins] {len(present)} metric(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
