#!/usr/bin/env python3
"""
scripts/plot_clash_test.py — Stage 2g report figure
=======================================================
Does OpenMM minimization reduce steric clashes? Takes the per-pair
clashes.csv files (scripts/count_clashes.py), keeps the (pair, sample) units
that have BOTH a raw and a minimized structure, and draws a paired violin
(raw vs minimized, one line per unit) for the total clash count and the
inter-chain (bait<->prey) clash count.

Test: paired Wilcoxon signed-rank (non-parametric), one-sided by default
(H1: raw > minimized). Zero differences are dropped (Wilcoxon's convention);
the matched-pairs rank-biserial correlation is reported as the effect size.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

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
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-stats", required=True)
    ap.add_argument("--formats", default="svg")
    ap.add_argument("--alternative", default="greater",
                    choices=["greater", "two-sided", "less"],
                    help="H1 on (raw - minimized); 'greater' = minimization reduces clashes")
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(t) for t in a.tables], ignore_index=True)
    stats_rows = []
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.6 * len(METRICS), 4.2),
                             constrained_layout=True, squeeze=False)
    for ax, (col, label) in zip(axes[0], METRICS.items()):
        wide = (df.pivot_table(index=["pair", "sample"], columns="state", values=col)
                .dropna(subset=["raw", "minimized"]))
        raw, mini = wide["raw"].to_numpy(float), wide["minimized"].to_numpy(float)
        if len(wide) == 0:
            ax.text(0.5, 0.5, "no paired samples", ha="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        res = paired_test(raw, mini, a.alternative)
        stats_rows.append({"metric": col, "alternative": a.alternative, **res})

        for pos, vals in ((1, raw), (2, mini)):
            if len(vals) > 1 and np.ptp(vals) > 0:
                parts = ax.violinplot(vals, positions=[pos], showmedians=True, showextrema=False)
                for body in parts["bodies"]:
                    body.set_alpha(0.5)
        ax.plot([1, 2], [raw, mini], color="grey", alpha=0.25, lw=0.6, zorder=2)
        ax.scatter(np.full(len(raw), 1), raw, s=6, color="black", alpha=0.5, zorder=3)
        ax.scatter(np.full(len(mini), 2), mini, s=6, color="black", alpha=0.5, zorder=3)
        ax.set_xticks([1, 2], ["raw", "minimized"])
        ax.set_ylabel("clash count")
        p = res["p_value"]
        ptxt = "p = n/a (no differences)" if np.isnan(p) else f"Wilcoxon p = {p:.2g}"
        ax.set_title(f"{label} (n={len(wide)})\n{ptxt}", fontsize=9)
        print(f"[plot_clash_test] {col}: n={len(wide)} median {res['median_raw']:.0f} -> "
              f"{res['median_minimized']:.0f}, {ptxt}, rank-biserial={res['rank_biserial']:.2f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for fmt in a.formats.split(","):
        fig.savefig(f"{out.with_suffix('')}.{fmt.strip()}")
    plt.close(fig)
    pd.DataFrame(stats_rows).to_csv(a.out_stats, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
