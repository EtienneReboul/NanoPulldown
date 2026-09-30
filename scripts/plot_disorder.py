#!/usr/bin/env python3
"""
scripts/plot_disorder.py — Stage 1 report figure
================================================
Superposed per-residue disorder / binding(MoRF) curves for one protein
chain, from AIUPred's two tracks, with residues where the "disorder" track
exceeds --cutoff shaded.

Adapted from ab_initio_pipeline's scripts/plot_disorder.py: one TSV (two
tracks: disorder, binding) instead of a separate disorder.tsv + morf.tsv,
and a single-track disorder mask instead of a multi-tool consensus (there's
only one disorder predictor now).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

TRACK_STYLE = {
    "disorder": ("#c44e52", "-"),
    "binding":  ("#55a868", "-"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disorder", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--cutoff", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="svg")
    args = ap.parse_args()

    dis = pd.read_csv(args.disorder, sep="\t")
    dis = dis[dis["chain"] == args.chain]

    fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    if dis.empty:
        ax.text(0.5, 0.5, f"chain {args.chain}: no disorder tracks",
                ha="center", va="center", transform=ax.transAxes)
    else:
        wide = dis.pivot_table(index="resi", columns="track", values="score").sort_index()
        for t in wide.columns:
            c, ls = TRACK_STYLE.get(t, ("#333333", "-"))
            ax.plot(wide.index, wide[t], color=c, ls=ls, lw=1.2, label=t)
        if "disorder" in wide.columns:
            strong = wide["disorder"] >= args.cutoff
            ax.fill_between(wide.index, 0, 1, where=strong, color="#dd8452",
                            alpha=0.15, step="mid", label="disordered")
        ax.axhline(args.cutoff, color="grey", lw=.7, ls=":")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)
    ax.set_xlabel("residue"); ax.set_ylabel("score")
    ax.set_title(f"chain {args.chain} — AIUPred disorder / binding", fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    for fmt in args.formats.split(","):
        fig.savefig(f"{stem}.{fmt.strip()}", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_disorder] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
