#!/usr/bin/env python3
"""
scripts/plot_plip_heatmap.py — Stage 2e report figure
=========================================================
Domain x domain mean-PLIP-contact heatmap for one pair, from
results/<pair>/plip_summary.csv (pliparser's common columns -- resnr,
restype, reschain, resnr_lig, restype_lig, reschain_lig, dist,
interaction_type -- plus the `sample` column added by
scripts/aggregate_plip_summaries.py). One panel, averaged over all
esmfold2.num_diffusion_samples minimized samples.

Adapted from ab_initio_pipeline's scripts/plot_plip_heatmaps.py: nanopulldown
has one backend (ESMFold2) and no pose clustering, so the backend=/cluster=
strata don't apply -- just the one "mean contacts per sample" panel.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402


def domain_map(spec, chain):
    segs = (spec.get("domains") or {}).get(chain)
    if segs:
        def _lookup(resnr):
            for d in segs:
                if int(d["start"]) <= resnr <= int(d["end"]) and d["kind"] != "morf":
                    return f"{chain}:{d['name']}"
            return f"{chain}:?"
        return _lookup
    return lambda r: f"{chain}:{((r - 1) // 50) * 50 + 1}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--summary", required=True, help="results/<pair>/plip_summary.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    spec = yaml.safe_load(Path(a.spec).read_text())
    df = pd.read_csv(a.summary)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.2), constrained_layout=True)

    if df.empty:
        ax.text(0.5, 0.5, f"{a.pair}: no PLIP contacts", ha="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        n_samples = max(df["sample"].nunique(), 1)
        chains = sorted(set(df["reschain"]) | set(df["reschain_lig"]))
        dmap = {c: domain_map(spec, c) for c in chains}
        df = df.assign(
            rdom=df.apply(lambda x: dmap[x["reschain"]](int(x["resnr"])), axis=1),
            ldom=df.apply(lambda x: dmap[x["reschain_lig"]](int(x["resnr_lig"])), axis=1),
        )
        piv = df.pivot_table(index="rdom", columns="ldom", values="dist",
                             aggfunc="count", fill_value=0) / n_samples
        im = ax.imshow(piv.values, cmap="magma", aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=7)
        for (i, j), v in np.ndenumerate(piv.values):
            if v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < piv.values.max() * 0.6 else "black")
        fig.colorbar(im, ax=ax, label=f"mean contacts / sample (n={n_samples})")

    ax.set_title(f"{a.pair}: PLIP domain x domain contacts", fontsize=9)
    for fmt in a.formats.split(","):
        fig.savefig(f"{out.with_suffix('')}.{fmt.strip()}")
    plt.close(fig)
    print(f"[plot_plip_heatmap] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
