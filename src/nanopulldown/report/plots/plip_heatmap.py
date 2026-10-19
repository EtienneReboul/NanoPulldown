#!/usr/bin/env python3
"""
nanopulldown.report.plots.plip_heatmap — Stage 2e report figure
=========================================================
Domain x domain mean-PLIP-contact heatmap for one pair, from
results/<pair>/plip_contacts.parquet (pliparser's common columns -- resnr,
restype, reschain, resnr_lig, restype_lig, reschain_lig, dist,
interaction_type -- plus the `pair`/`sample` columns added by
nanopulldown.plip.aggregate). One panel, averaged over all
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from nanopulldown.core.domains import domain_map
from nanopulldown.core.proteins import chain_labels, load_proteins, pair_labels
from nanopulldown.core.specs import load_pair
from nanopulldown.report.theme import lazy_theme

TH = lazy_theme()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--contacts", required=True, help="results/<pair>/plip_contacts.parquet")
    ap.add_argument("--annotation", default=None,
                    help="data/annotation/<pair>/annotation.yaml; proposed domains used "
                         "for chains the spec has no curated `domains:` for")
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    TH.apply()
    spec = load_pair(a.pair)
    proteins = load_proteins()
    clabel = chain_labels(spec, proteins)
    names = pair_labels(spec, proteins)
    df = pd.read_parquet(a.contacts)
    if a.annotation and Path(a.annotation).exists():
        proposed = (yaml.safe_load(Path(a.annotation).read_text()) or {}).get("domains") or {}
        spec["domains"] = {**proposed, **(spec.get("domains") or {})}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.2), constrained_layout=True)

    if df.empty:
        ax.text(0.5, 0.5, f"{names['label']}: no PLIP contacts", ha="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        n_samples = max(df["sample"].nunique(), 1)
        chains = sorted(set(df["reschain"]) | set(df["reschain_lig"]))
        dmap = {c: domain_map(spec, c, clabel.get(c)) for c in chains}
        df = df.assign(
            rdom=df.apply(lambda x: dmap[x["reschain"]](int(x["resnr"])), axis=1),
            ldom=df.apply(lambda x: dmap[x["reschain_lig"]](int(x["resnr_lig"])), axis=1),
        )
        piv = df.pivot_table(index="rdom", columns="ldom", values="dist",
                             aggfunc="count", fill_value=0) / n_samples
        vmax = max(float(piv.values.max()), 1e-9)
        im = ax.imshow(piv.values, cmap=TH.cmap(), aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=7)
        for (i, j), v in np.ndenumerate(piv.values):
            if v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color=TH.cell_ink(v / vmax))
        fig.colorbar(im, ax=ax, label=f"mean contacts / sample (n={n_samples})")

    ax.set_title(f"{names['label']}: PLIP domain x domain contacts\n[{a.pair}]", fontsize=9)
    TH.save_all(fig, out, a.formats)
    plt.close(fig)
    print(f"[plot_plip_heatmap] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
