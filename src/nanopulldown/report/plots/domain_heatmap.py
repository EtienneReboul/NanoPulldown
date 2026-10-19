#!/usr/bin/env python3
"""
nanopulldown.report.plots.domain_heatmap — Stage 2e report figure
==========================================================
One bait-domain x prey-domain heatmap PER diffusion sample (no clustering or
aggregation needed -- 5 samples is few enough to just look at each one
directly, per the project plan). Cell = mean inverse-PAE
(max(0, 1 - PAE/pae_cutoff), i.e. AlphaPulldown's PAE-plot confidence sense:
low PAE -> bright/high value -> confident relative position) between every
residue pair whose chains map to that (bait-domain, prey-domain) cell,
averaged over the pair -- the domain-aggregation idea is ported directly
from ab_initio_pipeline's scripts/plot_plip_heatmaps.py `domain_map()`, with
PLIP contact counts swapped for PAE confidence since nanopulldown has no
PLIP step.

Axes = domain names from configs/curated_domains.yaml `domains:`, else the
proposed ones from data/annotation/<pair>/annotation.yaml (--annotation),
ordered by residue position; a chain with neither falls back to 50-residue
buckets (same fallback as
plot_plip_heatmaps.py, minus its RNA/DNA nt-bucket branch -- protein-protein
only).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from nanopulldown.core.domains import domain_map
from nanopulldown.core.fold_table import read_fold
from nanopulldown.core.proteins import chain_labels, load_proteins, pair_labels
from nanopulldown.core.specs import load_pair
from nanopulldown.report.theme import lazy_theme

TH = lazy_theme()


def one_heatmap(ax, pae, chain_ids, res_idx, bait_chains, prey_chains, dmap, pae_cutoff, title):
    conf = np.clip(1.0 - pae / pae_cutoff, 0.0, None)
    row_labels, col_labels = [], []
    row_of = {}
    col_of = {}
    for i, (c, r) in enumerate(zip(chain_ids, res_idx)):
        if c in bait_chains:
            lbl = dmap[c](int(r))
            row_of.setdefault(lbl, []).append(i)
        elif c in prey_chains:
            lbl = dmap[c](int(r))
            col_of.setdefault(lbl, []).append(i)
    # sequence order (chain, then first residue), not alphabetical -- "A:101"
    # must not sort before "A:51"
    def _order(groups):
        return sorted(groups, key=lambda l: (chain_ids[groups[l][0]], int(res_idx[groups[l][0]])))
    row_labels = _order(row_of)
    col_labels = _order(col_of)
    if not row_labels or not col_labels:
        ax.text(0.5, 0.5, "no bait/prey residues found", ha="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        return None
    mat = np.zeros((len(row_labels), len(col_labels)))
    for ri, rl in enumerate(row_labels):
        ii = row_of[rl]
        for ci, cl in enumerate(col_labels):
            jj = col_of[cl]
            mat[ri, ci] = conf[np.ix_(ii, jj)].mean()
    # single-hue ramp: 0 (no confident relative placement, the common case for a
    # non-binder) recedes into the surface, confident cells stand out
    im = ax.imshow(mat, cmap=TH.cmap(), aspect="auto", vmin=0, vmax=1)
    def _short(lbl, n=28):                      # long InterPro names squash the grid
        return lbl if len(lbl) <= n else lbl[:n - 1] + "…"
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels([_short(l) for l in col_labels], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels([_short(l) for l in row_labels], fontsize=7)
    ax.set_title(title, fontsize=8)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                color=TH.cell_ink(v))
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--fold", required=True, help="results/<pair>/raw/fold.parquet")
    ap.add_argument("--annotation", default=None,
                    help="data/annotation/<pair>/annotation.yaml; proposed domains used "
                         "for chains the spec has no curated `domains:` for")
    ap.add_argument("--pae-cutoff", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    TH.apply()
    spec = load_pair(a.pair)
    if a.annotation and Path(a.annotation).exists():
        proposed = (yaml.safe_load(Path(a.annotation).read_text()) or {}).get("domains") or {}
        spec["domains"] = {**proposed, **(spec.get("domains") or {})}
    bait_chains = set(spec.get("bait_chains", []))
    prey_chains = set(spec.get("prey_chains", []))
    proteins = load_proteins()
    clabel = chain_labels(spec, proteins)
    names = pair_labels(spec, proteins)
    dmap = {c: domain_map(spec, c, clabel.get(c)) for c in bait_chains | prey_chains}

    panels = []
    for r in read_fold(Path(a.fold)):
        if r["pae"] is None:
            continue
        n = r["pae"].shape[0]
        chain_ids = r["token_chain"] if len(r["token_chain"]) == n else np.array([""] * n)
        # ESMFold2's residue_index is 0-based per chain; domains are 1-based
        res_idx = np.asarray(r["residue_index"], dtype=int).copy()
        for c in np.unique(chain_ids):
            m = chain_ids == c
            if m.any() and res_idx[m].min() == 0:
                res_idx[m] += 1
        panels.append((f"sample {r['sample_index']}", r["pae"], chain_ids, res_idx))

    if not panels:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, f"{names['label']}: no PAE arrays available", ha="center")
        TH.save_all(fig, a.out, a.formats)
        return 0

    ncol = min(3, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(6.5, 4.4 * ncol), 3.8 * nrow + 0.4),
                             squeeze=False, constrained_layout=True)
    im = None
    for k, (title, pae, chain_ids, res_idx) in enumerate(panels):
        im = one_heatmap(axes[k // ncol][k % ncol], pae, chain_ids, res_idx,
                         bait_chains, prey_chains, dmap, a.pae_cutoff, title) or im
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label=f"1 - PAE/{a.pae_cutoff:g}")
    fig.suptitle(f"{names['label']}  —  bait x prey domain confidence (1 - PAE/{a.pae_cutoff:g})\n"
                 f"{names['long']}  [{a.pair}]", fontsize=9)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    TH.save_all(fig, out, a.formats, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_domain_heatmap] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
