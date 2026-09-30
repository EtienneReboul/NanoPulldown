#!/usr/bin/env python3
"""
scripts/plot_domain_heatmap.py — Stage 2e report figure
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

Axes = domain names from configs/<pair>.yaml `domains:`; a chain with no
curated domains falls back to 50-residue buckets (same fallback as
plot_plip_heatmaps.py, minus its RNA/DNA nt-bucket branch -- protein-protein
only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
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
    row_labels = sorted(row_of)
    col_labels = sorted(col_of)
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
    im = ax.imshow(mat, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_title(title, fontsize=8)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                color="white" if v < mat.max() * 0.6 else "black")
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--metadata-root", default="results/metadata")
    ap.add_argument("--pae-cutoff", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    spec = yaml.safe_load(Path(a.spec).read_text())
    bait_chains = set(spec.get("bait_chains", []))
    prey_chains = set(spec.get("prey_chains", []))
    dmap = {c: domain_map(spec, c) for c in bait_chains | prey_chains}

    meta = pd.read_parquet(Path(a.metadata_root) / a.pair / "model_metadata.parquet")
    meta = meta[meta["has_array"]].sort_values("sample_index")
    h5path = Path(a.metadata_root) / a.pair / "arrays.h5"

    panels = []
    with h5py.File(h5path, "r") as h5:
        topo_key = f"_topology/{a.pair}/chain_id"
        chain_id_int = np.asarray(h5[topo_key][:], dtype=int) if topo_key in h5 else None
        chain_lookup = (json.loads(h5[topo_key].attrs.get("chain_lookup", "{}"))
                        if topo_key in h5 else {})
        for _, r in meta.iterrows():
            key = r["array_key"]
            if f"{key}/pae" not in h5:
                continue
            pae = np.asarray(h5[f"{key}/pae"][:], dtype=np.float32)
            extra = json.loads(r["extra_json"]) if r.get("extra_json") else {}
            res_idx = extra.get("residue_index")
            n = pae.shape[0]
            if chain_id_int is not None and len(chain_id_int) == n:
                chain_ids = np.array([chain_lookup.get(str(c), str(c)) for c in chain_id_int])
            else:
                chain_ids = np.array([""] * n)
            res_idx = np.asarray(res_idx[:n]) if res_idx and len(res_idx) >= n else np.arange(1, n + 1)
            panels.append((f"sample {int(r['sample_index'])}", pae, chain_ids, res_idx))

    if not panels:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, f"{a.pair}: no PAE arrays available", ha="center")
        for fmt in a.formats.split(","):
            fig.savefig(f"{Path(a.out).with_suffix('')}.{fmt.strip()}")
        return 0

    ncol = min(3, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.8 * nrow),
                             squeeze=False, constrained_layout=True)
    for k, (title, pae, chain_ids, res_idx) in enumerate(panels):
        one_heatmap(axes[k // ncol][k % ncol], pae, chain_ids, res_idx,
                    bait_chains, prey_chains, dmap, a.pae_cutoff, title)
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"{a.pair}: bait x prey domain confidence (1 - PAE/{a.pae_cutoff:g}), per diffusion sample")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for fmt in a.formats.split(","):
        fig.savefig(f"{out.with_suffix('')}.{fmt.strip()}")
    plt.close(fig)
    print(f"[plot_domain_heatmap] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
