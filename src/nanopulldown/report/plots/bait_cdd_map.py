#!/usr/bin/env python3
"""
nanopulldown.report.plots.bait_cdd_map — Stage 2g report figure (per bait)
==========================================================================
Where every prey lands on ONE bait protein, against the bait's CDD
annotation: the cross-pair view the per-pair plip-cdd figure can't give.
Bait residues are in full-length UniProt numbering, so fragment pairs (bait
fragments numbered from 1) line up (nanopulldown.report.cdd.to_full).

  profile      mean PLIP contact frequency per bait residue, one line per
               control group (known partners / preys / decoys)
  CDD rows     the bait's CDD domain locations and conserved-feature sites,
               the union over its pairs (a fragment only carries the part
               it covers)
  heatmap      one row per prey protein: per bait residue, the fraction of
               samples whose PLIP bait-prey interactions include it (a prey
               folded as several fragment pairs: the max over them). Rows
               grouped as known partners / preys / decoys, then ordered by
               the share of the bait's site residues they contact, then by
               total contact

Groups come from the bait's results_table.csv (prey_sources), as in the
violin and calibration pages (violins.control_groups).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd

from nanopulldown.core.proteins import load_proteins, protein_name
from nanopulldown.core.specs import load_pairs
from nanopulldown.report.cdd import contacts_by_sample, read_cdd, to_full
from nanopulldown.report.plots.domain_bar import AX_W_IN, FIG_W, INK, INK2, LEFT_IN, ROW_IN, RULE, TH, _bar, _clip
from nanopulldown.report.plots.violins import GROUPS, control_groups

GROUP_SLOT = {"known partners": "green", "preys": "blue", "decoys": "gray"}
MAX_LABELLED_ROWS = 80
MAP_W = FIG_W + 0.7                  # the domain-bar geometry plus room for the colorbar


def bait_cdd(cdd: pd.DataFrame, specs: dict, pairs: list[str]) -> tuple[list, dict]:
    """The bait's CDD annotation in full-length numbering, united over pairs:
    ([(accession, name, start, end)], {(accession, site): {residue}})."""
    locs: dict[tuple, tuple] = {}
    sites: dict[tuple, set] = defaultdict(set)
    for r in cdd[(cdd["role"] == "bait") & cdd["pair"].isin(pairs)].itertuples():
        reg = specs[r.pair]["bait"].get("regions")
        if r.site is None or pd.isna(r.site):
            locs.setdefault((r.cdd_accession, r.cdd_name), []).append(
                (to_full(int(r.start), reg), to_full(int(r.end), reg)))
        else:
            sites[(r.cdd_accession, r.site)] |= {to_full(int(x), reg) for x in r.residues}
    merged = []
    for (acc, name), spans in locs.items():               # overlapping pieces of one model -> one span
        for s, e in sorted(spans):
            if merged and merged[-1][0] == acc and s <= merged[-1][3] + 1:
                merged[-1] = (acc, name, merged[-1][2], max(merged[-1][3], e))
            else:
                merged.append((acc, name, s, e))
    return sorted(merged, key=lambda x: x[2]), dict(sites)


def prey_matrix(t: pd.DataFrame, specs: dict, results_root: Path, n_samples: int):
    """(preys [{key, label, group}], {key: {full residue: frequency}}, max residue,
    (first, last) full-length residue any pair's bait chain covers)."""
    freq: dict[str, dict[int, float]] = defaultdict(dict)
    preys, L, lo = {}, 0, None
    for pair, g in t.groupby("pair", sort=False):
        spec = specs.get(pair)
        path = results_root / pair / "plip_contacts.parquet"
        if spec is None or not path.exists():
            continue
        reg = spec["bait"].get("regions")
        chains = set(spec.get("bait_chains", []))
        L = max([L] + [to_full(len(c["sequence"]), reg) for c in spec["sequences"] if c["id"] in chains])
        lo = min(lo or L, to_full(1, reg))
        key = str(g["prey_uniprot"].iloc[0])
        preys.setdefault(key, {"key": key, "label": str(g["prey"].iloc[0]).split(" ")[0],
                               "group": g["group"].iloc[0]})
        pl = pd.read_parquet(path, columns=["sample", "resnr", "reschain", "resnr_lig", "reschain_lig"])
        counts: dict[int, int] = defaultdict(int)
        for res in contacts_by_sample(pl, chains).values():
            for p in {to_full(r, reg) for _, r in res}:      # bait copies: a residue once per sample
                counts[p] += 1
        row = freq[key]
        for p, n in counts.items():
            row[p] = max(row.get(p, 0.0), n / max(n_samples, 1))
    return list(preys.values()), freq, L, (lo or 1, L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bait", required=True, help="bait UniProt accession")
    ap.add_argument("--table", required=True, help="the bait's results_table.csv")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--cdd", required=True, help="<annotation dir>/cdd_sites.parquet")
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--decoy-tag", default="decoy")
    ap.add_argument("--positive-tag", default="known")
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    TH.apply()
    specs = load_pairs()
    t = pd.read_csv(a.table)
    t = t.assign(group=control_groups(t, a.decoy_tag, a.positive_tag))
    t = t[t["group"].notna()]
    cdd = read_cdd(a.cdd)
    pairs = list(t["pair"].unique())
    locs, sites = bait_cdd(cdd, specs, pairs) if cdd is not None else ([], {})
    preys, freq, L, (x_lo, x_hi) = prey_matrix(t, specs, Path(a.results_root), a.num_samples)
    L = max([L, 1] + [e for *_, e in locs] + [max(r) for r in sites.values() if r])

    site_res = sorted({r for rs in sites.values() for r in rs})
    M = np.zeros((len(preys), L))
    for i, p in enumerate(preys):
        for r, v in freq[p["key"]].items():
            if 1 <= r <= L:
                M[i, r - 1] = v
    cov = (M[:, [r - 1 for r in site_res]] > 0).mean(axis=1) if site_res else np.zeros(len(preys))
    order = sorted(range(len(preys)), key=lambda i: (GROUPS.index(preys[i]["group"]), -cov[i], -M[i].sum()))
    M, preys, cov = M[order], [preys[i] for i in order], cov[order]

    # geometry (inches): title, profile, CDD rows, heatmap, x label, legend
    trk_rows = len(locs) + len(sites) + 1
    n = len(preys)
    title_in, prof_in, trk_in = 0.55, 1.1, ROW_IN * (trk_rows + 0.4)
    heat_in = float(np.clip(n * 0.022, 1.6, 14.0))
    bot_in = 0.85
    H = title_in + prof_in + 0.1 + trk_in + 0.1 + heat_in + bot_in
    x0, w = LEFT_IN / MAP_W, AX_W_IN / MAP_W
    fig = plt.figure(figsize=(MAP_W, H))
    axh = fig.add_axes([x0, bot_in / H, w, heat_in / H])
    axt = fig.add_axes([x0, (bot_in + heat_in + 0.1) / H, w, trk_in / H], sharex=axh)
    axp = fig.add_axes([x0, (bot_in + heat_in + 0.2 + trk_in) / H, w, prof_in / H], sharex=axh)
    axg = fig.add_axes([x0 - 0.12 / MAP_W, bot_in / H, 0.08 / MAP_W, heat_in / H])

    gcolor = {g: TH.slot(s) for g, s in GROUP_SLOT.items()}
    name = protein_name(a.bait, load_proteins())
    label = name if a.bait in name else f"{name} ({a.bait})"
    axp.set_title(f"{label}: where PLIP puts each prey, against the bait's CDD sites",
                  fontsize=9, loc="left", color=INK, pad=16)
    n_by = {g: sum(p["group"] == g for p in preys) for g in GROUPS}
    groups = ", ".join(f"{v} {g}" for g, v in n_by.items() if v)
    axp.text(0, 1.04, f"{n} prey proteins{f' ({groups})' if sum(map(bool, n_by.values())) > 1 else ''} · "
             f"{len(site_res)} CDD site residues · residue numbering: full-length UniProt"
             + (f" · folded bait residues {x_lo}–{x_hi}" if (x_lo, x_hi) != (1, L) else "")
             + (" · row label %: the site residues that prey contacts" if site_res and n <= MAX_LABELLED_ROWS else ""),
             transform=axp.transAxes, fontsize=7, color=INK2, va="bottom")

    # profile: mean contact frequency per group
    x = np.arange(1, L + 1)
    for g in GROUPS:
        rows = [i for i, p in enumerate(preys) if p["group"] == g]
        if rows:
            axp.plot(x, M[rows].mean(axis=0), color=gcolor[g], lw=1.1, label=f"{g} (n={len(rows)})")
    for r in site_res:
        axp.axvline(r, color=TH.color("site"), lw=0.6, alpha=0.35, zorder=0)
    axp.set_ylabel("mean contact\nfrequency", fontsize=7.5, color=INK2)
    axp.tick_params(labelsize=7, colors=INK2)
    axp.legend(loc="upper right", bbox_to_anchor=(-0.08, 1.0), fontsize=6.5, frameon=False, handlelength=1.6)
    plt.setp(axp.get_xticklabels(), visible=False)
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)

    # CDD rows
    tr = axt.get_yaxis_transform()
    y = 0.0
    for acc, nm, s, e in locs:
        axt.axhline(y, color=RULE, lw=0.6, zorder=0)
        _bar(axt, y, s, e, TH.slot("aqua"), L, None, 0.5, 0.9)
        axt.text(-0.012, y + 0.17, _clip(nm, 46), transform=tr, ha="right", va="center", fontsize=7, color=INK)
        axt.text(-0.012, y - 0.24, f"CDD {acc} · {s}–{e}", transform=tr, ha="right", va="center",
                 fontsize=5.6, color=INK2)
        y -= 1
        for (sacc, desc), res in sites.items():
            if sacc != acc:
                continue
            axt.axhline(y, color=RULE, lw=0.6, zorder=0)
            axt.vlines(sorted(res), y - 0.3, y + 0.3, color=TH.color("site"), lw=1.4)
            axt.text(-0.012, y, _clip(desc, 46), transform=tr, ha="right", va="center", fontsize=7, color=INK)
            y -= 1
    if not locs:
        axt.text(0.5, 0.5, "no CDD domain on this bait", transform=axt.transAxes, ha="center",
                 va="center", fontsize=7.5, color=INK2)
    axt.set_ylim(min(y, -1) + 0.4, 0.6)
    axt.set_yticks([])
    plt.setp(axt.get_xticklabels(), visible=False)
    for sp in ("top", "right", "left", "bottom"):
        axt.spines[sp].set_visible(False)
    axt.tick_params(axis="x", length=0)

    # heatmap + group strip
    if n:
        im = axh.imshow(M, cmap=TH.cmap(), vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                        extent=(0.5, L + 0.5, n, 0))
        strip = np.array([[GROUPS.index(p["group"])] for p in preys])
        axg.imshow(strip, cmap=ListedColormap([gcolor[g] for g in GROUPS]), vmin=0, vmax=len(GROUPS) - 1,
                   aspect="auto", interpolation="nearest", extent=(0, 1, n, 0))
        for i in range(1, n):
            if preys[i]["group"] != preys[i - 1]["group"]:
                axh.axhline(i, color=INK, lw=0.8)
        if n <= MAX_LABELLED_ROWS:
            axg.set_yticks(np.arange(n) + 0.5)
            axg.set_yticklabels([f"{p['label']}  {c:.0%}" if site_res else p["label"]
                                 for p, c in zip(preys, cov)], fontsize=5.5)
        else:
            axg.set_yticks([])
            axg.set_ylabel(f"{n} prey proteins", fontsize=7.5, color=INK2)
        cax = fig.add_axes([(LEFT_IN + AX_W_IN + 0.12) / MAP_W, bot_in / H, 0.1 / MAP_W, min(heat_in, 2.0) / H])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("fraction of samples contacting the residue", fontsize=6.5, color=INK2)
        cb.ax.tick_params(labelsize=6, colors=INK2)
    else:
        axh.text(0.5, 0.5, "no PLIP contacts for this bait", transform=axh.transAxes, ha="center",
                 va="center", fontsize=8, color=INK2)
        axg.set_yticks([])
    axg.set_xticks([])
    for sp in axg.spines.values():
        sp.set_visible(False)
    axh.set_xlim(x_lo - 0.5, x_hi + 0.5)       # shared: only what any pair folded
    axh.set_yticks([])
    axh.tick_params(axis="x", labelsize=7, colors=INK2)
    axh.set_xlabel(f"bait residue ({name}, full-length numbering)", fontsize=8, color=INK2)

    handles = [mlines.Line2D([], [], color=gcolor[g], lw=4, label=g) for g in GROUPS if n_by[g]]
    handles.append(mlines.Line2D([], [], color=TH.color("site"), lw=1.4, label="CDD site residue"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5 + (LEFT_IN - 0.3) / MAP_W / 2, 0.005), handlelength=1.6)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    TH.save_all(fig, out, a.formats, png_dpi=150)
    plt.close(fig)
    print(f"[plot_bait_cdd_map] {n} prey(s) x {L} residue(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
