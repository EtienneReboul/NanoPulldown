#!/usr/bin/env python3
"""
nanopulldown.report.plots.plip_cdd — Stage 2e report figure
===========================================================
Where PLIP puts the interface on each chain of a pair, against the chain's
CDD annotation (<annotation dir>/cdd_sites.parquet). One panel per chain,
bait chains first, residue on x:

  contacts       per-residue PLIP contact frequency (fraction of the
                 --num-samples samples whose bait-prey interactions include
                 the residue); CDD site residues in the site colour, CDD
                 domain locations shaded behind
  Domain map     the segments the domain heatmaps use (curated or proposed,
                 as in the domain-bar figure)
  CDD rows       each CDD domain location, then one row per conserved-feature
                 site of it: a tick per site residue, filled when PLIP
                 contacts it in at least one sample, with k/n contacted

The panel header summarises the chain with nanopulldown.report.cdd's
statistics (sites contacted in any sample; enrichment and share of contacts
in CDD domains, averaged over samples).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from nanopulldown.core.specs import load_pair
from nanopulldown.report.cdd import SIDES, cdd_residues, cdd_stats, contacts_by_sample, read_cdd
from nanopulldown.report.plots.domain_bar import (
    AX_W_IN, FIG_W, INK, INK2, KIND_COLOR, LEFT_IN, ROW_IN, RULE, TH, _bar, _clip, chain_segments,
    protein_title,
)

CDD_TABLE = ["pair", "role", "chain", "cdd_accession", "cdd_name", "start", "end", "site", "residues"]
BAR_IN, TITLE_IN, XLAB_IN, PANEL_GAP_IN = 0.85, 0.5, 0.45, 0.3


def _colors():
    return {"site": TH.color("site"), "cdd": TH.slot("aqua"), "contact": INK2}


def chain_rows(segs: list[dict], cdd: pd.DataFrame, hit_any: set[int]) -> list[dict]:
    """Track rows of one chain: {label, sub, kind, bars | ticks}."""
    col = _colors()
    rows = [{"label": "Domain map", "sub": None, "kind": "bars",
             "bars": [(d["start"], d["end"], KIND_COLOR.get(d["kind"], INK2),
                       d["name"] if d["kind"] == "domain" else None, 0.78, 1.0)
                      for d in segs if d["kind"] != "morf"]}]
    for (acc, name, s, e), g in cdd.groupby(["cdd_accession", "cdd_name", "start", "end"], sort=False):
        rows.append({"label": name, "sub": f"CDD {acc} · {s}–{e}", "kind": "bars", "gap": True,
                     "bars": [(s, e, col["cdd"], None, 0.5, 0.9)]})
        for site, res in zip(g["site"], g["residues"]):
            if site is None or pd.isna(site):
                continue
            res = [int(r) for r in res]
            k = sum(r in hit_any for r in res)
            rows.append({"label": site, "sub": f"{k}/{len(res)} residues contacted", "kind": "ticks",
                         "ticks": res, "hit": hit_any})
    return rows


def draw_panel(fig, H, y_top, role, cid, spec, ann, cdd, by_smp, n_samples) -> float:
    """One chain's panel with its top edge at y_top (inches from the bottom);
    returns its bottom edge."""
    col = _colors()
    seq = next(c["sequence"] for c in spec["sequences"] if c["id"] == cid)
    L = len(seq)
    segs, _ = chain_segments(spec, ann, cid)
    cdd_c = cdd[cdd["chain"] == cid]
    sites, spans = cdd_residues(cdd_c, {cid})
    hit_any = {r for s in by_smp.values() for _, r in s}
    rows = chain_rows(segs, cdd_c, hit_any)

    freq = np.zeros(L + 1)
    for s in by_smp.values():
        for c, r in s:
            if r <= L:
                freq[r] += 1
    freq /= max(n_samples, 1)

    # header: the chain's own statistics (mean over samples)
    stats = [cdd_stats(by_smp.get(i, set()), sites, spans, L)
             for i in range(n_samples)]
    enrich = np.nanmean([s["cdd_sites_enrich"] for s in stats]) if sites and hit_any else np.nan
    in_cdd = np.nanmean([s["contacts_in_cdd"] for s in stats]) if spans and hit_any else np.nan
    n_hit = len({r for c, r in sites} & hit_any)
    summary = (f"CDD sites contacted: {n_hit}/{len(sites)} residues (any of {n_samples} samples)"
               if sites else "no CDD site annotated")
    if sites and not np.isnan(enrich):
        summary += f" · site enrichment {enrich:.1f}×"
    summary += (f" · contacts in CDD domains {in_cdd:.0%}" if not np.isnan(in_cdd)
                else " · no CDD domain" if not spans else "")
    if not hit_any:
        summary += " · no PLIP contact on this chain"

    n_gap = sum(1 for r in rows if r.get("gap"))
    trk_in = ROW_IN * (len(rows) + 0.6 * n_gap + 0.6)
    bar_y = y_top - TITLE_IN - BAR_IN
    trk_y = bar_y - 0.08 - trk_in
    axb = fig.add_axes([LEFT_IN / FIG_W, bar_y / H, AX_W_IN / FIG_W, BAR_IN / H])
    ax = fig.add_axes([LEFT_IN / FIG_W, trk_y / H, AX_W_IN / FIG_W, trk_in / H], sharex=axb)

    site_res = {r for c, r in sites}
    x = np.arange(1, L + 1)
    colors = [col["site"] if i in site_res else col["contact"] for i in x]
    for s, e in zip(cdd_c.loc[cdd_c["site"].isna(), "start"], cdd_c.loc[cdd_c["site"].isna(), "end"]):
        axb.axvspan(s - 0.5, e + 0.5, color=col["cdd"], alpha=0.13, lw=0)
    axb.bar(x, freq[1:], width=1.0, color=colors, lw=0)
    axb.set_ylim(0, 1.02)
    axb.set_yticks([0, 0.5, 1])
    axb.set_ylabel("contact\nfrequency", fontsize=7.5, color=INK2)
    axb.tick_params(labelsize=7, colors=INK2)
    plt.setp(axb.get_xticklabels(), visible=False)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.set_title(f"{role.upper()} · {protein_title(spec, cid)} · chain {cid}", fontsize=9, loc="left",
                  color=INK, pad=14)
    axb.text(0, 1.04, summary, transform=axb.transAxes, fontsize=7, color=INK2, va="bottom")

    tr = ax.get_yaxis_transform()
    y = 0.0
    for r in rows:
        if r.get("gap"):
            y -= 0.6
        ax.axhline(y, color=RULE, lw=0.6, zorder=0)
        if r["kind"] == "bars":
            for s, e, c, text, h, alpha in r["bars"]:
                _bar(ax, y, int(s), int(e), c, L, text, h, alpha)
        else:
            hit = [t for t in r["ticks"] if t in r["hit"]]
            miss = [t for t in r["ticks"] if t not in r["hit"]]
            ax.vlines(miss, y - 0.22, y + 0.22, color=INK2, lw=0.9, alpha=0.8)
            ax.vlines(hit, y - 0.34, y + 0.34, color=col["site"], lw=1.8)
        dy = 0.17 if r.get("sub") else 0
        ax.text(-0.012, y + dy, _clip(r["label"], 46), transform=tr, ha="right", va="center",
                fontsize=7, color=INK)
        if r.get("sub"):
            ax.text(-0.012, y - 0.24, _clip(r["sub"], 60), transform=tr, ha="right", va="center",
                    fontsize=5.6, color=INK2)
        y -= 1
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_xlim(0.5, L + 0.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7, colors=INK2)
    ax.set_xlabel("residue", fontsize=8, color=INK2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    return trk_y - XLAB_IN


def panel_height(cdd, cid) -> float:
    c = cdd[cdd["chain"] == cid]
    n = 1 + len(c) + 0.6 * int(c["site"].isna().sum())
    return TITLE_IN + BAR_IN + 0.08 + ROW_IN * (n + 0.6) + XLAB_IN


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--contacts", required=True, help="results/<pair>/plip_contacts.parquet")
    ap.add_argument("--cdd", required=True, help="<annotation dir>/cdd_sites.parquet")
    ap.add_argument("--annotation", required=True, help="data/annotation/<pair>/annotation.yaml")
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    TH.apply()
    spec = load_pair(a.pair)
    ann = yaml.safe_load(Path(a.annotation).read_text()) or {}
    cdd = read_cdd(a.cdd)
    cdd = cdd[cdd["pair"] == a.pair] if cdd is not None else pd.DataFrame(columns=CDD_TABLE)
    pl = pd.read_parquet(a.contacts, columns=["sample", "resnr", "reschain", "resnr_lig", "reschain_lig"])
    panels = [(side, cid) for side in SIDES for cid in spec.get(f"{side}_chains", [])]
    by_smp = {cid: contacts_by_sample(pl, {cid}) for _, cid in panels}

    top_in, legend_in = 0.2, 0.4
    heights = [panel_height(cdd, cid) for _, cid in panels]
    H = top_in + sum(heights) + PANEL_GAP_IN * (len(panels) - 1) + legend_in
    fig = plt.figure(figsize=(FIG_W, H))
    y = H - top_in
    for (side, cid), h in zip(panels, heights):
        draw_panel(fig, H, y, side, cid, spec, ann, cdd, by_smp[cid], a.num_samples)
        y -= h + PANEL_GAP_IN

    col = _colors()
    handles = [mpatches.Patch(color=col["contact"], label="PLIP contact frequency"),
               mpatches.Patch(color=col["cdd"], label="CDD domain"),
               mlines.Line2D([], [], color=col["site"], lw=1.8, label="CDD site residue, contacted (≥1 sample)"),
               mlines.Line2D([], [], color=INK2, lw=0.9, label="CDD site residue, not contacted")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5 + (LEFT_IN - 0.3) / FIG_W / 2, 0.005), handlelength=1.4,
               columnspacing=1.0)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    TH.save_all(fig, out, a.formats, png_dpi=150)
    plt.close(fig)
    print(f"[plot_plip_cdd] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
