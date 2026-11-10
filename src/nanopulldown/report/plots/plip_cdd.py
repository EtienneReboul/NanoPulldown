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
  ELM rows       (--elm, <annotation dir>/elm_partner_matches.parquet) on the
                 motif's chain, one row per disordered ELM motif whose binding
                 Pfam domain is on the other side (nanopulldown.annotate.elm):
                 a tick per motif residue, filled when contacted, thick when it
                 contacts the partner's binding domain, and in how many samples
                 the motif touches that domain. On the partner's chain, one row
                 per binding Pfam domain, naming the motifs it would bind.
  Coiled coils   one row per COILS segment of the chain (InterPro extra
                 feature, annotation.yaml `interpro.<chain>.features.coils`):
                 a tick per residue, filled when contacted, solid when it
                 contacts a coiled coil of the other side, and in how many
                 samples the two coils pair -- coiled coils are often predicted
                 disordered yet drive IDP-IDP pairing, with no folded domain
                 on either side for CDD or ELM to see.

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
    return {"site": TH.color("site"), "cdd": TH.slot("aqua"), "contact": INK2, "slim": TH.slot("violet"), "coil": TH.color("coils")}


MAX_ELM_ROWS = 20                    # per chain, best-ranked first; the rest are counted in the header


def contact_pairs_by_sample(pl: pd.DataFrame) -> dict[int, set]:
    """{sample: {((chain, resnr), (chain, resnr))}} of PLIP contacts, both orders."""
    out: dict[int, set] = {}
    for smp, c1, r1, c2, r2 in zip(pl["sample"], pl["reschain"].astype(str), pl["resnr"],
                                   pl["reschain_lig"].astype(str), pl["resnr_lig"]):
        a, b = (c1, int(r1)), (c2, int(r2))
        out.setdefault(int(smp), set()).update({(a, b), (b, a)})
    return out


def coil_segments(ann: dict, cid: str) -> list[tuple[int, int]]:
    """COILS segments of one chain (chain numbering) from annotation.yaml."""
    feats = (((ann.get("interpro") or {}).get(cid) or {}).get("features") or {})
    return [(int(a), int(b)) for a, b in feats.get("coils") or []]


def coil_pairing(coils: dict, panels: list, pairs_by_smp: dict) -> dict[str, list[dict]]:
    """{chain: [{start, end, res, to_cc, smp_cc, partner}]}: per coiled-coil
    segment, its residues contacting a coiled coil of the OTHER side, and the
    samples in which it does."""
    side_of = {cid: side for side, cid in panels}
    out = {}
    for side, cid in panels:
        partner = {(c, r) for c, segs in coils.items() if side_of.get(c) not in (None, side)
                   for a, b in segs for r in range(a, b + 1)}
        rows = []
        for a, b in coils.get(cid, []):
            res = set(range(a, b + 1))
            to_cc, smp_cc = set(), 0
            for pairs in pairs_by_smp.values():
                hit = {x[1] for x, y in pairs if x[0] == cid and x[1] in res and y in partner}
                to_cc |= hit
                smp_cc += bool(hit)
            rows.append({"start": a, "end": b, "res": sorted(res), "to_cc": to_cc, "smp_cc": smp_cc,
                         "partner": bool(partner)})
        out[cid] = rows
    return out


def elm_motifs(elm: pd.DataFrame, cid: str, pairs_by_smp: dict) -> list[dict]:
    """The partner-matched ELM motifs on chain `cid` -- those touching the
    partner domain in the most samples first, then by rank: one per
    motif hit, with its partner domains and the residues / samples touching them."""
    out = []
    g = elm[elm["motif_chain"] == cid].sort_values("rank")
    for (eid, s, e), m in g.groupby(["elm_identifier", "start", "end"], sort=False):
        res = set(range(int(s), int(e) + 1))
        doms = list(zip(m["partner_chain"], m["partner_pfam"], m["partner_pfam_name"],
                        m["partner_domain_start"], m["partner_domain_end"]))
        to_dom, smp_dom = set(), 0
        for pairs in pairs_by_smp.values():
            hit = {a[1] for a, b in pairs if a[0] == cid and a[1] in res
                   and any(b[0] == pc and ds <= b[1] <= de for pc, _, _, ds, de in doms)}
            to_dom |= hit
            smp_dom += bool(hit)
        out.append({"elm": eid, "start": int(s), "end": int(e), "motif": m["motif"].iloc[0],
                    "res": sorted(res), "doms": doms, "to_dom": to_dom, "smp_dom": smp_dom,
                    "known": m["known_instance"].notna().any()})
    return sorted(out, key=lambda m: -m["smp_dom"])   # stable: rank within ties


def elm_partner_domains(elm: pd.DataFrame, cid: str) -> list[dict]:
    """Binding Pfam domains on chain `cid` that a motif of the other side matches."""
    g = elm[elm["partner_chain"] == cid]
    return [{"pfam": acc, "name": name, "start": int(s), "end": int(e),
             "motifs": sorted(set(m["elm_identifier"])), "chains": sorted(set(m["motif_chain"]))}
            for (acc, name, s, e), m in g.groupby(["partner_pfam", "partner_pfam_name",
                                                    "partner_domain_start", "partner_domain_end"])]


def chain_rows(segs: list[dict], cdd: pd.DataFrame, hit_any: set[int], motifs=(), pdoms=(),
               n_samples: int = 0, coils=()) -> list[dict]:
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
    for i, c in enumerate(coils):
        k = sum(r in hit_any for r in c["res"])
        pair = f"↔ partner coil in {c['smp_cc']}/{n_samples}" if c["partner"] else "partner has no coiled coil"
        rows.append({"label": f"Coiled coil {c['start']}–{c['end']}", "gap": i == 0, "kind": "ticks",
                     "sub": f"COILS · {k}/{len(c['res'])} contacted · {pair}",
                     "ticks": c["res"], "hit": hit_any, "strong": c["to_cc"], "color": col["coil"]})
    for i, d in enumerate(pdoms):
        rows.append({"label": f"ELM partner: {d['name']}", "gap": i == 0, "kind": "bars",
                     "sub": f"{d['pfam']} · binds {', '.join(d['motifs'])} (chain {', '.join(d['chains'])})",
                     "bars": [(d["start"], d["end"], col["slim"], None, 0.5, 0.9)]})
    for i, m in enumerate(motifs[:MAX_ELM_ROWS]):
        k = sum(r in hit_any for r in m["res"])
        pf = ",".join(sorted({p for _, p, _, _, _ in m["doms"]}))
        rows.append({"label": m["elm"] + (" (known)" if m["known"] else ""), "gap": i == 0, "kind": "ticks",
                     "sub": f"{m['start']}–{m['end']} {m['motif']} · {k}/{len(m['res'])} contacted · "
                            f"→{pf} in {m['smp_dom']}/{n_samples}",
                     "ticks": m["res"], "hit": hit_any, "strong": m["to_dom"], "color": col["slim"]})
    return rows


def draw_panel(fig, H, y_top, role, cid, spec, ann, cdd, by_smp, n_samples, motifs=(), pdoms=(),
               coils=()) -> float:
    """One chain's panel with its top edge at y_top (inches from the bottom);
    returns its bottom edge."""
    col = _colors()
    seq = next(c["sequence"] for c in spec["sequences"] if c["id"] == cid)
    L = len(seq)
    segs, _ = chain_segments(spec, ann, cid)
    cdd_c = cdd[cdd["chain"] == cid]
    sites, spans = cdd_residues(cdd_c, {cid})
    hit_any = {r for s in by_smp.values() for _, r in s}
    rows = chain_rows(segs, cdd_c, hit_any, motifs, pdoms, n_samples, coils)

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
    if coils and any(c["partner"] for c in coils):
        summary += f" · coiled coils paired with the partner's: {sum(c['smp_cc'] > 0 for c in coils)}/{len(coils)}"
    if motifs:
        n_dom = sum(m["smp_dom"] > 0 for m in motifs)
        summary += f" · ELM motifs on the partner domain: {n_dom}/{len(motifs)}"
        if len(motifs) > MAX_ELM_ROWS:
            summary += f" (best {MAX_ELM_ROWS} shown)"
    if not hit_any:
        summary += " · no PLIP contact on this chain"

    n_gap = sum(1 for r in rows if r.get("gap"))
    trk_in = ROW_IN * (len(rows) + 0.6 * n_gap + 0.6)
    bar_y = y_top - TITLE_IN - BAR_IN
    trk_y = bar_y - 0.08 - trk_in
    axb = fig.add_axes([LEFT_IN / FIG_W, bar_y / H, AX_W_IN / FIG_W, BAR_IN / H])
    ax = fig.add_axes([LEFT_IN / FIG_W, trk_y / H, AX_W_IN / FIG_W, trk_in / H], sharex=axb)

    site_res = {r for c, r in sites}
    slim_res = {r for m in motifs for r in m["res"]}
    x = np.arange(1, L + 1)
    colors = [col["site"] if i in site_res else col["slim"] if i in slim_res else col["contact"] for i in x]
    for c in coils:
        axb.axvspan(c["start"] - 0.5, c["end"] + 0.5, color=col["coil"], alpha=0.13, lw=0)
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
            strong = r.get("strong", set())
            hit = [t for t in r["ticks"] if t in r["hit"] and t not in strong]
            miss = [t for t in r["ticks"] if t not in r["hit"]]
            ax.vlines(miss, y - 0.22, y + 0.22, color=INK2, lw=0.9, alpha=0.8)
            # ELM rows: contacted elsewhere faded, contacted on the partner's binding domain solid
            ax.vlines(hit, y - 0.34, y + 0.34, color=r.get("color", col["site"]), lw=1.8,
                      alpha=0.4 if "strong" in r else 1.0)
            ax.vlines(sorted(strong), y - 0.34, y + 0.34, color=r.get("color", col["site"]), lw=1.8)
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


def panel_height(cdd, cid, n_motifs: int = 0, n_pdoms: int = 0, n_coils: int = 0) -> float:
    c = cdd[cdd["chain"] == cid]
    n_elm = min(n_motifs, MAX_ELM_ROWS)
    n = (1 + len(c) + 0.6 * int(c["site"].isna().sum())
         + n_elm + n_pdoms + n_coils + 0.6 * (bool(n_elm) + bool(n_pdoms) + bool(n_coils)))
    return TITLE_IN + BAR_IN + 0.08 + ROW_IN * (n + 0.6) + XLAB_IN


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--contacts", required=True, help="results/<pair>/plip_contacts.parquet")
    ap.add_argument("--cdd", required=True, help="<annotation dir>/cdd_sites.parquet")
    ap.add_argument("--annotation", required=True, help="data/annotation/<pair>/annotation.yaml")
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--elm", default=None, help="<annotation dir>/elm_partner_matches.parquet (optional)")
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
    elm = pd.read_parquet(a.elm) if a.elm and Path(a.elm).exists() else None
    elm = elm[elm["pair"] == a.pair] if elm is not None else None
    coils = {cid: coil_segments(ann, cid) for _, cid in panels}
    pairs_by_smp = contact_pairs_by_sample(pl) if (elm is not None and len(elm)) or any(coils.values()) else {}
    coil_rows = coil_pairing(coils, panels, pairs_by_smp)
    if elm is not None and len(elm):
        motifs = {cid: elm_motifs(elm, cid, pairs_by_smp) for _, cid in panels}
        pdoms = {cid: elm_partner_domains(elm, cid) for _, cid in panels}
    else:
        motifs, pdoms = {}, {}

    col = _colors()
    handles = [mpatches.Patch(color=col["contact"], label="PLIP contact frequency"),
               mpatches.Patch(color=col["cdd"], label="CDD domain"),
               mlines.Line2D([], [], color=col["site"], lw=1.8, label="CDD site residue, contacted (≥1 sample)"),
               mlines.Line2D([], [], color=INK2, lw=0.9, label="CDD site / motif / coil residue, not contacted")]
    if any(coil_rows.values()):
        handles += [mlines.Line2D([], [], color=col["coil"], lw=1.8, alpha=0.4, label="coiled coil, contacted elsewhere"),
                    mlines.Line2D([], [], color=col["coil"], lw=1.8, label="coiled coil on the partner's coil")]
    if any(motifs.values()) or any(pdoms.values()):
        handles += [mlines.Line2D([], [], color=col["slim"], lw=1.8, alpha=0.4,
                                  label="ELM motif, contacted elsewhere"),
                    mlines.Line2D([], [], color=col["slim"], lw=1.8,
                                  label="ELM motif on the partner's binding domain"),
                    mpatches.Patch(color=col["slim"], alpha=0.9, label="ELM binding domain")]
    ncol = 3
    top_in, legend_in = 0.2, 0.25 + 0.17 * -(-len(handles) // ncol)
    heights = [panel_height(cdd, cid, len(motifs.get(cid, [])), len(pdoms.get(cid, [])), len(coil_rows[cid]))
               for _, cid in panels]
    H = top_in + sum(heights) + PANEL_GAP_IN * (len(panels) - 1) + legend_in
    fig = plt.figure(figsize=(FIG_W, H))
    y = H - top_in
    for (side, cid), h in zip(panels, heights):
        draw_panel(fig, H, y, side, cid, spec, ann, cdd, by_smp[cid], a.num_samples,
                   motifs.get(cid, []), pdoms.get(cid, []), coil_rows[cid])
        y -= h + PANEL_GAP_IN

    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5 + (LEFT_IN - 0.3) / FIG_W / 2, 0.005), handlelength=1.4,
               columnspacing=1.2)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    TH.save_all(fig, out, a.formats, png_dpi=150)
    plt.close(fig)
    print(f"[plot_plip_cdd] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
