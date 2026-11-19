#!/usr/bin/env python3
"""
nanopulldown.report.plots.plip_cdd — Stage 2e report figures
============================================================
PLIP contacts on sequences of interest: do the predicted complexes actually
touch what the sequence analyses point at? One figure per analysis for one
pair, written into --out-dir, all drawn the same way: one panel per chain,
bait chains first, residue on x, the per-residue PLIP contact frequency
(fraction of the --num-samples samples whose bait-prey interactions include
the residue) over the domain map, then one row per sequence of interest -- a
tick per residue, filled when PLIP contacts it in at least one sample, solid
when it contacts the partner region the analysis names, with in how many
samples it does.

  cdd.svg       (always) each CDD domain location, then one row per
                conserved-feature site of it (k/n residues contacted); the
                coiled-coil segments (annotation.yaml `features.coils`:
                DeepCoil where it ran, else InterPro COILS): solid = knobs in
                the partner coil's holes in samples passing the
                knobs-into-holes test (--coiled-coil, nanopulldown.structure.
                coiled_coil), else contacting a coil of the other side.
                Header: nanopulldown.report.cdd's statistics (sites contacted
                in any sample; enrichment and share of contacts in CDD
                domains, averaged over samples).
  elm.svg       (--elm, pairs with a match) one row per disordered ELM motif
                whose binding Pfam domain is on the other side
                (nanopulldown.annotate.elm): solid = contacting the partner's
                binding domain; on the partner's chain, one bar per binding
                Pfam domain naming the motifs it would bind.
  sparrow.svg   (--idr, pairs with scored IDR pairs) one row per bait IDR x
                prey IDR pair (nanopulldown.annotate.idr_pairs, best ranked
                first) on each chain: solid = contacting the partner IDR;
                the sub-line gives the dominant mechanism, its background
                percentile and the window behind it.
  minipae.svg   (--minipae, pairs with a segment) one row per min-iPAE
                segment (nanopulldown.structure.minipae): solid = contacting
                the partner residues the fold places it on (argmin of the
                best sample); on the partner's chain, one bar per such target
                region.

Sequence analyses alone (ELM regex, SPARROW physicochemistry, even min-iPAE,
a confidence) do not say the residues touch: these figures check it on the
same models PLIP read.
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
    return {"site": TH.color("site"), "cdd": TH.slot("aqua"), "contact": INK2, "slim": TH.slot("violet"),
            "coil": TH.color("coils"), "idr": TH.slot("orange"), "minipae": TH.slot("magenta")}


MAX_ELM_ROWS = 20                    # per chain, best-ranked first; the rest are counted in the header
MAX_IDR_ROWS = 10                    # IDR pairs per pair, best-ranked first
ANALYSES = ("cdd", "elm", "sparrow", "minipae")


def contact_pairs_by_sample(pl: pd.DataFrame) -> dict[int, set]:
    """{sample: {((chain, resnr), (chain, resnr))}} of PLIP contacts, both orders."""
    out: dict[int, set] = {}
    for smp, c1, r1, c2, r2 in zip(pl["sample"], pl["reschain"].astype(str), pl["resnr"],
                                   pl["reschain_lig"].astype(str), pl["resnr_lig"]):
        a, b = (c1, int(r1)), (c2, int(r2))
        out.setdefault(int(smp), set()).update({(a, b), (b, a)})
    return out


def coil_segments(ann: dict, cid: str) -> list[tuple[int, int]]:
    """Coiled-coil segments of one chain (chain numbering) from annotation.yaml."""
    feats = (((ann.get("interpro") or {}).get(cid) or {}).get("features") or {})
    return [(int(a), int(b)) for a, b in feats.get("coils") or []]


def coil_source(ann: dict, cid: str) -> str:
    """DeepCoil or COILS (annotation.yaml from before DeepCoil: COILS)."""
    return ((ann.get("interpro") or {}).get(cid) or {}).get("coils_source") or "COILS"


def _kih(geo: pd.DataFrame | None, cid: str, a: int, b: int) -> tuple[set[int], int] | None:
    """(knob residues of segment cid:a-b in passing samples, number of samples
    where it forms a coiled coil with a partner segment), None without a table."""
    if geo is None:
        return None
    knobs, smps = set(), set()
    for s, (ch, st, en, kcol) in (("a", ("chain_a", "start_a", "end_a", "knob_residues_a")),
                                  ("b", ("chain_b", "start_b", "end_b", "knob_residues_b"))):
        g = geo[(geo[ch] == cid) & (geo[st] == a) & (geo[en] == b) & geo["coiled_coil"]]
        smps |= set(g["sample"])
        knobs |= {int(r) for v in g[kcol] for r in str(v).split(",") if r}
    return knobs, len(smps)


def coil_pairing(coils: dict, panels: list, pairs_by_smp: dict,
                 geo: pd.DataFrame | None = None) -> dict[str, list[dict]]:
    """{chain: [{start, end, res, to_cc, smp_cc, partner, kih}]}: per
    coiled-coil segment, its residues contacting a coiled coil of the OTHER
    side and the samples in which it does -- or, with the geometry table, its
    knobs-into-holes residues and the samples forming a coiled coil (kih)."""
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
            kih = _kih(geo, cid, a, b) if partner else None
            if kih is not None:
                to_cc, smp_cc = kih
            rows.append({"start": a, "end": b, "res": sorted(res), "to_cc": to_cc, "smp_cc": smp_cc,
                         "partner": bool(partner), "kih": kih is not None})
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


def _contacting(pairs_by_smp: dict, cid: str, res: set[int], partner) -> tuple[set[int], int]:
    """(residues of `res` on chain cid contacting a residue for which
    partner((chain, resnr)) is true in any sample, number of such samples)."""
    hit, n = set(), 0
    for pairs in pairs_by_smp.values():
        h = {a[1] for a, b in pairs if a[0] == cid and a[1] in res and partner(b)}
        hit |= h
        n += bool(h)
    return hit, n


def _span(r: dict, side: str) -> tuple[int, int] | None:
    s, e = r.get(f"{side}_mech_start"), r.get(f"{side}_mech_end")
    return (int(s), int(e)) if pd.notna(s) and pd.notna(e) else None


def idr_pairs_on_chain(idr: pd.DataFrame, cid: str, pairs_by_smp: dict) -> list[dict]:
    """The pair's scored IDR pairs seen from chain cid (best ranked first,
    MAX_IDR_ROWS of them): the IDR, the partner IDR, the residues / samples
    contacting it, and the window behind the dominant mechanism."""
    out = []
    for r in idr.sort_values("rank").head(MAX_IDR_ROWS).to_dict("records"):
        for me, other in (("a", "b"), ("b", "a")):
            if str(r[f"{me}_chain"]) != cid:
                continue
            s, e = int(r[f"{me}_start"]), int(r[f"{me}_end"])
            pc, ps, pe = str(r[f"{other}_chain"]), int(r[f"{other}_start"]), int(r[f"{other}_end"])
            to_p, smp = _contacting(pairs_by_smp, cid, set(range(s, e + 1)),
                                    lambda b, pc=pc, ps=ps, pe=pe: b[0] == pc and ps <= b[1] <= pe)
            win = _span(r, me)
            pwin = _span(r, other)
            win_smp = None
            if win and pwin:          # the complementary windows themselves in contact
                _, win_smp = _contacting(pairs_by_smp, cid, set(range(win[0], win[1] + 1)),
                                         lambda b, pc=pc, w=pwin: b[0] == pc and w[0] <= b[1] <= w[1])
            out.append({"rank": int(r["rank"]), "start": s, "end": e, "res": list(range(s, e + 1)),
                        "partner": (pc, ps, pe), "to_p": to_p, "smp": smp,
                        "mech": r.get("dominant_mechanism") or "none",
                        "pct": r.get("max_mechanism_pct"), "win": win, "pwin": pwin, "win_smp": win_smp})
    return out


def minipae_segments_on_chain(seg: pd.DataFrame, cid: str, pairs_by_smp: dict) -> list[dict]:
    """The pair's min-iPAE segments on chain cid, with the residues / samples
    contacting the partner region the fold places them on."""
    out = []
    for r in seg[seg["motif_chain"].astype(str) == cid].sort_values("rank").to_dict("records"):
        s, e = int(r["start"]), int(r["end"])
        pcs = set(str(r["partner_chain"]).split(";"))
        ps, pe = int(r["partner_start"]), int(r["partner_end"])
        to_p, smp = _contacting(pairs_by_smp, cid, set(range(s, e + 1)),
                                lambda b: b[0] in pcs and ps <= b[1] <= pe)
        out.append({"start": s, "end": e, "res": list(range(s, e + 1)), "segment": r["segment"],
                    "min_pae": float(r["min_pae"]), "support": float(r["support"]),
                    "slim_like": bool(r["slim_like"]), "partner": (sorted(pcs), ps, pe),
                    "domains": r.get("partner_domains"), "to_p": to_p, "smp": smp})
    return out


def minipae_targets_on_chain(seg: pd.DataFrame, cid: str) -> list[dict]:
    """Partner regions on chain cid that a min-iPAE segment of the other side is placed on."""
    g = seg[seg["partner_chain"].astype(str).str.split(";").map(lambda cs: cid in cs)]
    return [{"start": int(r["partner_start"]), "end": int(r["partner_end"]), "from": str(r["motif_chain"]),
             "seg": (int(r["start"]), int(r["end"])), "domains": r.get("partner_domains")}
            for r in g.sort_values("rank").to_dict("records")]


def domain_map_row(segs: list[dict]) -> dict:
    return {"label": "Domain map", "sub": None, "kind": "bars",
            "bars": [(d["start"], d["end"], KIND_COLOR.get(d["kind"], INK2),
                      d["name"] if d["kind"] == "domain" else None, 0.78, 1.0)
                     for d in segs if d["kind"] != "morf"]}


def cdd_rows(cdd: pd.DataFrame, hit_any: set[int], n_samples: int = 0, coils=(),
             coil_src: str = "COILS") -> list[dict]:
    col = _colors()
    rows = []
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
        pair = ("partner has no coiled coil" if not c["partner"]
                else f"KIH coiled coil {c['smp_cc']}/{n_samples}" if c.get("kih")
                else f"↔ partner coil in {c['smp_cc']}/{n_samples}")
        rows.append({"label": f"Coiled coil {c['start']}–{c['end']}", "gap": i == 0, "kind": "ticks",
                     "sub": f"{coil_src} · {k}/{len(c['res'])} contacted · {pair}",
                     "ticks": c["res"], "hit": hit_any, "strong": c["to_cc"], "color": col["coil"]})
    return rows


def elm_rows(hit_any: set[int], motifs=(), pdoms=(), n_samples: int = 0) -> list[dict]:
    col = _colors()
    rows = []
    for i, d in enumerate(pdoms):
        rows.append({"label": f"ELM partner: {d['name']}", "gap": i == 0, "kind": "bars",
                     "sub": f"{d['pfam']} · binds {', '.join(d['motifs'])} (chain {', '.join(d['chains'])})",
                     "bars": [(d["start"], d["end"], col["slim"], None, 0.5, 0.9)]})
    for i, m in enumerate(motifs[:MAX_ELM_ROWS]):
        k = sum(r in hit_any for r in m["res"])
        pf = ",".join(sorted({p for _, p, _, _, _ in m["doms"]}))
        rows.append({"label": m["elm"] + (" (known)" if m["known"] else ""), "gap": i == 0, "kind": "ticks",
                     "sub": f"→{pf} in {m['smp_dom']}/{n_samples} · {k}/{len(m['res'])} contacted · "
                            f"{m['start']}–{m['end']} {m['motif']}",
                     "ticks": m["res"], "hit": hit_any, "strong": m["to_dom"], "color": col["slim"]})
    return rows


def idr_rows(hit_any: set[int], idrs=(), n_samples: int = 0) -> list[dict]:
    """One row per IDR pair: ticks over the IDR (solid on the partner IDR), the
    window behind the dominant mechanism underlined, and in how many samples
    that window contacts the partner's window -- what SPARROW actually predicts."""
    col = _colors()
    rows = []
    for i, d in enumerate(idrs):
        k = sum(r in hit_any for r in d["res"])
        pc, ps, pe = d["partner"]
        mech = (f"{d['mech']} p{d['pct']:.0f}" if d["mech"] != "none" and pd.notna(d["pct"])
                else "no labelled mechanism")
        ww = (f"window↔window {d['win_smp']}/{n_samples}" if d["win_smp"] is not None else "no window")
        rows.append({"label": f"#{d['rank']} IDR {d['start']}–{d['end']} × {pc}:{ps}–{pe}", "gap": i == 0,
                     "kind": "ticks", "win": d["win"],
                     "sub": f"{ww} · ↔IDR {d['smp']}/{n_samples} · {k}/{len(d['res'])} contacted · {mech}",
                     "ticks": d["res"], "hit": hit_any, "strong": d["to_p"], "color": col["idr"]})
    return rows


def minipae_rows(hit_any: set[int], segs=(), targets=(), n_samples: int = 0) -> list[dict]:
    col = _colors()
    rows = []
    for i, t in enumerate(targets):
        dom = f" · {t['domains']}" if t["domains"] and pd.notna(t["domains"]) else ""
        rows.append({"label": f"min-iPAE target {t['start']}–{t['end']}", "gap": i == 0, "kind": "bars",
                     "sub": f"segment {t['from']}:{t['seg'][0]}–{t['seg'][1]} placed here{dom}",
                     "bars": [(t["start"], t["end"], col["minipae"], None, 0.5, 0.9)]})
    for i, s in enumerate(segs):
        k = sum(r in hit_any for r in s["res"])
        pcs, ps, pe = s["partner"]
        rows.append({"label": f"{s['start']}–{s['end']} {s['segment']}" + (" (SLiM-like)" if s["slim_like"] else ""),
                     "gap": i == 0, "kind": "ticks",
                     "sub": f"→{','.join(pcs)}:{ps}–{pe} in {s['smp']}/{n_samples} · {k}/{len(s['res'])} "
                            f"contacted · min-iPAE {s['min_pae']:.1f} Å · support {s['support']:.0%}",
                     "ticks": s["res"], "hit": hit_any, "strong": s["to_p"], "color": col["minipae"]})
    return rows


def cdd_summary(cdd_c, sites, spans, by_smp, hit_any, n_samples, L, coils) -> str:
    stats = [cdd_stats(by_smp.get(i, set()), sites, spans, L) for i in range(n_samples)]
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
        what = ("in a knobs-into-holes coiled coil with the partner" if any(c.get("kih") for c in coils)
                else "paired with the partner's")
        summary += f" · coils {what}: {sum(c['smp_cc'] > 0 for c in coils)}/{len(coils)}"
    return summary


def draw_panel(fig, H, y_top, role, cid, spec, rows, by_smp, n_samples, summary,
               bar_colors=(), shades=()) -> float:
    """One chain's panel with its top edge at y_top (inches from the bottom);
    returns its bottom edge. bar_colors: [(residue set, colour)] for the
    contact bars, first match wins; shades: [(start, end, colour)] behind them."""
    col = _colors()
    seq = next(c["sequence"] for c in spec["sequences"] if c["id"] == cid)
    L = len(seq)
    hit_any = {r for s in by_smp.values() for _, r in s}
    if not hit_any:
        summary += " · no PLIP contact on this chain"

    freq = np.zeros(L + 1)
    for s in by_smp.values():
        for c, r in s:
            if r <= L:
                freq[r] += 1
    freq /= max(n_samples, 1)

    n_gap = sum(1 for r in rows if r.get("gap"))
    trk_in = ROW_IN * (len(rows) + 0.6 * n_gap + 0.6)
    bar_y = y_top - TITLE_IN - BAR_IN
    trk_y = bar_y - 0.08 - trk_in
    axb = fig.add_axes([LEFT_IN / FIG_W, bar_y / H, AX_W_IN / FIG_W, BAR_IN / H])
    ax = fig.add_axes([LEFT_IN / FIG_W, trk_y / H, AX_W_IN / FIG_W, trk_in / H], sharex=axb)

    x = np.arange(1, L + 1)
    colors = [next((c for res, c in bar_colors if i in res), col["contact"]) for i in x]
    for s, e, c in shades:
        axb.axvspan(s - 0.5, e + 0.5, color=c, alpha=0.13, lw=0)
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
            # rows with a partner region: contacted elsewhere faded, on the partner region solid
            ax.vlines(hit, y - 0.34, y + 0.34, color=r.get("color", col["site"]), lw=1.8,
                      alpha=0.4 if "strong" in r else 1.0)
            ax.vlines(sorted(strong), y - 0.34, y + 0.34, color=r.get("color", col["site"]), lw=1.8)
            if r.get("win"):          # SPARROW: the window behind the dominant mechanism
                ax.hlines(y - 0.42, r["win"][0] - 0.5, r["win"][1] + 0.5, color=INK, lw=2.2)
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


def panel_height(rows: list[dict]) -> float:
    n_gap = sum(1 for r in rows if r.get("gap"))
    return TITLE_IN + BAR_IN + 0.08 + ROW_IN * (len(rows) + 0.6 * n_gap + 0.6) + XLAB_IN


def render(panels: list[dict], spec, n_samples: int, handles: list, out: Path, formats: str) -> None:
    """panels: [{side, cid, rows, by_smp, summary, bar_colors, shades}] -> one figure."""
    ncol = 3
    top_in, legend_in = 0.2, 0.25 + 0.17 * -(-len(handles) // ncol)
    heights = [panel_height(p["rows"]) for p in panels]
    H = top_in + sum(heights) + PANEL_GAP_IN * (len(panels) - 1) + legend_in
    fig = plt.figure(figsize=(FIG_W, H))
    y = H - top_in
    for p, h in zip(panels, heights):
        draw_panel(fig, H, y, p["side"], p["cid"], spec, p["rows"], p["by_smp"], n_samples, p["summary"],
                   p.get("bar_colors", ()), p.get("shades", ()))
        y -= h + PANEL_GAP_IN
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5 + (LEFT_IN - 0.3) / FIG_W / 2, 0.005), handlelength=1.4,
               columnspacing=1.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    TH.save_all(fig, out, formats, png_dpi=150)
    plt.close(fig)
    print(f"[plot_plip_soi] -> {out}")


def _read_pair(path: str | None, pair: str) -> pd.DataFrame | None:
    if not path or not Path(path).exists():
        return None
    import pyarrow.parquet as pq
    df = pq.read_table(path, filters=[("pair", "=", pair)]).to_pandas()
    return df if len(df) else None


def main() -> int:
    ap = argparse.ArgumentParser(description="PLIP contacts of one pair on its sequences of interest "
                                             "(CDD, ELM, SPARROW IDR pairs, min-iPAE segments)")
    ap.add_argument("--pair", required=True)
    ap.add_argument("--contacts", required=True, help="results/<pair>/plip_contacts.parquet")
    ap.add_argument("--cdd", required=True, help="<annotation dir>/cdd_sites.parquet")
    ap.add_argument("--annotation", required=True, help="data/annotation/<pair>/annotation.yaml")
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--elm", default=None, help="<annotation dir>/elm_partner_matches.parquet (optional)")
    ap.add_argument("--idr", default=None, help="<annotation dir>/idr_pair_scores.parquet (optional)")
    ap.add_argument("--minipae", default=None, help="<results>/_minipae/minipae_segments.parquet (optional)")
    ap.add_argument("--coiled-coil", default=None, help="results/<pair>/coiled_coil.parquet (optional)")
    ap.add_argument("--out-dir", required=True, help="cdd.svg, and elm / sparrow / minipae.svg when they have rows")
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    TH.apply()
    col = _colors()
    n = a.num_samples
    spec = load_pair(a.pair)
    ann = yaml.safe_load(Path(a.annotation).read_text()) or {}
    cdd = read_cdd(a.cdd)
    cdd = cdd[cdd["pair"] == a.pair] if cdd is not None else pd.DataFrame(columns=CDD_TABLE)
    pl = pd.read_parquet(a.contacts, columns=["sample", "resnr", "reschain", "resnr_lig", "reschain_lig"])
    chains = [(side, cid) for side in SIDES for cid in spec.get(f"{side}_chains", [])]
    by_smp = {cid: contacts_by_sample(pl, {cid}) for _, cid in chains}
    hit_any = {cid: {r for s in by_smp[cid].values() for _, r in s} for _, cid in chains}
    pairs_by_smp = contact_pairs_by_sample(pl)
    elm, idr, seg = _read_pair(a.elm, a.pair), _read_pair(a.idr, a.pair), _read_pair(a.minipae, a.pair)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):                    # a re-run must not leave a stale analysis behind
        if old.is_file():
            old.unlink()
    fmt = lambda name: out / f"{name}.svg"       # noqa: E731  (save_all writes every format next to it)
    not_hit = mlines.Line2D([], [], color=INK2, lw=0.9, label="residue of interest, not contacted")
    contact = mpatches.Patch(color=col["contact"], label="PLIP contact frequency")

    def faded(color, what):
        return [mlines.Line2D([], [], color=color, lw=1.8, alpha=0.4, label=f"{what}, contacted elsewhere"),
                mlines.Line2D([], [], color=color, lw=1.8, label=f"{what}, contacting the partner region")]

    # CDD (+ coiled coils): always
    geo = pd.read_parquet(a.coiled_coil) if a.coiled_coil and Path(a.coiled_coil).exists() else None
    coils = {cid: coil_segments(ann, cid) for _, cid in chains}
    coil_rows = coil_pairing(coils, chains, pairs_by_smp, geo)
    panels = []
    for side, cid in chains:
        L = len(next(c["sequence"] for c in spec["sequences"] if c["id"] == cid))
        segs, _ = chain_segments(spec, ann, cid)
        cdd_c = cdd[cdd["chain"] == cid]
        sites, spans = cdd_residues(cdd_c, {cid})
        rows = [domain_map_row(segs)] + cdd_rows(cdd_c, hit_any[cid], n, coil_rows[cid], coil_source(ann, cid))
        shades = ([(c["start"], c["end"], col["coil"]) for c in coil_rows[cid]]
                  + [(s, e, col["cdd"]) for s, e in zip(cdd_c.loc[cdd_c["site"].isna(), "start"],
                                                         cdd_c.loc[cdd_c["site"].isna(), "end"])])
        panels.append({"side": side, "cid": cid, "rows": rows, "by_smp": by_smp[cid], "shades": shades,
                       "summary": cdd_summary(cdd_c, sites, spans, by_smp[cid], hit_any[cid], n, L, coil_rows[cid]),
                       "bar_colors": [({r for _, r in sites}, col["site"])]})
    handles = [contact, mpatches.Patch(color=col["cdd"], label="CDD domain"),
               mlines.Line2D([], [], color=col["site"], lw=1.8, label="CDD site residue, contacted (≥1 sample)"),
               not_hit]
    if any(coil_rows.values()):
        kih = any(c.get("kih") for v in coil_rows.values() for c in v)
        handles += [mlines.Line2D([], [], color=col["coil"], lw=1.8, alpha=0.4,
                                  label="coiled coil, contacted" + ("" if kih else " elsewhere")),
                    mlines.Line2D([], [], color=col["coil"], lw=1.8,
                                  label="coil knob in the partner coil's holes" if kih
                                  else "coiled coil on the partner's coil")]
    render(panels, spec, n, handles, fmt("cdd"), a.formats)

    def side_panels(make):
        ps = []
        for side, cid in chains:
            segs, _ = chain_segments(spec, ann, cid)
            rows, summary, res = make(cid)
            ps.append({"side": side, "cid": cid, "rows": [domain_map_row(segs)] + rows, "by_smp": by_smp[cid],
                       "summary": summary, "bar_colors": [(res, col[analysis])]})
        return ps

    # ELM
    if elm is not None:
        analysis = "slim"
        motifs = {cid: elm_motifs(elm, cid, pairs_by_smp) for _, cid in chains}
        pdoms = {cid: elm_partner_domains(elm, cid) for _, cid in chains}

        def make(cid):
            m = motifs[cid]
            s = (f"ELM motifs on the partner domain: {sum(x['smp_dom'] > 0 for x in m)}/{len(m)}"
                 + (f" (best {MAX_ELM_ROWS} shown)" if len(m) > MAX_ELM_ROWS else "") if m
                 else "no partner-matched ELM motif on this chain")
            return elm_rows(hit_any[cid], m, pdoms[cid], n), s, {r for x in m for r in x["res"]}
        render(side_panels(make), spec, n,
               [contact, *faded(col["slim"], "ELM motif"),
                mpatches.Patch(color=col["slim"], alpha=0.9, label="ELM binding domain (partner)"), not_hit],
               fmt("elm"), a.formats)

    # SPARROW IDR pairs
    if idr is not None:
        analysis = "idr"
        idrs = {cid: idr_pairs_on_chain(idr, cid, pairs_by_smp) for _, cid in chains}

        def make(cid):
            d = idrs[cid]
            s = (f"IDR pairs whose complementary windows touch: {sum((x['win_smp'] or 0) > 0 for x in d)}/{len(d)}"
                 f" · touching the partner IDR at all: {sum(x['smp'] > 0 for x in d)}/{len(d)}"
                 if d else "no scored IDR on this chain")
            return idr_rows(hit_any[cid], d, n), s, {r for x in d for r in x["res"]}
        render(side_panels(make), spec, n,
               [contact, *faded(col["idr"], "IDR"),
                mlines.Line2D([], [], color=INK, lw=2.2, label="window behind the dominant mechanism"), not_hit],
               fmt("sparrow"), a.formats)

    # min-iPAE segments
    if seg is not None:
        analysis = "minipae"
        msegs = {cid: minipae_segments_on_chain(seg, cid, pairs_by_smp) for _, cid in chains}
        tgts = {cid: minipae_targets_on_chain(seg, cid) for _, cid in chains}

        def make(cid):
            d = msegs[cid]
            s = (f"min-iPAE segments contacting their placed partner region: {sum(x['smp'] > 0 for x in d)}/{len(d)}"
                 if d else "no min-iPAE segment on this chain")
            return minipae_rows(hit_any[cid], d, tgts[cid], n), s, {r for x in d for r in x["res"]}
        render(side_panels(make), spec, n,
               [contact, *faded(col["minipae"], "min-iPAE segment"),
                mpatches.Patch(color=col["minipae"], alpha=0.9, label="min-iPAE target region (partner)"), not_hit],
               fmt("minipae"), a.formats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
