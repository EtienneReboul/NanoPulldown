#!/usr/bin/env python3
"""
nanopulldown.report.plots.domain_bar — Stage 1 report figure
==================================================
One figure per protein, laid out like InterPro's protein viewer:

  AIUPred panel     disorder / binding / linker tracks (per-sequence cache)
  Family            the protein's family (InterPro family entry, from PANTHER ...)
  Domain map        the segments the heatmaps use: hand-curated `domains:`
                    (configs/curated_domains.yaml, `annotation_reviewed: true`)
                    or the proposal of nanopulldown.annotate.merge --
                    domain / linker / disordered
  InterPro entries  one row per integrated InterPro entry (domain, repeat,
                    homologous superfamily, site), with the member databases
                    that agree on it
  Unintegrated      member-database signatures no InterPro entry holds
  Features          MobiDB-lite, coiled coil, signal peptide, TM helix
                    (InterPro extra features) and AIUPred MoRFs

Everything below the AIUPred panel comes from data/annotation/<pair>/
annotation.yaml (its `domains:` and `interpro:` blocks), so the figure needs
no network. Row identity is written in the left margin (name + a grey line
with accession and sources); colour encodes the entry type.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import yaml

from nanopulldown.core.cache import cached_disorder_tracks
from nanopulldown.core.config import load_config
from nanopulldown.core.proteins import load_proteins
from nanopulldown.core.specs import chain_by_id, load_pair
from nanopulldown.report.theme import lazy_theme

# colours by role, from the report theme (nanopulldown.report.theme; one hue per entry type)
TH = lazy_theme()
TYPE_COLOR = {k: TH.color(k) for k in
              ("family", "domain", "homologous_superfamily", "repeat", "site", "unintegrated")}
KIND_COLOR = {k: TH.region(k) for k in ("domain", "linker", "disordered")}
TRACK_COLOR = {"disorder": TH.color("disorder"), "binding": TH.color("binding"),
               "linker": TH.color("linker_track")}
FEATURES = [("disorder", "MobiDB-lite", TH.color("disorder")), ("coils", "Coiled coil", TH.color("coils")),
            ("signal_peptide", "Signal peptide", TH.color("signal_peptide")), ("tm", "TM helix", TH.color("tm"))]
INK, INK2, RULE = TH.ink, TH.ink2, TH.rule

AX_W_IN, LEFT_IN, FIG_W = 6.7, 2.95, 10.0     # fixed geometry: label fitting is computable
ROW_IN = 0.30


def protein_title(spec: dict, chain: str) -> str:
    """'BPM6 — BTB/POZ and MATH domain-containing protein 6 (A1L4W5, AT3G43700, 415 aa)'."""
    c = chain_by_id(spec, chain) or {}
    acc = (spec.get(c.get("role")) or {}).get("uniprot")
    if not acc:
        return f"chain {chain}"
    r = load_proteins().get(acc) or {}
    short, long = r.get("short_name") or acc, r.get("long_name")
    ids = [x for x in (acc, r.get("locus")) if x and x not in short]
    ids.append(f"{len(c.get('sequence', ''))} aa")
    return short + (f" — {long}" if long and long != short else "") + f"  ({', '.join(ids)})"


def _fit(label: str | None, s: int, e: int, seqlen: int, size: float = 6.5) -> str | None:
    """Label truncated to the bar's width, or None when fewer than 5 chars fit."""
    if not label:
        return None
    n = int(((e - s + 1) / max(seqlen, 1)) * AX_W_IN * 72 / (size * 0.58))
    if n >= len(label):
        return label
    return label[:n - 1] + "…" if n >= 5 else None


def _bar(ax, y, s, e, color, seqlen, label=None, h=0.62, alpha=1.0):
    ax.add_patch(mpatches.FancyBboxPatch(
        (s, y - h / 2), e - s + 1, h, boxstyle="round,pad=0,rounding_size=2",
        mutation_aspect=0.08, facecolor=color, alpha=alpha, edgecolor=TH.surface, lw=1.0))
    t = _fit(label, s, e, seqlen)
    if t:
        ax.text((s + e) / 2, y, t, ha="center", va="center", fontsize=6.5, color="white")


def _pack(items: list[dict]) -> list[list[dict]]:
    """Greedy interval packing: overlapping bars of one track go to sub-rows."""
    rows: list[list[dict]] = []
    for it in sorted(items, key=lambda x: (x["start"], -x["end"])):
        for r in rows:
            if r[-1]["end"] + 3 < it["start"]:
                r.append(it)
                break
        else:
            rows.append([it])
    return rows


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def build_rows(segs: list[dict], ipr: dict, source_label: str) -> list[dict]:
    """Track rows top to bottom: {label, sub, bars: [(s, e, color, text, h, alpha)], gap_before}."""
    rows = []
    fam = ipr.get("family") or []
    for k, r in enumerate(_pack(fam)):
        rows.append({"label": "Family" if k == 0 else "",
                     "sub": " · ".join(ipr.get("family_sources") or []) if k == 0 else None,
                     "bars": [(f["start"], f["end"], TYPE_COLOR["family"], f.get("name"), 0.62, 1.0)
                              for f in r]})
    rows.append({"label": "Domain map", "sub": source_label,
                 "bars": [(d["start"], d["end"], KIND_COLOR.get(d["kind"], INK2),
                           d["name"] if d["kind"] == "domain" else None, 0.78, 1.0)
                          for d in segs if d["kind"] != "morf"]})
    first = True
    for e in ipr.get("entries") or []:
        color = TYPE_COLOR.get(e.get("type"), INK2)
        rows.append({"label": e["name"], "sub": " · ".join([e["accession"], *e.get("members", [])]),
                     "bars": [(s, t, color, None, 0.5, 0.9) for s, t in e["locations"]],
                     "header": "InterPro entries" if first else None})
        first = False
    for r in _pack(ipr.get("unintegrated") or []):
        rows.append({"label": r[0]["name"] if len(r) == 1 else f"{len(r)} signatures",
                     "sub": "unintegrated · " + " · ".join(f"{u['db']} {u['accession']}" for u in r),
                     "bars": [(u["start"], u["end"], TYPE_COLOR["unintegrated"], None, 0.45, 1.0) for u in r]})
    feats = ipr.get("features") or {}
    first = True
    for key, label, color in FEATURES:
        if feats.get(key):
            rows.append({"label": label, "sub": None, "gap": first,
                         "bars": [(s, e, color, None, 0.35, 0.8) for s, e in feats[key]]})
            first = False
    morfs = [d for d in segs if d["kind"] == "morf"]
    if morfs:
        rows.append({"label": "MoRF (AIUPred)", "sub": None, "gap": first,
                     "bars": [(d["start"], d["end"], TRACK_COLOR["binding"], None, 0.35, 0.8) for d in morfs]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--title", default=None,
                    help="figure title; default: the protein's display name + accession/locus")
    ap.add_argument("--no-disorder", action="store_true",
                    help="skip the AIUPred panel (tracks read from the per-sequence annotation cache)")
    ap.add_argument("--cutoff", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="svg")
    args = ap.parse_args()

    spec = load_pair(args.pair)
    ann = yaml.safe_load(Path(args.annotation).read_text()) or {}
    reviewed = spec.get("annotation_reviewed", False)
    src = spec.get("domains") if reviewed else None
    segs = (src or ann.get("domains", {})).get(args.chain, [])
    source_label = ("curated (configs/curated_domains.yaml)" if (reviewed and src)
                    else "auto-proposed from InterPro — review needed")
    ipr = (ann.get("interpro") or {}).get(args.chain) or {}
    chain_seq = (chain_by_id(spec, args.chain) or {}).get("sequence", "")
    seqlen = max([len(chain_seq)] + [int(d["end"]) for d in segs])
    tracks = {} if args.no_disorder or not chain_seq else cached_disorder_tracks(load_config(), chain_seq)

    TH.apply()
    rows = build_rows(segs, ipr, source_label)
    n_extra = sum(0.6 for r in rows if r.get("header") or r.get("gap"))
    top_in, aiu_in, gap_in, bot_in = 0.45, (0 if args.no_disorder else 1.35), 0.35, 0.75
    trk_in = ROW_IN * (len(rows) + n_extra + 0.6)
    H = top_in + aiu_in + (gap_in if aiu_in else 0) + trk_in + bot_in
    fig = plt.figure(figsize=(FIG_W, H))
    ax = fig.add_axes([LEFT_IN / FIG_W, bot_in / H, AX_W_IN / FIG_W, trk_in / H])
    title = args.title or protein_title(spec, args.chain)

    if aiu_in:
        axc = fig.add_axes([LEFT_IN / FIG_W, (bot_in + trk_in + gap_in) / H, AX_W_IN / FIG_W, aiu_in / H],
                           sharex=ax)
        plt.setp(axc.get_xticklabels(), visible=False)
        if not tracks:
            axc.text(0.5, 0.5, "no AIUPred tracks", ha="center", va="center", transform=axc.transAxes,
                     color=INK2)
        for t, vals in tracks.items():
            vals = vals[:seqlen]
            axc.plot(range(1, len(vals) + 1), vals, color=TRACK_COLOR.get(t, INK2), lw=1.2,
                     ls="--" if t == "linker" else "-", label=f"AIUPred {t}")
        axc.axhline(args.cutoff, color=INK2, lw=0.6, ls=":")
        axc.set_ylim(0, 1.05)
        axc.set_ylabel("AIUPred score", fontsize=8, color=INK2)
        if tracks:
            axc.legend(loc="upper right", bbox_to_anchor=(-0.08, 1.0), fontsize=6.5, frameon=False,
                       handlelength=1.6, borderaxespad=0.1)
        axc.tick_params(labelsize=7, colors=INK2)
        for sp in ("top", "right"):
            axc.spines[sp].set_visible(False)
        axc.set_title(title, fontsize=9, loc="left", color=INK)
    else:
        ax.set_title(title, fontsize=9, loc="left", color=INK)

    tr = ax.get_yaxis_transform()
    y = 0.0
    for r in rows:
        if r.get("header") or r.get("gap"):
            y -= 0.3
            if r.get("header"):
                ax.text(-0.012, y + 0.5, r["header"], transform=tr, fontsize=7.5, color=INK,
                        va="center", ha="right", fontweight="bold")
            y -= 0.3
        ax.axhline(y, color=RULE, lw=0.6, zorder=0)
        for s, e, color, text, h, alpha in r["bars"]:
            _bar(ax, y, int(s), int(e), color, seqlen, text, h, alpha)
        dy = 0.17 if r.get("sub") else 0
        if r["label"]:
            ax.text(-0.012, y + dy, _clip(r["label"], 46), transform=tr, ha="right", va="center",
                    fontsize=7, color=INK)
        if r.get("sub"):
            ax.text(-0.012, y - 0.24, _clip(r["sub"], 60), transform=tr, ha="right", va="center",
                    fontsize=5.6, color=INK2)
        y -= 1
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_xlim(1, max(seqlen, 1))
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7, colors=INK2)
    ax.set_xlabel("residue", fontsize=8, color=INK2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    handles = [mpatches.Patch(color=c, label=k.replace("_", " ")) for k, c in TYPE_COLOR.items()]
    handles += [mpatches.Patch(color=KIND_COLOR["disordered"], label="disordered region"),
                mpatches.Patch(color=KIND_COLOR["linker"], label="linker")]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5 + (LEFT_IN - 0.3) / FIG_W / 2, 0.005), handlelength=1.2,
               columnspacing=1.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    TH.save_all(fig, out, args.formats, png_dpi=150)
    plt.close(fig)
    print(f"[plot_domain_bar] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
