#!/usr/bin/env python3
"""
scripts/plot_domain_bar.py — Stage 1 report figure
==================================================
One horizontal residue bar per protein chain, split into coloured segments
by `kind` (domain / linker / disordered / morf). Uses the hand-curated
`domains:` block from configs/<pair>.yaml when `annotation_reviewed:
true`, otherwise the proposed block from data/annotation/<pair>/annotation.yaml.

Copied verbatim from ab_initio_pipeline's scripts/plot_domain_bar.py -- the
schema (`domains:` list of {name,start,end,kind}) is unchanged, only the
config's own terminology moved from "system" to "pair".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

KIND_COLOR = {
    "domain": "#4c72b0",
    "linker": "#c7c7c7",
    "disordered": "#dd8452",
    "morf": "#55a868",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="svg")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.spec).read_text())
    ann = yaml.safe_load(Path(args.annotation).read_text())
    reviewed = spec.get("annotation_reviewed", False)
    src = spec.get("domains") if reviewed else None
    segs = (src or ann.get("domains", {})).get(args.chain, [])
    source_label = "config (reviewed)" if (reviewed and src) else "auto-proposed (review needed)"

    seqlen = 0
    for s in spec.get("sequences", []):
        if s.get("id") == args.chain:
            seqlen = len(s.get("sequence", ""))
    if segs:
        seqlen = max(seqlen, max(int(d["end"]) for d in segs))

    fig, ax = plt.subplots(figsize=(9, 1.9), constrained_layout=True)
    ax.add_patch(mpatches.Rectangle((1, 0.0), max(seqlen, 1), 1.0,
                                    facecolor="#f0f0f0", edgecolor="none"))
    morfs = [d for d in segs if d["kind"] == "morf"]
    for d in segs:
        if d["kind"] == "morf":
            continue
        ax.add_patch(mpatches.Rectangle(
            (int(d["start"]), 0.0), int(d["end"]) - int(d["start"]) + 1, 1.0,
            facecolor=KIND_COLOR.get(d["kind"], "#999999"), edgecolor="white", lw=.6))
        mid = (int(d["start"]) + int(d["end"])) / 2
        if int(d["end"]) - int(d["start"]) > seqlen * 0.06:
            ax.text(mid, 0.5, str(d["name"])[:18], ha="center", va="center",
                    fontsize=7, color="white")
    for d in morfs:                              # MoRFs as a thin overlay band
        ax.add_patch(mpatches.Rectangle(
            (int(d["start"]), 1.02), int(d["end"]) - int(d["start"]) + 1, 0.22,
            facecolor=KIND_COLOR["morf"], edgecolor="none"))

    ax.set_xlim(1, max(seqlen, 1)); ax.set_ylim(-0.1, 1.35)
    ax.set_yticks([]); ax.set_xlabel("residue")
    ax.set_title(f"chain {args.chain} domain map — {source_label}", fontsize=9)
    ax.legend(handles=[mpatches.Patch(color=c, label=k) for k, c in KIND_COLOR.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=4, frameon=False, fontsize=7)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    for fmt in args.formats.split(","):
        fig.savefig(f"{stem}.{fmt.strip()}", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_domain_bar] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
