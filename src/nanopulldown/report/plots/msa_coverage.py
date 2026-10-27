#!/usr/bin/env python3
"""
nanopulldown.report.plots.msa_coverage — Stage 2a report figure (ColabFold only)
================================================================================
ColabFold-style "Sequence coverage" plot of one pair's complex MSA, read from
the cached a3m colabfold_msa wrote (nanopulldown.fold.colabfold.a3m_path):
`#len1,len2<TAB>card1,card2` header, then the paired rows (every chain filled)
and each chain's unpaired rows (gaps over the other chains).

Same figure as colabfold.plot.plot_msa_v2: match-state columns only (a3m
insertions dropped), rows grouped by which chains they cover (paired block
first, then one block per chain) and sorted by identity to the query inside
each block, coloured by that identity wherever the row has a residue, blank
where it has a gap. Identity is over the covered chains' full length, so a
short perfect partial hit still reads low. Black line = per-position depth
(non-gap rows); vertical rules = chain boundaries, horizontal = blocks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanopulldown.core.proteins import chain_labels, load_proteins, pair_labels
from nanopulldown.core.specs import load_pair, protein_chains
from nanopulldown.report.theme import lazy_theme

TH = lazy_theme()
# rows drawn at most: imshow embeds the image at data resolution, and a deep
# MSA (tens of thousands of rows) would make a many-MB SVG. Depth and block
# sizes are still computed on every row.
MAX_ROWS = 2000


def read_a3m(path: Path) -> tuple[list[int], list[int], list[str]]:
    """(unique-sequence lengths, their cardinalities, match-state rows)."""
    lens, cards, seqs, cur = None, None, [], []
    with open(path) as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if ln.startswith("#") and lens is None and not seqs:
                head = ln[1:].split("\t")
                lens = [int(x) for x in head[0].split(",")]
                cards = [int(x) for x in head[1].split(",")] if len(head) > 1 else [1] * len(lens)
            elif ln.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            elif ln.strip():
                cur.append(ln.strip())
    if cur:
        seqs.append("".join(cur))
    rows = ["".join(c for c in s if not c.islower()) for s in seqs]
    L = len(rows[0]) if rows else 0
    if lens is None:                                   # single-chain a3m: no header
        lens, cards = [L], [1]
    return lens, cards, [r for r in rows if len(r) == L]


def segments(spec: dict, lens: list[int], cards: list[int]) -> list[str]:
    """One label per a3m column segment: ColabFold lays the unique sequences
    out in first-appearance order, each repeated by its cardinality."""
    names = chain_labels(spec, load_proteins())
    by_seq: dict[str, list[str]] = {}
    for c in protein_chains(spec):
        by_seq.setdefault(c["sequence"], []).append(c["id"])
    groups = list(by_seq.values())
    if [len(g) for g in groups] != cards or [len(s) for s in by_seq] != lens:
        return [f"chain {i + 1}" for i in range(sum(cards))]   # a3m not of this spec
    return [f"{cid}: {names.get(cid, cid)}" for g in groups for cid in g]


def coverage_lines(msa: np.ndarray, bounds: np.ndarray):
    """plot_msa_v2's rows: (identity-coloured rows with NaN gaps, block edges)."""
    gap = msa != ord("-")
    qid = msa == msa[0]
    n_seg = len(bounds) - 1
    gapid = np.stack([gap[:, bounds[i]:bounds[i + 1]].any(-1) for i in range(n_seg)], -1)
    # paired rows (more chains covered) first, then chain 1's block, chain 2's, ...
    patterns = sorted({tuple(g) for g in gapid}, key=lambda g: (-sum(g), [not x for x in g]))
    lines, edges = [], [0]
    for g in patterns:
        idx = np.where((gapid == np.array(g)).all(-1))[0]
        seqid = np.stack([qid[idx, bounds[i]:bounds[i + 1]].mean(-1) for i in range(n_seg)], -1).sum(-1) \
            / (sum(g) + 1e-8)
        order = idx[np.argsort(seqid)]
        lines.append(np.where(gap[order], np.sort(seqid)[:, None], np.nan))
        edges.append(edges[-1] + len(idx))
    return np.concatenate(lines, 0), edges, patterns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    TH.apply()
    spec = load_pair(a.pair)
    try:
        label = pair_labels(spec, load_proteins())["label"]
    except KeyError:
        label = a.pair
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    a3m = Path(a.a3m)
    lens, cards, rows = read_a3m(a3m) if a3m.exists() else ([], [], [])
    if len(rows) < 2:
        ax.text(0.5, 0.5, "no MSA (query only)" if rows else f"no a3m at {a3m}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{label}: MSA coverage\n[{a.pair}]", fontsize=9)
    else:
        msa = np.frombuffer("".join(rows).encode(), dtype=np.uint8).reshape(len(rows), -1)
        seg_len = [n for n, c in zip(lens, cards) for _ in range(c)]
        bounds = np.concatenate([[0], np.cumsum(seg_len)])
        names = segments(spec, lens, cards)
        lines, edges, patterns = coverage_lines(msa, bounds)
        N, L = lines.shape
        depth = (~np.isnan(lines)).sum(0)

        shown = lines if N <= MAX_ROWS else lines[np.linspace(0, N - 1, MAX_ROWS).round().astype(int)]
        cmap = matplotlib.colormaps["rainbow_r"].copy()
        cmap.set_bad(alpha=0)
        im = ax.imshow(np.ma.masked_invalid(shown), interpolation="nearest", aspect="auto",
                       cmap=cmap, vmin=0, vmax=1, origin="lower", extent=(0.5, L + 0.5, 0, N))
        for x in bounds[1:-1]:
            ax.axvline(x + 0.5, color=TH.ink, lw=0.8)
        for y in edges[1:-1]:
            ax.axhline(y, color=TH.ink, lw=0.8)
        ax.plot(np.arange(1, L + 1), depth, color=TH.ink, lw=1)
        ax.set_xlim(0.5, L + 0.5)
        ax.set_ylim(0, N)
        ax.set_xlabel("positions (complex, chains concatenated)")
        ax.set_ylabel("sequences")

        # block sizes (paired = rows covering every chain) and per-chain depth
        sizes = {p: e1 - e0 for p, e0, e1 in zip(patterns, edges[:-1], edges[1:])}
        n_paired = sizes.get(tuple([True] * len(seg_len)), 0) - 1 if len(seg_len) > 1 else 0
        stats = [f"{nm}: median depth {int(np.median(depth[bounds[i]:bounds[i + 1]]))}"
                 for i, nm in enumerate(names)]
        head = f"{N} sequences" + (f", {max(n_paired, 0)} paired" if len(seg_len) > 1 else "")
        ax.set_title(f"{label}: MSA coverage ({head})\n" + " · ".join(stats) + f"\n[{a.pair}]",
                     fontsize=9)
        if len(seg_len) > 1:
            sec = ax.secondary_xaxis("top")
            sec.set_xticks([(bounds[i] + bounds[i + 1]) / 2 + 0.5 for i in range(len(names))])
            sec.set_xticklabels(names, fontsize=7)
            sec.tick_params(length=0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cbar.set_label("sequence identity to query")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    TH.save_all(fig, out, a.formats)
    plt.close(fig)
    print(f"[plot_msa_coverage] {a.pair} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
