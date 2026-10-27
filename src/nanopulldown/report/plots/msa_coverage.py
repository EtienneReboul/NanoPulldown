#!/usr/bin/env python3
"""
nanopulldown.report.plots.msa_coverage — Stage 2a report figures (ColabFold only)
=================================================================================
ColabFold-style "Sequence coverage" plots, read from the complex a3m
colabfold_msa cached for a pair (nanopulldown.fold.colabfold.a3m_path):
`#len1,len2<TAB>card1,card2` header -- one column segment per UNIQUE
sequence (copies of a homo-oligomer share it; ColabFold expands them at fold
time) -- then the paired rows (every segment filled) and each sequence's
unpaired rows (gaps over the others). Two figures:

  `plot msa-unpaired --pair P --chain C`  one protein's unpaired MSA (its
      own column segment of the a3m, its unpaired rows only) -- per distinct
      protein, next to its domain map + AIUPred figure (same numbering: the
      folded chain sequence, 1..L).
  `plot msa-paired --pair P`  the pair's paired rows only, chains side by
      side -- the coevolution signal AF2-multimer gets across the interface.

Drawn as colabfold.plot.plot_msa_v2: match-state columns only (a3m
insertions dropped), rows sorted by identity to the query, coloured by it
wherever the row has a residue, blank where it has a gap. Identity is over
the full length (of each chain, averaged over chains for paired rows), so a
short perfect partial hit still reads low. Black line = per-position depth.
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
# MSA (tens of thousands of rows) would make a many-MB SVG. Depth and counts
# are still computed on every row.
MAX_ROWS = 2000
GAP = ord("-")


def read_a3m(path: Path) -> tuple[list[int], list[int], np.ndarray]:
    """(unique-sequence lengths, their cardinalities, match-state rows as uint8 (N, L))."""
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
    rows = [r for r in rows if len(r) == L]
    return lens, cards, np.frombuffer("".join(rows).encode(), dtype=np.uint8).reshape(len(rows), L)


def segment_chains(spec: dict, lens: list[int], cards: list[int]) -> list[list[str]]:
    """Chain ids of each a3m column segment: ColabFold lays the unique sequences
    out in first-appearance order, one segment each. Empty lists when the a3m
    does not match the spec."""
    by_seq: dict[str, list[str]] = {}
    for c in protein_chains(spec):
        by_seq.setdefault(c["sequence"], []).append(c["id"])
    if [len(g) for g in by_seq.values()] != cards or [len(s) for s in by_seq] != lens:
        return [[] for _ in lens]
    return list(by_seq.values())


def load(pair: str, a3m: Path):
    """(spec, segment bounds, segment chain ids, rows, covered[N, n_seg])."""
    spec = load_pair(pair)
    if not a3m.exists():
        return spec, None, None, None, None
    lens, cards, msa = read_a3m(a3m)
    bounds = np.concatenate([[0], np.cumsum(lens)])
    covered = np.stack([(msa[:, bounds[i]:bounds[i + 1]] != GAP).any(-1)
                        for i in range(len(bounds) - 1)], -1)
    return spec, bounds, segment_chains(spec, lens, cards), msa, covered


def draw(ax, fig, msa: np.ndarray, query: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Rows of `msa` sorted and coloured by identity to `query` (averaged over
    the segments in `bounds`); returns the per-position depth."""
    qid = msa == query[None, :]
    seqid = np.mean([qid[:, bounds[i]:bounds[i + 1]].mean(-1) for i in range(len(bounds) - 1)], 0)
    order = np.argsort(seqid, kind="stable")
    lines = np.where(msa[order] != GAP, seqid[order][:, None], np.nan)
    N, L = lines.shape
    depth = (msa != GAP).sum(0)
    shown = lines if N <= MAX_ROWS else lines[np.linspace(0, N - 1, MAX_ROWS).round().astype(int)]
    cmap = matplotlib.colormaps["rainbow_r"].copy()
    cmap.set_bad(alpha=0)
    im = ax.imshow(np.ma.masked_invalid(shown), interpolation="nearest", aspect="auto",
                   cmap=cmap, vmin=0, vmax=1, origin="lower", extent=(0.5, L + 0.5, 0, N))
    for x in bounds[1:-1]:
        ax.axvline(x + 0.5, color=TH.ink, lw=0.8)
    ax.plot(np.arange(1, L + 1), depth, color=TH.ink, lw=1)
    ax.set_xlim(0.5, L + 0.5)
    ax.set_ylim(0, N)
    ax.set_ylabel("sequences")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("sequence identity to query")
    return depth


def empty(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])


def _args(chain: bool):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--a3m", required=True)
    if chain:
        ap.add_argument("--chain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="svg")
    return ap.parse_args()


def _save(fig, out: str, formats: str, what: str) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    TH.save_all(fig, out, formats)
    plt.close(fig)
    print(f"[plot_msa_coverage] {what} -> {out}")


def unpaired_main() -> int:
    a = _args(chain=True)
    TH.apply()
    spec, bounds, seg_chains, msa, covered = load(a.pair, Path(a.a3m))
    name = chain_labels(spec, load_proteins()).get(a.chain, a.chain)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    title = f"{name}: unpaired MSA coverage"
    if msa is None:
        empty(ax, f"no a3m at {a.a3m}")
    elif not any(a.chain in g for g in seg_chains):
        empty(ax, f"chain {a.chain} not found in the a3m")
    else:
        i = next(k for k, g in enumerate(seg_chains) if a.chain in g)
        lo, hi = bounds[i], bounds[i + 1]
        # its unpaired block: rows covering this segment only (query included)
        only = covered[:, i] & (covered.sum(-1) == 1) if len(seg_chains) > 1 else covered[:, i]
        rows = msa[only][:, lo:hi]
        query = msa[0, lo:hi]
        if len(rows) == 0 or not (rows == query).all(-1).any():
            rows = np.vstack([query[None, :], rows])
        depth = draw(ax, fig, rows, query, np.array([0, hi - lo]))
        ax.set_xlabel("residue")
        title += f" ({len(rows) - 1} sequences, median depth {int(np.median(depth))})"
    ax.set_title(f"{title}\n[{a.pair} chain {a.chain}]", fontsize=9)
    _save(fig, a.out, a.formats, f"{a.pair} {a.chain}")
    return 0


def paired_main() -> int:
    a = _args(chain=False)
    TH.apply()
    spec, bounds, seg_chains, msa, covered = load(a.pair, Path(a.a3m))
    try:
        label = pair_labels(spec, load_proteins())["label"]
    except KeyError:
        label = a.pair
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    title = f"{label}: paired MSA coverage"
    if msa is None:
        empty(ax, f"no a3m at {a.a3m}")
    elif len(bounds) <= 2:
        # ColabFold pairs only between different sequences
        empty(ax, "homo-oligomer (one unique sequence): no paired MSA")
    else:
        rows = msa[covered.all(-1)]                 # query first (a3m order)
        if len(rows) < 2:
            empty(ax, "no paired sequences (ColabFold found no pairing, or pair_mode: unpaired)")
            title += " (0 paired)"
        else:
            depth = draw(ax, fig, rows, msa[0], bounds)
            names = chain_labels(spec, load_proteins())
            segs = [f"{'/'.join(g)}: {names.get(g[0], g[0])}" if g else f"sequence {k + 1}"
                    for k, g in enumerate(seg_chains)]
            sec = ax.secondary_xaxis("top")
            sec.set_xticks([(bounds[k] + bounds[k + 1]) / 2 + 0.5 for k in range(len(segs))])
            sec.set_xticklabels(segs, fontsize=7)
            sec.tick_params(length=0)
            ax.set_xlabel("positions (chains concatenated)")
            stats = [f"{s.split(': ', 1)[-1]} median depth {int(np.median(depth[bounds[k]:bounds[k + 1]]))}"
                     for k, s in enumerate(segs)]
            title += f" ({len(rows) - 1} paired sequences)\n" + " · ".join(stats)
    ax.set_title(f"{title}\n[{a.pair}]", fontsize=9)
    _save(fig, a.out, a.formats, a.pair)
    return 0


if __name__ == "__main__":
    raise SystemExit(paired_main())
