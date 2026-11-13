#!/usr/bin/env python3
"""
nanopulldown.report.plots.idr_pairs — IDR complementarity figures
=================================================================
Drawn by nanopulldown.annotate.idr_pairs at the end of its run (the profiles
are recomputed from the chain sequences, numpy only):

  mechanism_heatmap.svg  the top IDR pairs (rows, run-wide ranking) x the four
                         mechanisms (columns): background percentile
  top/<NN>_<pair>_<bait IDR>_<prey IDR>.svg
                         per top IDR pair, bait chain left / prey chain right,
                         three panels sharing the residue axis: windowed NCPR,
                         windowed aromatic (F Y W) and R + K fractions, AIUPred
                         disorder + MoRFs. The IDR is tinted; the best
                         local-charge windows are marked, plus the windows
                         behind the dominant mechanism when that is not charge.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nanopulldown.annotate import idr_scores as S
from nanopulldown.report.theme import lazy_theme

TH = lazy_theme()
MECH_LABEL = {"charge": "charge", "cation_pi": "cation–π", "pi": "π–π", "hydrophobic": "hydrophobic"}


def _centered(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(window-centre residue numbers, window means) of a per-residue vector."""
    m = S.window_means(x, w)
    return np.arange(len(m)) + 1 + (w - 1) / 2, m


def _label(row, side: str) -> str:
    name = row["bait_protein" if side == "a" else "prey_protein"] or row[f"{side}_accession"] or row[f"{side}_chain"]
    return f"{name} {row[f'{side}_start']}–{row[f'{side}_end']}"


def mechanism_heatmap(view, out: Path, top_n: int, formats: str, meta: str) -> None:
    top = view.head(top_n)
    if top.empty:
        return
    mechs = list(S.MECHANISMS)
    vals = top[[f"pct_{m}" for m in mechs]].to_numpy(dtype=float)
    labels = [f"{r['pair']}  ·  {_label(r, 'a')} × {_label(r, 'b')}" for _, r in top.iterrows()]
    TH.apply()
    fig, ax = plt.subplots(figsize=(4.2 + 0.07 * max(map(len, labels)), 0.9 + 0.3 * len(top)))
    im = ax.imshow(vals, cmap=TH.cmap(), vmin=0, vmax=1, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=7, color=TH.cell_ink(vals[i, j]))
    ax.set_xticks(range(len(mechs)), [MECH_LABEL[m] for m in mechs])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("background percentile", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title("IDR pair complementarity by mechanism (top pairs, run-wide)", fontsize=9, pad=24)
    fig.text(0.01, 0.005, meta, fontsize=5, color=TH.ink2, wrap=True)
    fig.tight_layout()
    TH.save_all(fig, out, formats, bbox_inches="tight")
    plt.close(fig)


def _side_panels(axes, seq, disorder, morfs, idr, segs, w, title):
    """Three panels (NCPR, stickers, disorder) of one chain."""
    v = S.property_vectors(seq)
    pos_c, neg_c = TH.slot("blue"), TH.slot("orange")
    tint = TH.region("disordered")
    ax1, ax2, ax3 = axes
    for ax in axes:
        ax.axvspan(idr[0] - 0.5, idr[1] + 0.5, color=tint, alpha=0.45, lw=0, zorder=0)
        for (s, e), style in segs:
            ax.axvspan(s - 0.5, e + 0.5, zorder=1, lw=0, **style)
        ax.grid(axis="y", lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    x, m = _centered(v["charge"], w)
    ax1.fill_between(x, 0, m, where=m >= 0, color=pos_c, lw=0, step="mid", label="net +")
    ax1.fill_between(x, 0, m, where=m < 0, color=neg_c, lw=0, step="mid", label="net −")
    ax1.axhline(0, color=TH.ink2, lw=0.6)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_ylabel(f"NCPR ({w} aa)", fontsize=8)
    ax1.set_title(title, fontsize=9, loc="left")
    xa, ma = _centered(v["arom"], w)
    xc, mc = _centered(v["cat"], w)
    ax2.plot(xa, ma, color=TH.slot("violet"), lw=1.5, label="F Y W")
    ax2.plot(xc, mc, color=TH.slot("aqua"), lw=1.5, label="R K")
    ax2.set_ylim(0, max(0.3, float(np.max(np.r_[ma, mc])) * 1.1))
    ax2.set_ylabel(f"fraction ({w} aa)", fontsize=8)
    if len(disorder) == len(seq):
        ax3.plot(np.arange(1, len(seq) + 1), disorder, color=TH.color("disorder"), lw=1.2, label="AIUPred disorder")
    for s, e in morfs:
        ax3.plot([s, e], [1.04, 1.04], color=TH.color("binding"), lw=3, solid_capstyle="butt")
    ax3.axhline(0.5, color=TH.ink2, lw=0.6, ls=":")
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("disorder", fontsize=8)
    ax3.set_xlabel("residue (chain numbering)", fontsize=8)
    ax3.set_xlim(0.5, len(seq) + 0.5)


def pair_figure(row, chain_a, chain_b, w: int, out: Path, formats: str) -> None:
    """chain_x = (sequence, disorder track, MoRFs) of one chain, chain numbering."""
    TH.apply()
    fig, axes = plt.subplots(3, 2, figsize=(11, 5.6), sharex="col",
                             gridspec_kw={"height_ratios": [1.2, 1, 0.8], "hspace": 0.12, "wspace": 0.18})
    seg_style = {"color": TH.ink, "alpha": 0.16}
    mech_style = {"facecolor": "none", "edgecolor": TH.slot("magenta"), "hatch": "////", "alpha": 0.6}
    mech = row["dominant_mechanism"]
    for k, (side, (seq, dis, morfs)) in enumerate((("a", chain_a), ("b", chain_b))):
        segs = [((row[f"{side}_seg_start_w{w}"], row[f"{side}_seg_end_w{w}"]), seg_style)]
        if mech not in ("charge", "none"):
            segs.append(((row[f"{side}_mech_start"], row[f"{side}_mech_end"]), mech_style))
        role = "bait" if side == "a" else "prey"
        _side_panels(axes[:, k], seq, dis, morfs, (row[f"{side}_start"], row[f"{side}_end"]), segs, w,
                     f"{role}: {_label(row, side)} (chain {row[f'{side}_chain']})")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    h3, l3 = axes[2, 0].get_legend_handles_labels()
    from matplotlib.patches import Patch
    extra = [Patch(color=TH.region("disordered"), alpha=0.45, label="IDR"),
             Patch(color=TH.ink, alpha=0.16, label=f"best charge windows ({w} aa)")]
    if mech not in ("charge", "none"):
        extra.append(Patch(facecolor="none", edgecolor=TH.slot("magenta"), hatch="////",
                           label=f"{MECH_LABEL[mech]} windows"))
    extra.append(plt.Line2D([], [], color=TH.color("binding"), lw=3, label="MoRF (AIUPred)"))
    fig.legend(h1 + h2 + h3 + extra, l1 + l2 + l3 + [e.get_label() for e in extra],
               loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    pct = "  ".join(f"{MECH_LABEL[m]} {row[f'pct_{m}']:.2f}" for m in S.MECHANISMS)
    fig.suptitle(f"{row['pair']} — dominant mechanism: {MECH_LABEL.get(mech, mech)}\n"
                 f"background percentiles: {pct} · local charge z(shuffle) "
                 f"{row[f'local_charge_w{w}_z_shuffle']:.1f}", fontsize=9)
    fig.subplots_adjust(bottom=0.16, top=0.88)
    TH.save_all(fig, out, formats, bbox_inches="tight")
    plt.close(fig)


def draw_all(view, chains, run_idrs, prots, scorer, fig_dir: Path, top_n: int, formats: str, meta: str) -> None:
    """Heatmap + one profile figure per top IDR pair (`view`: ranked scores)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    top_dir = fig_dir / "top"
    top_dir.mkdir(exist_ok=True)
    for old in top_dir.glob("*"):
        old.unlink()
    mechanism_heatmap(view, fig_dir / "mechanism_heatmap.svg", max(top_n, 40), formats, meta)
    w = scorer.w0

    def chain_data(pair, cid):
        for ch, full, regions in chains[pair]:
            if ch["id"] != cid:
                continue
            pr = prots[full]
            fp = (np.concatenate([np.arange(a, b + 1) for a, b in regions]) if regions
                  else np.arange(1, len(full) + 1))
            dis = pr.disorder[fp - 1] if len(pr.disorder) == len(full) else np.array([])
            from nanopulldown.annotate.idr_pairs import _project
            return ch["sequence"], dis, _project(pr.morfs, fp)
        raise KeyError(cid)

    for i, (_, row) in enumerate(view.head(top_n).iterrows(), start=1):
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                      f"{i:02d}_{row['pair']}_{row['a_chain']}{row['a_start']}-{row['a_end']}_"
                      f"{row['b_chain']}{row['b_start']}-{row['b_end']}")
        pair_figure(row, chain_data(row["pair"], row["a_chain"]), chain_data(row["pair"], row["b_chain"]),
                    w, top_dir / f"{name}.svg", formats)
