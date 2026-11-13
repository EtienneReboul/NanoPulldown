#!/usr/bin/env python3
"""
nanopulldown.annotate.idr_scores — IDR x IDR complementarity, the arithmetic
===========================================================================
numpy only (no SPARROW): IDR calling from a disorder track, per-residue
property vectors, windowed profiles, the pairwise complementarity scores and
their two null models. nanopulldown.annotate.idr_pairs adds the SPARROW
features and the pipeline plumbing around it.

Every pairwise score is a function of a few per-IDR summary statistics
(global fractions + the extremes of the windowed profiles), so the same
formulas (`pair_scores`) serve the observed pair, the shuffle null (arrays
over shuffles) and the background null (arrays over background IDRs) by
broadcasting.

Scores (a, b = the two IDRs; f = fraction of the IDR's residues):
  s_charge      -NCPR_a * NCPR_b                      > 0 when net charges are opposite
  local_charge  max_ij -NCPR_win_a[i] * NCPR_win_b[j] the most complementary pair of
                                                      charge blocks (one window per IDR);
                                                      bilinear -> at the profiles' extremes
  s_catpi       f_arom_a * f_RK_b + f_arom_b * f_RK_a cation-pi stickers (arom = F+Y+W)
  local_catpi   max(max_win arom_a * max_win RK_b, the reverse)
  s_pi          f_arom_a * f_arom_b                   pi-pi stickers
  local_pi      max_win arom_a * max_win arom_b
  hydro_local   max_win hyd_a * max_win hyd_b         hyd = Kyte-Doolittle scaled to [0, 1]
                                                      (SPARROW's scale / 9)
Charges: K, R +1; D, E -1; H 0 (SPARROW's NCPR); a phosphosite (pS/pT/pY) -2.
Windows are the L - w + 1 full windows (no end padding), as SPARROW's
linear_sequence_profile(..., end_mode="").
"""
from __future__ import annotations

import hashlib

import numpy as np

AROMATIC = "FYW"
CATIONIC = "RK"
# Kyte-Doolittle, as SPARROW (calculate_parameters) shifts it: KD + 4.5, here / 9 -> [0, 1]
KD = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7,
      "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5,
      "N": -3.5, "K": -3.9, "R": -4.5}
HYD = {aa: (v + 4.5) / 9.0 for aa, v in KD.items()}

# mechanism -> the scores whose background percentile calls it (the max of them)
MECHANISMS = {
    "charge": ("s_charge", "local_charge"),
    "cation_pi": ("s_catpi", "local_catpi"),
    "pi": ("s_pi", "local_pi"),
    "hydrophobic": ("hydro_local",),
}
LOCAL_SCORES = ("local_charge", "local_catpi", "local_pi", "hydro_local")
GLOBAL_SCORES = ("s_charge", "s_catpi", "s_pi")


# ── IDR calling ───────────────────────────────────────────────────────────

def define_idrs(disorder, cutoff: float = 0.5, max_gap: int = 10, min_len: int = 30) -> list[tuple[int, int]]:
    """1-based inclusive IDRs: runs of disorder >= cutoff, merged across
    ordered gaps of <= max_gap residues, kept when >= min_len long."""
    runs, start = [], None
    for i, v in enumerate(list(disorder) + [-1.0]):
        if v >= cutoff and start is None:
            start = i
        elif v < cutoff and start is not None:
            runs.append([start, i - 1])
            start = None
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s + 1, e + 1) for s, e in merged if e - s + 1 >= min_len]


def segments_overlap(s: int, e: int, segs) -> int:
    """Residues of [s, e] inside any of `segs` (1-based inclusive, disjoint)."""
    return sum(max(0, min(e, b) - max(s, a) + 1) for a, b in segs)


# ── per-residue vectors and windows ───────────────────────────────────────

def charge_vector(seq: str, phospho=()) -> np.ndarray:
    """Per-residue charge; `phospho` = 0-based positions carrying a phosphate (-2)."""
    q = np.array([1.0 if c in "KR" else -1.0 if c in "DE" else 0.0 for c in seq])
    for i in phospho:
        q[i] = -2.0
    return q


def property_vectors(seq: str, phospho=()) -> dict[str, np.ndarray]:
    return {
        "charge": charge_vector(seq, phospho),
        "arom": np.array([c in AROMATIC for c in seq], dtype=float),
        "cat": np.array([c in CATIONIC for c in seq], dtype=float),
        "hyd": np.array([HYD.get(c, 0.5) for c in seq]),
    }


def window_means(x: np.ndarray, w: int) -> np.ndarray:
    """Means of the full windows of width w along the last axis: (..., L - w + 1)."""
    c = np.cumsum(np.concatenate([np.zeros(x.shape[:-1] + (1,)), x], axis=-1), axis=-1)
    return (c[..., w:] - c[..., :-w]) / w


def summary(vecs: dict[str, np.ndarray], windows) -> dict:
    """The per-IDR statistics every score is made of. Works on 1-D vectors
    (one sequence) or 2-D (rows = shuffles); extremes along the last axis.
    Window-dependent keys are suffixed with the window size; the argmax
    positions (0-based window starts) ride along for 1-D input."""
    q = vecs["charge"]
    out = {"ncpr": q.mean(-1), "fpos": (q > 0).mean(-1), "fneg": (q < 0).mean(-1),
           "arom": vecs["arom"].mean(-1), "cat": vecs["cat"].mean(-1)}
    for w in windows:
        prof = {"ncpr": window_means(q, w), "arom": window_means(vecs["arom"], w),
                "cat": window_means(vecs["cat"], w), "hyd": window_means(vecs["hyd"], w)}
        out[f"ncpr_max_{w}"], out[f"ncpr_min_{w}"] = prof["ncpr"].max(-1), prof["ncpr"].min(-1)
        for k in ("arom", "cat", "hyd"):
            out[f"{k}_max_{w}"] = prof[k].max(-1)
        if q.ndim == 1:
            out[f"pos_{w}"] = {"ncpr_max": int(prof["ncpr"].argmax()), "ncpr_min": int(prof["ncpr"].argmin()),
                               **{f"{k}_max": int(prof[k].argmax()) for k in ("arom", "cat", "hyd")}}
    return out


def shuffle_summary(seq: str, n: int, windows, phospho=(), seed: int | None = None) -> dict:
    """summary() of n composition-preserving shuffles (rows), reproducible:
    the default seed is derived from the sequence."""
    if seed is None:
        seed = int(hashlib.md5(seq.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    perm = rng.permuted(np.tile(np.arange(len(seq)), (n, 1)), axis=1)
    vecs = property_vectors(seq, phospho)
    return summary({k: v[perm] for k, v in vecs.items()}, windows)


# ── pairwise scores ───────────────────────────────────────────────────────

def pair_scores(a: dict, b: dict, w: int) -> dict:
    """All scores of IDR a x IDR b at window w; a and b are summary() dicts
    whose values may be scalars or broadcastable arrays."""
    amax, amin, bmax, bmin = a[f"ncpr_max_{w}"], a[f"ncpr_min_{w}"], b[f"ncpr_max_{w}"], b[f"ncpr_min_{w}"]
    return {
        "s_charge": -a["ncpr"] * b["ncpr"],
        "local_charge": np.maximum.reduce(np.broadcast_arrays(-amax * bmin, -amin * bmax, -amax * bmax, -amin * bmin)),
        "s_catpi": a["arom"] * b["cat"] + b["arom"] * a["cat"],
        "local_catpi": np.maximum(a[f"arom_max_{w}"] * b[f"cat_max_{w}"], b[f"arom_max_{w}"] * a[f"cat_max_{w}"]),
        "s_pi": a["arom"] * b["arom"],
        "local_pi": a[f"arom_max_{w}"] * b[f"arom_max_{w}"],
        "hydro_local": a[f"hyd_max_{w}"] * b[f"hyd_max_{w}"],
    }


def local_charge_windows(a: dict, b: dict, w: int) -> tuple[int, int]:
    """0-based window starts (in a, in b) of the best local_charge of two
    1-D summaries: which extremes of the two NCPR profiles realise it."""
    pa, pb = a[f"pos_{w}"], b[f"pos_{w}"]
    cands = [(-a[f"ncpr_max_{w}"] * b[f"ncpr_min_{w}"], pa["ncpr_max"], pb["ncpr_min"]),
             (-a[f"ncpr_min_{w}"] * b[f"ncpr_max_{w}"], pa["ncpr_min"], pb["ncpr_max"]),
             (-a[f"ncpr_max_{w}"] * b[f"ncpr_max_{w}"], pa["ncpr_max"], pb["ncpr_max"]),
             (-a[f"ncpr_min_{w}"] * b[f"ncpr_min_{w}"], pa["ncpr_min"], pb["ncpr_min"])]
    _, i, j = max(cands, key=lambda c: c[0])
    return i, j


def mechanism_windows(mech: str, a: dict, b: dict, w: int) -> tuple[int, int]:
    """0-based window starts (in a, in b) of the local score behind `mech`."""
    pa, pb = a[f"pos_{w}"], b[f"pos_{w}"]
    if mech == "charge":
        return local_charge_windows(a, b, w)
    if mech == "cation_pi":
        if a[f"arom_max_{w}"] * b[f"cat_max_{w}"] >= b[f"arom_max_{w}"] * a[f"cat_max_{w}"]:
            return pa["arom_max"], pb["cat_max"]
        return pa["cat_max"], pb["arom_max"]
    if mech == "pi":
        return pa["arom_max"], pb["arom_max"]
    return pa["hyd_max"], pb["hyd_max"]


# ── null models ───────────────────────────────────────────────────────────

def shuffle_null(obs: float, null: np.ndarray) -> tuple[float, float]:
    """(z-score, one-sided empirical p = (1 + #null >= obs) / (n + 1))."""
    null = np.asarray(null, dtype=float)
    sd = null.std()
    z = float((obs - null.mean()) / sd) if sd > 0 else float("nan")
    return z, float((1 + np.sum(null >= obs - 1e-12)) / (len(null) + 1))


def percentile(obs: float, null: np.ndarray) -> float:
    """Fraction of `null` below obs, ties counted half (mid-rank), in [0, 1]."""
    null = np.asarray(null, dtype=float)
    if not len(null):
        return float("nan")
    return float((np.sum(null < obs - 1e-12) + 0.5 * np.sum(np.abs(null - obs) <= 1e-12)) / len(null))


def stack_summaries(summaries: list[dict], windows) -> dict:
    """List of 1-D summary() dicts -> one dict of arrays (the background set)."""
    keys = [k for k in summaries[0] if not k.startswith("pos_")]
    return {k: np.array([s[k] for s in summaries], dtype=float) for k in keys}


def dominant_mechanism(pcts: dict[str, float], raw: dict[str, float], min_pct: float) -> tuple[str, dict[str, float]]:
    """(label, {mechanism: percentile}): a mechanism's percentile is the max of
    its scores' background percentiles among those > 0; the label is the
    highest one when >= min_pct, else "none"."""
    mp = {}
    for m, keys in MECHANISMS.items():
        vals = [pcts[k] for k in keys if raw[k] > 0 and np.isfinite(pcts[k])]
        mp[m] = max(vals) if vals else 0.0
    best = max(mp, key=mp.get)
    return (best if mp[best] >= min_pct else "none"), mp
