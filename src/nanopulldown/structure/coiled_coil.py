#!/usr/bin/env python3
"""
nanopulldown.structure.coiled_coil — Stage 2 (coiled-coil geometry)
===================================================================
Did the predicted coiled coils actually pair in the models? For every
minimized sample of a pair and every (bait-side coil, prey-side coil) pair of
segments from annotation.yaml (`interpro.<chain>.features.coils`: DeepCoil
where it ran, else InterPro COILS -- nanopulldown.annotate.coils), a
knobs-into-holes test after SOCKET (Walshaw & Woolfson 2001, J Mol Biol
307:1427):

  helix      a residue is helical when CA(i)-CA(i+3) and CA(i)-CA(i+4) are
             alpha-helical (ca_i3 / ca_i4 ranges below; ideal 5.0-5.4 and
             6.2 A); residues i..i+4 of each such window are marked
  knob       a helical residue of one helix whose side-chain centroid (heavy
             atoms from CB on; CA for Gly) is within `packing_cutoff` (SOCKET:
             7.0 A) of the centroids of ALL FOUR residues j, j+3, j+4, j+7 of
             the other helix -- the hole -- all four helical
  coiled     knobs on BOTH helices, at least `min_knobs` each (complementary
  coil       packing; one heptad of a dimer gives two knobs per helix, a and
             d), i.e. the two segments wrap around each other rather than
             touching tip to side

Descriptors (not part of the call), over the knob region of each helix
(first to last knob or hole residue, +-3): helix axes from the mean of four
consecutive CAs, fitted as lines (PCA); `axis_distance` the mean distance of
each axis' points to the other line (dimers ~9.5-10.5 A, trimers ~6.5 A from
the bundle axis so ~11 A apart); `crossing_angle` between the two lines,
0-90 degrees (coiled coils: ~20 or less over a short stretch);
`orientation` parallel / antiparallel from the N->C directions.
`knobs_at_ad`: fraction of the knobs on DeepCoil a/d positions
(annotation.yaml coil_register; NaN for COILS segments), a check that the
model packs with the register DeepCoil predicted.

Segments are padded by `pad` residues on each side (a coil call boundary is
not precise). Residue numbers are the chain's own (1-based), as everywhere.

Output (Parquet, one row per sample x segment pair; schema in COLUMNS):
  results/<pair>/coiled_coil.parquet

Usage:
  nanopulldown coiled-coil --pair <pair> --annotation <annotation.yaml> \
      --results-root <results> --num-samples N --out <parquet>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from nanopulldown.core.config import LazyConfig
from nanopulldown.core.io import write_parquet_with_metadata
from nanopulldown.core.specs import load_pair

CFG = LazyConfig()
BACKBONE = {"N", "CA", "C", "O", "OXT"}
DEFAULTS = {"packing_cutoff": 7.0, "min_knobs": 3, "pad": 3,
            "ca_i3": [4.5, 5.9], "ca_i4": [5.5, 6.9]}

COLUMNS = {
    "pair": "Pair name",
    "sample": "Sample index (results/<pair>/minimized/sample_<i>.pdb.gz)",
    "chain_a": "Bait-side chain of the coil segment pair",
    "start_a": "First residue of the bait-side coil segment (chain numbering, before padding)",
    "end_a": "Last residue of the bait-side coil segment",
    "chain_b": "Prey-side chain",
    "start_b": "First residue of the prey-side coil segment",
    "end_b": "Last residue of the prey-side coil segment",
    "coil_source": "Where the segments come from: DeepCoil or COILS (annotation.yaml coils_source)",
    "helical_frac_a": "Fraction of the bait-side segment residues that are alpha-helical in the model (CA i/i+3/i+4 distances)",
    "helical_frac_b": "Same, prey-side segment",
    "knobs_a": "Number of bait-side knobs: helical residues whose side-chain centroid is within packing_cutoff of all four centroids of a prey-side hole j, j+3, j+4, j+7",
    "knobs_b": "Number of prey-side knobs into bait-side holes",
    "knob_residues_a": "Bait-side knob residues, comma-separated",
    "knob_residues_b": "Prey-side knob residues, comma-separated",
    "knobs_at_ad_a": "Fraction of bait-side knobs on DeepCoil heptad a/d positions (NaN without a DeepCoil register or knobs)",
    "knobs_at_ad_b": "Same, prey side",
    "axis_distance": "Mean distance (A) between the two helix axes over the knob region (NaN without knobs on both sides)",
    "crossing_angle": "Angle (degrees, 0-90) between the two helix axes over the knob region",
    "orientation": "parallel / antiparallel (N->C directions of the two axes), empty without knobs",
    "coiled_coil": "True when both helices have at least min_knobs knobs (complementary knobs-into-holes)",
}


def geometry_cfg() -> dict:
    return {**DEFAULTS, **(CFG.get("coiled_coil") or {})}


# ── structure ────────────────────────────────────────────────────────────

def load_chains(path: Path) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """{chain: {resnum: (CA xyz, side-chain centroid xyz)}} of the first model."""
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    out: dict[str, dict] = {}
    for ch in st[0]:
        d = {}
        for res in ch:
            ca = res.find_atom("CA", "*")
            if ca is None:
                continue
            sc = [a.pos.tolist() for a in res if a.name not in BACKBONE]
            ca_xyz = np.array(ca.pos.tolist())
            d[res.seqid.num] = (ca_xyz, np.mean(sc, axis=0) if sc else ca_xyz)
        out[ch.name] = d
    return out


def helical_residues(res: dict[int, tuple], gcfg: dict) -> set[int]:
    lo3, hi3 = gcfg["ca_i3"]
    lo4, hi4 = gcfg["ca_i4"]
    out = set()
    for i in res:
        if i + 4 not in res or i + 3 not in res:
            continue
        d3 = np.linalg.norm(res[i][0] - res[i + 3][0])
        d4 = np.linalg.norm(res[i][0] - res[i + 4][0])
        if lo3 <= d3 <= hi3 and lo4 <= d4 <= hi4:
            out.update(range(i, i + 5))
    return out


def knobs(res_k: dict, ks: list[int], res_h: dict, hs: list[int], cutoff: float) -> tuple[set, set]:
    """(knob residues among ks, hole residues among hs they pack into): a knob's
    centroid within cutoff of all of j, j+3, j+4, j+7 of the other helix."""
    if not ks or not hs:
        return set(), set()
    hset = set(hs)
    tree = cKDTree(np.array([res_h[j][1] for j in hs]))
    out, holes = set(), set()
    for k in ks:
        near = {hs[i] for i in tree.query_ball_point(res_k[k][1], cutoff)}
        for j in near:
            hole = (j, j + 3, j + 4, j + 7)
            if all(h in near and h in hset for h in hole):
                out.add(k)
                holes.update(hole)
    return out, holes


def _axis(res: dict, lo: int, hi: int) -> np.ndarray:
    """Helix axis points: mean of CA i..i+3 over [lo, hi]."""
    pts = [np.mean([res[i + d][0] for d in range(4)], axis=0)
           for i in range(lo, hi - 2) if all(i + d in res for d in range(4))]
    return np.array(pts)


def _line(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = pts.mean(axis=0)
    v = np.linalg.svd(pts - c)[2][0]
    if np.dot(pts[-1] - pts[0], v) < 0:       # N -> C
        v = -v
    return c, v


def _dist_to_line(pts, c, v) -> np.ndarray:
    d = pts - c
    return np.linalg.norm(d - np.outer(d @ v, v), axis=1)


def axis_geometry(res_a, region_a, res_b, region_b) -> tuple[float, float, str]:
    pa, pb = _axis(res_a, *region_a), _axis(res_b, *region_b)
    if len(pa) < 3 or len(pb) < 3:
        return np.nan, np.nan, ""
    (ca, va), (cb, vb) = _line(pa), _line(pb)
    dist = float(np.mean(np.concatenate([_dist_to_line(pa, cb, vb), _dist_to_line(pb, ca, va)])))
    cos = float(np.dot(va, vb))
    ang = float(np.degrees(np.arccos(min(1.0, abs(cos)))))
    return dist, ang, "parallel" if cos > 0 else "antiparallel"


def _ad_fraction(ks: set[int], ad: set[int] | None) -> float:
    return float(np.mean([k in ad for k in ks])) if ad and ks else np.nan


def test_pair(res_a, seg_a, ad_a, res_b, seg_b, ad_b, gcfg) -> dict:
    """The knobs-into-holes test of one segment pair in one model."""
    pad = int(gcfg["pad"])
    hel_a, hel_b = helical_residues(res_a, gcfg), helical_residues(res_b, gcfg)
    span_a = [r for r in range(seg_a[0] - pad, seg_a[1] + pad + 1) if r in res_a]
    span_b = [r for r in range(seg_b[0] - pad, seg_b[1] + pad + 1) if r in res_b]
    ha, hb = [r for r in span_a if r in hel_a], [r for r in span_b if r in hel_b]
    cut = float(gcfg["packing_cutoff"])
    ka, holes_b = knobs(res_a, ha, res_b, hb, cut)
    kb, holes_a = knobs(res_b, hb, res_a, ha, cut)
    seg_len = lambda s: s[1] - s[0] + 1
    row = {
        "helical_frac_a": sum(seg_a[0] <= r <= seg_a[1] for r in ha) / seg_len(seg_a),
        "helical_frac_b": sum(seg_b[0] <= r <= seg_b[1] for r in hb) / seg_len(seg_b),
        "knobs_a": len(ka), "knobs_b": len(kb),
        "knob_residues_a": ",".join(map(str, sorted(ka))), "knob_residues_b": ",".join(map(str, sorted(kb))),
        "knobs_at_ad_a": _ad_fraction(ka, ad_a), "knobs_at_ad_b": _ad_fraction(kb, ad_b),
        "axis_distance": np.nan, "crossing_angle": np.nan, "orientation": "",
        "coiled_coil": len(ka) >= int(gcfg["min_knobs"]) and len(kb) >= int(gcfg["min_knobs"]),
    }
    if ka and kb:
        ra, rb = ka | holes_a, kb | holes_b
        row["axis_distance"], row["crossing_angle"], row["orientation"] = axis_geometry(
            res_a, (min(ra) - 3, max(ra) + 3), res_b, (min(rb) - 3, max(rb) + 3))
    return row


# ── pair ─────────────────────────────────────────────────────────────────

def chain_coils(ann: dict, cid: str) -> tuple[list[tuple[int, int]], dict, str]:
    """(segments, {(start, end): a/d residue set}, source) of one chain."""
    ip = (ann.get("interpro") or {}).get(cid) or {}
    segs = [(int(a), int(b)) for a, b in (ip.get("features") or {}).get("coils") or []]
    reg = {(int(r["start"]), int(r["end"])): set(r.get("a") or []) | set(r.get("d") or [])
           for r in ip.get("coil_register") or []}
    return segs, reg, ip.get("coils_source") or ("COILS" if segs else "")


def pair_rows(pair: str, spec: dict, ann: dict, samples: list[tuple[int, Path]], gcfg: dict) -> list[dict]:
    coils = {c: chain_coils(ann, c) for side in ("bait", "prey") for c in spec.get(f"{side}_chains", [])}
    combos = [(ca, sa, cb, sb) for ca in spec.get("bait_chains", []) for sa in coils[ca][0]
              for cb in spec.get("prey_chains", []) for sb in coils[cb][0]]
    rows = []
    if not combos:
        return rows
    for smp, path in samples:
        chains = load_chains(path)
        for ca, sa, cb, sb in combos:
            if ca not in chains or cb not in chains:
                print(f"[coiled_coil] {path}: chain {ca} or {cb} missing", file=sys.stderr)
                continue
            src = coils[ca][2] if coils[ca][2] == coils[cb][2] else f"{coils[ca][2]}/{coils[cb][2]}"
            rows.append({"pair": pair, "sample": smp, "chain_a": ca, "start_a": sa[0], "end_a": sa[1],
                         "chain_b": cb, "start_b": sb[0], "end_b": sb[1], "coil_source": src,
                         **test_pair(chains[ca], sa, coils[ca][1].get(sa), chains[cb], sb,
                                     coils[cb][1].get(sb), gcfg)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1] if __doc__ else None)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--annotation", required=True, help="data/annotation/<pair>/annotation.yaml")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gcfg = geometry_cfg()
    spec = load_pair(a.pair)
    ann = yaml.safe_load(Path(a.annotation).read_text()) or {}
    root = Path(a.results_root) / a.pair / "minimized"
    samples = [(i, root / f"sample_{i}.pdb.gz") for i in range(a.num_samples)]
    rows = pair_rows(a.pair, spec, ann, samples, gcfg)
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    for c in ("sample", "start_a", "end_a", "start_b", "end_b", "knobs_a", "knobs_b"):
        df[c] = df[c].astype("int64")
    df["coiled_coil"] = df["coiled_coil"].astype(bool)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_with_metadata(
        df, out,
        table_description=("Knobs-into-holes (SOCKET-style) test of every bait-side x prey-side predicted "
                           f"coiled-coil segment pair in each minimized sample of {a.pair}; parameters: "
                           + ", ".join(f"{k}={v}" for k, v in sorted(gcfg.items()))),
        column_descriptions=COLUMNS,
        table_references="Walshaw J, Woolfson DN (2001) J Mol Biol 307:1427 (SOCKET)",
    )
    n_cc = df.loc[df["coiled_coil"], "sample"].nunique() if len(df) else 0
    print(f"[coiled_coil] {a.pair}: {len(df)} segment-pair x sample row(s); "
          f"coiled coil in {n_cc}/{a.num_samples} sample(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
