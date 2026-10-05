#!/usr/bin/env python3
"""
scripts/count_clashes.py — Stage 2d (minimization QC)
=========================================================
Re-implements ChimeraX's `clashes` detection scheme in plain Python so the raw
ESMFold2 sample and its OpenMM-minimized counterpart can be scored the same
way (no ChimeraX needed).

ChimeraX scheme (defaults of `clashes` / `contacts`):
  overlap(a, b) = vdW(a) + vdW(b) - dist(a, b)
  a pair is a clash when overlap - hbond_allowance >= overlap_cutoff
    * overlap_cutoff   = 0.6 A
    * hbond_allowance  = 0.4 A, subtracted only for pairs that could form an
      H-bond (donor/acceptor)
    * pairs separated by <= 4 bonds (bond_separation) are ignored
    * every atom pair in the model is considered (intra- AND inter-chain)

Deliberate deviations (documented, kept in the output so they can be audited):
  * Heavy atoms only. ESMFold2's raw CIF has no hydrogens while the minimized
    PDB has them (PDBFixer, pH 7); counting H on one side only would bias the
    paired comparison. Both states are therefore stripped of H (and waters).
  * H-bond capability is approximated at the heavy-atom level: any N/O--N/O
    pair gets the allowance (ChimeraX uses per-atom donor/acceptor types).
  * Bonds are inferred from covalent radii (no topology is carried in the
    files), which also covers peptide bonds and disulfides.
  * vdW radii are plain element radii (C 1.70, N 1.55, O 1.52, S 1.80).

Output: one row per (sample, state in {raw, minimized}) in a CSV.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "SE": 1.90}
COV = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "SE": 1.20}
HBOND_ELEMENTS = {"N", "O"}
BOND_TOL = 0.45          # A added to the sum of covalent radii
DEFAULT_VDW, DEFAULT_COV = 1.70, 0.76


def load_heavy_atoms(path: Path):
    """Return (xyz, element, chain, bond-graph-ready arrays) of non-H, non-water atoms."""
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    st.remove_waters()
    xyz, elem, chain = [], [], []
    for ch in st[0]:
        for res in ch:
            for at in res:
                xyz.append(at.pos.tolist())
                elem.append(at.element.name.upper())
                chain.append(ch.name)
    return np.asarray(xyz, dtype=float), np.asarray(elem), np.asarray(chain)


def bond_graph(xyz: np.ndarray, elem: np.ndarray) -> list[set[int]]:
    cov = np.array([COV.get(e, DEFAULT_COV) for e in elem])
    adj: list[set[int]] = [set() for _ in range(len(xyz))]
    tree = cKDTree(xyz)
    for i, j in tree.query_pairs(r=2 * cov.max() + BOND_TOL):
        if np.linalg.norm(xyz[i] - xyz[j]) <= cov[i] + cov[j] + BOND_TOL:
            adj[i].add(j)
            adj[j].add(i)
    return adj


def within_bonds(adj: list[set[int]], start: int, max_sep: int) -> set[int]:
    seen, frontier = {start}, {start}
    for _ in range(max_sep):
        frontier = {n for a in frontier for n in adj[a]} - seen
        seen |= frontier
    return seen


def count_clashes(path: Path, cutoff: float, hbond_allowance: float,
                  bond_separation: int) -> dict:
    xyz, elem, chain = load_heavy_atoms(path)
    vdw = np.array([VDW.get(e, DEFAULT_VDW) for e in elem])
    polar = np.isin(elem, list(HBOND_ELEMENTS))
    adj = bond_graph(xyz, elem)

    # A pair can only clash if dist <= vdw_i + vdw_j - cutoff + allowance.
    reach = 2 * vdw.max() - cutoff + hbond_allowance
    n_total = n_inter = 0
    max_overlap = 0.0
    per_atom: dict[int, int] = defaultdict(int)
    for i, j in cKDTree(xyz).query_pairs(r=reach):
        d = float(np.linalg.norm(xyz[i] - xyz[j]))
        overlap = vdw[i] + vdw[j] - d
        if polar[i] and polar[j]:
            overlap -= hbond_allowance
        if overlap < cutoff:
            continue
        if j in within_bonds(adj, i, bond_separation):
            continue
        n_total += 1
        n_inter += int(chain[i] != chain[j])
        max_overlap = max(max_overlap, overlap)
        per_atom[i] += 1
        per_atom[j] += 1
    return {"n_atoms": len(xyz), "n_clashes": n_total, "n_clashes_interchain": n_inter,
            "n_clashing_atoms": len(per_atom), "max_overlap": round(max_overlap, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlap-cutoff", type=float, default=0.6)
    ap.add_argument("--hbond-allowance", type=float, default=0.4)
    ap.add_argument("--bond-separation", type=int, default=4)
    a = ap.parse_args()

    base = Path(a.results_root) / a.pair
    rows = []
    for i in range(a.num_samples):
        for state, path in (("raw", base / "raw" / f"sample_{i}.cif"),
                            ("minimized", base / "minimized" / f"sample_{i}.pdb")):
            if not path.exists():
                print(f"[count_clashes] missing {path} — skipped")
                continue
            res = count_clashes(path, a.overlap_cutoff, a.hbond_allowance, a.bond_separation)
            rows.append({"pair": a.pair, "sample": i, "state": state, **res})
            print(f"[count_clashes] {a.pair} sample_{i} {state}: "
                  f"{res['n_clashes']} clashes ({res['n_clashes_interchain']} inter-chain)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pair", "sample", "state", "n_atoms", "n_clashes", "n_clashes_interchain",
              "n_clashing_atoms", "max_overlap"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
