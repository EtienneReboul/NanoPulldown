#!/usr/bin/env python3
"""
tools/pinc/pinc.py — faithful Python port of badonyi/Pinc's Pinc.R
=================================================================
Pinc ("probabilistic AlphaFold interaction score", Badonyi 2026,
Protein Science / bioRxiv 2026.03.02.708997). NOT a trained model — a
closed-form geometric calculation:

  * per residue: centre of mass (element masses S/P/O/N, else C; H excluded)
  * CONTACT_RADIUS = 12 A (fixed, not configurable — part of the definition)
  * contact probability p(i->j) = (intersection volume of a radius-12 sphere
    at residue i and a radius-PAE(i,j) "uncertainty sphere" at residue j,
    given their COM distance) / volume of the uncertainty sphere
  * Pinc(chainA, chainB) = mean of the symmetrised contact probability over
    residue pairs (i in A, j in B) with COM distance < 12 A

Public API:
    pinc_from_arrays(pae, coms, chain_ids) -> dict[(cA,cB)] = {"Pinc","Pinc1","Pinc2"}
      pae       (N,N) float   predicted aligned error, token/residue order
      coms      (N,3) float   residue centre-of-mass coordinates
      chain_ids (N,) str-like chain id per residue

Standalone CLI mirrors Pinc.R:  python pinc.py <pae.json> <model.cif>
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

CONTACT_RADIUS = 12.0
_ELEMENT_MASS = {"S": 32.0650, "P": 30.9738, "O": 15.9994, "N": 14.0067}
_DEFAULT_MASS = 12.0107


def _vol_sphere(r):
    return (4.0 / 3.0) * np.pi * np.power(r, 3)


def _vol_intersect(ru, d, rc=CONTACT_RADIUS):
    """Volume of intersection of a fixed radius-rc sphere and a radius-ru
    sphere whose centres are distance d apart. Vectorised over ru, d."""
    ru = np.asarray(ru, dtype=float)
    d = np.asarray(d, dtype=float)
    vol = np.zeros(np.broadcast(ru, d).shape, dtype=float)
    ok = np.isfinite(ru) & np.isfinite(d) & (ru > 0)
    if not np.any(ok):
        return vol
    ruo, dd = np.broadcast_arrays(ru, d)
    ruo, dd = ruo.copy(), dd.copy()
    disjoint = dd >= (rc + ruo)
    contain = dd <= np.abs(rc - ruo)
    lens = ~(disjoint | contain) & ok
    c = contain & ok
    vol[c] = _vol_sphere(np.minimum(rc, ruo[c]))
    dl, rl = dd[lens], ruo[lens]
    vol[lens] = np.maximum(
        0.0,
        np.pi * (rc + rl - dl) ** 2
        * (dl ** 2 + 2 * dl * (rl + rc) - 3 * (rl - rc) ** 2)
        / (12 * dl),
    )
    return vol


def pae_to_contact_prob(pae: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Asymmetric contact-probability matrix. cont[i, j] ~ p(j -> i) as in
    Pinc.R (rows carry the uncertainty sphere). diag = 1."""
    n = pae.shape[0]
    p = _vol_intersect(pae, dist) / _vol_sphere(pae)
    p[~np.isfinite(p)] = 0.0
    p = np.clip(p, 0.0, 1.0)
    cont_asym = p.T.copy()
    np.fill_diagonal(cont_asym, 1.0)
    return cont_asym


def pinc_from_arrays(pae: np.ndarray, coms: np.ndarray, chain_ids) -> dict:
    pae = np.asarray(pae, dtype=float)
    coms = np.asarray(coms, dtype=float)
    chain_ids = np.asarray([str(c) for c in chain_ids])
    diff = coms[:, None, :] - coms[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    cont = pae_to_contact_prob(pae, dist)

    out = {}
    uchains = list(dict.fromkeys(chain_ids.tolist()))
    for ca, cb in itertools.combinations(uchains, 2):
        i = np.where(chain_ids == ca)[0]
        j = np.where(chain_ids == cb)[0]
        sub = dist[np.ix_(i, j)]
        keep = np.isfinite(sub) & (sub < CONTACT_RADIUS)
        if not keep.any():
            out[(ca, cb)] = {"Pinc": 0.0, "Pinc1": 0.0, "Pinc2": 0.0}
            continue
        pinc1 = float(cont[np.ix_(j, i)].T[keep].mean())
        pinc2 = float(cont[np.ix_(i, j)][keep].mean())
        out[(ca, cb)] = {"Pinc1": pinc1, "Pinc2": pinc2,
                         "Pinc": 0.5 * (pinc1 + pinc2)}
    return out


# ── minimal standalone CLI (COM from a CIF, PAE from JSON) ─────────────────

def _coms_from_cif(cif_path: str):
    import gemmi
    st = gemmi.read_structure(cif_path)
    m = st[0]
    chain_ids, coms = [], []
    for ch in m:
        for res in ch:
            mass, acc = 0.0, np.zeros(3)
            for at in res:
                if at.element.name == "H":
                    continue
                w = _ELEMENT_MASS.get(at.element.name.upper(), _DEFAULT_MASS)
                acc += w * np.array([at.pos.x, at.pos.y, at.pos.z])
                mass += w
            if mass:
                coms.append(acc / mass)
                chain_ids.append(ch.name)
    return np.array(coms), chain_ids


def main(argv):
    if len(argv) < 3:
        print("usage: python pinc.py <pae.json> <model.cif>")
        return 1
    j = json.loads(Path(argv[1]).read_text())
    pae = np.asarray(j.get("pae") or j.get("predicted_aligned_error"), dtype=float)
    coms, chain_ids = _coms_from_cif(argv[2])
    res = pinc_from_arrays(pae, coms, chain_ids)
    print("chain1,chain2,Pinc1,Pinc2,Pinc")
    for (a, b), v in sorted(res.items(), key=lambda kv: -kv[1]["Pinc"]):
        print(f"{a},{b},{v['Pinc1']:.4f},{v['Pinc2']:.4f},{v['Pinc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
