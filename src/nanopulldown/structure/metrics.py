#!/usr/bin/env python3
"""
nanopulldown.structure.metrics — Stage 2c
==================================================
PAE-based interface metrics for every ESMFold2-Fast diffusion sample of one
bait/prey pair, one row per (sample_index, chain_i, chain_j):

  lis, clis, ilis   Local Interaction Score family (Kim et al 2024; iLIS =
                    sqrt(LIS*cLIS), Kim et al 2026). Same pure-Python
                    implementation as ab_initio_pipeline's
                    compute_interface_metrics.py -- PAE + Cb contacts, backend-
                    agnostic.
  pinc              Badonyi 2026 probabilistic interaction score -- via the
                    vendored port tools/pinc/pinc.py (unchanged).
  ipsae, pdockq,    Dunbrack ipSAE v4 + pDockQ + pDockQ2 -- via the vendored
  pdockq2           reference tools/ipsae/ipsae.py (subprocess, unchanged).
                    Left NaN with a `note` if that call can't run for a sample.
  pair_iptm         ESMFold2's own per-chain-pair ipTM (pair_chains_iptm),
                    read straight from fold.parquet -- an
                    additional bait-prey confidence score AlphaPulldown-style
                    ipTM-based ranking mirrors.

Adapted from ab_initio_pipeline: the PAE + per-token chain ids come straight
from ESMFold2's own output (results/<pair>/raw/fold.parquet, see
nanopulldown.core.fold_table) instead of being reconstructed from a multi-backend
JSON schema -- no per-backend normalisation needed.

Output: results/<pair>/interface_metrics.parquet
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

from nanopulldown.core.fold_table import read_fold
from nanopulldown.core.io import write_parquet_with_metadata
from nanopulldown.core.metric_refs import column_references
from nanopulldown.structure.vendor.pinc import _DEFAULT_MASS, _ELEMENT_MASS, pinc_from_arrays

IPSAE = Path(__file__).resolve().parent / "vendor" / "ipsae.py"

TABLE_DESCRIPTION = (
    "PAE-based interface metrics of one nanopulldown pair: one row per sample x "
    "bait-prey chain pair. Each score column's `references` field metadata cites "
    "its publication (nanopulldown.core.metric_refs). Written by "
    "nanopulldown.structure.metrics, joined with fold.parquet's pTM / ipTM / pLDDT "
    "by `table results`.")
COLUMN_DESCRIPTIONS = {
    "pair": "Pair id (<bait>__<prey>).",
    "seed": "Random seed of the run the sample comes from.",
    "sample_index": "Sample index within the pair (sample_<i>.cif.gz in raw/).",
    "cif_path": "The sample's gzipped mmCIF, relative to the results root.",
    "chain_i": "First chain of the chain pair (sorted).",
    "chain_j": "Second chain of the chain pair (sorted).",
    "lis": "Local Interaction Score: mean (cutoff - PAE)/cutoff over inter-chain pairs with PAE <= cutoff, 0-1.",
    "clis": "Contact LIS: LIS over the pairs also within the Cb-Cb contact cutoff, 0-1.",
    "ilis": "Integrated LIS, sqrt(lis * clis), 0-1.",
    "pinc": "Pinc: mean PAE-derived contact probability over inter-chain residue pairs within 12 A, 0-1.",
    "ipsae": "ipSAE (ipsae.py v4): the highest of the chain pair's ipsae.py rows, 0-1.",
    "pdockq": "pDockQ, from the same ipsae.py row as ipsae, 0-1.",
    "pdockq2": "pDockQ2, from the same ipsae.py row as ipsae, 0-1.",
    "pair_iptm": "The fold engine's ipTM of this chain pair (fold.parquet pair_chains_iptm), 0-1.",
    "note": "Why ipSAE / pDockQ / pDockQ2 are missing, empty otherwise.",
}


# ── structure helpers (unchanged from ab_initio_pipeline) ─────────────────

def residue_geometry(cif_path: Path):
    """Per residue: chain id, Cb coord (Ca fallback), and centre of mass."""
    st = gemmi.read_structure(str(cif_path))
    m = st[0]
    chains, cb, com = [], [], []
    for ch in m:
        for res in ch:
            atoms = {a.name: a for a in res}
            piv = atoms.get("CB") or atoms.get("CA") or next(iter(atoms.values()), None)
            if piv is None:
                continue
            cb.append([piv.pos.x, piv.pos.y, piv.pos.z])
            mass, acc = 0.0, np.zeros(3)
            for a in res:
                if a.element.name == "H":
                    continue
                w = _ELEMENT_MASS.get(a.element.name.upper(), _DEFAULT_MASS)
                acc += w * np.array([a.pos.x, a.pos.y, a.pos.z])
                mass += w
            com.append(acc / mass if mass else [piv.pos.x, piv.pos.y, piv.pos.z])
            chains.append(ch.name)
    return np.array(chains), np.array(cb, float), np.array(com, float)


# ── LIS / cLIS / iLIS (unchanged) ───────────────────────────────────────

def lis_family(pae, chains, cb, pae_cut, dist_cut):
    out = {}
    diff = cb[:, None, :] - cb[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    uch = list(dict.fromkeys(chains.tolist()))
    for ca, cbid in combinations(uch, 2):
        i = np.where(chains == ca)[0]
        j = np.where(chains == cbid)[0]
        if len(i) == 0 or len(j) == 0:
            continue
        block = np.concatenate([pae[np.ix_(i, j)].ravel(), pae[np.ix_(j, i)].ravel()])
        dblock = np.concatenate([dist[np.ix_(i, j)].ravel(), dist[np.ix_(j, i)].ravel()])
        lia = block <= pae_cut
        lis = float(((pae_cut - block[lia]) / pae_cut).mean()) if lia.any() else 0.0
        clia = lia & (dblock <= dist_cut)
        clis = float(((pae_cut - block[clia]) / pae_cut).mean()) if clia.any() else 0.0
        out[(ca, cbid)] = {"lis": lis, "clis": clis, "ilis": float(np.sqrt(lis * clis))}
    return out


# ── ipSAE (reference script, subprocess -- unchanged) ──────────────────

def _atom_plddts(cif_path: Path, token_plddt):
    """Expand per-residue pLDDT (0-100) to a per-atom list indexed by atom serial - 1,
    using the same residue order as residue_geometry()."""
    st = gemmi.read_structure(str(cif_path))
    per_atom, ri = {}, 0
    for ch in st[0]:
        for res in ch:
            if not len(res):
                continue
            for a in res:
                per_atom[a.serial - 1] = float(token_plddt[min(ri, len(token_plddt) - 1)])
            ri += 1
    out = np.zeros(max(per_atom) + 1)
    for k, v in per_atom.items():
        out[k] = v
    return out.tolist()


def run_ipsae(pae, token_chain_ids, token_res_ids, cif_path: Path, pae_cut, dist_cut,
              plddt=None):
    """Returns {(cA,cB): {"ipsae","pdockq","pdockq2"}} or {} on any failure."""
    if not IPSAE.exists():
        return {}, "ipsae.py not vendored"
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cif_link = td / "model.cif"
            raw = cif_path.read_bytes()
            cif_link.write_bytes(gzip.decompress(raw) if cif_path.suffix == ".gz" else raw)
            js = {"pae": np.asarray(pae, float).tolist(),
                  "token_chain_ids": [str(x) for x in token_chain_ids],
                  "token_res_ids": [int(x) for x in token_res_ids]}
            # ipsae.py picks AF3 mode for .cif input and takes pDockQ / pDockQ2 pLDDT
            # from "atom_plddts" (indexed by CIF atom serial). Without it, it silently
            # falls back to zeros and both metrics collapse to a constant floor.
            if plddt is not None:
                js["atom_plddts"] = _atom_plddts(cif_path, plddt)
            jpath = td / "model_full_data.json"
            jpath.write_text(json.dumps(js))
            subprocess.run([sys.executable, str(IPSAE), str(jpath), str(cif_link),
                            str(int(pae_cut)), str(int(dist_cut))],
                           cwd=td, capture_output=True, text=True, timeout=300, check=True)
            ps = int(pae_cut); ds = int(dist_cut)
            txt = td / f"model_{ps:02d}_{ds:02d}.txt"
            if not txt.exists():
                cands = list(td.glob("model_*_*.txt"))
                txt = cands[0] if cands else None
            if not txt or not txt.exists():
                return {}, "ipsae.py produced no .txt"
            return _parse_ipsae_txt(txt.read_text()), ""
    except subprocess.CalledProcessError as e:                 # noqa: BLE001
        return {}, f"ipsae.py exit {e.returncode}: {(e.stderr or '')[:120]}"
    except Exception as e:                                     # noqa: BLE001
        return {}, f"ipsae.py: {type(e).__name__}: {str(e)[:120]}"


def _parse_ipsae_txt(txt: str) -> dict:
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    hdr_i = next((k for k, ln in enumerate(lines)
                  if "Chn1" in ln and "ipSAE" in ln), None)
    if hdr_i is None:
        return {}
    hdr = lines[hdr_i].split()
    col = {name: hdr.index(name) for name in hdr}

    def g(parts, name):
        try:
            return float(parts[col[name]])
        except (KeyError, ValueError, IndexError):
            return float("nan")

    out: dict = {}
    for ln in lines[hdr_i + 1:]:
        p = ln.split()
        if len(p) < len(hdr):
            continue
        c1, c2 = p[col["Chn1"]], p[col["Chn2"]]
        key = tuple(sorted((c1, c2)))
        rec = {"ipsae": g(p, "ipSAE"),
               "pdockq": g(p, "pDockQ") if "pDockQ" in col else float("nan"),
               "pdockq2": g(p, "pDockQ2") if "pDockQ2" in col else float("nan")}
        if key not in out or (rec["ipsae"] == rec["ipsae"] and rec["ipsae"] > out[key]["ipsae"]):
            out[key] = rec
    return out


# ── pair_chains_iptm (ESMFold2's own per-chain-pair ipTM) ──────────────

def pair_iptm_lookup(mat, chain_order: list[str]) -> dict:
    """ESMFold2's pair_chains_iptm is a square matrix over chains in
    chain_order (its integer chain order). Returns {} (all pairs NaN
    downstream) if the shape doesn't match."""
    if mat is None or not chain_order:
        return {}
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != len(chain_order) or mat.shape[1] != len(chain_order):
        return {}
    out = {}
    for i, ca in enumerate(chain_order):
        for j, cb in enumerate(chain_order):
            if i < j:
                out[tuple(sorted((ca, cb)))] = float(mat[i, j])
    return out


# ── main ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--fold", required=True, help="results/<pair>/raw/fold.parquet")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--enabled", default="ipsae,ilis,pinc")
    ap.add_argument("--pae-cutoff", type=float, default=12.0)
    ap.add_argument("--contact-cutoff-cb", type=float, default=8.0)
    ap.add_argument("--pairs", default="bait_prey")
    ap.add_argument("--bait-chains", default="")
    ap.add_argument("--prey-chains", default="")
    a = ap.parse_args()

    enabled = set(a.enabled.split(","))
    want_pairs = None
    if a.pairs == "bait_prey" and a.bait_chains and a.prey_chains:
        B = set(a.bait_chains.split(","))
        P = set(a.prey_chains.split(","))
        want_pairs = {tuple(sorted(p)) for p in
                      [(x, y) for x in B for y in P if x != y]}

    results_root = Path(a.results_root)
    rows = []
    for r in read_fold(Path(a.fold)):
        pae = r["pae"]
        if pae is None:
            continue
        cif_path = results_root / r["structure_path"]
        if not cif_path.exists():
            continue
        try:
            chains, cb, com = residue_geometry(cif_path)
        except Exception:                                      # noqa: BLE001
            continue

        # ESMFold2's own per-token chain letters; fall back to gemmi's chain
        # names if they don't line up (same length guard as
        # ab_initio_pipeline's own defensive trim).
        chain_ids = r["token_chain"] if len(r["token_chain"]) == pae.shape[0] else chains
        n = min(len(chain_ids), pae.shape[0], len(chains))
        chain_ids, pae, chains, cb = chain_ids[:n], pae[:n, :n], chains[:n], cb[:n]
        token_res_ids = r["residue_index"][:n]
        plddt = r["plddt"][:n] * 100.0                         # stored 0-1; ipsae.py wants 0-100

        lis = lis_family(pae, chain_ids, cb, a.pae_cutoff, a.contact_cutoff_cb) if "ilis" in enabled else {}
        pin = pinc_from_arrays(pae, com[:n], chain_ids) if "pinc" in enabled else {}
        ips, note = ({}, "")
        if "ipsae" in enabled:
            ips, note = run_ipsae(pae, chain_ids, token_res_ids, cif_path,
                                   a.pae_cutoff, 15.0, plddt)
        pit = pair_iptm_lookup(r["pair_chains_iptm"], r["chain_order"])

        pair_set = set(lis) | set(pin) | set(ips) | set(pit)
        for pr in sorted(pair_set):
            if want_pairs is not None and pr not in want_pairs:
                continue
            rec = {
                "pair": a.pair, "seed": r["seed"], "sample_index": r["sample_index"],
                "cif_path": r["structure_path"], "chain_i": pr[0], "chain_j": pr[1],
                "lis": lis.get(pr, {}).get("lis", np.nan),
                "clis": lis.get(pr, {}).get("clis", np.nan),
                "ilis": lis.get(pr, {}).get("ilis", np.nan),
                "pinc": pin.get(pr, {}).get("Pinc", np.nan),
                "ipsae": ips.get(pr, {}).get("ipsae", np.nan),
                "pdockq": ips.get(pr, {}).get("pdockq", np.nan),
                "pdockq2": ips.get(pr, {}).get("pdockq2", np.nan),
                "pair_iptm": pit.get(pr, np.nan),
                "note": note,
            }
            rows.append(rec)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=list(COLUMN_DESCRIPTIONS))
    write_parquet_with_metadata(df, out, table_description=TABLE_DESCRIPTION,
                                column_descriptions=COLUMN_DESCRIPTIONS,
                                column_references=column_references(df.columns))
    print(f"[interface_metrics] {a.pair}: {len(df)} (sample,pair) rows -> {out}")
    if not df.empty:
        print(df.groupby(["chain_i", "chain_j"])[["ilis", "pinc", "ipsae", "pair_iptm"]]
              .mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
