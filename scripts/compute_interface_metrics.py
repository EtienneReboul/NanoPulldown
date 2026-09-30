#!/usr/bin/env python3
"""
scripts/compute_interface_metrics.py — Stage 2c
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
                    read straight from each sample's extra_json -- an
                    additional bait-prey confidence score AlphaPulldown-style
                    ipTM-based ranking mirrors.

Adapted from ab_initio_pipeline: the PAE + per-token chain ids come straight
from ESMFold2's own output (arrays.h5's `_topology/<pair>/chain_id` +
chain_lookup attribute) instead of being reconstructed from a multi-backend
JSON schema -- no per-backend normalisation needed.

Output: results/<pair>/interface_metrics.parquet
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import gemmi
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "pinc"))
from pinc import _ELEMENT_MASS, _DEFAULT_MASS, pinc_from_arrays  # noqa: E402

IPSAE = Path(__file__).resolve().parent.parent / "tools" / "ipsae" / "ipsae.py"


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

def run_ipsae(pae, token_chain_ids, token_res_ids, cif_path: Path, pae_cut, dist_cut):
    """Returns {(cA,cB): {"ipsae","pdockq","pdockq2"}} or {} on any failure."""
    if not IPSAE.exists():
        return {}, "ipsae.py not vendored"
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cif_link = td / "model.cif"
            cif_link.write_bytes(cif_path.read_bytes())
            js = {"pae": np.asarray(pae, float).tolist(),
                  "token_chain_ids": [str(x) for x in token_chain_ids],
                  "token_res_ids": [int(x) for x in token_res_ids]}
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

def pair_iptm_lookup(extra_json: dict, chain_ids_unique: list[str]) -> dict:
    """Best-effort: ESMFold2's pair_chains_iptm is assumed to be a square
    matrix over chains in the same order as chain_lookup's sorted integer
    keys. Returns {} (all pairs NaN downstream) if the shape doesn't match
    what's expected -- this mapping is flagged for verification against a
    live run in the project plan."""
    mat = extra_json.get("pair_chains_iptm")
    lookup = extra_json.get("chain_lookup") or {}
    if mat is None or not lookup:
        return {}
    mat = np.asarray(mat, dtype=float)
    ordered_ids = [lookup[str(k)] for k in sorted((int(x) for x in lookup), key=int)]
    if mat.ndim != 2 or mat.shape[0] != len(ordered_ids) or mat.shape[1] != len(ordered_ids):
        return {}
    out = {}
    for i, ca in enumerate(ordered_ids):
        for j, cb in enumerate(ordered_ids):
            if i < j:
                out[tuple(sorted((ca, cb)))] = float(mat[i, j])
    return out


# ── main ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--metadata-root", default="results/metadata")
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

    meta = pd.read_parquet(Path(a.metadata_root) / a.pair / "model_metadata.parquet")
    h5path = Path(a.metadata_root) / a.pair / "arrays.h5"
    results_root = Path(a.results_root)
    if not h5path.exists():
        raise FileNotFoundError(f"{h5path} — run stage 2 metadata compression first")

    chain_lookup = {}
    chain_id_int = None
    with h5py.File(h5path, "r") as h5:
        topo_key = f"_topology/{a.pair}/chain_id"
        if topo_key in h5:
            chain_id_int = np.asarray(h5[topo_key][:], dtype=int)
            chain_lookup = json.loads(h5[topo_key].attrs.get("chain_lookup", "{}"))

    rows = []
    with h5py.File(h5path, "r") as h5:
        for _, r in meta.iterrows():
            if not r.get("has_array"):
                continue
            key = r["array_key"]
            if f"{key}/pae" not in h5:
                continue
            pae = np.asarray(h5[f"{key}/pae"][:], dtype=np.float32)
            extra = json.loads(r["extra_json"]) if r.get("extra_json") else {}

            cif_path = results_root / r["cif_path"]
            if not cif_path.exists():
                continue
            try:
                chains, cb, com = residue_geometry(cif_path)
            except Exception:                                  # noqa: BLE001
                continue

            # map integer chain_id topology onto the letter ids gemmi read
            # from the CIF; fall back to gemmi's own chain names if the
            # topology array doesn't line up (same length guard as
            # ab_initio_pipeline's own defensive trim).
            if chain_id_int is not None and len(chain_id_int) == pae.shape[0]:
                chain_ids = np.array([chain_lookup.get(str(c), str(c)) for c in chain_id_int])
            else:
                chain_ids = chains
            n = min(len(chain_ids), pae.shape[0], len(chains))
            chain_ids, pae, chains, cb = chain_ids[:n], pae[:n, :n], chains[:n], cb[:n]

            res_idx = extra.get("residue_index")
            token_res_ids = (np.asarray(res_idx[:n]) if res_idx and len(res_idx) >= n
                              else np.arange(1, n + 1))

            lis = lis_family(pae, chain_ids, cb, a.pae_cutoff, a.contact_cutoff_cb) if "ilis" in enabled else {}
            pin = pinc_from_arrays(pae, com[:n], chain_ids) if "pinc" in enabled else {}
            ips, note = ({}, "")
            if "ipsae" in enabled:
                ips, note = run_ipsae(pae, chain_ids, token_res_ids, cif_path,
                                       a.pae_cutoff, 15.0)
            pit = pair_iptm_lookup(extra, list(dict.fromkeys(chain_ids.tolist())))

            pair_set = set(lis) | set(pin) | set(ips) | set(pit)
            for pr in sorted(pair_set):
                if want_pairs is not None and pr not in want_pairs:
                    continue
                rec = {
                    "pair": a.pair, "seed": r["seed"], "sample_index": r["sample_index"],
                    "cif_path": r["cif_path"], "chain_i": pr[0], "chain_j": pr[1],
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
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"[interface_metrics] {a.pair}: {len(df)} (sample,pair) rows -> {out}")
    if not df.empty:
        print(df.groupby(["chain_i", "chain_j"])[["ilis", "pinc", "ipsae", "pair_iptm"]]
              .mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
