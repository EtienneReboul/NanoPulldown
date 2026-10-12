#!/usr/bin/env python3
"""
scripts/collect_results_table.py — Stage 2g
==============================================
Joins each pair's raw/fold.parquet (ptm/iptm/mean_plddt per sample)
with its interface_metrics.parquet (ipSAE/iLIS/Pinc/pDockQ/pair_iptm per
bait-prey chain pair) into one flat, sortable results table across every
screened pair -- AlphaPulldown's "Results table with confidence scores and
interaction metrics" (its README's stated output), built from
ab_initio_pipeline's own rescoring stack instead of AlphaJudge.

PLIP contacts (results/<pair>/plip_contacts.parquet) are counted per row's
sample and chain pair: `plip_contacts` = all bait<->prey contacts, and one
`plip_<type>` column per PLIP interaction type (PLIP_TYPES). A pair whose
PLIP table exists but has no contact for a row gets 0, not NaN -- "no
contacts" is a real value when ranking; NaN means PLIP never ran.

Each row also carries the pair's display names (bait / prey short labels,
prey protein name and locus, both accessions -- lib_pipeline.identity_columns)
and, from the run's configs/inputs_<run>.parquet, the input sources that
picked the prey plus any IP-MS columns they carry (<source>_<column>).

Output: reports/folding/results_table.csv (fed to the datavzrd view in
report/datavzrd/nanopulldown.datavzrd.yaml)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fold_io import FOLD_FILE, read_fold_scalars  # noqa: E402
from lib_pipeline import identity_columns, load_pairs, load_proteins  # noqa: E402

# PLIP interaction_type -> results-table column (all eight PLIP types, so a
# type absent from a run still reads 0)
PLIP_TYPES = {
    "hydrogen_bonds": "plip_hbond", "hydrophobic_interactions": "plip_hydrophobic",
    "salt_bridges": "plip_salt_bridge", "water_bridges": "plip_water_bridge",
    "pi-stacking": "plip_pi_stacking", "pi-cation_interactions": "plip_pi_cation",
    "halogen_bonds": "plip_halogen", "metal_complexes": "plip_metal",
}
PLIP_COLS = ["plip_contacts", *PLIP_TYPES.values()]


def plip_counts(path: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """PLIP_COLS for each (sample_index, chain_i, chain_j) of `rows` (one pair)."""
    out = rows[["sample_index", "chain_i", "chain_j"]].copy()
    if not path.exists():
        return out.assign(**{c: float("nan") for c in PLIP_COLS})
    pl = pd.read_parquet(path, columns=["sample", "reschain", "reschain_lig", "interaction_type"])
    # (sample, unordered chain pair) -> Counter(interaction_type); A-B and B-A count together
    cnt: dict[tuple, Counter] = defaultdict(Counter)
    for smp, c1, c2, t in zip(pl["sample"], pl["reschain"].astype(str),
                              pl["reschain_lig"].astype(str), pl["interaction_type"]):
        cnt[(int(smp), tuple(sorted((c1, c2))))][t] += 1
    vals = []
    for smp, ci, cj in zip(out["sample_index"], out["chain_i"], out["chain_j"]):
        if pd.isna(ci) or pd.isna(cj):                 # no interface_metrics row
            vals.append({c: float("nan") for c in PLIP_COLS})
            continue
        c = cnt.get((int(smp), tuple(sorted((str(ci), str(cj))))), Counter())
        d = {col: int(c.get(t, 0)) for t, col in PLIP_TYPES.items()}
        d["plip_contacts"] = sum(c.values())
        vals.append(d)
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(vals, columns=PLIP_COLS)], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--inputs", default=None, help="configs/inputs_<run>.parquet (optional)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for pair in a.pairs:
        mm_path = Path(a.results_root) / pair / "raw" / FOLD_FILE
        im_path = Path(a.results_root) / pair / "interface_metrics.parquet"
        if not mm_path.exists():
            continue
        mm = read_fold_scalars(mm_path)[["pair", "sample_index", "ptm", "iptm", "mean_plddt"]]
        if im_path.exists():
            im = pd.read_parquet(im_path)
            joined = im.merge(mm, on=["pair", "sample_index"], how="left")
        else:
            joined = mm.assign(chain_i=None, chain_j=None, lis=None, clis=None,
                               ilis=None, pinc=None, ipsae=None, pdockq=None,
                               pdockq2=None, pair_iptm=None, note="no interface_metrics")
        pc = plip_counts(Path(a.results_root) / pair / "plip_contacts.parquet", joined)
        joined = joined.reset_index(drop=True)
        joined[PLIP_COLS] = pc[PLIP_COLS].to_numpy()
        rows.append(joined)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.concat(rows, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["pair", "sample_index", "chain_i", "chain_j", "ptm", "iptm",
                                    "mean_plddt", "lis", "clis", "ilis", "pinc", "ipsae",
                                    "pdockq", "pdockq2", "pair_iptm", "note", *PLIP_COLS])

    specs, proteins = load_pairs(), load_proteins()
    ident = pd.DataFrame([{"pair": p, **identity_columns(specs[p], proteins)}
                          for p in a.pairs if p in specs])
    if not ident.empty:
        df = ident.merge(df, on="pair", how="right")
    if a.inputs and Path(a.inputs).exists() and not df.empty:
        inp = pd.read_parquet(a.inputs)
        inp = inp[inp["role"] == "prey"].drop(columns="role")
        inp["sources"] = inp["sources"].map(lambda v: ", ".join(v) if v is not None else "")
        inp = inp.rename(columns={"accession": "prey_uniprot", "sources": "prey_sources"})
        df = df.merge(inp, on="prey_uniprot", how="left")
    df.to_csv(out, index=False)
    print(f"[collect_results_table] {len(df)} row(s) across {len(a.pairs)} pair(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
