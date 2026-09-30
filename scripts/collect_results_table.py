#!/usr/bin/env python3
"""
scripts/collect_results_table.py — Stage 2g
==============================================
Joins each pair's model_metadata.parquet (ptm/iptm/mean_plddt per sample)
with its interface_metrics.parquet (ipSAE/iLIS/Pinc/pDockQ/pair_iptm per
bait-prey chain pair) into one flat, sortable results table across every
screened pair -- AlphaPulldown's "Results table with confidence scores and
interaction metrics" (its README's stated output), built from
ab_initio_pipeline's own rescoring stack instead of AlphaJudge.

Output: reports/folding/results_table.csv (fed to the datavzrd view in
report/datavzrd/nanopulldown.datavzrd.yaml)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--metadata-root", default="results/metadata")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for pair in a.pairs:
        mm_path = Path(a.metadata_root) / pair / "model_metadata.parquet"
        im_path = Path(a.results_root) / pair / "interface_metrics.parquet"
        if not mm_path.exists():
            continue
        mm = pd.read_parquet(mm_path)[["pair", "sample_index", "ptm", "iptm", "mean_plddt"]]
        if im_path.exists():
            im = pd.read_parquet(im_path)
            joined = im.merge(mm, on=["pair", "sample_index"], how="left")
        else:
            joined = mm.assign(chain_i=None, chain_j=None, lis=None, clis=None,
                               ilis=None, pinc=None, ipsae=None, pdockq=None,
                               pdockq2=None, pair_iptm=None, note="no interface_metrics")
        rows.append(joined)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.concat(rows, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["pair", "sample_index", "chain_i", "chain_j", "ptm", "iptm",
                                    "mean_plddt", "lis", "clis", "ilis", "pinc", "ipsae",
                                    "pdockq", "pdockq2", "pair_iptm", "note"])
    df.to_csv(out, index=False)
    print(f"[collect_results_table] {len(df)} row(s) across {len(a.pairs)} pair(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
