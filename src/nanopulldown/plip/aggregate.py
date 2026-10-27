#!/usr/bin/env python3
"""
nanopulldown.plip.aggregate — Stage 2e
=================================================
Merges one pair's per-sample PLIP contact tables (pliparser's `plip2csv`
summary.csv, one per results/<pair>/minimized/sample_<i>.pdb.gz) into a
single results/<pair>/plip_contacts.parquet, tagging each row with the pair
and the sample index parsed from its file name (sample_<i>_summary.csv, or
the older sample_<i>_report/csv/summary.csv layout).

Adapted from ab_initio_pipeline's scripts/aggregate_summaries.py: nanopulldown
has one backend (ESMFold2) and no pose clustering, so there's just one axis
to tag rows with (sample index) instead of (replica, model).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from nanopulldown.core.io import write_parquet_with_metadata
from nanopulldown.core.metric_refs import references_text

SAMPLE_RE = re.compile(r"sample_(\d+)(?:_report/csv/summary|_summary)\.csv$")

PLIP_TABLE_DESCRIPTION = (
    "PLIP bait<->prey contacts of one nanopulldown pair, one row per contact, "
    "over all minimized ESMFold2 diffusion samples (PLIP --chains [bait, prey]; "
    "pliparser plip2csv summary columns). The receptor side (res*) and the "
    "'ligand' side (*_lig) are both protein residues. Written by "
    "nanopulldown.plip.aggregate."
)
PLIP_COLUMNS = {
    "pair": ("string", "Bait__prey pair identifier"),
    "sample": ("int32", "Diffusion sample index the contact was found in"),
    "resnr": ("int32", "Residue number, receptor side"),
    "restype": ("string", "Residue name, receptor side"),
    "reschain": ("string", "Chain id, receptor side"),
    "resnr_lig": ("int32", "Residue number, partner side"),
    "restype_lig": ("string", "Residue name, partner side"),
    "reschain_lig": ("string", "Chain id, partner side"),
    "dist": ("float64", "Interaction distance reported by PLIP (A)"),
    "ligcoo": ("object", "Partner-side interaction coordinates [x, y, z] (A)"),
    "protcoo": ("object", "Receptor-side interaction coordinates [x, y, z] (A)"),
    "interaction_type": ("string", "PLIP interaction type, e.g. hydrogen_bonds, salt_bridges"),
}


def sample_index(path: Path) -> int:
    m = SAMPLE_RE.search(path.as_posix())
    if not m:
        raise ValueError(f"cannot parse sample index from '{path}' -- expected "
                         ".../sample_<i>_summary.csv")
    return int(m.group(1))


def _coords(v) -> list[float] | None:
    return None if pd.isna(v) else [float(x) for x in str(v).split(",")]


def read_summary(path: Path, pair: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    df = df.reindex(columns=[c for c in PLIP_COLUMNS if c not in ("pair", "sample")])
    df.insert(0, "sample", sample_index(path))
    df.insert(0, "pair", pair)
    for col in ("ligcoo", "protcoo"):
        df[col] = df[col].map(_coords)
    return df.astype({c: t for c, (t, _) in PLIP_COLUMNS.items() if t != "object"})


def write_contacts(df: pd.DataFrame, out: Path) -> None:
    write_parquet_with_metadata(
        df, out, table_description=PLIP_TABLE_DESCRIPTION,
        column_descriptions={c: d for c, (_, d) in PLIP_COLUMNS.items()},
        table_references=references_text("plip_contacts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="per-sample summary.csv paths")
    ap.add_argument("--pair", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = []
    for p in args.inputs:
        df = read_summary(Path(p), args.pair)
        if df.empty:
            print(f"[aggregate_plip] '{p}': no contacts", file=sys.stderr)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    write_contacts(merged, Path(args.out))
    print(f"[aggregate_plip] -> {args.out} ({len(merged)} contacts across {len(frames)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
