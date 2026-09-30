#!/usr/bin/env python3
"""
scripts/aggregate_plip_summaries.py — Stage 2e
=================================================
Merges one pair's per-sample PLIP summary.csv files (pliparser's `plip2csv`
output, one per results/<pair>/minimized/sample_<i>.pdb) into a single
results/<pair>/plip_summary.csv, tagging each row with the sample index
parsed from its report directory name (sample_<i>_report/csv/summary.csv).

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

SAMPLE_RE = re.compile(r"^sample_(\d+)_report$")


def sample_index(path: Path) -> int:
    """Recover the sample index from .../sample_<i>_report/csv/summary.csv."""
    report_dir = path.parent.parent.name
    m = SAMPLE_RE.match(report_dir)
    if not m:
        raise ValueError(
            f"cannot parse sample index from '{path}' -- expected "
            ".../sample_<i>_report/csv/summary.csv"
        )
    return int(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="per-sample summary.csv paths")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = []
    for p in args.inputs:
        path = Path(p)
        df = pd.read_csv(path)
        if df.empty:
            print(f"[aggregate_plip] '{path}': no contacts", file=sys.stderr)
        df.insert(0, "sample", sample_index(path))
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[aggregate_plip] -> {out_path} ({len(merged)} contacts across {len(frames)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
