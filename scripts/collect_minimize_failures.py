#!/usr/bin/env python3
"""
scripts/collect_minimize_failures.py — Stage 2g
===================================================
Concatenates every pair's minimize_failure_rate.csv (written per-pair by
scripts/plot_minimize_energy.py) into one table for the report's datavzrd
view.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    frames = [pd.read_csv(p) for p in a.tables if Path(p).exists()]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["pair", "num_samples", "num_succeeded", "num_failed", "failure_rate"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[collect_minimize_failures] {len(df)} pair(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
