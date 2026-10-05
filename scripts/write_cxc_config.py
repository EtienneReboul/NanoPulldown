#!/usr/bin/env python3
"""
scripts/write_cxc_config.py — Stage 2e
==========================================
Writes the JSON config `pliparser csv2cxc --config` consumes: the minimized
PDB (absolute path, so the .cxc opens from any working directory), bait chains
as one colored chain group and prey chains as the other, and the PLIP CSV
directory as the single source. Stdlib only (runs in the pliparser env).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", type=Path, required=True)
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bait-chains", required=True, help="comma-joined chain ids")
    ap.add_argument("--prey-chains", required=True, help="comma-joined chain ids")
    ap.add_argument("--bait-color", default="gray")
    ap.add_argument("--prey-color", default="cornflowerblue")
    ap.add_argument("--transparency", type=int, default=65)
    ap.add_argument("--model-id", type=int, default=1)
    a = ap.parse_args()

    config = {
        "pdb": str(a.pdb.resolve()),
        "model_id": a.model_id,
        "chains": [
            {"chain": a.bait_chains, "color": a.bait_color, "transparency": a.transparency},
            {"chain": a.prey_chains, "color": a.prey_color, "transparency": a.transparency},
        ],
        "sources": [
            {"name": "plip", "input": str(a.csv_dir.resolve()), "issmalmol": False},
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(config, indent=4) + "\n")
    print(f"[write_cxc_config] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
