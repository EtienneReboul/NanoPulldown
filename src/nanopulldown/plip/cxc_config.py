"""
nanopulldown.plip.cxc_config — Stage 2e
=======================================
Writes the JSON config `pliparser csv2cxc --config` consumes: the minimized
PDB (path relative to the .cxc's directory, so the results tree can be copied
off the cluster and the .cxc still opens; ChimeraX resolves `open` paths in a
.cxc against the script's own folder), bait chains
as one colored chain group and prey chains as the other, and the PLIP CSV
directory as the single source. Stdlib only (runs in the pliparser env,
between the plip_parse rule's two `pliparser` calls).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", type=Path, required=True)
    ap.add_argument("--cxc-dir", type=Path, default=None,
                    help="directory the .cxc will be written to; --pdb is made relative to it "
                         "(default: absolute --pdb)")
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bait-chains", required=True, help="comma-joined chain ids")
    ap.add_argument("--prey-chains", required=True, help="comma-joined chain ids")
    ap.add_argument("--bait-color", default="gray")
    ap.add_argument("--prey-color", default="cornflowerblue")
    ap.add_argument("--transparency", type=int, default=65)
    ap.add_argument("--model-id", type=int, default=1)
    a = ap.parse_args()

    pdb = str(a.pdb.resolve())
    if a.cxc_dir is not None:
        pdb = os.path.relpath(pdb, a.cxc_dir.resolve())
    config = {
        "pdb": pdb,
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
