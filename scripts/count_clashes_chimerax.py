#!/usr/bin/env python3
"""
scripts/count_clashes_chimerax.py — run INSIDE ChimeraX (nogui) to score the
raw / minimized samples with ChimeraX's own `find_clashes` (the `clashes`
command), as a reference for scripts/count_clashes.py.

  ChimeraX --nogui --exit --script "scripts/count_clashes_chimerax.py \
      --results-root results --num-samples 5 --out clashes_chimerax.csv [--pairs A B ...]"

Settings are ChimeraX's defaults (cutoff 0.6, H-bond allowance 0.4, bond
separation 4, intra-residue pairs NOT counted, per-atom donor/acceptor
types). Hydrogens are deleted first so the raw CIF (no H) and the minimized
PDB (PDBFixer H) are scored on the same heavy-atom footing.
Output columns match count_clashes.py (same pair/sample/state key).
"""
import argparse
import csv
import sys
from pathlib import Path

from chimerax.clashes.clashes import find_clashes
from chimerax.core.commands import run

ap = argparse.ArgumentParser()
ap.add_argument("--results-root", default="results")
ap.add_argument("--num-samples", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pairs", nargs="*", default=None, help="default: every results/*/ with clashes.csv")
ap.add_argument("--bond-separation", type=int, default=4)
ap.add_argument("--overlap-cutoff", type=float, default=0.6)
ap.add_argument("--hbond-allowance", type=float, default=0.4)
a = ap.parse_args(sys.argv[1:])

root = Path(a.results_root)
pairs = a.pairs or sorted(p.parent.name for p in root.glob("*/clashes.csv"))
FIELDS = ["pair", "sample", "state", "n_atoms", "n_clashes", "n_clashes_interchain",
          "n_clashing_atoms", "max_overlap"]
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    for pair in pairs:
        for i in range(a.num_samples):
            for state, path in (("raw", root / pair / "raw" / f"sample_{i}.cif"),
                                ("minimized", root / pair / "minimized" / f"sample_{i}.pdb")):
                if not path.exists():
                    continue
                m = run(session, f"open {path} log false")[0]
                run(session, "delete H", log=False)
                clashes = find_clashes(
                    session, m.atoms, attr_name="_cxclash", inter_model=False, intra_model=True,
                    intra_mol=True, intra_res=False, clash_threshold=a.overlap_cutoff,
                    hbond_allowance=a.hbond_allowance, bond_separation=a.bond_separation)
                seen, n_inter, mx = set(), 0, 0.0
                for atom, nbrs in clashes.items():
                    for other, ov in nbrs.items():
                        key = frozenset((atom, other))
                        if key in seen:
                            continue
                        seen.add(key)
                        n_inter += atom.residue.chain_id != other.residue.chain_id
                        mx = max(mx, ov)
                w.writerow({"pair": pair, "sample": i, "state": state, "n_atoms": m.num_atoms,
                            "n_clashes": len(seen), "n_clashes_interchain": n_inter,
                            "n_clashing_atoms": len(clashes), "max_overlap": round(mx, 3)})
                fh.flush()
                run(session, "close", log=False)
