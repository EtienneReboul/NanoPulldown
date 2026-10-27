#!/usr/bin/env python3
"""
nanopulldown.structure.pydock — Stage 2d (energy rescoring)
==========================================================
pyDock binding energy (Cheng, Blundell & Fernandez-Recio 2007) between the
bait chains and the prey chains of every minimized sample of one pair, for the
combined AF-confidence / pyDock score of Rodriguez-Lumbreras et al 2026
(Proteins 94:1827, Eq. 1 -- the z-scores themselves are taken run-wide in
`table results`, nanopulldown.report.tables).

The energy terms come from LightDock's C port of the pyDock scoring function
(lightdock.scoring.cpydock, AMBER94 charges / vdW, ASA-based desolvation via
freesasa), called directly rather than through LightDock's model adapter:

  pydock_elec     Coulombic electrostatics, distance-dependent dielectric 4r,
                  each atom pair capped at -1/+1 kcal/mol
  pydock_vdw      Lennard-Jones van der Waals, each atom pair capped at
                  +1 kcal/mol (the rigid-body clash guard)
  pydock_desolv   ASA-based desolvation
  pydock          elec + desolv + 0.1 * vdw   (default pyDock, docking poses)
  pydock_1vdw     elec + desolv + 1.0 * vdw   (pyDock-1VDW, the paper's term
                  for AF models: their interfaces have few clashes)

kcal/mol, lower = more favourable. Deviations from LightDock's adapter:
  * His protonation follows the hydrogens OpenMM placed (HID / HIE / HIP)
    instead of LightDock's blanket HID; Cys without HG is CYX (disulfide).
  * The AMBER94 table has no terminal residue types: N-terminal H2/H3 are
    dropped and the C-terminal OXT takes its residue's O parameters (its
    desolvation already maps OXT -> Asp OD1).
  * The hydrogen mask is passed as the uint32 "1 = hydrogen" array the C code
    reads; LightDock passes an int64 "1 = heavy atom" array, which shifts the
    heavy-atom contacts behind desolvation (~2% on its own 1AY7 test case).

Input: the OpenMM-minimized samples (results/<pair>/minimized/sample_<i>.pdb.gz),
which carry the hydrogens the AMBER charges need -- the raw ESMFold2 CIF has none.

Output: results/<pair>/pydock_energy.parquet, one row per sample.
"""
from __future__ import annotations

import argparse
import gzip
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

TABLE_DESCRIPTION = (
    "pyDock binding energy (LightDock's cpydock terms) between the bait chains "
    "and the prey chains of each minimized sample of one pair. kcal/mol, lower = "
    "more favourable. Read by `table results`, which adds the combined "
    "AF-confidence / pyDock-1VDW z-scores.")
COLUMN_DESCRIPTIONS = {
    "pair": "Pair id (<bait>__<prey>).",
    "sample_index": "Sample index (sample_<i>.pdb.gz in minimized/).",
    "pydock_elec": "Electrostatics, kcal/mol (dielectric 4r, per atom pair capped at +-1).",
    "pydock_vdw": "Van der Waals, kcal/mol (per atom pair capped at +1), unweighted.",
    "pydock_desolv": "ASA-based desolvation, kcal/mol.",
    "pydock": "Default pyDock energy: elec + desolv + 0.1 * vdw, kcal/mol.",
    "pydock_1vdw": "pyDock-1VDW energy: elec + desolv + 1.0 * vdw, kcal/mol.",
    "n_interface_atoms": "Bait-prey atom pairs within 3.9 A (LightDock's interface cutoff).",
    "note": "Why the energy is missing, empty otherwise.",
}
INTERFACE_CUTOFF = 3.9           # A, LightDock's DEFAULT_CONTACT_RESTRAINTS_CUTOFF
N_TERM_H = {"H2", "H3"}


def protonation_names(atoms) -> dict:
    """(chain, resnum, insertion) -> AMBER94 residue name from the hydrogens present."""
    names: dict[tuple, set] = {}
    resname = {}
    for a in atoms:
        key = (a.chain_id, a.residue_number, a.residue_insertion)
        names.setdefault(key, set()).add(a.name)
        resname[key] = a.residue_name
    out = {}
    for key, rn in resname.items():
        h = names[key]
        if rn == "HIS":
            rn = {(True, True): "HIP", (False, True): "HIE"}.get(("HD1" in h, "HE2" in h), "HID")
        elif rn == "CYS" and "HG" not in h:
            rn = "CYX"
        out[key] = rn
    return out


def side_model(atoms):
    """cpydock's per-molecule arrays for one side (bait or prey chains) of the complex."""
    import freesasa
    import lightdock.scoring.cpydock.data.amber as amber
    import lightdock.scoring.cpydock.data.solvation as solvation
    import lightdock.scoring.cpydock.data.vdw as vdw

    kept = []
    for a in atoms:
        atom_id = f"{a.residue_name}-{'O' if a.name == 'OXT' else a.name}"
        if atom_id not in amber.amber_types:
            if a.name in N_TERM_H:
                continue
            raise KeyError(f"no AMBER94 type for {a.chain_id}:{a.residue_number} {atom_id}")
        a.amber_type = amber.amber_types[atom_id]
        a.charge = amber.charges[atom_id]
        kept.append(a)

    des_energy, des_radii = solvation.get_solvation(SimpleNamespace(atoms=kept))
    heavy = np.array([not a.is_hydrogen() for a in kept])
    structure = freesasa.Structure()
    for a in (a for a in kept if not a.is_hydrogen()):
        structure.addAtom(a.name, a.residue_name, a.residue_number, a.chain_id, a.x, a.y, a.z)
    structure.setRadii([float(r) for r in des_radii[heavy]])
    res = freesasa.calc(structure)
    sasa = np.full(len(kept), -1.0)
    sasa[heavy] = [res.atomArea(i) for i in range(int(heavy.sum()))]

    return SimpleNamespace(
        coordinates=np.array([[a.x, a.y, a.z] for a in kept], dtype=np.float64),
        charges=np.array([a.charge for a in kept], dtype=np.float64),
        vdw_energy=np.array([vdw.vdw_energy[a.amber_type] for a in kept], dtype=np.float64),
        vdw_radii=np.array([vdw.vdw_radii[a.amber_type] for a in kept], dtype=np.float64),
        hydrogens=(~heavy).astype(np.uint32),
        sasa=sasa, des_energy=np.asarray(des_energy, dtype=np.float64),
    )


def pydock_energy(pdb_gz: Path, bait_chains: set[str], prey_chains: set[str]) -> dict:
    import freesasa
    import lightdock.scoring.cpydock.energy.c.cpydock as cpydock
    from lightdock.pdbutil.PDBIO import parse_complex_from_file

    freesasa.setVerbosity(freesasa.silent)
    # LightDock's parser reads a plain-text path
    with tempfile.NamedTemporaryFile("wb", suffix=".pdb") as tmp:
        tmp.write(gzip.decompress(pdb_gz.read_bytes()))
        tmp.flush()
        atoms, _, _ = parse_complex_from_file(tmp.name)

    resnames = protonation_names(atoms)
    for a in atoms:
        a.residue_name = resnames[(a.chain_id, a.residue_number, a.residue_insertion)]
    rec = side_model([a for a in atoms if a.chain_id in bait_chains])
    lig = side_model([a for a in atoms if a.chain_id in prey_chains])
    if not len(rec.charges) or not len(lig.charges):
        raise ValueError(f"no atoms for bait chains {sorted(bait_chains)} or prey chains {sorted(prey_chains)}")

    elec, vdw, solv_rec, solv_lig, intf_rec, _ = cpydock.calculate_energy(
        rec, lig, rec.charges, lig.charges, rec.vdw_energy, lig.vdw_energy,
        rec.vdw_radii, lig.vdw_radii, rec.hydrogens, lig.hydrogens,
        rec.sasa, lig.sasa, rec.des_energy, lig.des_energy, INTERFACE_CUTOFF)
    desolv = -(solv_rec + solv_lig)        # LightDock's sign: pyDock = elec + w*vdw + desolv
    return {"pydock_elec": elec, "pydock_vdw": vdw, "pydock_desolv": desolv,
            "pydock": elec + desolv + 0.1 * vdw, "pydock_1vdw": elec + desolv + vdw,
            "n_interface_atoms": len(intf_rec), "note": ""}


def main() -> int:
    ap = argparse.ArgumentParser(description="pyDock bait-prey energy of every minimized sample of one pair")
    ap.add_argument("--pair", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--bait-chains", required=True, help="comma-separated")
    ap.add_argument("--prey-chains", required=True, help="comma-separated")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from nanopulldown.core.io import write_parquet_with_metadata
    from nanopulldown.core.metric_refs import column_references

    bait, prey = set(a.bait_chains.split(",")), set(a.prey_chains.split(","))
    rows = []
    for i in range(a.num_samples):
        pdb = Path(a.results_root) / a.pair / "minimized" / f"sample_{i}.pdb.gz"
        row = {"pair": a.pair, "sample_index": i}
        try:
            row |= pydock_energy(pdb, bait, prey)
            print(f"[pydock] {a.pair} sample {i}: pydock={row['pydock']:.2f} "
                  f"pydock_1vdw={row['pydock_1vdw']:.2f} (elec {row['pydock_elec']:.2f}, "
                  f"desolv {row['pydock_desolv']:.2f}, vdw {row['pydock_vdw']:.2f})")
        except Exception as e:                                # noqa: BLE001
            row |= {c: np.nan for c in ("pydock_elec", "pydock_vdw", "pydock_desolv",
                                        "pydock", "pydock_1vdw")}
            row |= {"n_interface_atoms": -1, "note": f"{type(e).__name__}: {e}"}
            print(f"[pydock] {a.pair} sample {i}: FAILED {row['note']}", file=sys.stderr)
        rows.append(row)

    write_parquet_with_metadata(pd.DataFrame(rows), Path(a.out), table_description=TABLE_DESCRIPTION,
                                column_descriptions=COLUMN_DESCRIPTIONS,
                                column_references=column_references(COLUMN_DESCRIPTIONS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
