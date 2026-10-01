#!/usr/bin/env python3
"""
scripts/minimize_openmm.py — Stage 2d
========================================
Replaces ab_initio_pipeline's ChimeraX scripts/minimize_cif.py with a plain
OpenMM script: no ChimeraX, no antechamber/GAFF (protein-protein only, so a
standard protein forcefield is all that's needed -- no small-molecule
parameterization).

Pipeline: PDBFixer (load CIF, add missing atoms/terminal residues, add
hydrogens at pH 7) -> OpenMM ForceField (amber14-all + a GBSA/implicit-
solvent term, from config.yaml's `minimize.forcefield`) -> a short probe
minimization to screen for divergence (mirrors the ChimeraX script's
probe-then-continue guard, scripts/minimize_cif.py:156-188) -> a longer
minimization to convergence -> minimized PDB + a 3-point energy trace CSV
(initial / post-probe / final) for scripts/plot_minimize_energy.py.

Platform: tries `minimize.platform_preference` in order (OpenCL first, for
Mac Metal-backed GPU acceleration, then CPU) -- confirm OpenCL actually
initializes on the target Mac before relying on GPU accel; CPU is always a
safe fallback.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import load_config  # noqa: E402


def _get_platform(preference: list[str]):
    from openmm import Platform

    for name in preference:
        try:
            return Platform.getPlatformByName(name)
        except Exception:                                     # noqa: BLE001
            continue
    return Platform.getPlatformByName("Reference")


def minimize_one(cif_path: Path, pdb_path: Path, cfg: dict) -> None:
    from openmm import LocalEnergyMinimizer, unit
    from openmm.app import ForceField, Modeller, PDBFile
    from pdbfixer import PDBFixer

    mz = cfg["minimize"]

    if pdb_path.exists():
        print(f"[minimize] output already exists — skipping: {pdb_path.name}")
        return
    pdb_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[minimize] loading {cif_path.name}")
    # pdbxfile= wants a file OBJECT (it calls .seek() on it), not a path
    # string -- filename= is the string-path form and auto-detects mmCIF
    # from the .cif extension.
    fixer = PDBFixer(filename=str(cif_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    forcefield = ForceField(*mz["forcefield"])
    modeller = Modeller(fixer.topology, fixer.positions)
    # Default nonbondedMethod (NoCutoff) is correct for an implicit-solvent,
    # no-periodic-box minimization of a single complex.
    system = forcefield.createSystem(modeller.topology)

    from openmm import LangevinMiddleIntegrator
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                           0.002 * unit.picoseconds)
    platform = _get_platform(mz["platform_preference"])
    print(f"[minimize] platform: {platform.getName()}")

    from openmm.app import Simulation
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    def energy_kj() -> float:
        state = simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    trace = []
    e0 = energy_kj()
    trace.append((0, e0))
    print(f"[minimize] initial energy: {e0:.1f} kJ/mol")

    print(f"[minimize] probe ({mz['probe_steps']} iterations)...")
    LocalEnergyMinimizer.minimize(simulation.context, maxIterations=int(mz["probe_steps"]))
    e_probe = energy_kj()
    trace.append((int(mz["probe_steps"]), e_probe))

    diverged = (
        e_probe != e_probe                                    # NaN check
        or abs(e_probe) > float(mz["max_abs_energy"])
        or (e0 < 0 and e_probe > 0 and abs(e_probe) > float(mz["divergence_factor"]) * abs(e0))
    )
    if diverged:
        raise RuntimeError(
            f"Minimization diverged during the probe (initial={e0:.1f}, "
            f"post-probe={e_probe:.1f} kJ/mol) — refusing to continue or save "
            "a garbage structure."
        )
    print(f"[minimize] probe energy looks sane ({e_probe:.1f} kJ/mol) — continuing to convergence")

    remaining = max(0, int(mz["max_steps"]) - int(mz["probe_steps"]))
    if remaining:
        LocalEnergyMinimizer.minimize(simulation.context, maxIterations=remaining)
    e_final = energy_kj()
    trace.append((int(mz["max_steps"]), e_final))

    if e_final != e_final or abs(e_final) > float(mz["max_abs_energy"]):
        raise RuntimeError(
            f"Minimization diverged during the full run (final={e_final:.1f} "
            "kJ/mol) — refusing to save a garbage structure."
        )

    state = simulation.context.getState(getPositions=True)
    with open(pdb_path, "w") as fh:
        PDBFile.writeFile(modeller.topology, state.getPositions(), fh)
    print(f"[minimize] saved: {pdb_path.name} (final energy {e_final:.1f} kJ/mol)")

    energy_csv = pdb_path.with_name(pdb_path.stem + "_energy.csv")
    with open(energy_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "energy_kJ_mol"])
        writer.writeheader()
        for step, energy in trace:
            writer.writerow({"step": step, "energy_kJ_mol": energy})
    print(f"[minimize] energy trace -> {energy_csv.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cif", type=Path)
    ap.add_argument("pdb", type=Path)
    args = ap.parse_args()

    cfg = load_config()
    minimize_one(args.cif, args.pdb, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
