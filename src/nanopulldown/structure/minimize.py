#!/usr/bin/env python3
"""
nanopulldown.structure.minimize — Stage 2d
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
minimization to convergence -> minimized PDB (gzipped, sample_<i>.pdb.gz --
ChimeraX/gemmi read it as-is) + an energy trace parquet (sample_<i>_energy.parquet,
one point every `minimize.trace_interval` iterations) for nanopulldown.report.plots.minimize_energy.

Platform: tries `minimize.platform_preference` in order (OpenCL first, for
Mac Metal-backed GPU acceleration, then CPU) -- confirm OpenCL actually
initializes on the target Mac before relying on GPU accel; CPU is always a
safe fallback.
"""
from __future__ import annotations

import argparse
import gzip
import io
import sys
from pathlib import Path

ENERGY_TABLE_DESCRIPTION = (
    "Potential-energy trace of one OpenMM minimization (one diffusion "
    "sample): one row per recorded point, from the unminimized start "
    "to the final step. Read by nanopulldown.report.plots.minimize_energy.")
ENERGY_COLUMN_DESCRIPTIONS = {
    "step": "Cumulative L-BFGS iterations performed so far (0 = initial structure).",
    "energy_kJ_mol": "Potential energy of the system at that step, kJ/mol.",
}


def energy_path(pdb_path: Path) -> Path:
    """minimized/sample_<i>.pdb.gz -> minimized/sample_<i>_energy.parquet"""
    stem = pdb_path.name.split(".")[0]
    return pdb_path.with_name(f"{stem}_energy.parquet")


def _opencl_gpu_platform(platform) -> int | None:
    """Index of the first OpenCL platform whose default device is a GPU
    vendor's (NVIDIA / AMD / Apple), probed on a 1-particle system."""
    from openmm import Context, System, VerletIntegrator

    probe = System()
    probe.addParticle(1.0)
    for i in range(8):
        try:
            ctx = Context(probe, VerletIntegrator(0.001), platform, {"OpenCLPlatformIndex": str(i)})
        except Exception:                                     # noqa: BLE001
            break                                             # past the last platform
        vendor = platform.getPropertyValue(ctx, "OpenCLPlatformName")
        del ctx
        if any(v in vendor for v in ("NVIDIA", "AMD Accelerated", "Apple")):
            return i
    return None


def _make_simulation(topology, system, preference: list[str]):
    """Simulation on the first platform of `preference` that both exists AND
    starts: a platform can be listed yet fail on its first Context -- e.g.
    OpenMM's CUDA kernels built for a newer CUDA than the node's driver
    (CUDA_ERROR_UNSUPPORTED_PTX_VERSION on IFB). Reference is the last resort."""
    import os

    # OpenMM's CPU platform otherwise starts one thread per core of the whole
    # node, not of the SLURM allocation
    if os.environ.get("SLURM_CPUS_PER_TASK") and not os.environ.get("OPENMM_CPU_THREADS"):
        os.environ["OPENMM_CPU_THREADS"] = os.environ["SLURM_CPUS_PER_TASK"]
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import Simulation

    for name in [*preference, "Reference"]:
        try:
            platform = Platform.getPlatformByName(name)
            # OpenCL: a GPU vendor's OpenCL platform, never a CPU one. CAIUS nodes
            # also register Intel's CPU OpenCL, which OpenMM may silently pick --
            # minimization then crawls on 2 cores past the job's walltime.
            props = {}
            if name == "OpenCL":
                idx = _opencl_gpu_platform(platform)
                if idx is None:
                    raise RuntimeError("no GPU OpenCL platform")
                props = {"OpenCLPlatformIndex": str(idx)}
            # an integrator belongs to one Context: a fresh one per attempt
            integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                  0.002 * unit.picoseconds)
            simulation = Simulation(topology, system, integrator, platform, props)
        except Exception as e:                                # noqa: BLE001
            print(f"[minimize] platform {name} unusable ({str(e).splitlines()[0][:120]}) -- trying the next",
                  flush=True)
            continue
        device = ""
        if name in ("OpenCL", "CUDA"):
            device = " (" + platform.getPropertyValue(simulation.context, "DeviceName").strip() + ")"
        print(f"[minimize] platform: {name}{device}", flush=True)
        return simulation
    raise RuntimeError("no OpenMM platform could start, not even Reference")


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
    # pdbxfile= wants a seekable file OBJECT; filename= can't read gzip.
    if cif_path.suffix == ".gz":
        with gzip.open(cif_path, "rt") as fh:
            fixer = PDBFixer(pdbxfile=io.StringIO(fh.read()))
    else:
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

    simulation = _make_simulation(modeller.topology, system, mz["platform_preference"])
    simulation.context.setPositions(modeller.positions)

    def energy_kj() -> float:
        state = simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    trace = []
    e0 = energy_kj()
    trace.append((0, e0))
    print(f"[minimize] initial energy: {e0:.1f} kJ/mol")

    # Minimize in chunks of `trace_interval` iterations, recording the energy
    # after each one (L-BFGS restarts per call, which is harmless here). The
    # divergence guard runs on every chunk; the stricter probe check (energy
    # blowing up relative to the start) runs once, when probe_steps is reached.
    interval = max(1, int(mz.get("trace_interval", 50)))
    max_steps = int(mz["max_steps"])
    probe_steps = int(mz["probe_steps"])
    max_abs = float(mz["max_abs_energy"])
    done, probed, prev = 0, False, e0
    while done < max_steps:
        n = min(interval, max_steps - done)
        LocalEnergyMinimizer.minimize(simulation.context, maxIterations=n)
        done += n
        e = energy_kj()
        trace.append((done, e))

        bad = e != e or abs(e) > max_abs
        if not probed and done >= probe_steps:
            probed = True
            bad = bad or (e0 < 0 and e > 0 and abs(e) > float(mz["divergence_factor"]) * abs(e0))
            if not bad:
                print(f"[minimize] probe energy looks sane ({e:.1f} kJ/mol) — continuing to convergence")
        if bad:
            raise RuntimeError(
                f"Minimization diverged at step {done} (initial={e0:.1f}, "
                f"current={e:.1f} kJ/mol) — refusing to continue or save a garbage structure."
            )
        # A chunk that moves nothing means L-BFGS converged before its cap.
        if done < max_steps and abs(e - prev) < float(mz.get("convergence_delta", 1e-3)):
            print(f"[minimize] converged at step {done}")
            break
        prev = e
    e_final = trace[-1][1]

    state = simulation.context.getState(getPositions=True)
    with gzip.open(pdb_path, "wt") as fh:
        PDBFile.writeFile(modeller.topology, state.getPositions(), fh)
    print(f"[minimize] saved: {pdb_path.name} (final energy {e_final:.1f} kJ/mol)")

    import pandas as pd

    from nanopulldown.core.io import write_parquet_with_metadata

    energy_pq = energy_path(pdb_path)
    df = pd.DataFrame(trace, columns=["step", "energy_kJ_mol"])
    write_parquet_with_metadata(
        df, energy_pq,
        table_description=ENERGY_TABLE_DESCRIPTION,
        column_descriptions=ENERGY_COLUMN_DESCRIPTIONS,
    )
    print(f"[minimize] energy trace ({len(df)} points) -> {energy_pq.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cif", type=Path)
    ap.add_argument("pdb", type=Path)
    args = ap.parse_args()

    from nanopulldown.core.config import load_config

    cfg = load_config()
    minimize_one(args.cif, args.pdb, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
