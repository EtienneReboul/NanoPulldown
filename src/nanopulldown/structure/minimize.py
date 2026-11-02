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

Platform: the first of `minimize.platform_preference` that starts (OpenCL: a
GPU vendor's platform, never a CPU one). There is NO implicit CPU/Reference
fallback: a CPU minimization of a clashed 45k-atom complex runs for hours
without a guarantee of converging, so no GPU platform = the job fails.

Fallback (`minimize.cg2all_fallback`): a model so clashed that the very first
energy/force evaluation is infinite or NaN (atoms ~0.01 A apart, 1e31 kJ/mol
and up) is re-built with cg2all (Heo & Feig; CA trace in, all-atom out, CA
positions kept via --fix) and minimized again. Only that error triggers it:
every other model goes through the plain route, unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import io
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ENERGY_TABLE_DESCRIPTION = (
    "Potential-energy trace of one OpenMM minimization (one diffusion "
    "sample): one row per recorded point, from the unminimized start "
    "to the final step. Read by nanopulldown.report.plots.minimize_energy.")
ENERGY_COLUMN_DESCRIPTIONS = {
    "step": "Cumulative L-BFGS iterations performed so far, pre-relaxation included "
            "(0 = initial structure; energies are always the full implicit-solvent system's).",
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
    (CUDA_ERROR_UNSUPPORTED_PTX_VERSION on IFB). No CPU/Reference platform is
    tried unless the config lists it explicitly (see the module docstring)."""
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import Simulation

    for name in preference:
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
    raise RuntimeError(f"no OpenMM GPU platform could start (tried {', '.join(preference)}); "
                       "there is no CPU fallback")


def _load_fixer(cif_path: Path):
    """PDBFixer of a raw model with its missing heavy atoms added (no hydrogens)."""
    from pdbfixer import PDBFixer

    # pdbxfile= wants a seekable file OBJECT; filename= can't read gzip.
    if cif_path.suffix == ".gz":
        with gzip.open(cif_path, "rt") as fh:
            fixer = PDBFixer(pdbxfile=io.StringIO(fh.read()))
    else:
        fixer = PDBFixer(filename=str(cif_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    return fixer


def _cg2all_cli(name: str) -> str:
    """Path of a cg2all console script: next to this interpreter (the rule's
    conda env), else on PATH."""
    local = Path(sys.executable).parent / name
    found = str(local) if local.exists() else shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found -- cg2all is not installed in this environment "
                           "(use envs/minimize_cg2all.yaml, see minimize.cg2all_fallback)")
    return found


def _cg2all_rebuild(cif_path: Path, workdir: Path, fb: dict) -> Path:
    """Heavy-atom PDB of `cif_path` rebuilt by cg2all: all-atom -> CA trace ->
    all-atom with the CA coordinates fixed (--fix), IUPAC atom names, hydrogens
    dropped again (PDBFixer re-adds its own protonation)."""
    from openmm.app import PDBFile

    model = fb.get("model", "CalphaBasedModel")
    all_pdb, cg_pdb, rebuilt = workdir / "input.pdb", workdir / "cg.pdb", workdir / "rebuilt_all.pdb"
    fixer = _load_fixer(cif_path)
    with open(all_pdb, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    steps = (
        ([_cg2all_cli("convert_all2cg"), "-p", str(all_pdb), "-o", str(cg_pdb), "--cg", model], cg_pdb),
        ([_cg2all_cli("convert_cg2all"), "-p", str(cg_pdb), "-o", str(rebuilt), "--cg", model,
          "--fix", "--standard-name", "--device", fb.get("device", "cpu")], rebuilt),
    )
    for cmd, out in steps:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(fb.get("timeout_s", 900)))
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"{Path(cmd[0]).name} failed ({proc.returncode}): {proc.stderr[-400:]}")
    heavy = workdir / "rebuilt.pdb"
    with open(rebuilt) as src, open(heavy, "w") as dst:
        for line in src:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip().startswith("H"):
                continue
            dst.write(line)
    return heavy


def _minimize(cif_path: Path, pdb_path: Path, cfg: dict, table_note: str = "") -> None:
    from openmm import LocalEnergyMinimizer, unit
    from openmm.app import ForceField, Modeller, PDBFile

    mz = cfg["minimize"]

    if pdb_path.exists():
        print(f"[minimize] output already exists — skipping: {pdb_path.name}", flush=True)
        return
    pdb_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[minimize] loading {cif_path.name}", flush=True)
    fixer = _load_fixer(cif_path)
    fixer.addMissingHydrogens(7.0)

    forcefield = ForceField(*mz["forcefield"])
    modeller = Modeller(fixer.topology, fixer.positions)
    # Default nonbondedMethod (NoCutoff) is correct for an implicit-solvent,
    # no-periodic-box minimization of a single complex.
    system = forcefield.createSystem(modeller.topology)

    simulation = _make_simulation(modeller.topology, system, mz["platform_preference"])
    simulation.context.setPositions(modeller.positions)

    def energy_kj(sim=simulation) -> float:
        state = sim.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    trace = []
    e0 = energy_kj()
    trace.append((0, e0))
    print(f"[minimize] initial energy: {e0:.1f} kJ/mol", flush=True)

    interval = max(1, int(mz.get("trace_interval", 50)))
    max_steps = int(mz["max_steps"])
    probe_steps = int(mz["probe_steps"])
    max_abs = float(mz["max_abs_energy"])
    # Wall-clock budget (pre-relaxation included), checked between chunks: a
    # badly interpenetrated model (dozens of sub-1 A overlaps) can take hours of
    # line searches; past the budget the sane, already-relaxed structure is
    # saved as not converged.
    budget_s = 60.0 * float(mz.get("max_minutes") or 0)
    t_start = time.monotonic()

    # Pre-relaxation: vacuum, 1 nm cutoff (minimize.pre_relax). Raw models start
    # at ~1e12 kJ/mol (median; up to 1e19): with forces that large OpenMM's
    # minimizer evaluates them in double precision on one CPU thread, and with
    # NoCutoff GB that is O(N^2) per evaluation -- hours for a 45k-atom complex.
    # The cutoff system is cheap on any platform and pulls the overlaps apart.
    pre = mz.get("pre_relax") or {}
    pre_done = 0
    if pre.get("enabled"):
        from openmm.app import CutoffNonPeriodic

        pre_system = ForceField(*pre.get("forcefield", ["amber14-all.xml"])).createSystem(
            modeller.topology, nonbondedMethod=CutoffNonPeriodic,
            nonbondedCutoff=float(pre.get("cutoff_nm", 1.0)) * unit.nanometer)
        # Charges off: with overlapping atoms, vacuum Coulomb lets opposite
        # charges collapse onto each other (-3e7 kJ/mol, then nan in the GB
        # pass). Lennard-Jones + bonded terms alone push the overlaps apart.
        from openmm import NonbondedForce

        for force in pre_system.getForces():
            if isinstance(force, NonbondedForce):
                for i in range(force.getNumParticles()):
                    _, sigma, eps = force.getParticleParameters(i)
                    force.setParticleParameters(i, 0.0, sigma, eps)
                for k in range(force.getNumExceptions()):
                    p1, p2, _, sigma, eps = force.getExceptionParameters(k)
                    force.setExceptionParameters(k, p1, p2, 0.0, sigma, eps)
        pre_sim = _make_simulation(modeller.topology, pre_system, mz["platform_preference"])
        pre_sim.context.setPositions(modeller.positions)
        pre_steps = int(pre.get("max_steps", 500))
        e_pre = energy_kj(pre_sim)
        print(f"[minimize] pre-relax (vacuum, cutoff): start {e_pre:.1f} kJ/mol", flush=True)
        while pre_done < pre_steps:
            n = min(interval, pre_steps - pre_done)
            LocalEnergyMinimizer.minimize(pre_sim.context, maxIterations=n)
            pre_done += n
            e = energy_kj(pre_sim)
            if e != e:
                raise RuntimeError(f"Pre-relaxation diverged (nan) at step {pre_done}")
            if abs(e - e_pre) < float(mz.get("convergence_delta", 1e-3)):
                break
            e_pre = e
        positions = pre_sim.context.getState(getPositions=True).getPositions()
        del pre_sim
        simulation.context.setPositions(positions)
        e_start = energy_kj()
        trace.append((pre_done, e_start))
        print(f"[minimize] pre-relaxed in {pre_done} steps, {time.monotonic() - t_start:.0f}s: "
              f"full-system energy {e_start:.1f} kJ/mol", flush=True)
    else:
        e_start = e0

    # Minimize in chunks of `trace_interval` iterations, recording the energy
    # after each one (L-BFGS restarts per call, which is harmless here). The
    # divergence guard runs on every chunk; the stricter probe check (energy
    # blowing up relative to the start) runs once, when probe_steps is reached.
    done, probed, prev = 0, False, e_start
    while done < max_steps:
        n = min(interval, max_steps - done)
        LocalEnergyMinimizer.minimize(simulation.context, maxIterations=n)
        done += n
        e = energy_kj()
        trace.append((pre_done + done, e))

        bad = e != e or abs(e) > max_abs
        if not probed and done >= probe_steps:
            probed = True
            bad = bad or (e_start < 0 and e > 0 and abs(e) > float(mz["divergence_factor"]) * abs(e_start))
            if not bad:
                print(f"[minimize] probe energy looks sane ({e:.1f} kJ/mol) — continuing to convergence", flush=True)
        if bad:
            raise RuntimeError(
                f"Minimization diverged at step {done} (initial={e0:.1f}, "
                f"current={e:.1f} kJ/mol) — refusing to continue or save a garbage structure."
            )
        # A chunk that moves nothing means L-BFGS converged before its cap.
        if done < max_steps and abs(e - prev) < float(mz.get("convergence_delta", 1e-3)):
            print(f"[minimize] converged at step {done}", flush=True)
            break
        if probed and budget_s and time.monotonic() - t_start > budget_s:
            print(f"[minimize] time budget ({budget_s / 60:.0f} min) reached at step {done} — "
                  f"saving unconverged ({e:.1f} kJ/mol)", flush=True)
            break
        prev = e
    e_final = trace[-1][1]

    state = simulation.context.getState(getPositions=True)
    with gzip.open(pdb_path, "wt") as fh:
        PDBFile.writeFile(modeller.topology, state.getPositions(), fh)
    print(f"[minimize] saved: {pdb_path.name} (final energy {e_final:.1f} kJ/mol)", flush=True)

    import pandas as pd

    from nanopulldown.core.io import write_parquet_with_metadata

    energy_pq = energy_path(pdb_path)
    df = pd.DataFrame(trace, columns=["step", "energy_kJ_mol"])
    write_parquet_with_metadata(
        df, energy_pq,
        table_description=ENERGY_TABLE_DESCRIPTION + table_note,
        column_descriptions=ENERGY_COLUMN_DESCRIPTIONS,
    )
    print(f"[minimize] energy trace ({len(df)} points) -> {energy_pq.name}", flush=True)


def minimize_one(cif_path: Path, pdb_path: Path, cfg: dict) -> None:
    fb = cfg["minimize"].get("cg2all_fallback") or {}
    try:
        _minimize(cif_path, pdb_path, cfg)
        return
    except Exception as e:                                    # noqa: BLE001
        # OpenMM's "Energy or force at minimization starting point is infinite or NaN"
        # (and the pre-relaxation's own nan guard): atoms so close the first
        # evaluation overflows. Nothing else is retried.
        if not (fb.get("enabled") and "NaN" in str(e)):
            raise
        first_error = str(e).splitlines()[0]
    print(f"[minimize] {first_error} -- rebuilding with cg2all ({fb.get('model', 'CalphaBasedModel')}, "
          "CA coordinates fixed) and minimizing that", flush=True)
    with tempfile.TemporaryDirectory() as td:
        t0 = time.monotonic()
        rebuilt = _cg2all_rebuild(cif_path, Path(td), fb)
        print(f"[minimize] cg2all rebuild: {time.monotonic() - t0:.0f}s", flush=True)
        _minimize(rebuilt, pdb_path, cfg,
                  table_note=" Minimized from a cg2all rebuild of the raw model (the raw model's "
                             "first energy evaluation was infinite or NaN): step 0 is the rebuilt "
                             "model's energy, not the raw model's.")


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
