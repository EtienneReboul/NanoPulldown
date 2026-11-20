"""pyKVFinder Trp-pocket analysis of AGO1 x FBW2 models (worker for the fbw2_ago1_trp_pockets*.ipynb notebooks).

For one complex model (AGO1 = chain A, FBW2 = chain B): remove FBW2 and the low-pLDDT AGO1 residues, then
  * global pyKVFinder detection  -> which AGO1 cavity each FBW2 Trp indole sits in (and its volume);
  * ligand-mode detection per Trp (side chain as ligand, LIGAND_CUTOFF Å) -> the Trp sub-pocket:
    volume, area, depth, hydropathy, lining residues, fraction of the indole inside the cavity;
  * AGO1 heavy atoms within CONTACT Å of the indole.

Run all models in parallel from the shell (results cached as parquet, re-runs skip finished models):
    python notebooks/trp_pockets.py abcfold --workers 12
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_WARNINGS", "0")
import argparse, csv, logging, tempfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import gemmi

TRP = [225, 258, 275, 295, 313]
INDOLE = ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]
STEP, PROBE_IN, PROBE_OUT = 0.6, 1.4, 8.0
LIGAND_CUTOFF, CONTACT, PLDDT_TRIM, CHANNEL_MIN_VOL = 5.0, 4.5, 50.0, 5000.0
BAIT, PREY = "A", "B"

ABCFOLD = Path("/Users/ereboul/projects/AGO1_modelling/ABCfold_ifb_AGO1_MiRs_complexes/results/ago1_fbw2")
ABCFOLD_OUT = Path(__file__).resolve().parents[1] / "reports_ago1_fbw2_abcfold" / "trp_pockets"


def model_to_uniprot(n: int) -> int:
    """ABCfold AGO1 (1,050 aa) has an IR insertion at 332-333 vs UniProt O04379 (1,048 aa)."""
    return n if n <= 331 else (n - 2 if n >= 334 else -1)


def _write(chains, path):
    st = gemmi.Structure(); st.add_model(gemmi.Model("1"))
    for name, residues in chains.items():
        ch = gemmi.Chain(name)
        for r in residues: ch.add_residue(r)
        st[0].add_chain(ch)
    st.setup_entities(); st.write_pdb(str(path))


def analyse_model(model_id: str, pdb: str, plddt: dict[tuple[str, int], float], workdir: str,
                  export_dir: str | None = None):
    """plddt: {(chain, resnum): pLDDT 0-100}. Returns (per-Trp rows, per-residue rows).
    export_dir: also write each Trp sub-pocket grid as pocket_W<n>.pdb (for viewing)."""
    import pyKVFinder
    logging.getLogger().setLevel(logging.ERROR)
    wd = Path(workdir)
    st = gemmi.read_structure(pdb); st.remove_hydrogens(); m = st[0]
    A, B = m[BAIT], m[PREY]
    keep = [r for r in A if plddt.get((BAIT, r.seqid.num), 100.0) >= PLDDT_TRIM]
    rec = wd / "rec.pdb"; _write({BAIT: keep}, rec)
    atomic = pyKVFinder.read_pdb(str(rec))
    vert = pyKVFinder.get_vertices(atomic, probe_out=PROBE_OUT, step=STEP)
    o = vert[0]; ax = np.array([(vert[i] - o) / np.linalg.norm(vert[i] - o) for i in (1, 2, 3)])

    def tags_at(grid, xyz, r=1.2):
        ijk = np.rint(((xyz - o) @ ax.T) / STEP).astype(int); k = int(np.ceil(r / STEP)); out = []
        for d in np.ndindex(2 * k + 1, 2 * k + 1, 2 * k + 1):
            q = ijk + np.array(d) - k
            if np.all(q >= 0) and np.all(q < grid.shape) and grid[tuple(q)] > 1: out.append(int(grid[tuple(q)]))
        return out

    kw = dict(step=STEP, probe_in=PROBE_IN, probe_out=PROBE_OUT, nthreads=1)
    _, cav = pyKVFinder.detect(atomic, vert, **kw)
    _, gvol, _ = pyKVFinder.spatial(cav, step=STEP, nthreads=1)
    ns = gemmi.NeighborSearch(m, st.cell, 6).populate()
    T, R = [], []
    for w in TRP:
        res = B[str(w)][0]
        ind = np.array([a.pos.tolist() for a in res if a.name in INDOLE])
        flat = [t for x in ind for t in tags_at(cav, x)]
        gname = pyKVFinder.grid._get_cavity_name(max(set(flat), key=flat.count) - 2) if flat else None
        gv = gvol[gname] if gname else np.nan
        sc = res.clone()
        for name in ("N", "C", "O"): sc.remove_atom(name, "\0")
        lig = wd / "lig.pdb"; _write({PREY: [sc]}, lig)
        n2, sub = pyKVFinder.detect(atomic, vert, latomic=pyKVFinder.read_pdb(str(lig)),
                                    ligand_cutoff=LIGAND_CUTOFF, **kw)
        lining, vol, area, dmax, davg, hyd = set(), 0.0, 0.0, 0.0, 0.0, np.nan
        if n2 > 0:
            surf, v, a_ = pyKVFinder.spatial(sub, step=STEP, nthreads=1)
            _, maxd, avgd = pyKVFinder.depth(sub, step=STEP, nthreads=1)
            try:
                _, hydro = pyKVFinder.hydropathy(surf, atomic, vert, step=STEP, nthreads=1)
            except ValueError:   # pyKVFinder bug on some sub-pockets ("negative dimensions"): no hydropathy
                hydro = {k: np.nan for k in v}
            cons = pyKVFinder.constitutional(sub, atomic, vert, step=STEP, probe_in=PROBE_IN, nthreads=1)
            vol, area = sum(v.values()), sum(a_.values()); dmax = max(maxd.values())
            davg = sum(avgd[k] * v[k] for k in v) / vol; hyd = sum(hydro[k] * v[k] for k in v) / vol
            for k in cons: lining |= {int(r[0]) for r in cons[k]}
            if export_dir:
                pyKVFinder.export(str(Path(export_dir) / f"pocket_W{w}.pdb"), sub, surf, vert, step=STEP)
        occ = float(np.mean([len(tags_at(sub, x, 1.0)) > 0 for x in ind])) if n2 else 0.0
        contact = {}
        for a in res:
            if a.name not in INDOLE: continue
            for mk in ns.find_atoms(a.pos, "\0", radius=CONTACT):
                c = mk.to_cra(m)
                if c.chain.name == BAIT:
                    rn = c.residue.seqid.num; contact[rn] = min(contact.get(rn, 9e9), c.atom.pos.dist(a.pos))
        for rn in sorted(lining | set(contact)):
            R.append(dict(model_id=model_id, trp=w, ago1_resnr=rn, ago1_resname=A[str(rn)][0].name,
                          lines_subpocket=rn in lining, indole_contact=rn in contact,
                          min_dist=contact.get(rn, np.nan), plddt=plddt.get((BAIT, rn), np.nan)))
        T.append(dict(model_id=model_id, trp=w, trp_plddt=plddt.get((PREY, w), np.nan),
                      global_cavity=gname, global_vol=gv, in_channel=bool(gname) and gv >= CHANNEL_MIN_VOL,
                      sub_volume=vol, sub_area=area, sub_depth_max=dmax, sub_depth_avg=davg,
                      sub_hydropathy=hyd, indole_in_cavity=occ, n_lining=len(lining), n_contacts=len(contact),
                      n_ago1_kept=len(keep)))
    return pd.DataFrame(T), pd.DataFrame(R)


# ---------------------------------------------------------------- ABCfold dataset
def abcfold_models(root: Path = ABCFOLD) -> pd.DataFrame:
    """The 600 models of selected_models.csv (current pose clusters); minimized/ also holds stale ones."""
    sel = pd.read_csv(root / "selected_models.csv")
    sel["model_name"] = sel.staged_cif.map(lambda p: Path(p).stem)
    sel["model_id"] = sel.cluster.astype(str) + "/" + sel.model_name
    sel["pdb"] = [str(root / "minimized" / str(c) / n / f"{n}.pdb") for c, n in zip(sel.cluster, sel.model_name)]
    sel["cif"] = [str(root / "selected" / f"cluster_{c}" / f"{n}.cif") for c, n in zip(sel.cluster, sel.model_name)]
    return sel


def residue_plddt(cif: str) -> dict[tuple[str, int], float]:
    st = gemmi.read_structure(cif)
    return {(ch.name, r.seqid.num): float(np.mean([a.b_iso for a in r])) for ch in st[0] for r in ch}


def _job(model_id, pdb, cif):
    with tempfile.TemporaryDirectory() as wd:
        return analyse_model(model_id, pdb, residue_plddt(cif), wd)


def run_abcfold(workers: int = 12, out: Path = ABCFOLD_OUT, flush_every: int = 25):
    out.mkdir(parents=True, exist_ok=True)
    f_trp, f_res = out / "kvfinder_trp.parquet", out / "kvfinder_residues.parquet"
    models = abcfold_models()
    T = [pd.read_parquet(f_trp)] if f_trp.exists() else []
    R = [pd.read_parquet(f_res)] if f_res.exists() else []
    done = set(T[0].model_id) if T else set()
    todo = models[~models.model_id.isin(done)]
    print(f"{len(done)} done, {len(todo)} to do, {workers} workers", flush=True)

    def flush():
        pd.concat(T, ignore_index=True).to_parquet(f_trp, index=False)
        pd.concat(R, ignore_index=True).to_parquet(f_res, index=False)

    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_job, r.model_id, r.pdb, r.cif): r.model_id for r in todo.itertuples()}
        for fut in as_completed(futs):
            try:
                a, b = fut.result(); T.append(a); R.append(b); n += 1
            except Exception as e:  # keep going; failed models are retried on the next run
                print("FAILED", futs[fut], repr(e), flush=True)
            if n and n % flush_every == 0:
                flush(); print(f"{n}/{len(todo)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    if T: flush()
    print(f"finished {n} models in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["abcfold"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="test on the first N models (separate output dir)")
    a = ap.parse_args()
    if a.limit:
        _full = abcfold_models
        abcfold_models = lambda: _full().groupby("backend").head(max(1, a.limit // 6))  # noqa: E731
        run_abcfold(a.workers, ABCFOLD_OUT.parent / "trp_pockets_test")
    else:
        run_abcfold(a.workers)
