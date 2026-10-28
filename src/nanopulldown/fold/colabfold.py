"""
nanopulldown.fold.colabfold — Stage 2a with fold.engine: colabfold
==================================================================
ColabFold (AlphaFold2-multimer) instead of ESMFold2-Fast, in two steps: the
MSAs first, then the folds. The fold step writes exactly what
nanopulldown.fold.esmfold2 writes -- results/<pair>/raw/sample_<i>.cif.gz +
fold.parquet (schema: nanopulldown.core.fold_table) -- so metrics,
minimization, PLIP, pyDock and the report run unchanged.

  `nanopulldown colabfold msa --pairs P... --done F`   (rule colabfold_msa)
      One paired + unpaired MSA per COMPLEX (a3m with ColabFold's
      `#len1,len2<TAB>card1,card2` header), cached by content in
      colabfold.msa.cache_dir/<key>.a3m -- key = the chain sequences + MSA
      settings (msa_key), so a complex shared by two runs or renamed is
      searched once. Only missing keys are searched, in ONE batch:
        mode: server  `colabfold_batch --msa-only` against the ColabFold
                      MMseqs2 API (internet; small runs -- the public server
                      is shared, colabfold.msa.server_max_queries guards it)
        mode: local   `colabfold_search` against local ColabFold databases
                      (setup_databases.sh), on the GPU with MMseqs2-GPU
                      (colabfold.msa.gpu: 1) -- a GPU job on IFB / CAIUS.
      F (the .done file) is touched only when every pair has its a3m.

  `nanopulldown colabfold fold --pairs P... --out-root R`   (rule run_colabfold*)
      One `colabfold_batch` call per chunk (model weights compiled once per
      chunk) over a directory of the chunk's cached a3m files, into
      R/_colabfold/<chunk>/ -- ColabFold's own <job>.done.txt markers make a
      rerun skip finished pairs. Each finished pair is converted (convert())
      into R/_colabfold_cache/<pair>/ and hard-linked into R/<pair>/raw/, the
      crash-safety scheme of nanopulldown.fold.esmfold2 (its own cache dir,
      so an ESMFold2 cache in the same results root is never mistaken for it).

  `nanopulldown colabfold convert --pair P --run-dir D --out-dir O`
      The conversion alone, e.g. for folds run by hand.

Samples: colabfold.num_models x colabfold.num_seeds per pair, sample_index
seed-major (seed random_seed, model 1..N; then seed random_seed+1, ...) --
fixed by model and seed, not by ColabFold's confidence rank. Chains: ColabFold
letters them A, B, ... in input order; they are renamed to the pair spec's
chain ids. pLDDT is stored 0-1 and residue_index 0-based per chain, like
ESMFold2's (ColabFold writes 0-100 and numbers residues from 1).
pair_chains_iptm comes from --calc-extra-ptm's pairwise_iptm (per chain pair);
without it, a 2-chain pair gets the complex ipTM (its only interface) and
larger complexes none.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from nanopulldown.core.config import REPO_ROOT, load_config
from nanopulldown.core.fold_table import FOLD_FILE, engine_samples, fold_row, write_fold
from nanopulldown.core.specs import load_pairs, protein_chains

CACHE_DIR = "_colabfold_cache"
RUN_DIR = "_colabfold"
# <job>_scores_rank_001_alphafold2_multimer_v3_model_1_multimer_v3_seed_000.json
_SCORES = re.compile(r"_scores_rank_\d+_.*?_model_(\d+)\S*?_seed_(\d+)\.json$")


def safe_filename(name: str) -> str:
    """ColabFold's own job-name sanitizer (colabfold.input.safe_filename)."""
    return "".join(c if c.isalnum() or c in "_.-" else "_" for c in name)


# ── MSAs ──────────────────────────────────────────────────────────────────

def msa_settings(cf: dict) -> dict:
    """What an MSA depends on besides the sequences (part of its cache key)."""
    m = cf.get("msa") or {}
    out = {"mode": m.get("mode", "server"), "msa_mode": m.get("msa_mode", "mmseqs2_uniref_env"),
           "pair_mode": m.get("pair_mode", "unpaired_paired")}
    if out["mode"] == "local":
        out.update(db1=m.get("db1", "uniref30_2302_db"), db3=m.get("db3", "colabfold_envdb_202108_db"),
                   use_env=int(m.get("use_env", 1)))
    return out


def msa_key(spec: dict, cf: dict) -> str:
    seqs = [c["sequence"] for c in protein_chains(spec)]
    blob = json.dumps({"seqs": seqs, **msa_settings(cf)}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def msa_dir(cf: dict) -> Path:
    return REPO_ROOT / (cf.get("msa") or {}).get("cache_dir", "data/msa")


def a3m_path(spec: dict, cf: dict) -> Path:
    return msa_dir(cf) / f"{msa_key(spec, cf)}.a3m"


def _valid_a3m(path: Path, n_chains: int) -> bool:
    """A complex a3m: `#lens<TAB>cards` header, then at least the query."""
    try:
        with open(path) as fh:
            head = [fh.readline() for _ in range(3)]
    except OSError:
        return False
    if len(head) < 3 or not head[2].strip():
        return False
    return n_chains == 1 or (head[0].startswith("#") and len(head[0][1:].split("\t")) == 2)


def _run(cmd: list[str], env: dict | None = None) -> int:
    print("[colabfold] $ " + shlex.join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


def build_msas(pairs: list[str], cfg: dict, work_tag: str | None = None) -> int:
    """Search the MSAs the cache lacks for `pairs` (one batch); 0 when all exist.

    `work_tag` gives the batch its own work dir, so several batches can search at once."""
    cf = cfg["colabfold"]
    m = cf.get("msa") or {}
    mode = msa_settings(cf)["mode"]
    specs = load_pairs(REPO_ROOT)
    todo: dict[str, dict] = {}
    for p in pairs:
        spec = specs[p]
        if not _valid_a3m(a3m_path(spec, cf), len(protein_chains(spec))):
            todo.setdefault(msa_key(spec, cf), spec)
    print(f"[colabfold msa] {len(pairs)} pair(s), {len(todo)} complex MSA(s) to search ({mode})", flush=True)
    if not todo:
        return 0
    if mode == "server" and len(todo) > int(m.get("server_max_queries", 500)):
        print(f"[colabfold msa] {len(todo)} queries > colabfold.msa.server_max_queries="
              f"{m.get('server_max_queries', 500)}: the public MMseqs2 server is shared -- use "
              "colabfold.msa.mode: local (GPU search) for a run this size, or raise the limit "
              "for a private server (colabfold.msa.host_url)", file=sys.stderr)
        return 1

    cache = msa_dir(cf)
    work = cache / "_work" / (f"{mode}_{work_tag}" if work_tag else mode)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    fasta = work.parent / f"queries_{work.name}.fasta"
    fasta.write_text("".join(f">{k}\n{':'.join(c['sequence'] for c in protein_chains(s))}\n"
                             for k, s in todo.items()))

    if mode == "server":
        cmd = [m.get("colabfold_batch", "colabfold_batch"), str(fasta), str(work), "--msa-only",
               "--msa-mode", m.get("msa_mode", "mmseqs2_uniref_env"),
               "--pair-mode", m.get("pair_mode", "unpaired_paired")]
        if m.get("host_url"):
            cmd += ["--host-url", m["host_url"]]
    elif mode == "local":
        if not m.get("db_dir"):
            print("[colabfold msa] colabfold.msa.mode: local needs colabfold.msa.db_dir "
                  "(setup_databases.sh output)", file=sys.stderr)
            return 1
        st = msa_settings(cf)
        cmd = [m.get("colabfold_search", "colabfold_search"), "--mmseqs", m.get("mmseqs", "mmseqs"),
               "--db1", st["db1"], "--db3", st["db3"], "--use-env", str(st["use_env"]),
               "--use-templates", "0", "--pair-mode", st["pair_mode"],
               "--threads", str(int(m.get("threads", 0)) or int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
                                or os.cpu_count() or 8),
               "--db-load-mode", str(int(m.get("db_load_mode", 0)))]
        if int(m.get("gpu", 0)):
            cmd += ["--gpu", "1", "--gpu-server", str(int(m.get("gpu_server", 0)))]
        cmd += [str(fasta), str(m["db_dir"]), str(work)]
    else:
        print(f"[colabfold msa] unknown colabfold.msa.mode {mode!r} (server | local)", file=sys.stderr)
        return 1
    cmd += [str(x) for x in m.get("extra_args") or []]

    t0 = time.time()
    rc = _run(cmd)
    # collect whatever finished, even after a failure: a rerun searches only the rest
    got = 0
    for k, spec in todo.items():
        f = work / f"{safe_filename(k)}.a3m"
        if f.exists() and _valid_a3m(f, len(protein_chains(spec))):
            os.replace(f, cache / f"{k}.a3m")
            got += 1
    print(f"[colabfold msa] {got}/{len(todo)} MSA(s) in {time.time() - t0:.0f}s (exit {rc})", flush=True)
    if got == len(todo):
        shutil.rmtree(work, ignore_errors=True)
        fasta.unlink(missing_ok=True)
        return 0
    return rc or 1


def msa_main() -> int:
    ap = argparse.ArgumentParser(description="ColabFold MSAs of pairs (server or local MMseqs2[-GPU])")
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--done", help="touched once every pair has its MSA")
    ap.add_argument("--work-tag", help="own work dir (data/msa/_work/<mode>_<tag>): concurrent batches")
    a = ap.parse_args()
    rc = build_msas(a.pairs, load_config(), a.work_tag)
    if rc == 0 and a.done:
        Path(a.done).parent.mkdir(parents=True, exist_ok=True)
        Path(a.done).touch()
    return rc


# ── folds ─────────────────────────────────────────────────────────────────

def run_params(cf: dict) -> dict:
    """Everything a fold depends on (fold.parquet metadata; a change refolds)."""
    return {"model_type": cf.get("model_type", "alphafold2_multimer_v3"),
            "num_models": int(cf.get("num_models", 5)), "num_seeds": int(cf.get("num_seeds", 1)),
            "random_seed": int(cf.get("random_seed", 0)), "num_recycle": cf.get("num_recycle"),
            "recycle_early_stop_tolerance": cf.get("recycle_early_stop_tolerance"),
            "use_dropout": bool(cf.get("use_dropout", False)),
            "calc_extra_ptm": bool(cf.get("calc_extra_ptm", True)),
            "fast_kernels": bool(cf.get("fast_kernels", False)),
            "kernel_backend": cf.get("kernel_backend", "auto"),
            "compile_mode": cf.get("compile_mode", "tuned"),
            "extra_args": [str(x) for x in cf.get("extra_args") or []],
            "msa": msa_settings(cf)}


def _batch_cmd(cf: dict, in_dir: Path, out_dir: Path) -> list[str]:
    rp = run_params(cf)
    cmd = [cf.get("colabfold_batch", "colabfold_batch"), str(in_dir), str(out_dir),
           "--model-type", rp["model_type"], "--num-models", str(rp["num_models"]),
           "--num-seeds", str(rp["num_seeds"]), "--random-seed", str(rp["random_seed"]),
           "--sort-queries-by", "length", "--skip-output", "plots,pae_json"]
    if rp["num_recycle"] is not None:
        cmd += ["--num-recycle", str(int(rp["num_recycle"]))]
    if rp["recycle_early_stop_tolerance"] is not None:
        cmd += ["--recycle-early-stop-tolerance", str(float(rp["recycle_early_stop_tolerance"]))]
    if rp["use_dropout"]:
        cmd.append("--use-dropout")
    if rp["calc_extra_ptm"]:
        cmd.append("--calc-extra-ptm")
    if rp["fast_kernels"]:       # fused kernels: Pallas on sm_80+, fp16 cuda_legacy below (V100/P100/RTX)
        cmd += ["--use-fast-kernels", "--kernel-backend", rp["kernel_backend"],
                "--compile-mode", rp["compile_mode"]]
    if cf.get("data_dir"):
        cmd += ["--data", str(cf["data_dir"])]
    return cmd + rp["extra_args"]


def _pair_matrix(scores: dict, letters: list[str], ids: list[str]) -> list | None:
    """Chain x chain ipTM in spec chain order: --calc-extra-ptm's pairwise_iptm
    ({"A-B": x}) with per_chain_ptm on the diagonal; else, for 2 chains, the
    complex ipTM (it IS their interface score); else None."""
    k = len(ids)
    pw = scores.get("pairwise_iptm")
    if isinstance(pw, dict) and pw:
        mat = np.full((k, k), np.nan, dtype=np.float32)
        pos = {c: i for i, c in enumerate(letters)}
        for key, v in pw.items():
            a, _, b = key.partition("-")
            if a in pos and b in pos and v is not None:
                mat[pos[a], pos[b]] = mat[pos[b], pos[a]] = float(v)
        for c, v in (scores.get("per_chain_ptm") or {}).items():
            if c in pos and v is not None:
                mat[pos[c], pos[c]] = float(v)
        return mat
    if k == 2 and scores.get("iptm") is not None:
        x = float(scores["iptm"])
        return np.array([[np.nan, x], [x, np.nan]], dtype=np.float32)
    return None


_ATOM_SITE = ("group_PDB", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id",
              "label_asym_id", "label_seq_id", "pdbx_PDB_ins_code", "auth_seq_id", "auth_comp_id",
              "auth_asym_id", "auth_atom_id", "B_iso_or_equiv", "occupancy", "label_entity_id",
              "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num", "id")


def _mmcif(model) -> str:
    """The `_atom_site` loop ESMFold2's to_mmcif() writes (same columns, label =
    auth ids, label_seq_id 1..L per chain, pLDDT 0-100 as B-factor), so every
    reader downstream -- ipsae.py's own column parser included -- sees one
    layout. gemmi's mmCIF writer would leave label_seq_id "." for a PDB
    without SEQRES (ColabFold's), which ipsae.py takes for ligand atoms."""
    lines = ["data_pred", "#", "loop_", *(f"_atom_site.{c}" for c in _ATOM_SITE)]
    serial = 0
    for ent, ch in enumerate(model, 1):
        for k, res in enumerate(ch, 1):
            for at in res:
                serial += 1
                p = at.pos
                lines.append(" ".join(map(str, (
                    "ATOM", at.element.name, at.name, ".", res.name, ch.name, k, ".", k, res.name,
                    ch.name, at.name, f"{at.b_iso:.2f}", f"{at.occ:.1f}", ent,
                    f"{p.x:.3f}", f"{p.y:.3f}", f"{p.z:.3f}", 1, serial))))
    return "\n".join(lines + ["#", ""])


def convert(pair: str, spec: dict, run_dir: Path, out_dir: Path, n_samples: int,
            params: dict | None = None) -> list[Path]:
    """ColabFold's per-model PDB + scores JSON of `pair` in run_dir ->
    out_dir/sample_<i>.cif.gz + fold.parquet (written last, atomically)."""
    import gemmi

    job = safe_filename(pair)
    found = []
    for f in run_dir.glob(f"{job}_scores_rank_*.json"):
        mt = _SCORES.search(f.name[len(job):])
        if mt:
            found.append((int(mt.group(2)), int(mt.group(1)), f))
    found.sort()
    if len(found) != n_samples:
        raise RuntimeError(f"{pair}: {len(found)} ColabFold model(s) in {run_dir}, expected {n_samples} "
                           "(colabfold.num_models x num_seeds)")

    ids = [c["id"] for c in protein_chains(spec)]
    letters = [chr(ord("A") + i) for i in range(len(ids))]   # ColabFold's input-order chain names
    rename = dict(zip(letters, ids))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, written = [], []
    for i, (seed, model, scores_path) in enumerate(found):
        pdb = scores_path.with_name(scores_path.name.replace("_scores_", "_unrelaxed_", 1)).with_suffix(".pdb")
        st = gemmi.read_structure(str(pdb))
        st.remove_ligands_and_waters()
        model0 = st[0]
        lens = [len(c["sequence"]) for c in protein_chains(spec)]
        if [len(ch) for ch in model0] != lens:
            raise RuntimeError(f"{pdb.name}: chain lengths {[len(ch) for ch in model0]}, "
                               f"the spec's are {lens}")
        for ch in model0:
            ch.name = rename.get(ch.name, ch.name)
        tok_chain = [ch.name for ch in model0 for _ in ch]
        # 0-based per chain, ESMFold2's residue_index convention (ColabFold numbers from 1)
        tok_res = [k for ch in model0 for k in range(len(ch))]
        cif_path = out_dir / f"sample_{i}.cif.gz"
        with gzip.GzipFile(cif_path, "wb", mtime=0) as fh:          # mtime=0: byte-reproducible
            fh.write(_mmcif(model0).encode())

        scores = json.loads(scores_path.read_text())
        plddt = np.asarray(scores["plddt"], dtype=np.float32) / 100.0
        pae = np.asarray(scores["pae"], dtype=np.float32)
        if len(tok_chain) != plddt.size or pae.shape != (plddt.size, plddt.size):
            raise RuntimeError(f"{scores_path.name}: {plddt.size} pLDDT / {pae.shape} PAE vs "
                               f"{len(tok_chain)} residues in {pdb.name}")
        chain_idx = {c: k for k, c in enumerate(ids)}
        rows.append(fold_row(
            pair, seed, i, f"{pair}/raw/{cif_path.name}",
            ptm=scores.get("ptm"), iptm=scores.get("iptm"), plddt=plddt, pae=pae,
            chain_id=[chain_idx[c] for c in tok_chain], chain_lookup=dict(enumerate(ids)),
            residue_index=tok_res, pair_chains_iptm=_pair_matrix(scores, letters, ids),
        ))
        written.append(cif_path)
        print(f"[colabfold] {pair} sample {i} (seed {seed}, model {model}): "
              f"ptm={scores.get('ptm')} iptm={scores.get('iptm')}", flush=True)
    write_fold(out_dir / FOLD_FILE, rows, run_params=params, engine="colabfold")
    return written


def fold_batch(pairs: list[str], out_root: Path) -> int:
    """One colabfold_batch call over the chunk's pairs (crash-safe, see the module docstring)."""
    from nanopulldown.fold.esmfold2 import _complete, _link_tree

    root = Path(out_root)
    cfg = load_config()
    cf = cfg["colabfold"]
    n_samples = engine_samples(cfg)
    params = run_params(cf)
    specs = load_pairs(REPO_ROOT)
    cache_root = root / CACHE_DIR

    def done(p):   # complete AND folded with these settings
        d = cache_root / p
        if not _complete(d, n_samples):
            return False
        import pyarrow.parquet as pq
        meta = pq.read_schema(d / FOLD_FILE).metadata or {}
        return json.loads(meta.get(b"colabfold", b"{}")) == json.loads(json.dumps(params))

    params_json = json.loads(json.dumps(params))

    def harvest(p, run_dir) -> bool | None:
        """Convert p's finished ColabFold run in run_dir into the cache. None: not
        there; False: there but unconvertible -- its done marker is removed, so
        the next attempt folds it again instead of skipping it forever."""
        job = safe_filename(p)
        if not (run_dir / f"{job}.done.txt").exists():
            return None
        cache = cache_root / p
        shutil.rmtree(cache, ignore_errors=True)
        try:
            convert(p, specs[p], run_dir, cache, n_samples, params)
        except Exception as e:                             # noqa: BLE001
            print(f"[colabfold fold] {p}: conversion FAILED: {e}", file=sys.stderr, flush=True)
            shutil.rmtree(cache, ignore_errors=True)
            (run_dir / f"{job}.done.txt").unlink()
            return False
        if not cf.get("keep_raw_outputs", False):          # PDB/JSON copies (the JSON PAE is large)
            for f in run_dir.glob(f"{job}_*"):
                f.unlink()
        (run_dir / f"{job}.done.txt").unlink(missing_ok=True)   # the cache is the record now
        return True

    def same_params(run_dir) -> bool:
        stamp = run_dir / "nanopulldown_params.json"
        return stamp.exists() and json.loads(stamp.read_text()) == params_json

    todo = [p for p in pairs if not done(p)]
    # pairs ColabFold finished in an earlier job that died before converting them
    # (walltime, preemption) -- possibly another chunk's run dir: convert, don't refold
    for d in sorted((root / RUN_DIR).glob("*/")):
        if same_params(d):
            todo = [p for p in todo if not harvest(p, d)]
    print(f"[colabfold fold] {len(pairs)} pair(s) in chunk, {len(todo)} to fold, "
          f"{len(pairs) - len(todo)} already done", flush=True)
    failed = []
    if todo:
        missing = [p for p in todo if not _valid_a3m(a3m_path(specs[p], cf), len(protein_chains(specs[p])))]
        if missing:
            print(f"[colabfold fold] no MSA for {missing} -- run the colabfold_msa rule "
                  "(`nanopulldown colabfold msa`) first", file=sys.stderr)
            return 1
        chunk = hashlib.sha1(" ".join(sorted(todo)).encode()).hexdigest()[:12]
        run_dir = root / RUN_DIR / chunk
        if run_dir.exists() and not same_params(run_dir):
            shutil.rmtree(run_dir)           # other settings: ColabFold's done markers no longer apply
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "nanopulldown_params.json").write_text(json.dumps(params, sort_keys=True))
        in_dir = Path(tempfile.mkdtemp(prefix="in_", dir=run_dir))
        for p in todo:                       # jobname = file stem = the pair
            os.symlink(a3m_path(specs[p], cf).resolve(), in_dir / f"{safe_filename(p)}.a3m")
        t0 = time.time()
        rc = _run(_batch_cmd(cf, in_dir, run_dir))
        shutil.rmtree(in_dir, ignore_errors=True)
        print(f"[colabfold fold] colabfold_batch exit {rc} after {time.time() - t0:.0f}s", flush=True)
        failed = [p for p in todo if not harvest(p, run_dir)]

    for p in pairs:
        if _complete(cache_root / p, n_samples):
            _link_tree(cache_root / p, root / p / "raw")
    if failed:
        print(f"[colabfold fold] {len(failed)} pair(s) failed: {failed}", file=sys.stderr)
        return 1
    return 0


def fold_main() -> int:
    ap = argparse.ArgumentParser(description="fold bait/prey pairs with ColabFold (AlphaFold2-multimer)")
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out-root", default="results", help="results root")
    a = ap.parse_args()
    return fold_batch(a.pairs, Path(a.out_root))


def convert_main() -> int:
    ap = argparse.ArgumentParser(description="ColabFold outputs of one pair -> sample_<i>.cif.gz + fold.parquet")
    ap.add_argument("--pair", required=True)
    ap.add_argument("--run-dir", required=True, help="colabfold_batch's result directory")
    ap.add_argument("--out-dir", required=True, help="e.g. results/<pair>/raw")
    a = ap.parse_args()
    cfg = load_config()
    convert(a.pair, load_pairs(REPO_ROOT)[a.pair], Path(a.run_dir), Path(a.out_dir),
            engine_samples(cfg), run_params(cfg["colabfold"]))
    return 0


if __name__ == "__main__":
    sys.exit(fold_main())
