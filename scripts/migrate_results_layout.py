#!/usr/bin/env python3
"""
scripts/migrate_results_layout.py — ONE-OFF
=============================================
Converts a results tree written before the fold.parquet / gzipped-structure
layout into the current one, in place. Safe to rerun (finished steps are
skipped) and DRY-RUN by default -- pass --apply to change anything.

Per pair (results/<pair>/):
  raw/sample_<i>.cif                -> raw/sample_<i>.cif.gz
  metadata/<pair>/{model_metadata.parquet, arrays.h5}
    (or, if never compressed, the raw/_fold_cache confidences JSON)
                                    -> raw/fold.parquet          (scripts/fold_io.py)
  raw/prediction.done, *_confidences.json                        -> deleted
  _fold_cache/<pair>/               -> exactly raw/'s files, hard-linked
                                       (drops the leftover confidences JSON)
  minimized/sample_<i>.pdb          -> minimized/sample_<i>.pdb.gz
  minimized/sample_<i>_energy.csv   -> sample_<i>_energy.parquet (if missing)
  plip/sample_<i>_report/sample_<i>_protonated.pdb               -> deleted
  plip/.../sample_<i>.cxc                                        -> points at the .pdb.gz
  plip_summary.csv + plip/.../csv/*.csv + cxc-config.json        -> plip_contacts.parquet
With --annotation-root (data/annotation/<pair>/):
  domains.raw.tsv, disorder.tsv     -> deleted (merge_annotation and the domain
                                       bar figures now read the per-sequence cache)

Every converted file is verified (gzip round-trip byte-identical; fold.parquet
arrays bit-identical to arrays.h5's float16) BEFORE its source is deleted,
and inherits its source's mtime so Snakemake doesn't see it as new. Snakemake
still notices the rules' changed code/inputs, so run the first time after
migrating with `--rerun-triggers mtime` (check with `-n` first).

Env: envs/metrics.yaml (numpy, pandas, pyarrow, h5py).

Usage:
  python scripts/migrate_results_layout.py --results-root results            # dry run
  python scripts/migrate_results_layout.py --results-root results --apply
  python scripts/migrate_results_layout.py --results-root results_q9m548_e3_fbox --apply
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_plip_summaries import read_summary, write_contacts  # noqa: E402
from fold_io import FOLD_FILE, fold_row, read_fold, write_fold  # noqa: E402
from minimize_openmm import ENERGY_COLUMN_DESCRIPTIONS, ENERGY_TABLE_DESCRIPTION  # noqa: E402

SAMPLE = re.compile(r"^sample_(\d+)[._]")


class Ctx:
    def __init__(self, apply: bool):
        self.apply = apply
        self.freed = 0
        self.actions: list[str] = []
        self._links_seen: dict[int, int] = {}

    def unlink(self, p: Path) -> None:
        """Count bytes only when the file's LAST link goes (cache/ and raw/ share
        inodes); a dry run tracks the links it would have removed."""
        st = p.stat()
        seen = self._links_seen[st.st_ino] = self._links_seen.get(st.st_ino, 0) + 1
        if (st.st_nlink == 1) if self.apply else (seen == st.st_nlink):
            self.freed += st.st_size
        if self.apply:
            p.unlink()


def _gzip_file(src: Path, ctx: Ctx) -> None:
    """src -> src.gz, verified byte-identical, mtime kept, then src removed."""
    dst = src.with_name(src.name + ".gz")
    data = src.read_bytes()
    if ctx.apply:
        if not (dst.exists() and gzip.decompress(dst.read_bytes()) == data):
            tmp = dst.with_name(f".{dst.name}.tmp")
            with gzip.GzipFile(tmp, "wb", mtime=0) as fh:
                fh.write(data)
            if gzip.decompress(tmp.read_bytes()) != data:
                tmp.unlink()
                raise RuntimeError(f"gzip round-trip mismatch for {src}")
            tmp.replace(dst)
        st = src.stat()
        os.utime(dst, (st.st_atime, st.st_mtime))
        ctx.freed -= dst.stat().st_size
    else:
        ctx.freed -= len(gzip.compress(data, compresslevel=6))
    ctx.actions.append(f"gzip {src.name}")
    ctx.unlink(src)


# ── fold.parquet from the legacy sources ─────────────────────────────────

def _rows_from_metadata(pair: str, meta_dir: Path) -> tuple[list[dict], dict]:
    import h5py
    import pandas as pd

    meta = pd.read_parquet(meta_dir / "model_metadata.parquet").sort_values("sample_index")
    rows, ref = [], {}
    with h5py.File(meta_dir / "arrays.h5", "r") as h5:
        topo = h5.get(f"_topology/{pair}/chain_id")
        topo_lookup = json.loads(topo.attrs.get("chain_lookup", "{}")) if topo is not None else {}
        for _, r in meta.iterrows():
            extra = json.loads(r["extra_json"]) if r["extra_json"] else {}
            key = r["array_key"]
            pae = h5[f"{key}/pae"][()] if f"{key}/pae" in h5 else None
            plddt = h5[f"{key}/plddt"][()] if f"{key}/plddt" in h5 else np.zeros(0, np.float16)
            chain_id = extra.get("chain_id") if extra.get("chain_id") is not None else topo[()]
            i = int(r["sample_index"])
            rows.append(fold_row(
                pair, extra.get("seed", r["seed"]), i, f"{pair}/raw/sample_{i}.cif.gz",
                ptm=extra.get("ptm", r["ptm"]), iptm=extra.get("iptm", r["iptm"]),
                plddt=plddt, pae=pae, chain_id=chain_id,
                chain_lookup=extra.get("chain_lookup") or topo_lookup,
                residue_index=extra.get("residue_index"),
                pair_chains_iptm=extra.get("pair_chains_iptm")))
            if r["mean_plddt"] == r["mean_plddt"]:            # keep the full-precision mean
                rows[-1]["mean_plddt"] = float(r["mean_plddt"])
            ref[i] = (pae, plddt)
    return rows, ref


def _rows_from_json(pair: str, jsons: dict[int, Path]) -> tuple[list[dict], dict]:
    rows, ref = [], {}
    for i, jp in sorted(jsons.items()):
        d = json.loads(jp.read_text())
        rows.append(fold_row(
            pair, d.get("seed", 0), i, f"{pair}/raw/sample_{i}.cif.gz",
            ptm=d.get("ptm"), iptm=d.get("iptm"), plddt=d["plddt"], pae=d.get("pae"),
            chain_id=d["chain_id"], chain_lookup=d["chain_lookup"],
            residue_index=d.get("residue_index"), pair_chains_iptm=d.get("pair_chains_iptm")))
        ref[i] = (None if d.get("pae") is None else np.asarray(d["pae"], np.float16),
                  np.asarray(d["plddt"], np.float16))
    return rows, ref


def _verify_fold(path: Path, ref: dict) -> None:
    got = {r["sample_index"]: r for r in read_fold(path)}
    if set(got) != set(ref):
        raise RuntimeError(f"{path}: samples {sorted(got)} != source {sorted(ref)}")
    for i, (pae, plddt) in ref.items():
        if pae is not None and not np.array_equal(
                got[i]["pae"], np.asarray(pae, np.float16).astype(np.float32)):
            raise RuntimeError(f"{path}: sample {i} pae differs from source")
        if not np.array_equal(got[i]["plddt"], np.asarray(plddt, np.float16).astype(np.float32)):
            raise RuntimeError(f"{path}: sample {i} plddt differs from source")


def migrate_fold(pair: str, root: Path, meta_root: Path, ctx: Ctx) -> None:
    raw, cache, meta_dir = root / pair / "raw", root / "_fold_cache" / pair, meta_root / pair
    fold = raw / FOLD_FILE
    jsons = {}
    for d in (cache, raw):
        for jp in (d.glob("sample_*_confidences.json") if d.exists() else []):
            jsons.setdefault(int(SAMPLE.match(jp.name).group(1)), jp)

    if not fold.exists():
        if (meta_dir / "model_metadata.parquet").exists() and (meta_dir / "arrays.h5").exists():
            rows, ref = _rows_from_metadata(pair, meta_dir)
            src = "metadata"
        elif jsons:
            rows, ref = _rows_from_json(pair, jsons)
            src = "json"
        else:
            raise RuntimeError(f"{pair}: no fold.parquet, metadata or confidences json")
        ctx.actions.append(f"fold.parquet from {src} ({len(rows)} sample(s))")
        if ctx.apply:
            write_fold(fold, rows)
            _verify_fold(fold, ref)
            ref_mtime = min((p.stat().st_mtime for p in raw.glob("sample_*.cif*")),
                            default=fold.stat().st_mtime)
            os.utime(fold, (ref_mtime, ref_mtime))

    for cif in sorted(raw.glob("sample_*.cif")):
        _gzip_file(cif, ctx)
    for f in [raw / "prediction.done", *raw.glob("sample_*_confidences.json")]:
        if f.exists():
            ctx.unlink(f)

    # cache := exactly raw/'s files (hard links)
    if cache.exists():
        for f in cache.iterdir():
            ctx.unlink(f)
    if ctx.apply:
        cache.mkdir(parents=True, exist_ok=True)
        for f in raw.iterdir():
            os.link(f, cache / f.name)
    if meta_dir.exists():
        for f in meta_dir.iterdir():
            ctx.unlink(f)
        if ctx.apply:
            meta_dir.rmdir()


def migrate_minimized(pair: str, root: Path, ctx: Ctx) -> None:
    mdir = root / pair / "minimized"
    if not mdir.exists():
        return
    for pdb in sorted(mdir.glob("sample_*.pdb")):
        _gzip_file(pdb, ctx)
    for csv in sorted(mdir.glob("sample_*_energy.csv")):
        pq_path = csv.with_name(csv.name.replace(".csv", ".parquet"))
        if not pq_path.exists():
            ctx.actions.append(f"energy {csv.name} -> parquet")
            if ctx.apply:
                import pandas as pd
                from parquet_utils import write_parquet_with_metadata

                df = pd.read_csv(csv)
                write_parquet_with_metadata(df, pq_path, table_description=ENERGY_TABLE_DESCRIPTION,
                                            column_descriptions=ENERGY_COLUMN_DESCRIPTIONS)
                if not pd.read_parquet(pq_path).equals(df):
                    raise RuntimeError(f"{pq_path}: round-trip mismatch")
                st = csv.stat()
                os.utime(pq_path, (st.st_atime, st.st_mtime))
        ctx.unlink(csv)


def _retarget(path: Path, pattern: re.Pattern, ctx: Ctx) -> None:
    """Rewrite .../sample_<i>.pdb references to .pdb.gz, keeping the file's mtime."""
    text = path.read_text()
    new = pattern.sub(r"\1.pdb.gz\2", text)
    if new != text:
        ctx.actions.append(f"retarget {path.name}")
        if ctx.apply:
            st = path.stat()
            path.write_text(new)
            os.utime(path, (st.st_atime, st.st_mtime))


def migrate_plip(pair: str, root: Path, ctx: Ctx) -> None:
    pdir = root / pair / "plip"
    if not pdir.exists():
        return
    ref = re.compile(r"(minimized/sample_\d+)\.pdb(\"|\s|$)", re.M)
    for rep in sorted(pdir.glob("sample_*_report")):
        for f in rep.glob("*_protonated.pdb"):
            ctx.unlink(f)
        for f in rep.glob("sample_*.cxc"):
            _retarget(f, ref, ctx)

    old = root / pair / "plip_summary.csv"
    new = root / pair / "plip_contacts.parquet"
    summaries = sorted(pdir.glob("sample_*_report/csv/summary.csv"))
    if old.exists() and not new.exists():
        import pandas as pd

        ctx.actions.append(f"plip_contacts.parquet from {len(summaries)} summary.csv")
        df = pd.concat([read_summary(f, pair) for f in summaries], ignore_index=True)
        legacy = pd.read_csv(old)
        if len(df) != len(legacy) or not (
                df["resnr"].tolist() == legacy["resnr"].tolist()
                and df["interaction_type"].tolist() == legacy["interaction_type"].tolist()):
            raise RuntimeError(f"{pair}: per-sample summaries disagree with plip_summary.csv")
        if ctx.apply:
            write_contacts(df, new)
            st = old.stat()
            os.utime(new, (st.st_atime, st.st_mtime))
    if new.exists() or not ctx.apply:
        for f in [old, *pdir.glob("sample_*_report/csv/*.csv"), *pdir.glob("sample_*_report/cxc-config.json")]:
            if f.exists():
                ctx.unlink(f)
        if ctx.apply:
            for d in pdir.glob("sample_*_report/csv"):
                d.rmdir()


def migrate_annotation(pair_dir: Path, ctx: Ctx) -> None:
    for f in (pair_dir / "domains.raw.tsv", pair_dir / "disorder.tsv"):
        if f.exists():
            ctx.unlink(f)


def migrate_pair(pair: str, root: Path, meta_root: Path, apply: bool) -> tuple[str, int, list[str], str]:
    ctx = Ctx(apply)
    try:
        migrate_fold(pair, root, meta_root, ctx)
        migrate_minimized(pair, root, ctx)
        migrate_plip(pair, root, ctx)
        return pair, ctx.freed, ctx.actions, ""
    except Exception as e:                                     # noqa: BLE001
        return pair, ctx.freed, ctx.actions, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--metadata-root", default=None, help="default: <results-root>/metadata")
    ap.add_argument("--pairs", nargs="*", default=None, help="default: every <results-root>/*/raw/")
    ap.add_argument("--annotation-root", default=None,
                    help="also drop the per-pair annotation TSVs under this dir (e.g. data/annotation)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--apply", action="store_true", help="actually change files (default: dry run)")
    a = ap.parse_args()

    root = Path(a.results_root)
    meta_root = Path(a.metadata_root) if a.metadata_root else root / "metadata"
    pairs = a.pairs or sorted(d.parent.name for d in root.glob("*/raw") if d.is_dir())
    print(f"[migrate] {'APPLY' if a.apply else 'DRY RUN'}: {len(pairs)} pair(s) under {root}", flush=True)

    freed, failed, n_actions = 0, [], 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(migrate_pair, p, root, meta_root, a.apply) for p in pairs]
        for k, f in enumerate(futs, 1):
            pair, b, actions, err = f.result()
            freed += b
            n_actions += len(actions)
            if err:
                failed.append(pair)
                print(f"[migrate] {pair} FAILED: {err}", file=sys.stderr, flush=True)
            if k % 100 == 0 or k == len(futs):
                print(f"[migrate] {k}/{len(futs)} pairs, {freed / 1e9:.2f} GB "
                      f"{'freed' if a.apply else 'would be freed'}", flush=True)

    if a.annotation_root:
        ctx = Ctx(a.apply)
        for d in sorted(Path(a.annotation_root).iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                migrate_annotation(d, ctx)
        freed += ctx.freed
        print(f"[migrate] annotation TSVs under {a.annotation_root}: {ctx.freed / 1e6:.0f} MB "
              f"{'freed' if a.apply else 'would be freed'}")

    all_models = meta_root / "all_models.parquet"
    if all_models.exists():
        freed += all_models.stat().st_size
        if a.apply:
            all_models.unlink()
    if a.apply and meta_root.exists() and not any(meta_root.iterdir()):
        meta_root.rmdir()
    if a.apply:
        cache_root = root / "_fold_cache"
        orphans = [d.name for d in cache_root.iterdir()
                   if d.is_dir() and not (root / d.name / "raw").is_dir()] if cache_root.exists() else []
        if orphans:
            print(f"[migrate] {len(orphans)} _fold_cache dir(s) with no results/<pair>/raw "
                  f"left untouched, e.g. {orphans[:3]}")

    print(f"[migrate] done: {n_actions} action(s), {freed / 1e9:.2f} GB "
          f"{'freed' if a.apply else 'would be freed'}, {len(failed)} failure(s)")
    if not a.apply:
        print("[migrate] dry run only -- rerun with --apply")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
