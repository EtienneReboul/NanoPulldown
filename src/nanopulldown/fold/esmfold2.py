"""
nanopulldown.fold.esmfold2 — Stage 2a
=====================================
Fold bait/prey pairs with ESMFold2-Fast on Apple Silicon (MLX/Metal): one
random seed, `esmfold2.num_diffusion_samples` diffusion samples (5 by
default), no MSA, no templates -- the -Fast checkpoint ignores an MSA even
if one were passed, so simply never passing one is the "no MSA" requirement.

Writes, per pair, into --out-dir (results/<pair>/raw/):
  sample_<i>.cif.gz   structure, gzipped (gemmi / ChimeraX read it as-is)
  fold.parquet        one row per sample: ptm, iptm, plddt, pae (float16),
                      per-token chain/residue index, pair_chains_iptm -- the
                      schema lives in nanopulldown.core.fold_table. Written LAST and
                      atomically, so it doubles as the "pair folded" marker.
No intermediate JSON: ESMFold2's result objects go straight to Parquet.

Two modes (`nanopulldown fold`):
  --pair P --out-dir D       one pair, one model load, straight into D.
  --pairs P... --out-root R  one Snakemake "chunk" (esmfold2.pairs_per_load):
      the model is loaded ONCE and the pairs folded in a loop, instead of one
      process -- and one ~25 GB transient model load, with its swap spike --
      per pair. Output per pair is identical to --pair.
      Crash safety: results are written to <out-root>/_fold_cache/<pair>/
      first and hard-linked into <out-root>/<pair>/raw/. Snakemake deletes a
      failed job's declared outputs (the links) but not the cache, so a rerun
      after a crash -- e.g. the Metal allocation abort, a C++ abort no
      `except` can catch -- relinks the finished pairs instantly and only
      folds the rest. The cache holds exactly the final outputs and nothing
      else (hard links: no extra disk), so it costs nothing to keep.

*** IMPLEMENTATION NOTE *** the notebook this is based on
(/Users/ereboul/projects/esmfold2_test/esmfold2_local_applesilicon.ipynb)
only exercises the MLX (Apple Silicon) path through builder.fold(); this
script assumes the MLX ESMFold2Model's forward() populates
output["pae"]/output["pair_chains_iptm"] exactly like the Torch/CUDA path
read in esm/models/esmfold2/processor.py:271-283 ("the same code" per the
notebook) -- confirm this once against a live run before trusting the
rescoring stage built on top of it.

*** ONE MODEL AT A TIME -- REQUIRED, NOT OPTIONAL ***
The -Fast checkpoint loads ~25GB into Apple Silicon's unified memory (shared
between CPU and GPU) per the reference notebook. Snakemake's `--cores` has
no idea that's a single shared GPU resource -- `--cores 4` will happily
dispatch 4 concurrent run_esmfold2 jobs, each loading its own ~25GB copy,
which is how this actually OOM-crashed the Mac once already. The Snakefile's
`resources: gpu=1` on this rule is what now prevents that, and it ONLY works
if the run is launched with `--resources gpu=1` (see README) -- it is a
no-op otherwise. There is no other guard: always pass that flag.
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from nanopulldown.core.config import REPO_ROOT, load_config
from nanopulldown.core.fold_table import FOLD_FILE, fold_row, write_fold
from nanopulldown.core.specs import load_pair, protein_chains


def _load_model_and_builder(checkpoint: str):
    from esm.models.esmfold2 import ESMFold2InputBuilder
    from mlx_lm.models.esmfold2 import ESMFold2Model

    model = ESMFold2Model.from_pretrained(checkpoint)
    builder = ESMFold2InputBuilder()
    return model, builder


def fold_pair(spec: dict, cfg: dict, out_dir: Path, loaded=None) -> list[Path]:
    """`loaded` = an already-loaded (model, builder) -- nanopulldown.fold.esmfold2
    loads once and folds many pairs; None loads (and discards) per call."""
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    ef_cfg = cfg["esmfold2"]
    model, builder = loaded or _load_model_and_builder(ef_cfg["checkpoint"])

    spi = StructurePredictionInput(sequences=[
        ProteinInput(id=ch["id"], sequence=ch["sequence"])
        for ch in protein_chains(spec)
    ])

    n_requested = int(ef_cfg["num_diffusion_samples"])
    results = builder.fold(
        model, spi,
        num_loops=int(ef_cfg["num_loops"]),
        num_sampling_steps=int(ef_cfg["num_sampling_steps"]),
        num_diffusion_samples=n_requested,
        seed=int(ef_cfg["seed"]),
    )
    print(f"[run_esmfold2] builder.fold() returned {type(results).__name__} "
          f"of length {len(results) if hasattr(results, '__len__') else 'n/a'} "
          f"(requested {n_requested})", flush=True)
    if isinstance(results, list) and len(results) != n_requested:
        # Seen in practice: under GPU memory pressure, builder.fold() can
        # silently return fewer samples than requested with no exception --
        # exit 0, no traceback, the loop below just has fewer items. Fail
        # loud here instead of writing a partial `raw/` dir that Snakemake
        # would only catch later (and less clearly) via MissingOutputException.
        raise RuntimeError(
            f"builder.fold() returned {len(results)} sample(s), expected "
            f"{n_requested} -- likely Metal/unified-memory pressure "
            "(silently dropped samples rather than raising). Try again, or "
            "lower esmfold2.num_diffusion_samples in config.yaml."
        )
    if not isinstance(results, list):
        results = [results]

    out_dir.mkdir(parents=True, exist_ok=True)
    pair = spec["name"]
    written, rows = [], []
    for i, result in enumerate(results):
        cif_path = out_dir / f"sample_{i}.cif.gz"
        # mtime=0: identical structures give byte-identical files
        with gzip.GzipFile(cif_path, "wb", mtime=0) as fh:
            fh.write(result.complex.to_mmcif().encode())

        rows.append(fold_row(
            # always the results/<pair>/raw/ location, even when written to
            # run_esmfold2_batch.py's cache dir (hard-linked there afterwards)
            pair, ef_cfg["seed"], i, f"{pair}/raw/{cif_path.name}",
            ptm=result.ptm, iptm=result.iptm, plddt=result.plddt, pae=result.pae,
            chain_id=result.complex.chain_id,
            chain_lookup=dict(result.complex.metadata.chain_lookup),
            residue_index=result.residue_index,
            pair_chains_iptm=result.pair_chains_iptm,
        ))
        written.append(cif_path)
        print(f"[run_esmfold2] sample {i}: ptm={result.ptm:.3f} iptm={result.iptm}"
              if result.ptm is not None else f"[run_esmfold2] sample {i} written")

    write_fold(out_dir / FOLD_FILE, rows, run_params={
        k: ef_cfg[k] for k in ("checkpoint", "num_loops", "num_sampling_steps",
                               "num_diffusion_samples", "seed")})
    return written


def _link_tree(src: Path, dst: Path) -> None:
    """Make dst hold exactly src's files (hard links): stray files in dst -- e.g.
    sample_1/2 left by an earlier 3-sample run -- are removed first."""
    dst.mkdir(parents=True, exist_ok=True)
    keep = {f.name for f in src.iterdir()}
    for f in dst.iterdir():
        if f.is_file() and f.name not in keep:
            f.unlink()
    for f in src.iterdir():
        t = dst / f.name
        t.unlink(missing_ok=True)
        try:
            os.link(f, t)
        except OSError:
            shutil.copy2(f, t)


def _complete(d: Path, n_samples: int) -> bool:
    """fold.parquet (written last, atomically) AND every sample's structure."""
    return (d / FOLD_FILE).exists() and all(
        (d / f"sample_{i}.cif.gz").exists() for i in range(n_samples))


def fold_batch(pairs: list[str], out_root: Path) -> int:
    """Load the model once and fold `pairs` in a loop (crash-safe, see the module docstring)."""
    root = Path(out_root)
    cache_root = root / "_fold_cache"

    import mlx.core as mx
    clear = getattr(mx, "clear_cache", None) or mx.metal.clear_cache
    peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    reset = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
    active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
    cfg = load_config()
    n_samples = int(cfg["esmfold2"]["num_diffusion_samples"])

    todo = [p for p in pairs if not _complete(cache_root / p, n_samples)]
    print(f"[fold_batch] {len(pairs)} pair(s) in chunk, {len(todo)} to fold, "
          f"{len(pairs) - len(todo)} already done", flush=True)

    loaded, failed = None, []
    if todo:
        t0 = time.time()
        loaded = _load_model_and_builder(cfg["esmfold2"]["checkpoint"])
        print(f"[fold_batch] model loaded once in {time.time() - t0:.0f}s, "
              f"{active() / 2**30:.1f} GiB resident", flush=True)

    for i, p in enumerate(todo, 1):
        cache = cache_root / p
        shutil.rmtree(cache, ignore_errors=True)
        t0 = time.time()
        reset()
        try:
            fold_pair(load_pair(p, REPO_ROOT), cfg, cache, loaded)
            print(f"[fold_batch] {i}/{len(todo)} {p}: {time.time() - t0:.0f}s, "
                  f"peak {peak() / 2**30:.1f} GiB", flush=True)
        except Exception:                                  # noqa: BLE001
            failed.append(p)
            print(f"[fold_batch] {i}/{len(todo)} {p} FAILED:\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)
        finally:
            clear()                                        # return the fold's buffers; keep the weights

    for p in pairs:
        if _complete(cache_root / p, n_samples):
            _link_tree(cache_root / p, root / p / "raw")

    if failed:
        print(f"[fold_batch] {len(failed)} pair(s) failed: {failed}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="fold bait/prey pairs with ESMFold2-Fast (MLX)")
    one = ap.add_mutually_exclusive_group(required=True)
    one.add_argument("--pair", help="fold ONE pair straight into --out-dir")
    one.add_argument("--pairs", nargs="+",
                     help="fold several pairs with ONE model load, crash-safe via <out-root>/_fold_cache")
    ap.add_argument("--out-dir", help="with --pair: results/<pair>/raw")
    ap.add_argument("--out-root", default="results", help="with --pairs: results root")
    args = ap.parse_args()

    if args.pairs:
        return fold_batch(args.pairs, Path(args.out_root))
    if not args.out_dir:
        ap.error("--pair needs --out-dir")
    fold_pair(load_pair(args.pair), load_config(), Path(args.out_dir))
    return 0

if __name__ == "__main__":
    sys.exit(main())
