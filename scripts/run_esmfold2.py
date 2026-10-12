#!/usr/bin/env python3
"""
scripts/run_esmfold2.py — Stage 2a
====================================
Fold one bait/prey pair with ESMFold2-Fast on Apple Silicon (MLX/Metal): one
random seed, `esmfold2.num_diffusion_samples` diffusion samples (5 by
default), no MSA, no templates -- the -Fast checkpoint ignores an MSA even
if one were passed, so simply never passing one is the "no MSA" requirement.

Writes, per pair, into --out-dir (results/<pair>/raw/):
  sample_<i>.cif.gz   structure, gzipped (gemmi / ChimeraX read it as-is)
  fold.parquet        one row per sample: ptm, iptm, plddt, pae (float16),
                      per-token chain/residue index, pair_chains_iptm -- the
                      schema lives in scripts/fold_io.py. Written LAST and
                      atomically, so it doubles as the "pair folded" marker.
No intermediate JSON: ESMFold2's result objects go straight to Parquet.

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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fold_io import FOLD_FILE, fold_row, write_fold  # noqa: E402
from lib_pipeline import load_config, load_pair, protein_chains  # noqa: E402


def _load_model_and_builder(checkpoint: str):
    from mlx_lm.models.esmfold2 import ESMFold2Model
    from esm.models.esmfold2 import ESMFold2InputBuilder

    model = ESMFold2Model.from_pretrained(checkpoint)
    builder = ESMFold2InputBuilder()
    return model, builder


def fold_pair(spec: dict, cfg: dict, out_dir: Path, loaded=None) -> list[Path]:
    """`loaded` = an already-loaded (model, builder) -- scripts/run_esmfold2_batch.py
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = load_config()
    spec = load_pair(args.pair)

    fold_pair(spec, cfg, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
