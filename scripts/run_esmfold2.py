#!/usr/bin/env python3
"""
scripts/run_esmfold2.py — Stage 2a
====================================
Fold one bait/prey pair with ESMFold2-Fast on Apple Silicon (MLX/Metal): one
random seed, `esmfold2.num_diffusion_samples` diffusion samples (5 by
default), no MSA, no templates -- the -Fast checkpoint ignores an MSA even
if one were passed, so simply never passing one is the "no MSA" requirement.

Writes, per pair, the same "raw sprawl before compression" shape
ab_initio_pipeline uses (see compress_abcfold_metadata.py's own docstring
for the AF3-style precedent this mirrors):
  results/<pair>/raw/sample_<i>.cif                structure
  results/<pair>/raw/sample_<i>_confidences.json   ptm, iptm, plddt, pae,
                                                    pair_chains_iptm, chain_id
                                                    (per token), chain_lookup
scripts/compress_esmfold2_metadata.py then turns that sprawl into
results/metadata/<pair>/{model_metadata.parquet, arrays.h5}.

*** IMPLEMENTATION NOTE *** the notebook this is based on
(/Users/ereboul/projects/esmfold2_test/esmfold2_local_applesilicon.ipynb)
only exercises the MLX (Apple Silicon) path through builder.fold(); this
script assumes the MLX ESMFold2Model's forward() populates
output["pae"]/output["pair_chains_iptm"] exactly like the Torch/CUDA path
read in esm/models/esmfold2/processor.py:271-283 ("the same code" per the
notebook) -- confirm this once against a live run before trusting the
rescoring stage built on top of it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import load_config, load_pair, protein_chains  # noqa: E402


def _load_model_and_builder(checkpoint: str):
    from mlx_lm.models.esmfold2 import ESMFold2Model
    from esm.models.esmfold2 import ESMFold2InputBuilder

    model = ESMFold2Model.from_pretrained(checkpoint)
    builder = ESMFold2InputBuilder()
    return model, builder


def fold_pair(spec: dict, cfg: dict, out_dir: Path) -> list[Path]:
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    ef_cfg = cfg["esmfold2"]
    model, builder = _load_model_and_builder(ef_cfg["checkpoint"])

    spi = StructurePredictionInput(sequences=[
        ProteinInput(id=ch["id"], sequence=ch["sequence"])
        for ch in protein_chains(spec)
    ])

    results = builder.fold(
        model, spi,
        num_loops=int(ef_cfg["num_loops"]),
        num_sampling_steps=int(ef_cfg["num_sampling_steps"]),
        num_diffusion_samples=int(ef_cfg["num_diffusion_samples"]),
        seed=int(ef_cfg["seed"]),
    )
    if not isinstance(results, list):
        results = [results]

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, result in enumerate(results):
        cif_path = out_dir / f"sample_{i}.cif"
        cif_path.write_text(result.complex.to_mmcif())

        chain_lookup = dict(result.complex.metadata.chain_lookup)
        payload = {
            "sample_index": i,
            "seed": int(ef_cfg["seed"]),
            "ptm": result.ptm,
            "iptm": result.iptm,
            "plddt": np.asarray(result.plddt).astype(np.float32).tolist(),
            "pae": (np.asarray(result.pae).astype(np.float32).tolist()
                    if result.pae is not None else None),
            "pair_chains_iptm": (np.asarray(result.pair_chains_iptm).astype(np.float32).tolist()
                                  if result.pair_chains_iptm is not None else None),
            "chain_id": np.asarray(result.complex.chain_id).astype(int).tolist(),
            "chain_lookup": {str(k): v for k, v in chain_lookup.items()},
            "residue_index": (np.asarray(result.residue_index).astype(int).tolist()
                               if result.residue_index is not None else None),
        }
        json_path = out_dir / f"sample_{i}_confidences.json"
        json_path.write_text(json.dumps(payload))
        written.append(cif_path)
        print(f"[run_esmfold2] sample {i}: ptm={result.ptm:.3f} iptm={result.iptm}"
              if result.ptm is not None else f"[run_esmfold2] sample {i} written")

    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--done", required=True, help="sentinel file to touch on success")
    args = ap.parse_args()

    cfg = load_config()
    spec = load_pair(args.pair, Path(args.spec).resolve().parent.parent)

    fold_pair(spec, cfg, Path(args.out_dir))
    Path(args.done).write_text("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
