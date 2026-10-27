#!/usr/bin/env python3
"""
nanopulldown.core.fold_table
===================
The ONE on-disk format for a pair's fold output besides its structures:
results/<pair>/raw/fold.parquet, one row per sample. Written straight from
ESMFold2's result objects by nanopulldown.fold.esmfold2, or parsed from
ColabFold's per-model scores JSON by nanopulldown.fold.colabfold (no
per-sample confidences JSON kept, no separate compression step, no HDF5), and
read by every downstream script through read_fold()/read_fold_scalars() so
nothing else needs to know the column layout.

Structures sit next to it as raw/sample_<i>.cif.gz (gzip: gemmi and ChimeraX
open it as-is; `structure_path` points at it).

Arrays are Arrow lists: pae is FLATTENED row-major (n_tokens x n_tokens --
read_fold() reshapes it), pae/plddt are float16 (pae is a continuous [0,
~31.75) A expectation, plddt a [0,1] confidence; float16 keeps ~0.01-0.03
resolution, far below anything ipSAE/LIS/Pinc resolve). Every column carries
its own description in the Arrow schema (nanopulldown.core.io), plus a
`schema` version and the engine's run parameters (key: esmfold2 | colabfold) at the table level --
inspect with core.io.read_schema_metadata() or Data Wrangler.

pyarrow + numpy only (no pandas needed to write), so the folding env stays lean.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nanopulldown.core.metric_refs import REFS, column_references, references_text

SCHEMA_VERSION = "fold/1"
FOLD_FILE = "fold.parquet"

FOLD_TABLE_DESCRIPTION = (
    "Fold output (ESMFold2-Fast or ColabFold; see the esmfold2/colabfold metadata key) "
    "for one nanopulldown bait/prey pair: one row per sample (ESMFold2 diffusion sample, "
    "or ColabFold model x seed). Scalars, per-token topology and the full "
    "pae/plddt arrays; the structure itself is the gzipped mmCIF at "
    "structure_path. Written by nanopulldown.fold.esmfold2 / nanopulldown.fold.colabfold, "
    "read via nanopulldown.core.fold_table."
)
_F16 = pa.list_(pa.float16())


ENGINES = ("esmfold2", "colabfold")


def fold_engine(cfg: dict) -> str:
    """`fold.engine`: esmfold2 (default) | colabfold. Both write this same table
    and raw/sample_<i>.cif.gz, so nothing downstream depends on the choice."""
    eng = ((cfg.get("fold") or {}).get("engine") or "esmfold2").lower()
    if eng not in ENGINES:
        raise ValueError(f"fold.engine must be one of {ENGINES}, not {eng!r}")
    return eng


def fold_max_tokens(cfg: dict) -> int:
    """The engine's max_tokens (0 = no limit): pairs above it are not expanded/folded."""
    return int(cfg[fold_engine(cfg)].get("max_tokens", 0))


def samples_per_pair(ef_cfg: dict) -> int:
    """Samples (sample_<i>.cif.gz) per pair: esmfold2.num_seeds separate runs
    x esmfold2.num_diffusion_samples each."""
    return int(ef_cfg.get("num_seeds", 1)) * int(ef_cfg["num_diffusion_samples"])


def engine_samples(cfg: dict) -> int:
    """Samples per pair under the configured engine: ESMFold2 as above;
    ColabFold colabfold.num_seeds x colabfold.num_models."""
    if fold_engine(cfg) == "colabfold":
        cf = cfg["colabfold"]
        return int(cf.get("num_seeds", 1)) * int(cf.get("num_models", 5))
    return samples_per_pair(cfg["esmfold2"])
FOLD_COLUMNS: dict[str, tuple[pa.DataType, str]] = {
    "pair": (pa.string(), "Bait__prey pair identifier"),
    "seed": (pa.int32(), "Random seed of the run this sample comes from (ESMFold2 fold() call / ColabFold seed)"),
    "sample_index": (pa.int32(), "Sample index within the pair, over all its seeds (0-based, seed-major)"),
    "structure_path": (pa.string(), "This sample's gzipped mmCIF, relative to the results root"),
    "ptm": (pa.float32(), "Predicted TM-score (pTM)"),
    "iptm": (pa.float32(), "Predicted interface TM-score (ipTM)"),
    "mean_plddt": (pa.float32(), "Mean per-token pLDDT (0-1 scale; ColabFold's 0-100 is rescaled)"),
    "n_tokens": (pa.int32(), "Number of tokens (= residues, protein-only); pae is n_tokens x n_tokens"),
    "token_chain": (pa.list_(pa.string()), "Chain letter of each token (the pair spec's chain ids)"),
    "residue_index": (pa.list_(pa.int32()), "Residue index of each token (empty if not reported)"),
    "chain_order": (pa.list_(pa.string()), "Chain letters in the engine's chain order; row/column order of pair_chains_iptm"),
    "pair_chains_iptm": (pa.list_(pa.list_(pa.float32())), "Chain x chain ipTM matrix (chain_order x chain_order; ColabFold: --calc-extra-ptm pairwise ipTM; empty if not reported)"),
    "plddt": (_F16, "Per-token pLDDT, 0-1, float16"),
    "pae": (_F16, "Predicted aligned error (A), float16, flattened row-major n_tokens x n_tokens"),
}
# the publications behind the confidence columns (core.metric_refs); the
# per-token plddt array is cited like its mean, the ipTM matrix like pair_iptm
FOLD_REFERENCES = {**column_references(["ptm", "iptm"]),
                   "mean_plddt": references_text("mean_plddt"), "plddt": references_text("mean_plddt"),
                   "pair_chains_iptm": references_text("pair_iptm"), "pae": str(REFS["jumper2021"])}
SCALARS = ["pair", "seed", "sample_index", "structure_path", "ptm", "iptm", "mean_plddt", "n_tokens"]


def fold_row(pair: str, seed: int, sample_index: int, structure_path: str, *,
             ptm, iptm, plddt, pae, chain_id, chain_lookup: dict,
             residue_index=None, pair_chains_iptm=None) -> dict:
    """One sample's row from ESMFold2's raw values (any array-likes).
    chain_id = per-token integer chain index, chain_lookup = {int|str: letter}."""
    lookup = {int(k): v for k, v in chain_lookup.items()}
    plddt = np.asarray(plddt, dtype=np.float32).ravel()
    pae = None if pae is None else np.asarray(pae, dtype=np.float32)
    return {
        "pair": pair, "seed": int(seed), "sample_index": int(sample_index),
        "structure_path": structure_path,
        "ptm": None if ptm is None else float(ptm),
        "iptm": None if iptm is None else float(iptm),
        "mean_plddt": float(plddt.mean()) if plddt.size else None,
        "n_tokens": int(plddt.size),
        "token_chain": [lookup.get(int(c), str(int(c))) for c in np.asarray(chain_id).ravel()],
        "residue_index": [] if residue_index is None else np.asarray(residue_index).astype(int).tolist(),
        "chain_order": [lookup[k] for k in sorted(lookup)],
        "pair_chains_iptm": ([] if pair_chains_iptm is None
                             else np.asarray(pair_chains_iptm, dtype=np.float32).tolist()),
        "plddt": plddt.astype(np.float16),
        "pae": np.zeros(0, np.float16) if pae is None else pae.astype(np.float16).ravel(),
    }


def _f16_list(arrays: list[np.ndarray]) -> pa.Array:
    offsets = np.concatenate([[0], np.cumsum([len(a) for a in arrays])]).astype(np.int32)
    flat = np.concatenate(arrays) if arrays else np.zeros(0, np.float16)
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat, pa.float16()))


def write_fold(path: Path, rows: list[dict], run_params: dict | None = None,
               engine: str = "esmfold2") -> None:
    """Write rows (from fold_row) atomically: the file appearing is the
    'this pair is folded' signal, so a crash never leaves a partial one.
    run_params go in the table metadata under the engine's name."""
    fields, cols = [], []
    for name, (typ, desc) in FOLD_COLUMNS.items():
        meta = {b"description": desc.encode()}
        if name in FOLD_REFERENCES:
            meta[b"references"] = FOLD_REFERENCES[name].encode()
        fields.append(pa.field(name, typ, metadata=meta))
        vals = [r[name] for r in rows]
        cols.append(_f16_list(vals) if typ == _F16 else pa.array(vals, typ))
    meta = {b"description": FOLD_TABLE_DESCRIPTION.encode(), b"schema": SCHEMA_VERSION.encode()}
    if run_params:
        meta[engine.encode()] = json.dumps(run_params, sort_keys=True).encode()
    table = pa.Table.from_arrays(cols, schema=pa.schema(fields, metadata=meta))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=9,
                   column_encoding={"pae": "BYTE_STREAM_SPLIT", "plddt": "BYTE_STREAM_SPLIT"},
                   use_dictionary=["pair", "structure_path"])
    tmp.replace(path)


def read_fold(path: Path) -> list[dict]:
    """Rows sorted by sample_index, arrays as numpy: pae (n, n) float32,
    plddt (n,) float32, token_chain (n,) str, residue_index (n,) int
    (1..n when not reported), pair_chains_iptm (k, k) float32 or None."""
    rows = pq.read_table(path).to_pylist()
    for r in rows:
        n = r["n_tokens"]
        pae = np.asarray(r["pae"], dtype=np.float32)
        r["pae"] = pae.reshape(n, n) if pae.size == n * n else None
        r["plddt"] = np.asarray(r["plddt"], dtype=np.float32)
        r["token_chain"] = np.asarray(r["token_chain"])
        ri = r["residue_index"]
        r["residue_index"] = np.asarray(ri, dtype=int) if len(ri) == n else np.arange(1, n + 1)
        m = r["pair_chains_iptm"]
        r["pair_chains_iptm"] = np.asarray(m, dtype=np.float32) if m else None
    return sorted(rows, key=lambda r: r["sample_index"])


def read_fold_scalars(path: Path):
    """The scalar columns only (no arrays decoded) as a pandas DataFrame."""
    return pq.read_table(path, columns=SCALARS).to_pandas()
