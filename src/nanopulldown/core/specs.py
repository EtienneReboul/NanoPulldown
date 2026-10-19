"""
nanopulldown.core.specs
=======================
Per-pair specs (configs/pairs.parquet + hand-curated configs/curated_domains.yaml,
see configs/_schema.md), AlphaPulldown-style protein specs (`ID[:N][:start-stop...]`),
the run inputs table path, and small chain helpers.

Stdlib + pyyaml; pyarrow is imported lazily by the table functions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .config import REPO_ROOT

# ── Protein specs ───────────────────────────────────────────────────────────

@dataclass
class ProteinSpec:
    uniprot: str
    copies: int = 1
    regions: list[tuple[int, int]] | None = None


def parse_protein_spec(spec: str) -> ProteinSpec:
    """Parse AlphaPulldown-style `ID`, `ID:N`, `ID:start-stop[:start2-stop2]`,
    or `ID:N:start-stop[:start2-stop2]` into a ProteinSpec."""
    parts = spec.strip().split(":")
    uniprot = parts[0]
    rest = parts[1:]
    copies = 1
    regions: list[tuple[int, int]] | None = None

    if rest and "-" not in rest[0]:
        copies = int(rest[0])
        rest = rest[1:]

    if rest:
        regions = []
        for token in rest:
            start, stop = token.split("-")
            regions.append((int(start), int(stop)))

    return ProteinSpec(uniprot=uniprot, copies=copies, regions=regions)


def apply_regions(sequence: str, regions: list[tuple[int, int]] | None) -> str:
    """1-based, inclusive AlphaPulldown-style region slicing, concatenating
    discontinuous regions (residue numbering gaps are not preserved here --
    nanopulldown has no mmCIF author-numbering step to carry them into)."""
    if not regions:
        return sequence
    return "".join(sequence[start - 1:stop] for start, stop in regions)


# ── Per-pair spec ────────────────────────────────────────────────────────────
# One row per pair in configs/pairs.parquet (written by nanopulldown.inputs.expand);
# the only hand-edited part of a spec -- `annotation_reviewed` + `domains` --
# lives in configs/curated_domains.yaml ({pair: {annotation_reviewed, domains}})
# and is merged in on load. load_pair() returns the same dict shape the old
# per-pair configs/<pair>.yaml had, so callers don't care where it came from.

PAIRS_TABLE = Path("configs") / "pairs.parquet"
CURATED_DOMAINS = Path("configs") / "curated_domains.yaml"

PAIRS_TABLE_DESCRIPTION = (
    "nanopulldown pair specs: one row per bait__prey pair, as expanded by "
    "nanopulldown.inputs.expand from AlphaPulldown-style bait/prey lists. Chain "
    "order is the ESMFold2 input order. Hand-curated domains are NOT here: see "
    "configs/curated_domains.yaml. Read through nanopulldown.core.specs.load_pair()."
)
PAIRS_COLUMNS = {
    "pair": "Pair name <bait>__<prey> (the entries of config.yaml's `pairs:`)",
    "bait_uniprot": "Bait UniProt accession",
    "bait_copies": "Number of bait copies (chains)",
    "bait_regions": "Bait residue regions [[start, stop], ...] (1-based, inclusive); null = full length",
    "prey_uniprot": "Prey UniProt accession",
    "prey_copies": "Number of prey copies (chains)",
    "prey_regions": "Prey residue regions [[start, stop], ...] (1-based, inclusive); null = full length",
    "bait_fragment": "Fragment-mode bait domain name (nanopulldown.inputs.fragments); null = not a domain fragment",
    "prey_fragment": "Fragment-mode prey domain name (nanopulldown.inputs.fragments); null = not a domain fragment",
    "chains": "Chains in ESMFold2 input order: {id, name, role (bait|prey), sequence}",
    "bait_chains": "Chain ids of the bait copies",
    "prey_chains": "Chain ids of the prey copies",
    "n_tokens": "Total tokens (residues) ESMFold2 folds for this pair",
}


def _pairs_schema():
    import pyarrow as pa

    regions = pa.list_(pa.list_(pa.int32()))
    chain = pa.struct([("id", pa.string()), ("name", pa.string()),
                       ("role", pa.string()), ("sequence", pa.string())])
    types = {"pair": pa.string(), "bait_uniprot": pa.string(), "bait_copies": pa.int32(),
             "bait_regions": regions, "prey_uniprot": pa.string(), "prey_copies": pa.int32(),
             "prey_regions": regions, "chains": pa.list_(chain),
             "bait_chains": pa.list_(pa.string()), "prey_chains": pa.list_(pa.string()),
             "n_tokens": pa.int32(), "bait_fragment": pa.string(), "prey_fragment": pa.string()}
    return pa.schema([pa.field(k, t, metadata={b"description": PAIRS_COLUMNS[k].encode()})
                      for k, t in types.items()],
                     metadata={b"description": PAIRS_TABLE_DESCRIPTION.encode(),
                               b"schema": b"pairs/2"})


def _regions(r):
    return None if r is None else [[int(a), int(b)] for a, b in r]


def spec_to_row(spec: dict) -> dict:
    row = {"pair": spec["name"]}
    for role in ("bait", "prey"):
        ent = spec.get(role) or {}
        row[f"{role}_uniprot"] = ent.get("uniprot")
        row[f"{role}_copies"] = int(ent.get("copies", 1))
        row[f"{role}_regions"] = _regions(ent.get("regions"))
    row["chains"] = [{k: c.get(k) for k in ("id", "name", "role", "sequence")}
                     for c in spec.get("sequences", [])]
    row["bait_chains"] = list(spec.get("bait_chains", []))
    row["prey_chains"] = list(spec.get("prey_chains", []))
    row["n_tokens"] = pair_tokens(spec)
    for role in ("bait", "prey"):
        row[f"{role}_fragment"] = (spec.get(role) or {}).get("fragment")
    return row


def _row_to_spec(row: dict) -> dict:
    spec = {"name": row["pair"]}
    for role in ("bait", "prey"):
        spec[role] = {"uniprot": row[f"{role}_uniprot"], "copies": row[f"{role}_copies"],
                      "regions": _regions(row[f"{role}_regions"]),
                      "fragment": row.get(f"{role}_fragment")}
    spec["sequences"] = [dict(c) for c in row["chains"]]
    spec["bait_chains"] = list(row["bait_chains"])
    spec["prey_chains"] = list(row["prey_chains"])
    return spec


def write_pairs(specs: list[dict], repo_root: Path | str = REPO_ROOT) -> Path:
    """(Over)write configs/pairs.parquet with `specs` (sorted by pair), atomically.
    Curated `annotation_reviewed`/`domains` keys, if present, are ignored --
    they belong in configs/curated_domains.yaml."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = sorted((spec_to_row(s) for s in specs), key=lambda r: r["pair"])
    table = pa.Table.from_pylist(rows, schema=_pairs_schema())
    path = Path(repo_root) / PAIRS_TABLE
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=9)
    tmp.replace(path)
    _load_pairs_cached.cache_clear()
    return path


@lru_cache(maxsize=4)
def _load_pairs_cached(table: str, table_mtime: float, curated: str, curated_mtime: float) -> dict:
    import pyarrow.parquet as pq

    specs = {}
    if table_mtime:
        for row in pq.read_table(table).to_pylist():
            specs[row["pair"]] = _row_to_spec(row)
    cur = (yaml.safe_load(Path(curated).read_text()) or {}) if curated_mtime else {}
    for name, spec in specs.items():
        c = cur.get(name) or {}
        spec["annotation_reviewed"] = bool(c.get("annotation_reviewed", False))
        spec["domains"] = c.get("domains") or {}
    return specs


def load_pairs(repo_root: Path | str = REPO_ROOT) -> dict[str, dict]:
    """{pair: spec} for every pair in configs/pairs.parquet (+ curated domains).
    Returns copies, so callers may mutate them."""
    root = Path(repo_root)
    t, c = root / PAIRS_TABLE, root / CURATED_DOMAINS
    specs = _load_pairs_cached(str(t), t.stat().st_mtime if t.exists() else 0.0,
                               str(c), c.stat().st_mtime if c.exists() else 0.0)
    return copy.deepcopy(specs)


def load_pair(name: str, repo_root: Path | str = REPO_ROOT) -> dict:
    """One pair's spec: name, bait/prey {uniprot, copies, regions}, sequences
    [{id, name, sequence, role}], bait_chains, prey_chains (lists),
    annotation_reviewed (default False), domains (default {})."""
    specs = load_pairs(repo_root)
    if name not in specs:
        raise KeyError(f"pair {name!r} is not in {Path(repo_root) / PAIRS_TABLE} "
                       "(run nanopulldown.inputs.expand)")
    return specs[name]


def spec_digest(spec: dict, part: str = "spec") -> str:
    """Short content hash of a spec, for Snakemake `params:` (a changed param
    re-runs the job): part="spec" = everything ESMFold2/annotation see (chains,
    sequences, roles), part="curation" = annotation_reviewed + domains."""
    if part == "curation":
        d = {"r": spec.get("annotation_reviewed", False), "d": spec.get("domains") or {}}
    else:   # fragment names are display only (label_digest), and pairs/1 rows had none
        d = {k: v for k, v in spec_to_row(spec).items() if not k.endswith("_fragment")}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

# ── Run inputs: where each protein came from ───────────────────────────────
# Per run (config `run_name`, default "main"): configs/inputs_<run>.parquet,
# one row per (accession, role) with the input sources that selected it and
# any columns an IP-MS source carries (as <source>_<column>). Written by
# nanopulldown.inputs.expand, joined into the results table.

def inputs_table_path(cfg: dict, repo_root: Path | str = REPO_ROOT) -> Path:
    return Path(repo_root) / "configs" / f"inputs_{cfg.get('run_name') or 'main'}.parquet"


# ── Chain helpers ──────────────────────────────────────────────────────────

def protein_chains(spec: dict) -> list[dict]:
    """Every chain in a pair spec's `sequences:` list (protein-protein only,
    so this is every entry — kept as a function for parity with call sites
    that read like ab_initio_pipeline's protein_chains(spec))."""
    return list(spec.get("sequences", []))


def pair_tokens(spec: dict) -> int:
    """Total token count (= residues, protein-only) ESMFold2 sees for a pair.
    Folding memory grows ~quadratically with this -- see esmfold2.max_tokens."""
    return sum(len(c["sequence"]) for c in protein_chains(spec))


def chain_by_id(spec: dict, cid: str) -> dict | None:
    for s in spec.get("sequences", []):
        if s.get("id") == cid:
            return s
    return None


def bait_chain_ids(spec: dict) -> list[str]:
    return list(spec.get("bait_chains", []))


def prey_chain_ids(spec: dict) -> list[str]:
    return list(spec.get("prey_chains", []))
