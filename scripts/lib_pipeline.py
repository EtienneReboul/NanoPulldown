#!/usr/bin/env python3
"""
scripts/lib_pipeline.py
========================
Shared helpers for both stage Snakefiles and their scripts: config loading
(config.yaml + optional config.local.yaml overlay), per-pair spec loading
(configs/pairs.parquet + hand-curated configs/curated_domains.yaml -- see
configs/_schema.md), protein display names (configs/proteins.parquet +
hand overrides in configs/protein_names.yaml), and small chain/sequence
utilities.

Adapted from ab_initio_pipeline's scripts/lib_pipeline.py: drops the HPC/
scheduler/PLIP normalisation (nanopulldown runs one ESMFold2 backend, one
seed, locally, protein-protein only) and renames anchor/partner chains to
bait/prey chains to match AlphaPulldown's own vocabulary.

Config loading is stdlib + pyyaml only, so it can be imported from any of the
per-rule conda envs; the pair-spec functions also need pyarrow (imported lazily).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Config ──────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(repo_root: Path | str = REPO_ROOT) -> dict:
    """config.yaml with config.local.yaml deep-merged on top when present,
    then the per-run overlay named by $NANOPULLDOWN_RUN_CONFIG (e.g.
    config_q9m548_e3_fbox.yaml: its own `pairs:` and `dirs:`) when set."""
    root = Path(repo_root)
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    local = root / "config.local.yaml"
    if local.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(local.read_text()) or {})
    run_cfg = os.environ.get("NANOPULLDOWN_RUN_CONFIG")
    if run_cfg:
        cfg = _deep_merge(cfg, yaml.safe_load((root / run_cfg).read_text()) or {})
    return cfg


# ── Per-pair spec ────────────────────────────────────────────────────────────
# One row per pair in configs/pairs.parquet (written by scripts/expand_pairs.py);
# the only hand-edited part of a spec -- `annotation_reviewed` + `domains` --
# lives in configs/curated_domains.yaml ({pair: {annotation_reviewed, domains}})
# and is merged in on load. load_pair() returns the same dict shape the old
# per-pair configs/<pair>.yaml had, so callers don't care where it came from.

PAIRS_TABLE = Path("configs") / "pairs.parquet"
CURATED_DOMAINS = Path("configs") / "curated_domains.yaml"

PAIRS_TABLE_DESCRIPTION = (
    "nanopulldown pair specs: one row per bait__prey pair, as expanded by "
    "scripts/expand_pairs.py from AlphaPulldown-style bait/prey lists. Chain "
    "order is the ESMFold2 input order. Hand-curated domains are NOT here: see "
    "configs/curated_domains.yaml. Read through scripts/lib_pipeline.load_pair()."
)
PAIRS_COLUMNS = {
    "pair": "Pair name <bait>__<prey> (the entries of config.yaml's `pairs:`)",
    "bait_uniprot": "Bait UniProt accession",
    "bait_copies": "Number of bait copies (chains)",
    "bait_regions": "Bait residue regions [[start, stop], ...] (1-based, inclusive); null = full length",
    "prey_uniprot": "Prey UniProt accession",
    "prey_copies": "Number of prey copies (chains)",
    "prey_regions": "Prey residue regions [[start, stop], ...] (1-based, inclusive); null = full length",
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
             "n_tokens": pa.int32()}
    return pa.schema([pa.field(k, t, metadata={b"description": PAIRS_COLUMNS[k].encode()})
                      for k, t in types.items()],
                     metadata={b"description": PAIRS_TABLE_DESCRIPTION.encode(),
                               b"schema": b"pairs/1"})


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
    return row


def _row_to_spec(row: dict) -> dict:
    spec = {"name": row["pair"]}
    for role in ("bait", "prey"):
        spec[role] = {"uniprot": row[f"{role}_uniprot"], "copies": row[f"{role}_copies"],
                      "regions": _regions(row[f"{role}_regions"])}
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
                       "(run scripts/expand_pairs.py)")
    return specs[name]


def spec_digest(spec: dict, part: str = "spec") -> str:
    """Short content hash of a spec, for Snakemake `params:` (a changed param
    re-runs the job): part="spec" = everything ESMFold2/annotation see (chains,
    sequences, roles), part="curation" = annotation_reviewed + domains."""
    if part == "curation":
        d = {"r": spec.get("annotation_reviewed", False), "d": spec.get("domains") or {}}
    else:
        d = spec_to_row(spec)
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


# ── Proteins: human-readable names for the reports ─────────────────────────
# Pairs, paths and tables stay keyed by UniProt accession (stable, unique,
# filesystem-safe); names are a DISPLAY layer only. configs/proteins.parquet
# holds one row per accession with UniProt's metadata (scripts/fetch_uniprot.py)
# and a resolved `short_name` (axes, titles, pair labels: gene -> source-given
# name -> gene-like protein name -> AGI locus -> accession, made unique) and
# `long_name` (the protein name, for subtitles / tables / tooltips).
# configs/protein_names.yaml ({accession: {short: .., long: ..}}, tracked)
# overrides either one by hand. Every lookup falls back to the accession, so a
# missing table never breaks a figure.

PROTEINS_TABLE = Path("configs") / "proteins.parquet"
PROTEIN_NAMES = Path("configs") / "protein_names.yaml"

PROTEINS_TABLE_DESCRIPTION = (
    "nanopulldown proteins: one row per UniProt accession in any pair, with "
    "UniProt metadata (scripts/fetch_uniprot.py) and the display names the "
    "reports use. Hand overrides: configs/protein_names.yaml. Read through "
    "scripts/lib_pipeline.load_proteins()."
)
PROTEINS_COLUMNS = {
    "accession": "UniProt accession (the key pairs and paths use)",
    "short_name": "Display label: gene > source name > gene-like protein name > AGI locus > accession, unique",
    "long_name": "Protein name (UniProt recommended, else submitted); null if absent or a placeholder",
    "protein_name": "UniProt recommended (Swiss-Prot) or submitted (TrEMBL) protein name; placeholders dropped",
    "gene_name": "UniProt primary gene name",
    "gene_synonyms": "UniProt gene name synonyms",
    "manual_name": "Name given by an input source (e.g. an IP-MS table's gene column)",
    "locus": "Ordered locus; Arabidopsis AGI code normalised to AT#G#####",
    "organism": "Organism scientific name",
    "taxon_id": "NCBI taxon id",
    "length": "Canonical sequence length (aa)",
    "reviewed": "Swiss-Prot (true) or TrEMBL (false)",
    "function": "UniProt FUNCTION comment (first text)",
    "subcellular_location": "UniProt SUBCELLULAR LOCATION values",
    "uniprot_release": "UniProt release the metadata was fetched from",
}


def _proteins_schema():
    import pyarrow as pa

    types = {"accession": pa.string(), "short_name": pa.string(), "long_name": pa.string(),
             "protein_name": pa.string(), "gene_name": pa.string(),
             "gene_synonyms": pa.list_(pa.string()), "manual_name": pa.string(),
             "locus": pa.string(), "organism": pa.string(), "taxon_id": pa.int64(),
             "length": pa.int32(), "reviewed": pa.bool_(), "function": pa.string(),
             "subcellular_location": pa.list_(pa.string()), "uniprot_release": pa.string()}
    return pa.schema([pa.field(k, t, metadata={b"description": PROTEINS_COLUMNS[k].encode()})
                      for k, t in types.items()],
                     metadata={b"description": PROTEINS_TABLE_DESCRIPTION.encode(),
                               b"schema": b"proteins/1"})


def _gene_like(name: str | None) -> bool:
    """'EBF2' yes, 'BTB/POZ domain protein' no: TrEMBL submitters often put a
    gene symbol in the protein-name slot."""
    return bool(name) and " " not in name and len(name) <= 15


def _base_short(r: dict) -> str:
    for cand in (r.get("gene_name"), r.get("manual_name"),
                 r.get("protein_name") if _gene_like(r.get("protein_name")) else None,
                 r.get("locus")):
        if cand:
            return str(cand)
    return r["accession"]


def resolve_short_names(rows: list[dict]) -> None:
    """Fill `short_name` / `long_name` in place. A short name shared by several
    accessions (TrEMBL duplicates of a Swiss-Prot entry, two loci with the
    same gene symbol) keeps its plain form for the one reviewed entry, if
    exactly one is, and gets ' (<accession>)' appended for the others."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        r["short_name"] = _base_short(r)
        r["long_name"] = r.get("protein_name")
        groups.setdefault(r["short_name"].casefold(), []).append(r)
    for grp in groups.values():
        if len(grp) < 2:
            continue
        reviewed = [r for r in grp if r.get("reviewed")]
        keep = reviewed[0] if len(reviewed) == 1 else None
        for r in grp:
            if r is not keep and r["short_name"] != r["accession"]:
                r["short_name"] = f"{r['short_name']} ({r['accession']})"


def update_proteins(records, repo_root: Path | str = REPO_ROOT) -> Path:
    """Merge metadata records ({accession, ...} as fetch_uniprot.fetch_metadata
    returns, optionally with `manual_name`) into configs/proteins.parquet
    (new values win, a None never erases a known one), re-resolve every short
    name over the whole table, and write it atomically."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(repo_root) / PROTEINS_TABLE
    rows = {r["accession"]: r for r in (pq.read_table(path).to_pylist() if path.exists() else [])}
    for rec in records:
        old = rows.get(rec["accession"], {})
        rows[rec["accession"]] = {**old, **{k: v for k, v in rec.items() if v is not None}}
    out = []
    for r in rows.values():
        r = {k: r.get(k) for k in PROTEINS_COLUMNS}
        for k in ("gene_synonyms", "subcellular_location"):
            r[k] = list(r[k] or [])
        out.append(r)
    resolve_short_names(out)
    out.sort(key=lambda r: r["accession"])
    table = pa.Table.from_pylist(out, schema=_proteins_schema())
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=9)
    tmp.replace(path)
    _load_proteins_cached.cache_clear()
    return path


@lru_cache(maxsize=4)
def _load_proteins_cached(table: str, table_mtime: float, names: str, names_mtime: float) -> dict:
    prots = {}
    if table_mtime:
        import pyarrow.parquet as pq
        prots = {r["accession"]: r for r in pq.read_table(table).to_pylist()}
    over = (yaml.safe_load(Path(names).read_text()) or {}) if names_mtime else {}
    for acc, o in over.items():
        r = prots.setdefault(acc, {"accession": acc, "short_name": acc, "long_name": None})
        if (o or {}).get("short"):
            r["short_name"] = str(o["short"])
        if (o or {}).get("long"):
            r["long_name"] = str(o["long"])
    return prots


def load_proteins(repo_root: Path | str = REPO_ROOT) -> dict[str, dict]:
    """{accession: proteins-table row}, hand overrides applied. Empty when the
    table does not exist yet -- every label helper then falls back to the
    accession."""
    root = Path(repo_root)
    t, n = root / PROTEINS_TABLE, root / PROTEIN_NAMES
    return copy.deepcopy(_load_proteins_cached(
        str(t), t.stat().st_mtime if t.exists() else 0.0,
        str(n), n.stat().st_mtime if n.exists() else 0.0))


def protein_name(acc: str, proteins: dict, kind: str = "short") -> str:
    """Short (default) or long display name of one accession; long falls back
    to short, short to the accession."""
    r = proteins.get(acc) or {}
    short = r.get("short_name") or acc
    return (r.get("long_name") or short) if kind == "long" else short


def _adorn(name: str, ent: dict) -> str:
    regions = ent.get("regions")
    if regions:
        name += " (" + ", ".join(f"{a}-{b}" for a, b in regions) + ")"
    if int(ent.get("copies", 1)) > 1:
        name = f"{int(ent['copies'])}×{name}"
    return name


def pair_labels(spec: dict, proteins: dict) -> dict:
    """Display names of one pair: bait / prey short labels (with copies and
    regions), `label` = 'BPM6 × MYC2', `long` = the two protein names."""
    out = {}
    for role in ("bait", "prey"):
        ent = spec.get(role) or {}
        out[role] = _adorn(protein_name(ent.get("uniprot", "?"), proteins), ent)
        out[f"{role}_long"] = protein_name(ent.get("uniprot", "?"), proteins, "long")
        out[f"{role}_uniprot"] = ent.get("uniprot")
    out["label"] = f"{out['bait']} × {out['prey']}"
    out["long"] = f"{out['bait_long']} × {out['prey_long']}"
    return out


def chain_labels(spec: dict, proteins: dict) -> dict[str, str]:
    """{chain id: label} unique within the pair: the protein's short name,
    plus ' (bait)'/' (prey)' when both sides are the same protein and a copy
    number when a side has several chains."""
    base = {}
    for c in protein_chains(spec):
        ent = spec.get(c["role"]) or {}
        base[c["id"]] = (protein_name(ent.get("uniprot", c["id"]), proteins), c["role"])
    out, seen = {}, {}
    for cid, (name, role) in base.items():
        lbl = name
        if len({r for n, r in base.values() if n == name}) > 1:
            lbl = f"{name} ({role})"
        if sum(1 for n, r in base.values() if n == name and r == role) > 1:
            seen[lbl] = seen.get(lbl, 0) + 1
            lbl = f"{lbl} #{seen[lbl]}"
        out[cid] = lbl
    return out


def identity_columns(spec: dict, proteins: dict) -> dict:
    """Name columns a report table puts next to `pair`, in display order."""
    lb = pair_labels(spec, proteins)
    prey = proteins.get(lb["prey_uniprot"]) or {}
    return {"bait": lb["bait"], "prey": lb["prey"], "prey_name": lb["prey_long"],
            "prey_locus": prey.get("locus"), "bait_uniprot": lb["bait_uniprot"],
            "prey_uniprot": lb["prey_uniprot"]}


def label_digest(spec: dict, proteins: dict) -> str:
    """Digest of everything a figure shows about a pair's identity, for
    Snakemake `params:` -- a renamed protein redraws just its figures."""
    d = {"p": pair_labels(spec, proteins), "c": chain_labels(spec, proteins)}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


# ── Run inputs: where each protein came from ───────────────────────────────
# Per run (config `run_name`, default "main"): configs/inputs_<run>.parquet,
# one row per (accession, role) with the input sources that selected it and
# any columns an IP-MS source carries (as <source>_<column>). Written by
# scripts/expand_pairs.py, joined into the results table.

def inputs_table_path(cfg: dict, repo_root: Path | str = REPO_ROOT) -> Path:
    return Path(repo_root) / "configs" / f"inputs_{cfg.get('run_name') or 'main'}.parquet"


# ── Chain helpers ──────────────────────────────────────────────────────────

def protein_chains(spec: dict) -> list[dict]:
    """Every chain in a pair spec's `sequences:` list (protein-protein only,
    so this is every entry — kept as a function for parity with call sites
    that read like ab_initio_pipeline's protein_chains(spec))."""
    return list(spec.get("sequences", []))


def seq_md5(seq: str) -> str:
    return hashlib.md5(seq.encode("ascii")).hexdigest()


def annotation_cache_dir(cfg: dict, kind: str) -> Path:
    """data/annotation/_cache/<kind>/ -- per-UNIQUE-SEQUENCE annotation cache
    (<md5(sequence)>.<ext>), shared by every pair that contains the sequence.
    The bait appears in every pair, so without this it would be re-annotated
    once per pair. Delete the directory to force a refresh."""
    d = Path(cfg["dirs"]["annotation"]) / "_cache" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_disorder_tracks(cfg: dict, seq: str) -> dict[str, list[float]]:
    """AIUPred tracks ({"disorder": [...], "binding": [...], "linker": [...]},
    one value per residue) of a sequence from the annotation cache; {} if absent."""
    path = annotation_cache_dir(cfg, "disorder") / f"{seq_md5(seq)}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + rename so concurrent jobs never see a partial cache entry."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


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
