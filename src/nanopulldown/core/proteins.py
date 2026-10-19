"""
nanopulldown.core.proteins
==========================
Protein display names (configs/proteins.parquet + hand overrides in
configs/protein_names.yaml) and the pair / chain labels the figures and
tables use. Stdlib + pyyaml; pyarrow is imported lazily.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import yaml

from .config import REPO_ROOT
from .specs import protein_chains

# ── Proteins: human-readable names for the reports ─────────────────────────
# Pairs, paths and tables stay keyed by UniProt accession (stable, unique,
# filesystem-safe); names are a DISPLAY layer only. configs/proteins.parquet
# holds one row per accession with UniProt's metadata (nanopulldown.services.uniprot)
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
    "UniProt metadata (nanopulldown.services.uniprot) and the display names the "
    "reports use. Hand overrides: configs/protein_names.yaml. Read through "
    "nanopulldown.core.proteins.load_proteins()."
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


def _short_fragment(name: str, width: int = 32) -> str:
    return name if len(name) <= width else name[:width - 1].rstrip() + "…"


def _adorn(name: str, ent: dict) -> str:
    if ent.get("fragment"):
        name += f" [{_short_fragment(ent['fragment'])}]"
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
    out = {"bait": lb["bait"], "prey": lb["prey"], "prey_name": lb["prey_long"],
           "prey_locus": prey.get("locus"), "bait_uniprot": lb["bait_uniprot"],
           "prey_uniprot": lb["prey_uniprot"]}
    for role in ("bait", "prey"):       # fragment mode only, so other runs' tables keep their columns
        if (spec.get(role) or {}).get("fragment"):
            out[f"{role}_fragment"] = spec[role]["fragment"]
    return out


def label_digest(spec: dict, proteins: dict) -> str:
    """Digest of everything a figure shows about a pair's identity, for
    Snakemake `params:` -- a renamed protein redraws just its figures."""
    d = {"p": pair_labels(spec, proteins), "c": chain_labels(spec, proteins)}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
