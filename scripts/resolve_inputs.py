#!/usr/bin/env python3
"""
scripts/resolve_inputs.py — Stage 1a
=====================================
Turns a run config's `inputs:` block into the bait and prey protein specs
scripts/expand_pairs.py pairs up, plus where each protein came from. Each
role is a list of sources whose proteins are UNIONed (deduplicated by spec),
then `exclude:` entries are dropped:

    inputs:
      baits:
        - list: configs/baits.txt              # AlphaPulldown specs, one per line
      preys:
        - list: configs/old_preys.txt
          name: curated                        # source tag (default: file stem)
        - accessions: [Q39204, Q9FIP9:2]       # inline specs
          name: manual
        - uniprot_query:                       # GO / InterPro / organism query
            name: e3_nuclear                   # snapshot: data/uniprot_retrieval/e3_nuclear.tsv
            go: [GO:0061630, GO:0005634]       # ANDed (child terms included)
            organism: 3702
        - ipms:                                # IP-MS (or any) table with UniProt ids
            name: pxd013906
            path: data/pxd013906_uniprot_mapping.csv   # .csv / .tsv / .xlsx (+ sheet:)
            id_column: uniprot                 # "Q1;Q2" cells -> first id
            name_column: genes                 # optional: names for proteins UniProt has no gene for
            filter: "adjp < 0.05 and logfc > 0"   # optional pandas query
            keep_columns: [logfc, adjp]        # carried into the results table as pxd013906_logfc ...
            annotate_only: false               # true: only add columns to preys other sources picked
        - exclude: [P56757]                    # never pair these

`uniprot_query` keys are documented in fetch_uniprot.build_query. Query
results are snapshotted and reused while the query is unchanged (pass
refresh=True / expand_pairs --refresh-queries to re-query).

Run standalone to preview what a config selects without expanding anything:
    NANOPULLDOWN_RUN_CONFIG=config_bpm6_curated.yaml python scripts/resolve_inputs.py
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_uniprot import build_query, parse_protein_spec, run_query  # noqa: E402
from lib_pipeline import REPO_ROOT, load_config  # noqa: E402

KINDS = ("list", "accessions", "uniprot_query", "ipms", "exclude")


@dataclass
class Resolved:
    specs: dict[str, list[str]] = field(default_factory=lambda: {"bait": [], "prey": []})
    # (accession, role) -> {"sources": [...], "<source>_<col>": value, ...}
    provenance: dict[tuple[str, str], dict] = field(default_factory=dict)
    manual_names: dict[str, str] = field(default_factory=dict)

    def add(self, role: str, spec: str, source: str, extra: dict | None = None) -> None:
        if spec not in self.specs[role]:
            self.specs[role].append(spec)
        acc = parse_protein_spec(spec).uniprot
        rec = self.provenance.setdefault((acc, role), {"sources": []})
        if source not in rec["sources"]:
            rec["sources"].append(source)
        rec.update(extra or {})


def read_specs(path: Path) -> list[str]:
    """AlphaPulldown-style list file: one spec per line, `#` comments and blank lines ignored."""
    if not path.exists():
        raise FileNotFoundError(f"input list {path} does not exist")
    specs = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            specs.append(line)
    return specs


def _read_table(src: dict):
    import pandas as pd

    path = REPO_ROOT / src["path"] if not Path(src["path"]).is_absolute() else Path(src["path"])
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=src.get("sheet", 0), header=int(src.get("header", 0)))
    else:
        sep = "\t" if path.suffix in (".tsv", ".tab", ".txt") else ","
        df = pd.read_csv(path, sep=sep, header=int(src.get("header", 0)))
    if src.get("filter"):
        df = df.query(src["filter"])
    return df


def _ipms(res: Resolved, role: str, src: dict) -> str:
    import pandas as pd

    name = src.get("name") or Path(src["path"]).stem
    df = _read_table(src)
    idcol = src.get("id_column", "uniprot")
    if idcol not in df.columns:
        raise KeyError(f"ipms source {name!r}: no column {idcol!r} in {src['path']} "
                       f"(columns: {list(df.columns)})")
    keep = list(src.get("keep_columns") or [])
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"ipms source {name!r}: keep_columns {missing} not in {src['path']}")
    for _, row in df.iterrows():
        cell = row[idcol]
        if pd.isna(cell) or not str(cell).strip():
            continue
        acc = re.split(r"[;,\s]+", str(cell).strip())[0]
        extra = {f"{name}_{c}": (None if pd.isna(row[c]) else row[c].item()
                                 if hasattr(row[c], "item") else row[c]) for c in keep}
        if src.get("annotate_only"):
            if (acc, role) not in res.provenance:
                continue
            res.provenance[(acc, role)].update(extra)
        else:
            res.add(role, acc, name, extra)
        nc = src.get("name_column")
        if nc and nc in df.columns and not pd.isna(row[nc]) and str(row[nc]).strip():
            res.manual_names.setdefault(acc, str(row[nc]).strip())
    return name


def resolve_inputs(cfg: dict, refresh: bool = False, verbose: bool = True) -> Resolved:
    inputs = cfg.get("inputs") or {}
    res = Resolved()
    for key, role in (("baits", "bait"), ("preys", "prey")):
        excluded: set[str] = set()
        for src in inputs.get(key) or []:
            kind = next((k for k in KINDS if k in src), None)
            if kind is None:
                raise ValueError(f"inputs.{key}: unknown source {src!r} (one of {KINDS})")
            before = len(res.specs[role])
            if kind == "list":
                tag = src.get("name") or Path(src["list"]).stem
                for spec in read_specs(REPO_ROOT / src["list"]):
                    res.add(role, spec, tag)
            elif kind == "accessions":
                tag = src.get("name") or "manual"
                for spec in src["accessions"]:
                    res.add(role, str(spec), tag)
            elif kind == "uniprot_query":
                q = src["uniprot_query"]
                tag = q.get("name")
                if not tag:
                    raise ValueError(f"inputs.{key}: uniprot_query needs a `name:` (its snapshot file)")
                for acc in run_query(tag, build_query(q), cfg, refresh=refresh):
                    res.add(role, acc, tag)
            elif kind == "ipms":
                tag = _ipms(res, role, src["ipms"])
            else:                                           # exclude
                excluded |= {parse_protein_spec(str(s)).uniprot for s in src["exclude"]}
                continue
            if verbose:
                print(f"[resolve_inputs] {key}: {kind} {tag!r}: "
                      f"+{len(res.specs[role]) - before} new spec(s)", file=sys.stderr)
        if excluded:
            res.specs[role] = [s for s in res.specs[role]
                               if parse_protein_spec(s).uniprot not in excluded]
            for acc in excluded:
                res.provenance.pop((acc, role), None)
            if verbose:
                print(f"[resolve_inputs] {key}: excluded {len(excluded)} accession(s)", file=sys.stderr)
    return res


def write_inputs_table(res: Resolved, path: Path) -> Path:
    """configs/inputs_<run>.parquet: one row per (accession, role) -- see
    lib_pipeline.inputs_table_path."""
    import pandas as pd

    from parquet_utils import write_parquet_with_metadata

    rows = [{"accession": acc, "role": role, **rec} for (acc, role), rec in res.provenance.items()]
    df = pd.DataFrame(rows, columns=None if rows else ["accession", "role", "sources"])
    cols = {"accession": "UniProt accession", "role": "bait | prey",
            "sources": "Input sources (config `inputs:`) that selected this protein"}
    for c in df.columns:
        cols.setdefault(c, "Input-source column, carried as <source>_<column>")
    write_parquet_with_metadata(
        df, path,
        table_description="nanopulldown run inputs: one row per (accession, role) with the "
                          "config `inputs:` sources that selected it and any columns those "
                          "sources carry (<source>_<column>). Written by scripts/expand_pairs.py.",
        column_descriptions=cols)
    return path


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="preview what the config's `inputs:` selects")
    ap.add_argument("--refresh-queries", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if not cfg.get("inputs"):
        print("[resolve_inputs] config has no `inputs:` block")
        return 1
    res = resolve_inputs(cfg, refresh=a.refresh_queries)
    print(f"{len(res.specs['bait'])} bait(s) x {len(res.specs['prey'])} prey(s) = "
          f"{len(res.specs['bait']) * len(res.specs['prey'])} pair(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
