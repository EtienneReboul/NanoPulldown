#!/usr/bin/env python3
"""
Map the AGI-keyed IP-MS tables of one experiment (several contrasts) to UniProt
and merge them into one CSV that an `ipms:` input source reads.

Each input is `<label>=<path>`: a ';'-separated, decimal-comma table (Excel FR
export) with `Accession` (AT1G24450.1, or a UniProt accession for non-plant
hits such as viral proteins), `log2FC`, `pvalue`, `adjp`, `Gene name` and an
UP/DOWN call column. Output: one row per accession, contrast columns
`<label>_log2fc`, `<label>_adjp`, `<label>_call` (empty when the protein is not
in that contrast).

The A. thaliana proteome (organism 3702, Swiss-Prot + TrEMBL) is streamed ONCE
from UniProt and mapped locally on the Araport cross-reference (accession
OR-lists get HTTP 429). One accession per locus: Swiss-Prot first, then an entry
specific to that locus, then the longest. The stream is snapshotted next to the
output (`<out stem>.proteome_3702.tsv.gz`) and reused unless --refresh.

    python scripts/map_agi_ipms.py \
        ni=data/ip_data_drb2/IP_DRB2_vs_Col_NI.csv \
        tbsv=data/ip_data_drb2/IP_DRB2_vs_Col_TBSV.csv \
        --out data/ip_data_drb2/ip_drb2_uniprot.csv
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests

STREAM = "https://rest.uniprot.org/uniprotkb/stream"
FIELDS = "accession,reviewed,gene_primary,gene_oln,xref_araport,length"
AGI = re.compile(r"^AT[1-5CM]G\d{5}$", re.I)
UNIPROT = re.compile(r"^([OPQ]\d[A-Z\d]{3}\d|[A-NR-Z]\d([A-Z][A-Z\d]{2}\d){1,2})$")


def load_contrast(label: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    call = next(c for c in df.columns if c.lower().startswith("enriched"))
    df = df.rename(columns={"Accession": "accession", "Gene name": "gene", call: "call",
                            "log2FC": "log2fc"})
    df["accession"] = df["accession"].astype(str).str.strip()
    df["call"] = df["call"].astype(str).str.strip().str.upper()
    df["agi"] = df["accession"].str.replace(r"\.\d+$", "", regex=True).str.upper()
    df.loc[~df["agi"].str.match(AGI), "agi"] = pd.NA
    # one row per locus: several gene models of a locus -> the most significant
    df["_key"] = df["agi"].fillna(df["accession"])
    df = df.sort_values("adjp").drop_duplicates("_key")
    out = df[["_key", "agi", "accession", "gene"]].copy()
    for c in ("log2fc", "adjp", "call"):
        out[f"{label}_{c}"] = df[c]
    return out


def fetch_proteome(snapshot: Path, refresh: bool) -> pd.DataFrame:
    if refresh or not snapshot.exists():
        r = requests.get(STREAM, params={"query": "organism_id:3702", "fields": FIELDS,
                                         "format": "tsv", "compressed": "true"}, timeout=600)
        r.raise_for_status()
        snapshot.write_bytes(r.content)
        print(f"[map_agi_ipms] proteome 3702 streamed (release "
              f"{r.headers.get('X-UniProt-Release')}) -> {snapshot}", file=sys.stderr)
    return pd.read_csv(snapshot, sep="\t", compression="gzip", dtype=str)


def locus_table(prot: pd.DataFrame) -> pd.DataFrame:
    """(agi, uniprot, reviewed, n_loci, length) rows, one per (locus, entry)."""
    rows = []
    for _, r in prot.iterrows():
        loci = set()
        for col in ("Araport", "Gene Names (ordered locus)"):
            loci |= {t.upper() for t in re.split(r"[;\s/,]+", str(r.get(col, "") or ""))
                     if AGI.match(t)}
        for loc in loci:
            rows.append({"agi": loc, "uniprot": r["Entry"],
                         "reviewed": str(r["Reviewed"]).lower() == "reviewed",
                         "uniprot_gene": r.get("Gene Names (primary)"),
                         "n_loci": len(loci), "length": int(r["Length"])})
    t = pd.DataFrame(rows)
    t = t.sort_values(["agi", "reviewed", "n_loci", "length"],
                      ascending=[True, False, True, False])
    return t.drop_duplicates("agi")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("contrasts", nargs="+", help="<label>=<path>")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--refresh", action="store_true", help="re-stream the proteome")
    a = ap.parse_args()

    merged = None
    for spec in a.contrasts:
        label, path = spec.split("=", 1)
        df = load_contrast(label, Path(path))
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="_key", how="outer", suffixes=("", "_r"))
            for c in ("agi", "accession", "gene"):
                merged[c] = merged[c].fillna(merged.pop(f"{c}_r"))

    snapshot = a.out.with_name(a.out.stem + ".proteome_3702.tsv.gz")
    loci = locus_table(fetch_proteome(snapshot, a.refresh))
    merged = merged.merge(loci, on="agi", how="left")
    # non-plant hits (viral proteins) are already UniProt accessions
    direct = merged["agi"].isna() & merged["accession"].str.match(UNIPROT)
    merged.loc[direct, "uniprot"] = merged.loc[direct, "accession"]

    unresolved = merged[merged["uniprot"].isna()]
    labels = [s.split("=", 1)[0] for s in a.contrasts]
    cols = ["uniprot", "agi", "gene", "uniprot_gene", "reviewed", "length"] + \
           [f"{l}_{c}" for l in labels for c in ("log2fc", "adjp", "call")]
    out = merged[merged["uniprot"].notna()][cols]
    out = out.sort_values([f"{labels[0]}_adjp", f"{labels[-1]}_adjp"], na_position="last")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    up = out[[f"{l}_call" for l in labels]].eq("UP").any(axis=1)
    shared = out["uniprot"][out.duplicated("uniprot", keep=False)].unique()
    print(f"[map_agi_ipms] {len(out)} accessions ({up.sum()} UP in >= 1 contrast) -> {a.out}",
          file=sys.stderr)
    if len(shared):
        print(f"[map_agi_ipms] {len(shared)} accession(s) shared by several loci: "
              f"{', '.join(shared)}", file=sys.stderr)
    if len(unresolved):
        print(f"[map_agi_ipms] {len(unresolved)} unresolved: "
              f"{', '.join(unresolved['accession'] + ' (' + unresolved[[f'{l}_call' for l in labels]].bfill(axis=1).iloc[:, 0].fillna('?') + ')')}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
