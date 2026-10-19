#!/usr/bin/env python3
"""
Resolve Arabidopsis AGI/Araport locus IDs in PXD013906 Table S3
to UniProt accessions using UniProt's asynchronous ID Mapping API.

The current UniProt API returns mapping results as:
    {"from": "AT1G01020", "to": "F4HQG4"}

When multiple UniProt accessions map to one AGI, this script retrieves
the corresponding UniProt entries and prefers a reviewed (Swiss-Prot)
entry when one exists.

Usage:
    python resolve_pxd013906_uniprot.py \
        --xlsx ~/Downloads/pnas.1912199117.sd01.xlsx \
        --out data/reference/pxd013906_uniprot_mapping.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://rest.uniprot.org/idmapping"
KB_API = "https://rest.uniprot.org/uniprotkb"
CHUNK = 500


def load_table_s3(xlsx_path: Path) -> pd.DataFrame:
    s3 = pd.read_excel(xlsx_path, sheet_name="Table S3", header=6)
    s3.columns = [
        "accession",
        "logfc",
        "pval",
        "adjp",
        "accession2",
        "annotation",
    ]

    # Ignore rows that are already UniProt accessions.
    s3 = s3[
        ~s3["accession"].astype(str).str.startswith("sp|", na=False)
    ].copy()

    # Strip protein isoform suffix, e.g. AT3G43700.1 -> AT3G43700.
    s3["agi"] = s3["accession"].astype(str).str.replace(
        r"\.\d+$", "", regex=True
    )

    return s3[["agi", "logfc", "pval", "adjp", "annotation"]]


def submit_mapping_job(ids: list[str]) -> str:
    r = requests.post(
        f"{API}/run",
        data={
            "from": "Araport",
            "to": "UniProtKB",
            "ids": ",".join(ids),
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["jobId"]


def wait_for_job(
    job_id: str,
    poll_s: float = 2.0,
    timeout_s: float = 300.0,
) -> None:
    start = time.time()

    while True:
        r = requests.get(
            f"{API}/status/{job_id}",
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        status = payload.get("jobStatus")

        if status == "FINISHED" or "results" in payload:
            return

        if status == "FAILED":
            raise RuntimeError(
                f"UniProt mapping job {job_id} failed: {payload}"
            )

        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"UniProt mapping job {job_id} timed out"
            )

        time.sleep(poll_s)


def fetch_all_results(job_id: str) -> list[dict]:
    """
    Fetch the simple ID-mapping results.

    Current UniProt response format:
        {"from": "AT1G01020", "to": "F4HQG4"}
    """
    url = f"{API}/results/{job_id}?size=500"
    out: list[dict] = []

    while url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        payload = r.json()

        for result in payload.get("results", []):
            out.append(
                {
                    "from": result.get("from"),
                    "to": result.get("to"),
                }
            )

        for failed_id in payload.get("failedIds", []):
            out.append(
                {
                    "from": failed_id,
                    "to": None,
                }
            )

        link = r.headers.get("Link", "")
        url = None

        if 'rel="next"' in link:
            url = link.split(";")[0].strip("<> ")

    return out


def resolve_chunk(ids: list[str]) -> list[dict]:
    job_id = submit_mapping_job(ids)
    wait_for_job(job_id)
    return fetch_all_results(job_id)


def get_uniprot_entry(accession: str) -> dict | None:
    """
    Retrieve one UniProtKB entry and extract the fields needed for
    reviewed/unreviewed selection and the output table.
    """
    url = f"{KB_API}/{accession}"

    r = requests.get(url, timeout=60)

    if r.status_code == 404:
        return None

    r.raise_for_status()
    data = r.json()

    entry_type = str(data.get("entryType", ""))
    reviewed = "reviewed" in entry_type.lower()

    protein_name = ""
    protein_description = data.get("proteinDescription", {})
    recommended = protein_description.get("recommendedName", {})
    full_name = recommended.get("fullName", {})
    protein_name = full_name.get("value", "")

    genes = []
    for gene in data.get("genes", []):
        gene_name = gene.get("geneName", {}).get("value")
        if gene_name:
            genes.append(gene_name)

    length = data.get("sequence", {}).get("length")

    organism = data.get("organism", {}).get("scientificName")

    return {
        "uniprot": accession,
        "reviewed": reviewed,
        "genes": ";".join(genes),
        "proteinDescription": protein_name,
        "length": length,
        "organism": organism,
    }


def choose_best_mapping(mapped: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieve UniProt metadata and prefer reviewed entries.

    If no reviewed entry exists for an AGI, the first available
    unreviewed entry is retained.
    """
    accessions = (
        mapped["uniprot"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    metadata: list[dict] = []

    print(
        f"[resolve] retrieving metadata for {len(accessions)} "
        "UniProt accessions...",
        flush=True,
    )

    for i, accession in enumerate(accessions, start=1):
        try:
            entry = get_uniprot_entry(accession)
            if entry is not None:
                metadata.append(entry)
        except requests.RequestException as exc:
            print(
                f"[resolve] warning: could not retrieve {accession}: {exc}",
                file=sys.stderr,
                flush=True,
            )

        if i % 100 == 0:
            print(
                f"[resolve] metadata {i}/{len(accessions)}",
                flush=True,
            )

        # Be polite to the API.
        time.sleep(0.05)

    meta = pd.DataFrame(metadata)

    if meta.empty:
        return pd.DataFrame(
            columns=[
                "agi",
                "uniprot",
                "reviewed",
                "genes",
                "proteinDescription",
                "length",
                "organism",
            ]
        )

    mapped = mapped.merge(meta, on="uniprot", how="left")

    # Reviewed first, then preserve the original mapping order.
    mapped["_order"] = range(len(mapped))
    mapped["_reviewed_sort"] = mapped["reviewed"].fillna(False)

    mapped = mapped.sort_values(
        ["agi", "_reviewed_sort", "_order"],
        ascending=[True, False, True],
    )

    best = mapped.drop_duplicates("agi", keep="first").copy()

    return best.drop(columns=["_order", "_reviewed_sort"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    xlsx = args.xlsx.expanduser()

    if not xlsx.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx}")

    s3 = load_table_s3(xlsx)

    agi_list = sorted(s3["agi"].dropna().unique())

    print(
        f"[resolve] {len(agi_list)} unique AGI loci to resolve",
        flush=True,
    )

    raw_rows: list[dict] = []

    n_chunks = -(-len(agi_list) // CHUNK)

    for i in range(0, len(agi_list), CHUNK):
        chunk = agi_list[i : i + CHUNK]

        print(
            f"[resolve] chunk {i // CHUNK + 1}/{n_chunks} "
            f"({len(chunk)} ids)...",
            flush=True,
        )

        raw_rows.extend(resolve_chunk(chunk))

    mapped = pd.DataFrame(raw_rows)

    if mapped.empty:
        raise RuntimeError("UniProt returned no mapping results.")

    mapped_ok = mapped[mapped["to"].notna()].copy()
    mapped_ok = mapped_ok.rename(
        columns={
            "from": "agi",
            "to": "uniprot",
        }
    )

    best = choose_best_mapping(mapped_ok)

    failed = sorted(set(agi_list) - set(best["agi"]))

    print(
        f"[resolve] resolved {len(best)}/{len(agi_list)} "
        f"({len(failed)} with no UniProt hit)",
        flush=True,
    )

    merged = s3.merge(best, on="agi", how="left")

    cols = [
        "agi",
        "uniprot",
        "reviewed",
        "genes",
        "proteinDescription",
        "length",
        "organism",
        "logfc",
        "pval",
        "adjp",
        "annotation",
    ]

    merged = merged[[c for c in cols if c in merged.columns]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    n_uniprot = merged["uniprot"].notna().sum()

    print(
        f"[resolve] wrote {args.out} "
        f"({len(merged)} rows, {n_uniprot} with a UniProt accession)",
        flush=True,
    )

    if failed:
        failed_path = args.out.with_name(
            args.out.stem + "_unresolved.txt"
        )
        failed_path.write_text("\n".join(failed) + "\n")
        print(
            f"[resolve] {len(failed)} unresolved AGI codes -> "
            f"{failed_path}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
