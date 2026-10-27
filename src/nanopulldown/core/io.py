"""
nanopulldown.core.io
====================
Small file helpers: atomic text writes, sequence digests, and the shared
helper for writing self-documenting Parquet files across the
pipeline: every column gets a short description embedded directly in the
file's own Arrow schema metadata, plus one table-level description of what
the file is and how it relates to its siblings (e.g. row-alignment with a
companion .npy, or "see fold.parquet for the full arrays"). Written once here
so the writers can't drift apart on how — or whether — they document their
own output.

Metadata is plain Arrow schema metadata (bytes key/value pairs), so it
travels with the file for any reader: pyarrow.parquet.read_schema(path),
DuckDB's parquet_schema(), Data Wrangler, etc. It costs nothing at read
time (pandas.read_parquet ignores it) and nothing at compress time (it's
schema, not data).

Usage:
    write_parquet_with_metadata(
        df, path,
        table_description="One row per ...",
        column_descriptions={"col_a": "...", "col_b": "...", ...},
    )
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def write_parquet_with_metadata(
    df: pd.DataFrame,
    path: Path,
    *,
    table_description: str,
    column_descriptions: dict[str, str],
    column_references: dict[str, str] | None = None,
    table_references: str | None = None,
    compression: str = "zstd",
) -> None:
    """Write `df` to `path` as Parquet, embedding `table_description` as
    schema-level metadata and each entry of `column_descriptions` as that
    column's own field-level metadata. Every column in `df` must have an
    entry in `column_descriptions` (raises otherwise) — deliberately
    strict, so a column added later can't silently ship undocumented.
    `column_references` / `table_references` (the publications behind a score,
    core.metric_refs) go under a `references` key next to `description`."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    missing = set(df.columns) - set(column_descriptions)
    if missing:
        raise ValueError(
            f"write_parquet_with_metadata({path}): no description for column(s): {sorted(missing)}"
        )

    table = pa.Table.from_pandas(df, preserve_index=False)
    refs = column_references or {}
    documented_fields = [
        field.with_metadata({b"description": column_descriptions[field.name].encode(),
                             **({b"references": refs[field.name].encode()} if field.name in refs else {})})
        for field in table.schema
    ]
    meta = {b"description": table_description.encode()}
    if table_references:
        meta[b"references"] = table_references.encode()
    documented_schema = pa.schema(documented_fields, metadata=meta)
    table = pa.Table.from_arrays(table.columns, schema=documented_schema)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=compression)


def read_schema_metadata(path: Path) -> tuple[str | None, dict[str, str]]:
    """(table_description, {column: description}) as written above —
    convenience for inspecting a file's documentation from a REPL/notebook
    without reaching for pyarrow.parquet directly."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(str(path))
    table_desc = (schema.metadata or {}).get(b"description")
    columns = {}
    for field in schema:
        desc = (field.metadata or {}).get(b"description") if field.metadata else None
        if desc is not None:
            columns[field.name] = desc.decode()
    return (table_desc.decode() if table_desc else None), columns

def seq_md5(seq: str) -> str:
    return hashlib.md5(seq.encode("ascii")).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + rename so concurrent jobs never see a partial cache entry."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)
