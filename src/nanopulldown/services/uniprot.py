#!/usr/bin/env python3
"""
nanopulldown.services.uniprot — Stage 1a
====================================
Everything nanopulldown asks UniProt's REST API, each answer cached on disk
so repeated pairs sharing a bait/prey only hit the network once:

* sequence  -- accession -> raw FASTA under `uniprot.cache_dir`
* metadata  -- accession -> names, gene, ordered locus (AGI), organism,
  function, subcellular location, as JSON under `uniprot.meta_cache_dir`;
  batched through /uniprotkb/accessions. nanopulldown.core turns these into the
  human-readable labels the reports use (see configs/_schema.md).
* queries   -- a UniProt query string (GO / InterPro / organism ...) ->
  matching accessions, snapshotted to `uniprot.query_dir` with the release
  it came from, so a GO-term input set is reproducible.

Deliberately small and dependency-light (`requests` only, no `alphapulldown-
input-parser`): nanopulldown only needs those three things, not that
package's full feature/MSA object model.

Also parses AlphaPulldown's compact protein-spec syntax
(`ID`, `ID:N` for N copies, `ID:start-stop[:start2-stop2...]` for one or more
regions, `ID:N:start-stop` for both) so bait/prey entries in
configs/baits.txt / configs/preys.txt can use the same conventions
AlphaPulldown's sample sheet does.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests

from nanopulldown.core.config import load_config
from nanopulldown.core.io import atomic_write
from nanopulldown.core.specs import apply_regions, parse_protein_spec


def fasta_to_sequence(fasta_text: str) -> str:
    lines = [ln.strip() for ln in fasta_text.splitlines() if ln.strip()]
    return "".join(ln for ln in lines if not ln.startswith(">"))


def fetch_uniprot_fasta(accession: str, cfg: dict, cache_dir: Path) -> str:
    """Fetch (or reuse a cached) raw FASTA for one UniProt accession."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{accession}.fasta"
    if cache_path.exists():
        return cache_path.read_text()

    ucfg = cfg["uniprot"]
    url = f"{ucfg['base_url']}/{accession}.fasta"
    last_err = None
    for attempt in range(int(ucfg.get("retries", 3))):
        try:
            r = requests.get(url, timeout=float(ucfg.get("timeout_s", 30)))
            r.raise_for_status()
            cache_path.write_text(r.text)
            return r.text
        except Exception as e:                    # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"UniProt fetch failed for {accession} ({url}): {last_err}")


def fetch_sequence(spec: str, cfg: dict, cache_dir: Path | None = None) -> str:
    """Full resolve: parse spec -> fetch FASTA (cached) -> apply regions."""
    if cache_dir is None:
        cache_dir = Path(cfg["dirs"]["sequences"])
    ps = parse_protein_spec(spec)
    fasta = fetch_uniprot_fasta(ps.uniprot, cfg, cache_dir)
    return apply_regions(fasta_to_sequence(fasta), ps.regions)


# ── metadata ────────────────────────────────────────────────────────────────

META_FIELDS = ("accession,reviewed,protein_name,gene_names,gene_oln,organism_name,"
               "length,cc_function,cc_subcellular_location")
# TrEMBL placeholders that say nothing -- treated as "no protein name"
_GENERIC_NAME = re.compile(r"^(putative |probable )?(uncharacteri[sz]ed|unknown|hypothetical)"
                           r"( conserved)? protein\b|^protein of unknown function|^expressed protein$",
                           re.IGNORECASE)
_AGI = re.compile(r"\bAt([1-5CM])g(\d{5})\b", re.IGNORECASE)


def _get(ucfg: dict, url: str, params: dict | None = None) -> requests.Response:
    last_err = None
    for attempt in range(int(ucfg.get("retries", 3))):
        try:
            r = requests.get(url, params=params, timeout=float(ucfg.get("timeout_s", 30)))
            r.raise_for_status()
            return r
        except Exception as e:                    # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"UniProt request failed ({url}): {last_err}")


def _agi(locus: str | None) -> str | None:
    """'At3g43700' / 'AXX17_At5g25230' -> 'AT3G43700' (Arabidopsis AGI code)."""
    m = _AGI.search(locus or "")
    return f"AT{m.group(1).upper()}G{m.group(2)}" if m else None


def parse_entry(e: dict) -> dict:
    """One UniProtKB JSON entry -> the flat metadata record the proteins table
    stores. Missing fields are None, never guessed."""
    desc = e.get("proteinDescription") or {}
    rec = (desc.get("recommendedName") or {}).get("fullName", {}).get("value")
    sub = next((n["fullName"]["value"] for n in desc.get("submissionNames", [])
                if n.get("fullName", {}).get("value")), None)
    name = rec or sub
    if name and _GENERIC_NAME.search(name):
        name = None
    genes = e.get("genes") or []
    gene = next((g["geneName"]["value"] for g in genes if g.get("geneName")), None)
    synonyms = [s["value"] for g in genes for s in g.get("synonyms", [])]
    loci = [o["value"] for g in genes for o in g.get("orderedLocusNames", [])]
    locus = next((a for a in map(_agi, loci) if a), None) or (loci[0] if loci else None)
    comments = e.get("comments") or []
    function = next((t["value"] for c in comments if c.get("commentType") == "FUNCTION"
                     for t in c.get("texts", [])), None)
    locations = [sl["location"]["value"] for c in comments
                 if c.get("commentType") == "SUBCELLULAR LOCATION"
                 for sl in c.get("subcellularLocations", []) if sl.get("location")]
    return {
        "accession": e.get("primaryAccession"),
        "reviewed": "reviewed" in (e.get("entryType") or "").lower()
                    and "unreviewed" not in (e.get("entryType") or "").lower(),
        "protein_name": name,
        "gene_name": gene,
        "gene_synonyms": synonyms,
        "locus": locus,
        "organism": (e.get("organism") or {}).get("scientificName"),
        "taxon_id": (e.get("organism") or {}).get("taxonId"),
        "length": (e.get("sequence") or {}).get("length"),
        "function": function,
        "subcellular_location": list(dict.fromkeys(locations)),
    }


def fetch_metadata(accessions, cfg: dict, cache_dir: Path | None = None,
                   batch: int = 200) -> dict[str, dict]:
    """{accession: metadata record} for every accession, cached per accession
    as JSON (delete the cache dir to refresh names after a UniProt release).
    Accessions UniProt does not return (obsolete, typo) get a record with
    only the accession, so callers always fall back to it."""
    ucfg = cfg["uniprot"]
    cache_dir = Path(cache_dir or ucfg.get("meta_cache_dir", "data/uniprot_meta"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for acc in dict.fromkeys(accessions):
        path = cache_dir / f"{acc}.json"
        if path.exists():
            out[acc] = json.loads(path.read_text())
        else:
            todo.append(acc)
    base = ucfg["base_url"].rstrip("/")
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        r = _get(ucfg, f"{base}/accessions",
                 {"accessions": ",".join(chunk), "fields": META_FIELDS, "format": "json"})
        release = r.headers.get("X-UniProt-Release")
        got = {}
        for e in r.json().get("results", []):
            md = parse_entry(e)
            md["uniprot_release"] = release
            got[md["accession"]] = md
        for acc in chunk:
            md = got.get(acc) or {"accession": acc, "uniprot_release": release, "missing": True}
            if not md.get("missing"):            # don't cache a miss: retried next time
                atomic_write(cache_dir / f"{acc}.json", json.dumps(md, indent=1))
            out[acc] = md
        print(f"[fetch_uniprot] metadata {min(i + batch, len(todo))}/{len(todo)} "
              f"(release {release})", file=sys.stderr)
    return out


# ── queries (GO terms, InterPro, organism ...) ──────────────────────────────

def build_query(q: dict) -> str:
    """Structured `uniprot_query:` source -> UniProt query string. Clauses are
    ANDed; within `go_any` / `interpro` the terms are ORed. `query:` (a raw
    UniProt query) is ANDed in as-is, so both forms combine.

      go: [GO:0061630, GO:0005634]   every term (incl. child terms)
      go_any: [...]                  at least one
      go_evidence: experimental      use go_exp: (experimental evidence only)
      interpro: [IPR001810, ...]     at least one
      organism: 3702                 taxon id
      proteome: UP000006548
      reviewed: true                 Swiss-Prot only
      query: "<raw clause>"
    """
    go_field = "go_exp" if str(q.get("go_evidence", "")).startswith("exp") else "go"

    def _go(t):
        return f"{go_field}:{str(t).upper().removeprefix('GO:')}"
    clauses = [f"({_go(t)})" for t in q.get("go") or []]
    if q.get("go_any"):
        clauses.append("(" + " OR ".join(_go(t) for t in q["go_any"]) + ")")
    if q.get("interpro"):
        clauses.append("(" + " OR ".join(f"xref:interpro-{t}" for t in q["interpro"]) + ")")
    if q.get("organism"):
        clauses.append(f"(organism_id:{q['organism']})")
    if q.get("proteome"):
        clauses.append(f"(proteome:{q['proteome']})")
    if q.get("reviewed") is not None:
        clauses.append(f"(reviewed:{str(bool(q['reviewed'])).lower()})")
    if q.get("query"):
        clauses.append(f"({q['query']})")
    if not clauses:
        raise ValueError(f"uniprot_query {q.get('name')!r} has no criteria")
    return " AND ".join(clauses)


def run_query(name: str, query: str, cfg: dict, refresh: bool = False) -> list[str]:
    """Accessions matching a UniProt query, snapshotted to
    <uniprot.query_dir>/<name>.tsv + <name>.json (query, release, date, count).
    The snapshot is reused while the query string is unchanged, so a run stays
    reproducible across UniProt releases; `refresh=True` re-queries."""
    ucfg = cfg["uniprot"]
    qdir = Path(ucfg.get("query_dir", "data/uniprot_retrieval"))
    qdir.mkdir(parents=True, exist_ok=True)
    tsv, meta = qdir / f"{name}.tsv", qdir / f"{name}.json"
    if not refresh and tsv.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if info.get("query") == query:
            return _tsv_accessions(tsv.read_text())
    base = ucfg["base_url"].rstrip("/")
    r = _get(ucfg, f"{base}/stream",
             {"query": query, "fields": "accession,gene_primary,protein_name", "format": "tsv"})
    atomic_write(tsv, r.text)
    accs = _tsv_accessions(r.text)
    atomic_write(meta, json.dumps({"name": name, "query": query, "n": len(accs),
                                   "uniprot_release": r.headers.get("X-UniProt-Release"),
                                   "retrieved": date.today().isoformat()}, indent=1))
    print(f"[fetch_uniprot] query {name}: {len(accs)} entries "
          f"(release {r.headers.get('X-UniProt-Release')})", file=sys.stderr)
    return accs


def _tsv_accessions(text: str) -> list[str]:
    lines = text.splitlines()[1:]                     # header: Entry ...
    return [ln.split("\t", 1)[0].strip() for ln in lines if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", help="UniProt accession or AlphaPulldown-style ID:N:start-stop")
    ap.add_argument("--out", help="write the resolved sequence (FASTA) here; default: print")
    ap.add_argument("--meta", action="store_true", help="print the spec's metadata record instead")
    ap.add_argument("--backfill", action="store_true",
                    help="fetch metadata for every accession in configs/pairs.parquet and "
                         "(re)write configs/proteins.parquet -- names for pairs expanded "
                         "before the proteins table existed")
    a = ap.parse_args()

    cfg = load_config()
    if a.backfill:
        from nanopulldown.core.proteins import update_proteins
        from nanopulldown.core.specs import load_pairs
        accs = sorted({s[r]["uniprot"] for s in load_pairs().values() for r in ("bait", "prey")})
        path = update_proteins(fetch_metadata(accs, cfg).values())
        print(f"[fetch_uniprot] {len(accs)} accession(s) -> {path}")
        return 0
    if not a.spec:
        ap.error("spec is required unless --backfill")
    if a.meta:
        acc = parse_protein_spec(a.spec).uniprot
        print(json.dumps(fetch_metadata([acc], cfg)[acc], indent=1))
        return 0
    seq = fetch_sequence(a.spec, cfg)
    record = f">{a.spec}\n{seq}\n"
    if a.out:
        Path(a.out).write_text(record)
        print(f"[fetch_uniprot] {a.spec}: {len(seq)} aa -> {a.out}")
    else:
        print(record, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
