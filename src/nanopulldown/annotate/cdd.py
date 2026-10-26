"""
nanopulldown.annotate.cdd — Stage 1c
====================================
CDD (NCBI Conserved Domain Database) annotation of every chain of every pair,
for the report to read PLIP contacts against (nanopulldown.report.cdd): each
CDD domain hit, and the conserved-feature SITES CDD curates per domain model
("dimer interface", "putative substrate binding site", "ATP binding site",
...) -- the residue-level annotation the InterPro API serves at
/protein/uniprot/<acc>?residues.

  * CDD spans come from the domain cache nanopulldown.annotate.domains fills
    (the `annotate_domains_batch` rule), so this step adds no domain lookups.
  * Sites are fetched here, once per full-length sequence with a plain
    UniProt accession (thread pool, the same InterPro rate limiter), and
    cached under data/annotation/_cache/sites_v1/<md5>.json. A site whose
    residue letters don't match the sequence (the pair's sequence is not
    the entry InterPro computed on) is dropped. A chain with no plain
    accession (isoform, InterProScan-fallback sequence) gets spans, no sites.
  * Region chains (fragments) get the full-length entry's spans and sites
    projected onto fragment numbering, like their domain hits.

Output: <annotation dir>/cdd_sites.parquet, one row per CDD domain location
(site = null) and one per site of it (site = the CDD description, residues =
its residues), chain numbering -- see COLUMNS.

Usage:
  nanopulldown annotate cdd --pairs <pair> [<pair> ...] --out <annotation dir>/cdd_sites.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from nanopulldown.annotate.domains import (
    _IPR_DB,
    CFG,
    STOP,
    _lim,
    _retry,
    _ua,
    annotation_targets,
    chain_record,
    interproscan_email,
    project_span,
    region_source,
    seq_accessions,
)
from nanopulldown.core.cache import annotation_cache_dir
from nanopulldown.core.io import atomic_write, seq_md5, write_parquet_with_metadata
from nanopulldown.core.specs import load_pairs, protein_chains
from nanopulldown.services.http import http_get

SITES_CACHE = "sites_v1"             # bump when the cached record's shape changes

COLUMNS = {
    "pair": "pair id",
    "role": "bait | prey",
    "chain": "chain id in the pair's structures",
    "cdd_accession": "CDD model accession (cdNNNNN)",
    "cdd_name": "CDD model name",
    "start": "first residue of the CDD domain location (chain numbering, 1-based)",
    "end": "last residue of the CDD domain location (inclusive)",
    "site": "null: the domain location itself; else the CDD conserved-feature description "
            "of a site inside that location",
    "residues": "site residues (chain numbering); empty for a domain row",
}


def interpro_sites(accession: str, seq: str, cfg: dict) -> list[dict]:
    """Residue-level sites of a UniProt entry (InterPro API `?residues`: CDD
    conserved features, PIRSR rules) -> [{db, accession, name, description,
    residues}]; sites whose residue letters don't match `seq` are dropped."""
    base = cfg.get("api_url", "https://www.ebi.ac.uk/interpro/api")
    r = http_get(_lim("interpro"), f"{base}/protein/uniprot/{accession}?residues",
                 headers={**_ua(), "Accept": "application/json"}, timeout=60)
    if r.status_code in (204, 404):
        return []
    r.raise_for_status()
    out = []
    for key, v in (r.json() or {}).items():
        db = (v.get("source_database") or "").lower()
        for loc in v.get("locations", []):
            res, ok = [], True
            for fr in loc.get("fragments", []):
                s, e = int(fr["start"]), int(fr["end"])
                if fr.get("residues") and seq[s - 1:e] != fr["residues"]:
                    ok = False
                    break
                res += range(s, e + 1)
            if ok and res:
                out.append({"db": _IPR_DB.get(db, v.get("source_database")),
                            "accession": v.get("accession") or key, "name": v.get("name") or key,
                            "description": loc.get("description") or "", "residues": sorted(set(res))})
    return out


def sequence_sites(seq: str, accession: str | None, cfg: dict, log=print) -> list[dict] | None:
    """interpro_sites of one full-length sequence, cached by md5(sequence);
    [] without an accession, None when the lookup failed (not cached, so the
    next run retries it)."""
    if not accession:
        return []
    cache = annotation_cache_dir(CFG, SITES_CACHE) / f"{seq_md5(seq)}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        sites = _retry(interpro_sites, accession, seq, cfg)
    except Exception as e:                           # noqa: BLE001
        log(f"  InterPro residues FAILED for {accession}: {e}")
        return None
    atomic_write(cache, json.dumps(sites))
    return sites


def chain_sites(spec: dict, ch: dict, cfg: dict, accs: dict) -> list[dict]:
    """One chain's sites: its own, or for a region chain the full-length
    entry's, projected onto fragment numbering (sites outside it dropped)."""
    log = lambda m: print(m, file=sys.stderr)      # noqa: E731
    src = region_source(spec, ch)
    if not src:
        return sequence_sites(ch["sequence"], accs.get(ch["sequence"]), cfg, log) or []
    full, acc, regions = src
    out = []
    for s in sequence_sites(full, acc, cfg, log) or []:
        res = [p[0] for r in s["residues"] for p in project_span(r, r, regions)]
        if res:
            out.append({**s, "residues": res})
    return out


def chain_rows(hits: list[dict], sites: list[dict]) -> list[dict]:
    """COLUMNS rows (minus pair/role/chain) of one chain: each CDD location,
    then its sites (a site's residues inside that location; one with none
    inside any location of its model gets a location of its own)."""
    locs = sorted({(h["accession"], int(h["start"]), int(h["end"])): h.get("name") or h["accession"]
                   for h in hits if h["db"] == "CDD"}.items())
    rows = {k: [] for k, _ in locs}
    names = dict(locs)
    for st in sites:
        if st["db"] != "CDD":
            continue
        placed = False
        for (acc, s, e) in rows:
            res = [r for r in st["residues"] if acc == st["accession"] and s <= r <= e]
            if res:
                rows[(acc, s, e)].append((st["description"], res))
                placed = True
        if not placed:
            k = (st["accession"], min(st["residues"]), max(st["residues"]))
            names.setdefault(k, st["name"])
            rows.setdefault(k, []).append((st["description"], st["residues"]))
    out = []
    for (acc, s, e), sts in sorted(rows.items(), key=lambda kv: kv[0][1:]):
        base = {"cdd_accession": acc, "cdd_name": names[(acc, s, e)], "start": s, "end": e}
        out.append({**base, "site": None, "residues": []})
        out += [{**base, "site": d, "residues": r} for d, r in sts]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    acfg = CFG["annotate"]
    workers = a.workers or int(acfg.get("web_workers", 4))
    email = interproscan_email(acfg)
    specs = load_pairs()

    # sites cache: once per full-length sequence (region chains read their entry's)
    targets: dict[str, str] = {}
    for p in a.pairs:
        targets.update({s: acc for s, acc in annotation_targets(specs[p]).items() if acc})
    cache = annotation_cache_dir(CFG, SITES_CACHE)
    todo = [s for s in targets if not (cache / f"{seq_md5(s)}.json").exists()]
    print(f"[annotate_cdd] {len(targets)} sequence(s) with an accession, {len(todo)} to fetch sites for",
          flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(sequence_sites, s, targets[s], acfg["interproscan"],
                            lambda m: print(m, file=sys.stderr, flush=True)) for s in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i % 50 == 0 or i == len(todo):
                print(f"[annotate_cdd] sites {i}/{len(todo)}", flush=True)
    if STOP.is_set():
        print("[annotate_cdd] ABORTED: InterPro kept throttling (circuit breaker). Progress is cached; "
              "wait a while and rerun.", file=sys.stderr)
        return 2
    missing = sum(not (cache / f"{seq_md5(s)}.json").exists() for s in targets)
    if missing:
        # fatal: a missing record would read as "no sites" in every table and figure
        print(f"[annotate_cdd] {missing} sequence(s) have no cached sites (service errors); rerun",
              file=sys.stderr)
        return 1

    rows = []
    for p in a.pairs:
        spec = specs[p]
        accs = seq_accessions(spec)
        for ch in protein_chains(spec):
            hits = chain_record(spec, ch, acfg, email, accs,
                                log=lambda m: print(m, file=sys.stderr))["hits"]
            for r in chain_rows(hits, chain_sites(spec, ch, acfg["interproscan"], accs)):
                rows.append({"pair": p, "role": ch["role"], "chain": ch["id"], **r})
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    write_parquet_with_metadata(
        df, Path(a.out),
        table_description="CDD domain locations and their conserved-feature sites, per chain of every "
                          "pair of the run (nanopulldown.annotate.cdd)",
        column_descriptions=COLUMNS)
    n_sites = int(df["site"].notna().sum())
    print(f"[annotate_cdd] {len(df) - n_sites} CDD location(s), {n_sites} site(s) across "
          f"{len(a.pairs)} pair(s) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
