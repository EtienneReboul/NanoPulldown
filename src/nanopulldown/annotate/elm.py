#!/usr/bin/env python3
"""
nanopulldown.annotate.elm — Stage 1d
====================================
Known short linear motifs (SLiMs) in the disordered regions of baits and
preys, from the ELM database (elm.eu.org; Kumar et al. 2024, NAR 52:D442),
and the bait x prey pairs where one side carries a motif and the other the
Pfam domain ELM says binds it -- motif-domain interaction hypotheses for the
disordered regions that CDD / InterPro domains do not cover.

  fetch  The three ELM downloads, once, into <annotation>/_cache/elm/
         (delete the directory to refresh; the ELM version is recorded in
         the output tables' metadata):
           elms_index.tsv           motif classes (ELMIdentifier, Regex, Probability)
           interaction_domains.tsv  ELM class -> Pfam domain(s) that bind it
           instances.tsv            literature instances (true positives kept)
  scan   Every ELM regex over every FULL-LENGTH protein of the run (a fragment
         chain is scanned through its full-length entry, then projected: the
         ^/$-anchored classes mean the protein's termini, not a fragment's,
         and ELM instances use full-length numbering). Matches overlap (a
         lookahead finds one per start position). Kept when at least
         annotate.elm.min_disorder_fraction of the motif's residues have
         AIUPred disorder >= annotate.consensus.disorder_cutoff; each hit
         records its overlap with the MoRF segments (AIUPred binding track,
         annotate.morfchibi thresholds, as the domain map draws them) and any
         ELM instance on the same entry it overlaps.
  pairs  Per pair, a motif on one side whose binding Pfam domain is on the
         other side (Pfam hits from the domain cache, projected onto
         fragments; a motif cut by a fragment edge is dropped).

Outputs (scan rule):
  <annotation>/elm_motifs.parquet           one row per disordered motif hit per protein
  <annotation>/elm_partner_matches.parquet  one row per (pair, motif hit, partner Pfam domain)
  <reports>/preprocessing/elm_partner_matches.csv  the same, ranked, with display names

Ranking (both tables): known ELM instance > partner domain match > MoRF
overlap > lower ELM probability (less degenerate regex). Hits are
hypotheses: short degenerate classes (MOD_ phosphosites, CLV_ cleavage
sites) match by chance often -- the probability column is the expected rate
of a random match. Only KNOWN classes are found, and only interactions with a
folded binding domain on one side.

Usage:
  nanopulldown annotate elm-fetch --out-dir <annotation>/_cache/elm
  nanopulldown annotate elm --pairs <pair> [...] --elm-dir <annotation>/_cache/elm \\
      --motifs <parquet> --matches <parquet> --csv <csv>
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from pathlib import Path

import requests

from nanopulldown.core.config import LazyConfig
from nanopulldown.core.io import atomic_write, seq_md5

CFG = LazyConfig()

ELM_TABLES = {
    # local file: path under the ELM site
    "elms_index.tsv": "elms/elms_index.tsv",
    "interaction_domains.tsv": "interactiondomains.tsv",
    "instances.tsv": "instances.tsv?q=*",       # without q=* the site serves only the 100 newest
}
ELM_REFERENCE = "Kumar M et al. (2024) ELM-the Eukaryotic Linear Motif resource-2024 update. NAR 52:D442-D455"


# ── fetch ───────────────────────────────────────────────────────────────

def fetch_main() -> int:
    ap = argparse.ArgumentParser(description="download the ELM tables (once; delete to refresh)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    ecfg = CFG["annotate"].get("elm") or {}
    base = ecfg.get("base_url", "http://elm.eu.org").rstrip("/")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, path in ELM_TABLES.items():
        dest = out / name
        if dest.exists():
            print(f"[elm] {name}: cached", flush=True)
            continue
        for attempt in range(1, 6):              # the instance export answers 500 now and then
            try:
                r = requests.get(f"{base}/{path}", timeout=300,
                                 headers={"User-Agent": "nanopulldown/annotate_elm"})
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 5:
                    raise
                print(f"[elm] {name}: {e}; retry {attempt}/4 in {30 * attempt} s", file=sys.stderr, flush=True)
                time.sleep(30 * attempt)
        # the site answers errors with an HTML page and status 200 at times
        if r.text.lstrip().startswith("<") or '"ELMIdentifier"' not in r.text and '"ELM identifier"' not in r.text:
            print(f"[elm] {name}: unexpected content from {base}/{path}: {r.text[:200]!r}", file=sys.stderr)
            return 1
        atomic_write(dest, r.text)
        print(f"[elm] {name}: {len(r.text.splitlines())} lines", flush=True)
    return 0


# ── ELM tables ──────────────────────────────────────────────────────────

def _read_tsv(path: Path) -> tuple[list[dict], dict[str, str]]:
    """(rows, {header key: value}) of an ELM download: `#Key: value` lines,
    then a quoted TSV."""
    head, body = {}, []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            k, _, v = line[1:].partition(":")
            head[k.strip()] = v.strip()
        elif line.strip():
            body.append(line)
    return list(csv.DictReader(io.StringIO("\n".join(body)), delimiter="\t")), head


def load_elm(elm_dir: Path, ecfg: dict) -> dict:
    classes, head = _read_tsv(elm_dir / "elms_index.tsv")
    skip = {t.upper() for t in ecfg.get("exclude_types") or []}
    pmax = ecfg.get("max_probability")
    out_classes = []
    for c in classes:
        etype = c["ELMIdentifier"].split("_", 1)[0]
        prob = float(c["Probability"] or "nan")
        if etype in skip or (pmax is not None and prob > float(pmax)):
            continue
        try:
            pat = re.compile(f"(?=({c['Regex']}))")
        except re.error as e:
            print(f"[elm] {c['ELMIdentifier']}: regex {c['Regex']!r} not usable ({e})", file=sys.stderr)
            continue
        out_classes.append({"id": c["ELMIdentifier"], "acc": c["Accession"], "type": etype,
                            "name": c["FunctionalSiteName"], "regex": c["Regex"], "prob": prob, "pat": pat})

    domains: dict[str, list[tuple[str, str]]] = {}
    for r in _read_tsv(elm_dir / "interaction_domains.tsv")[0]:
        domains.setdefault(r["ELM identifier"], []).append(
            (r["Interaction Domain Id"], r["Interaction Domain Description"]))

    instances: dict[str, list[tuple[str, int, int, str]]] = {}
    for r in _read_tsv(elm_dir / "instances.tsv")[0]:
        if r["InstanceLogic"] != "true positive":
            continue
        for acc in set(r["Accessions"].split()) | {r["Primary_Acc"]}:
            instances.setdefault(acc, []).append((r["ELMIdentifier"], int(r["Start"]), int(r["End"]),
                                                  r["Accession"]))
    version = f"ELM classes {head.get('ELM_Classes_Download_Date', '?')}"
    print(f"[elm] {len(out_classes)}/{len(classes)} classes used, {len(domains)} with a binding domain, "
          f"{sum(map(len, instances.values()))} instance entries ({version})", flush=True)
    return {"classes": out_classes, "domains": domains, "instances": instances, "version": version}


# ── scan ────────────────────────────────────────────────────────────────

def morf_segments(binding: list[float], threshold: float, min_len: int) -> list[tuple[int, int]]:
    """1-based inclusive runs of binding >= threshold, >= min_len long
    (as nanopulldown.annotate.merge calls MoRFs for the domain map)."""
    segs, start = [], None
    for i, v in enumerate(list(binding) + [-1.0], start=1):
        if v >= threshold and start is None:
            start = i
        elif v < threshold and start is not None:
            if i - start >= min_len:
                segs.append((start, i - 1))
            start = None
    return segs


def scan_protein(seq: str, acc: str | None, tracks: dict, elm: dict, cutoff: float,
                 min_frac: float, morf: dict) -> list[dict]:
    """Disordered ELM hits of one full-length sequence (1-based, inclusive)."""
    dis = tracks.get("disorder") or []
    if len(dis) != len(seq):
        return []
    bind = tracks.get("binding") or []
    morfs = morf_segments(bind, float(morf["score_threshold"]), int(morf["min_len"])) if bind else []
    known = elm["instances"].get(acc or "", [])
    rows = []
    for c in elm["classes"]:
        for m in c["pat"].finditer(seq):
            s0, e0 = m.start(1), m.end(1)
            if e0 <= s0:
                continue
            frac = sum(d >= cutoff for d in dis[s0:e0]) / (e0 - s0)
            if frac < min_frac:
                continue
            s, e = s0 + 1, e0
            inst = [i for elm_id, a, b, i in known if elm_id == c["id"] and a <= e and b >= s]
            rows.append({
                "elm_identifier": c["id"], "elm_accession": c["acc"], "elm_type": c["type"],
                "elm_name": c["name"], "start": s, "end": e, "motif": seq[s0:e0],
                "probability": c["prob"],
                "disorder_fraction": round(frac, 3),
                "mean_disorder": round(sum(dis[s0:e0]) / (e0 - s0), 3),
                "mean_binding": round(sum(bind[s0:e0]) / (e0 - s0), 3) if bind else None,
                "morf_overlap": sum(max(0, min(e, b) - max(s, a) + 1) for a, b in morfs),
                "known_instance": ";".join(inst) or None,
                "binding_pfam": ";".join(p for p, _ in elm["domains"].get(c["id"], [])) or None,
            })
    return rows


MOTIF_COLUMNS = {
    "accession": "UniProt accession of the full-length protein (null: no plain accession)",
    "seq_md5": "md5 of the scanned full-length sequence (the annotation cache key)",
    "length": "full-length sequence length",
    "elm_identifier": "ELM class identifier (e.g. LIG_14-3-3_CanoR_1)",
    "elm_accession": "ELM class accession (ELMExxxxxx)",
    "elm_type": "ELM class type: CLV cleavage, DEG degron, DOC docking, LIG ligand, MOD modification, "
                "TRG targeting",
    "elm_name": "ELM functional site name",
    "start": "first motif residue, full-length numbering (1-based)",
    "end": "last motif residue (inclusive)",
    "motif": "matched sequence",
    "probability": "ELM regex probability: expected rate of a chance match (lower = more specific)",
    "disorder_fraction": "fraction of motif residues with AIUPred disorder >= annotate.consensus.disorder_cutoff",
    "mean_disorder": "mean AIUPred disorder score over the motif",
    "mean_binding": "mean AIUPred binding (MoRF) score over the motif",
    "morf_overlap": "motif residues inside a MoRF segment (AIUPred binding track, annotate.morfchibi)",
    "known_instance": "ELM instance accession(s) (true positive) of this class overlapping the hit on this "
                      "UniProt entry; null if none",
    "binding_pfam": "Pfam domain(s) ELM lists as binding this class, ';'-joined; null if none",
}

MATCH_COLUMNS = {
    "pair": "pair id",
    "rank": "rank within the pair (1 = best): known instance > MoRF overlap > lower probability",
    "motif_role": "bait | prey: the side carrying the motif",
    "motif_chain": "chain id carrying the motif",
    "motif_accession": "UniProt accession of the motif protein",
    "elm_identifier": MOTIF_COLUMNS["elm_identifier"],
    "elm_type": MOTIF_COLUMNS["elm_type"],
    "start": "first motif residue, chain numbering (1-based)",
    "end": "last motif residue, chain numbering (inclusive)",
    "full_start": "first motif residue, full-length numbering",
    "full_end": "last motif residue, full-length numbering",
    "motif": MOTIF_COLUMNS["motif"],
    "probability": MOTIF_COLUMNS["probability"],
    "disorder_fraction": MOTIF_COLUMNS["disorder_fraction"],
    "morf_overlap": MOTIF_COLUMNS["morf_overlap"],
    "known_instance": MOTIF_COLUMNS["known_instance"],
    "partner_role": "bait | prey: the side carrying the binding domain",
    "partner_chain": "chain id carrying the binding domain",
    "partner_accession": "UniProt accession of the partner protein",
    "partner_pfam": "Pfam accession of the binding domain (ELM interaction domain)",
    "partner_pfam_name": "Pfam name of the binding domain",
    "partner_domain_start": "first residue of the partner's hits of this Pfam family, chain numbering",
    "partner_domain_end": "last residue of the partner's hits of this Pfam family, chain numbering",
    "partner_domain_copies": "number of Pfam hits of this family on the partner chain (repeats: one per copy)",
}


def _rank_key(r: dict) -> tuple:
    return (r["known_instance"] is None, -(r["morf_overlap"] > 0), r["probability"])


def main() -> int:
    import pandas as pd

    from nanopulldown.annotate.disorder import cache_is_current, run_aiupred_batch
    from nanopulldown.annotate.domains import (annotation_targets, chain_record, interproscan_email,
                                               project_span, region_source, seq_accessions)
    from nanopulldown.core.cache import annotation_cache_dir, cached_disorder_tracks
    from nanopulldown.core.io import write_parquet_with_metadata
    from nanopulldown.core.proteins import load_proteins, protein_name
    from nanopulldown.core.specs import load_pairs, protein_chains

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--elm-dir", required=True)
    ap.add_argument("--motifs", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()

    acfg = CFG["annotate"]
    ecfg = acfg.get("elm") or {}
    cutoff = float(acfg["consensus"]["disorder_cutoff"])
    min_frac = float(ecfg.get("min_disorder_fraction", 0.8))
    elm = load_elm(Path(a.elm_dir), ecfg)

    # full-length sequences to scan: {seq: accession or None}
    specs = load_pairs()
    targets: dict[str, str | None] = {}
    for p in a.pairs:
        for seq, acc in annotation_targets(specs[p]).items():
            if acc or seq not in targets:
                targets[seq] = acc

    # AIUPred of the full-length entries of fragment chains is not in the
    # disorder cache of runs annotated before the batch covered them
    dcache = annotation_cache_dir(CFG, "disorder")
    todo = [s for s in targets if not cache_is_current(dcache / f"{seq_md5(s)}.json")]
    if todo:
        print(f"[elm] AIUPred for {len(todo)} uncached sequence(s)", flush=True)
        for md5, tracks in run_aiupred_batch(todo, acfg["aiupred"]).items():
            atomic_write(dcache / f"{md5}.json", json.dumps(tracks))

    hits_by_seq: dict[str, list[dict]] = {}
    motif_rows, skipped = [], 0
    for seq, acc in targets.items():
        tracks = cached_disorder_tracks(CFG, seq)
        if len(tracks.get("disorder") or []) != len(seq):
            skipped += 1
            continue
        hits = scan_protein(seq, acc, tracks, elm, cutoff, min_frac, acfg["morfchibi"])
        hits_by_seq[seq] = hits
        motif_rows += [{"accession": acc, "seq_md5": seq_md5(seq), "length": len(seq), **h} for h in hits]
    if skipped:
        print(f"[elm] WARNING: {skipped} sequence(s) without AIUPred tracks were not scanned", file=sys.stderr)
    print(f"[elm] {len(motif_rows)} disordered motif hit(s) in {len(hits_by_seq)} protein(s)", flush=True)

    # per pair: motif on one side, its binding Pfam domain on the other
    email = interproscan_email(acfg)
    match_rows = []
    for p in a.pairs:
        spec = specs[p]
        accs = seq_accessions(spec)
        chains = []
        for ch in protein_chains(spec):
            src = region_source(spec, ch)
            full, regions = (src[0], src[2]) if src else (ch["sequence"], None)
            pfam = [h for h in chain_record(spec, ch, acfg, email, accs,
                                            log=lambda m: print(m, file=sys.stderr))["hits"]
                    if h.get("db") == "Pfam"]
            chains.append({"id": ch["id"], "role": ch["role"], "full": full, "regions": regions,
                           "acc": (spec.get(ch["role"]) or {}).get("uniprot"), "pfam": pfam})
        rows = []
        for mc in chains:
            for h in hits_by_seq.get(mc["full"], []):
                if not h["binding_pfam"]:
                    continue
                if mc["regions"]:
                    span = project_span(h["start"], h["end"], mc["regions"])
                    if len(span) != 1 or span[0][1] - span[0][0] != h["end"] - h["start"]:
                        continue                     # outside the fragment, or cut by its edge
                    s, e = span[0]
                else:
                    s, e = h["start"], h["end"]
                wanted = set(h["binding_pfam"].split(";"))
                for pc in chains:
                    if pc["role"] == mc["role"]:
                        continue
                    # one row per Pfam family: repeats (WD40 blades, ...) are one hit each
                    fams: dict[str, list[dict]] = {}
                    for d in pc["pfam"]:
                        if str(d.get("accession")).split(".")[0] in wanted:
                            fams.setdefault(d["accession"], []).append(d)
                    for pf_acc, ds in fams.items():
                        rows.append({
                            "pair": p, "motif_role": mc["role"], "motif_chain": mc["id"],
                            "motif_accession": mc["acc"], "elm_identifier": h["elm_identifier"],
                            "elm_type": h["elm_type"], "start": s, "end": e,
                            "full_start": h["start"], "full_end": h["end"], "motif": h["motif"],
                            "probability": h["probability"], "disorder_fraction": h["disorder_fraction"],
                            "morf_overlap": h["morf_overlap"], "known_instance": h["known_instance"],
                            "partner_role": pc["role"], "partner_chain": pc["id"],
                            "partner_accession": pc["acc"], "partner_pfam": pf_acc,
                            "partner_pfam_name": ds[0].get("name"),
                            "partner_domain_start": min(int(d["start"]) for d in ds),
                            "partner_domain_end": max(int(d["end"]) for d in ds),
                            "partner_domain_copies": len(ds),
                        })
        rows.sort(key=_rank_key)
        for i, r in enumerate(rows, start=1):
            r["rank"] = i
        match_rows += rows
    print(f"[elm] {len(match_rows)} motif-domain match(es) in "
          f"{len({r['pair'] for r in match_rows})}/{len(a.pairs)} pair(s)", flush=True)

    motif_rows.sort(key=lambda r: (r["accession"] or "", r["binding_pfam"] is None, *_rank_key(r)))
    motifs = pd.DataFrame(motif_rows, columns=list(MOTIF_COLUMNS))
    matches = pd.DataFrame(match_rows, columns=list(MATCH_COLUMNS))
    for path in (a.motifs, a.matches, a.csv):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    params = (f"{elm['version']}; disorder >= {cutoff} on >= {min_frac:.0%} of the motif; "
              f"exclude_types={ecfg.get('exclude_types') or []}; max_probability={ecfg.get('max_probability')}")
    write_parquet_with_metadata(
        motifs, Path(a.motifs), column_descriptions=MOTIF_COLUMNS, table_references=ELM_REFERENCE,
        table_description=f"ELM motif hits in the disordered regions of every full-length protein ({params})")
    write_parquet_with_metadata(
        matches, Path(a.matches), column_descriptions=MATCH_COLUMNS, table_references=ELM_REFERENCE,
        table_description=f"ELM motif on one side x its binding Pfam domain on the other, per pair ({params})")

    # human-readable: run-wide ranking, with display names
    proteins = load_proteins()
    view = matches.copy()
    view.insert(1, "motif_protein", [protein_name(x, proteins) if x else "" for x in view["motif_accession"]])
    view.insert(2, "partner_protein", [protein_name(x, proteins) if x else "" for x in view["partner_accession"]])
    if len(view):
        view = view.assign(_k=[_rank_key(r) for r in view.to_dict("records")]).sort_values(
            ["_k", "pair"], kind="stable").drop(columns="_k")
    view.to_csv(a.csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
