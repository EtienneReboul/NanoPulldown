#!/usr/bin/env python3
"""
nanopulldown.services.interactors — Stage 1a
========================================
Known interaction partners of one or more proteins, gathered from the
protein-protein interaction databases UniProt cross-references, as UniProt
accessions -- the `interactors:` input source of nanopulldown.inputs.resolve:

* uniprot       -- the entry's own INTERACTION comments (binary interactions
                   curated from IntAct; `numberOfExperiments` as score; no
                   method, type or PubMed id, so no MIscore)
* intact, mint, biogrid, imex, ...
                -- any PSICQUIC service named in `interactors.psicquic`
                   (MITAB 2.5; score = number of evidence rows, kept per
                   detection method x interaction type, with their PubMed
                   ids; `confidence` keeps column 15, IntAct's own
                   `intact-miscore:` -- BioGRID writes none). MINT is now
                   curated inside IntAct, and IMEx is the union of the IMEx
                   members (IntAct, MINT, DIP-IMEx ...), so `imex` is opt-in.
* string        -- STRING interaction_partners (physical subnetwork by default;
                   combined score 0-1000 as score; `confidence` keeps the
                   channels escore / dscore / tscore = experiments / curated
                   databases / text-mining, 0-1000). Non-UniProt STRING ids
                   (e.g. 9606.ENSP...) are mapped through UniProt ID mapping.
                   Its experiments channel is imported from BioGRID / IntAct,
                   so STRING never adds independent evidence: annotation only.
* complexportal -- the other protein members of every Complex Portal complex
                   the protein is part of (score = number of shared complexes)

CORUM (mammals, download only), DIP (service offline) and ELM (motifs, not
binary interactions) have no usable per-protein API and are not queried.

Evidence tiers (`min_tier`, the positive controls of
nanopulldown.inputs.controls), per (query, partner):

* curated -- listed by hand under `curated:` ({query: [{partner, pmids, note}]})
* high    -- direct evidence in >= `high_min_publications` (2) publications,
             or in >= `min_direct_publications` with MIscore >=
             `high_miscore` (0.6), or any `structure_methods` row (X-ray, NMR)
* medium  -- direct evidence in >= `min_direct_publications` (1) publications
* low     -- reported, nothing direct (co-IP, AP-MS, FRET/BiFC alone, Complex
             Portal co-membership, STRING, UniProt)

Only the PSICQUIC databases say how an interaction was seen. An evidence row
is direct when its detection method is one of `direct_methods` (default
MI:0018 two hybrid), or its PSI-MI interaction type one of `direct_types`
(default MI:0407 direct interaction: covalent binding, enzymatic reactions
...) and its method is not one of `proximity_methods` (default FRET, BRET,
BiFC: BioGRID labels them "direct interaction", but they only say the two
proteins are within ~10 nm). Each term counts with every PSI-MI descendant
(OLS, cached in <snapshot_dir>/psi_mi_*.json).

MIscore (`miscore`): IntAct's score (Villaveces et al. 2015, Database;
github.com/EBI-IntAct/miscore, default scoreCategories.properties),
recomputed over every PSICQUIC database so BioGRID-only interactions get
one: the mean of a method, a type and a publication score, 0-1. One evidence
unit = one (PubMed id, detection method), merged across databases with its
best-scoring interaction type, so a paper curated by BioGRID and IntAct
counts once (IntAct's own score, `intact_miscore`, counts each of its
evidence rows and can be slightly higher or lower).

Isoform / chain ids (`Q9LMA8-2`, `P04637-PRO_0000...`) collapse to the
canonical accession; non-protein interactors (ChEBI, sub-complexes, RNA)
are dropped. The long table -- one row per (query, partner, database) -- is
snapshotted to `<interactors.snapshot_dir>/<name>.tsv` + `<name>.json` and
reused while the query proteins, databases and STRING settings are unchanged,
so a run stays reproducible as the databases grow (`refresh=True` re-fetches;
a snapshot with a failed database is never reused; a format-2 snapshot, from
before `confidence`, is still reused: tiers and MIscore only need its
methods, types and PubMed ids). Filtering (tiers, min number of databases,
same organism, top-N) works on the snapshot and never re-fetches.

Preview from the command line:
    nanopulldown inputs interactors A1L4W5 --min-databases 2
    nanopulldown inputs interactors Q9M548 --min-tier high   # as the positive controls
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import requests

from nanopulldown.core.config import load_config
from nanopulldown.core.io import atomic_write
from nanopulldown.services.uniprot import fetch_metadata

BUILTIN = ("uniprot", "string", "complexportal")
DEFAULT_DATABASES = ["uniprot", "intact", "mint", "biogrid", "string", "complexportal"]
COLUMNS = ("query", "partner", "database", "score", "partner_taxon",
           "detection_method", "interaction_type", "publications", "confidence")
SNAPSHOT_FORMAT = 3                  # 2: + detection_method / interaction_type / publications
                                     # 3: + confidence (IntAct MIscore, STRING channels)
DIRECT_DEFAULTS = {"direct_types": ["MI:0407"],     # direct interaction (+ descendants)
                   "direct_methods": ["MI:0018"],   # two hybrid (+ descendants)
                   # FRET, BRET, BiFC: never direct on their own, whatever the type says
                   "proximity_methods": ["MI:0055", "MI:0012", "MI:0809"],
                   "min_direct_publications": 1}
TIER_DEFAULTS = {"high_min_publications": 2,        # direct evidence in >= N publications
                 "high_miscore": 0.6,               # or direct + MIscore >= this
                 "structure_methods": ["MI:0114", "MI:0077"]}   # or X-ray / NMR (+ descendants)
TIERS = ("curated", "high", "medium", "low")        # best first
# IntAct MIscore categories (miscore scoreCategories.properties): PSI-MI term ->
# (score, category). A term scores as its nearest listed ancestor, else `unknown`.
MISCORE_METHODS = {"MI:0013": (1.00, 1),            # biophysical
                   "MI:0090": (0.66, 2),            # protein complementation assay
                   "MI:0254": (0.10, 3),            # genetic interference
                   "MI:0255": (0.10, 4),            # post transcriptional interference
                   "MI:0401": (1.00, 5),            # biochemical
                   "MI:0428": (0.33, 6),            # imaging technique
                   "unknown": (0.05, 3)}
MISCORE_TYPES = {"MI:0208": (0.10, 1),              # genetic interaction
                 "MI:0403": (0.33, 2),              # colocalization
                 "MI:0914": (0.33, 3),              # association
                 "MI:0915": (0.66, 3),              # physical association
                 "MI:0407": (1.00, 3),              # direct interaction
                 "unknown": (0.05, 1)}
MISCORE_MAX_PUBLICATIONS = 7                        # publication score 1 above this
_MI = re.compile(r'MI:\d{4}')
_ACC = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
                  r"(?![A-Z0-9])")
# MITAB id prefixes that carry a UniProt accession (BioGRID writes uniprot/swiss-prot:)
_MITAB_UNIPROT = ("uniprotkb:", "uniprot/swiss-prot:", "uniprot/trembl:", "uniprot:")


def canonical(ident: str | None) -> str | None:
    """'Q9LMA8-2' / 'P04637-PRO_0000185703' / 'Q9LMA8' -> 'Q9LMA8'; anything
    that is not a UniProt accession -> None."""
    m = _ACC.match((ident or "").strip())
    return m.group(1) if m else None


def _get(icfg: dict, url: str, params: dict | None = None, method: str = "GET",
         data: dict | None = None) -> requests.Response:
    last_err = None
    for attempt in range(int(icfg.get("retries", 3))):
        try:
            r = requests.request(method, url, params=params, data=data,
                                 timeout=float(icfg.get("timeout_s", 60)))
            r.raise_for_status()
            return r
        except Exception as e:                    # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed ({url}): {last_err}")


# ── per-database fetchers: query accession ->
#    [(partner, score, partner_taxon[, detection_method, interaction_type, publications])] ──

def _uniprot(acc: str, cfg: dict, icfg: dict, taxon: int | None) -> list[tuple]:
    base = cfg["uniprot"]["base_url"].rstrip("/")
    r = _get(icfg, f"{base}/{acc}", {"fields": "accession,cc_interaction", "format": "json"})
    out = []
    for c in r.json().get("comments") or []:
        for it in c.get("interactions") or []:
            two = it.get("interactantTwo") or {}
            p = canonical(two.get("uniProtKBAccession"))
            if p:
                # organismDiffer: UniProt only says "other species", not which one
                out.append((p, it.get("numberOfExperiments"),
                            -1 if it.get("organismDiffer") else taxon))
    return out


def _mitab_accs(field: str) -> set[str]:
    accs = set()
    for tok in field.split("|"):
        for pre in _MITAB_UNIPROT:
            if tok.lower().startswith(pre):
                a = canonical(tok[len(pre):])
                if a:
                    accs.add(a)
    return accs


def _mitab_side(ids: str, alt: str) -> tuple[set[str], set[str]]:
    """(accessions identifying this side, accessions to report for it): the
    alt-id column also lists secondary accessions, so it only names the
    partner when the id column has no UniProt id (BioGRID: Entrez gene ids)."""
    primary, alt_accs = _mitab_accs(ids), _mitab_accs(alt)
    return primary | alt_accs, primary or alt_accs


def _mitab_taxon(field: str) -> int | None:
    m = re.match(r"taxid:(-?\d+)", field or "")
    return int(m.group(1)) if m else None


def _mitab_term(field: str) -> str:
    """'psi-mi:"MI:0018"(two hybrid)' -> 'MI:0018(two hybrid)' ('' if none)."""
    m = re.search(r'(MI:\d{4})"?\(([^)]*)\)', field or "")
    if m:
        return f"{m.group(1)}({m.group(2)})"
    m = _MI.search(field or "")
    return m.group(0) if m else ""


def _mitab_pubs(field: str) -> set[str]:
    return {tok.split(":", 1)[1] for tok in (field or "").split("|")
            if tok.lower().startswith("pubmed:") and tok.split(":", 1)[1].strip()}


def _psicquic(url: str, acc: str, icfg: dict) -> list[tuple]:
    page = int(icfg.get("psicquic_page", 500))
    rows, first = [], 0
    while True:
        r = _get(icfg, f"{url.rstrip('/')}/search/interactor/{acc}",
                 {"format": "tab25", "firstResult": first, "maxResults": page})
        lines = [ln for ln in r.text.splitlines() if ln.strip()]
        rows += lines
        if len(lines) < page:
            break
        first += page
    # (partner, detection method, interaction type) -> [n rows, taxon, pubmed ids, confidences]
    counts: dict[tuple, list] = {}
    for ln in rows:
        f = ln.split("\t")
        if len(f) < 12:
            continue
        (a, a_rep), (b, b_rep) = _mitab_side(f[0], f[2]), _mitab_side(f[1], f[3])
        if acc in a:
            partners, tax = b_rep or {None}, _mitab_taxon(f[10])
        elif acc in b:
            partners, tax = a_rep or {None}, _mitab_taxon(f[9])
        else:
            continue
        method, itype = _mitab_term(f[6]), _mitab_term(f[11])
        conf = {tok for tok in (f[14] if len(f) > 14 else "").split("|")
                if ":" in tok and tok.strip() != "-"}
        for p in partners:
            rec = counts.setdefault((p, method, itype), [0, tax, set(), set()])
            rec[0] += 1
            rec[2] |= _mitab_pubs(f[8])
            rec[3] |= conf
    unmapped = sum(rec[0] for (p, _, _), rec in counts.items() if p is None)
    if unmapped:
        print(f"[fetch_interactors]   {acc}: {unmapped} evidence row(s) with a partner "
              f"that has no UniProt id (skipped)", file=sys.stderr)
    return [(p, n, tax, method, itype, "|".join(sorted(pubs)), "|".join(sorted(conf)))
            for (p, method, itype), (n, tax, pubs, conf) in counts.items() if p is not None]


def _idmapping(ids: list[str], from_db: str, icfg: dict, cfg: dict) -> dict[str, str]:
    """UniProt ID mapping job: {source id: UniProtKB accession} (Swiss-Prot
    preferred when an id maps to several entries)."""
    if not ids:
        return {}
    api = cfg["uniprot"]["base_url"].rstrip("/").rsplit("/", 1)[0] + "/idmapping"
    job = _get(icfg, f"{api}/run", method="POST",
               data={"from": from_db, "to": "UniProtKB", "ids": ",".join(ids)}).json()["jobId"]
    for _ in range(60):
        st = _get(icfg, f"{api}/status/{job}", {"format": "json"}).json()
        if st.get("jobStatus") in (None, "FINISHED") or "results" in st:
            break
        time.sleep(2)
    res = _get(icfg, f"{api}/uniprotkb/results/stream/{job}",
               {"fields": "accession,reviewed", "format": "tsv"}).text.splitlines()[1:]
    out: dict[str, str] = {}
    for ln in res:
        src, acc, rev = (ln.split("\t") + ["", ""])[:3]
        if src not in out or rev == "reviewed":
            out[src] = acc
    return out


def _string(acc: str, cfg: dict, icfg: dict, taxon: int | None) -> list[tuple]:
    s = icfg.get("string") or {}
    params = {"identifiers": acc, "required_score": int(s.get("min_score", 400)),
              "limit": int(s.get("limit", 100)), "network_type": s.get("network_type", "physical"),
              "caller_identity": "nanopulldown"}
    if taxon:
        params["species"] = taxon
    try:
        r = _get(icfg, f"{s.get('base_url', 'https://string-db.org/api').rstrip('/')}"
                       "/json/interaction_partners", params)
    except RuntimeError as e:
        if "404" in str(e) or "400" in str(e):   # protein not in STRING: no partners
            print(f"[fetch_interactors]   {acc}: not found in STRING", file=sys.stderr)
            return []
        raise
    time.sleep(1.0)                               # STRING asks for <= 1 request/s
    hits = r.json() if isinstance(r.json(), list) else []
    best: dict[str, tuple] = {}
    for h in hits:
        sid = h["stringId_B"]
        tax, _, local = sid.partition(".")
        channels = "|".join(f"{c}:{round(float(h.get(c) or 0) * 1000)}"
                            for c in ("escore", "dscore", "tscore"))
        best[sid] = (round(float(h["score"]) * 1000), int(tax) if tax.isdigit() else taxon,
                     canonical(local), channels)
    mapped = _idmapping([sid for sid, v in best.items() if not v[2]], "STRING", icfg, cfg)
    out = []
    for sid, (score, tax, local, channels) in best.items():
        p = local or canonical(mapped.get(sid))
        if p:
            out.append((p, score, tax, "", "", "", channels))
    return out


def _complexportal(acc: str, cfg: dict, icfg: dict, taxon: int | None) -> list[tuple]:
    url = (icfg.get("complexportal_url") or "https://www.ebi.ac.uk/intact/complex-ws").rstrip("/")
    elements, first = [], 0
    while True:
        d = _get(icfg, f"{url}/search/{acc}", {"format": "json", "first": first,
                                               "number": 100}).json()
        elements += d.get("elements") or []
        first += 100
        if first >= int(d.get("totalNumberOfResults") or 0):
            break
    counts: dict[str, list] = {}
    for el in elements:
        members = {canonical(i.get("identifier")) for i in el.get("interactors") or []}
        if acc not in members:                    # full-text search hit, not a member
            continue
        m = re.search(r";\s*(\d+)", el.get("organismName") or "")
        tax = int(m.group(1)) if m else None
        for p in members - {None, acc}:
            rec = counts.setdefault(p, [0, tax])
            rec[0] += 1
    return [(p, n, tax) for p, (n, tax) in counts.items()]


# ── snapshot + filtering ────────────────────────────────────────────────────

def fetch_settings(src: dict, of: list[str], cfg: dict) -> dict:
    """Everything that changes what the databases return (the snapshot key)."""
    icfg = cfg.get("interactors") or {}
    s = {**(icfg.get("string") or {}), **(src.get("string") or {})}
    return {"format": SNAPSHOT_FORMAT, "of": sorted(dict.fromkeys(of)),
            "databases": sorted(src.get("databases") or DEFAULT_DATABASES),
            "string": {"min_score": int(s.get("min_score", 400)),
                       "limit": int(s.get("limit", 100)),
                       "network_type": s.get("network_type", "physical")}}


def fetch_interactions(name: str, src: dict, of: list[str], cfg: dict,
                       refresh: bool = False) -> list[dict]:
    """Long table [{query, partner, database, score, partner_taxon}] for
    every query accession x database, from the snapshot when it matches."""
    icfg = dict(cfg.get("interactors") or {})
    settings = fetch_settings(src, of, cfg)
    icfg["string"] = {**(icfg.get("string") or {}), **settings["string"]}
    sdir = Path(icfg.get("snapshot_dir", "data/interactors"))
    sdir.mkdir(parents=True, exist_ok=True)
    tsv, meta = sdir / f"{name}.tsv", sdir / f"{name}.json"
    if not refresh and tsv.exists() and meta.exists():
        info = json.loads(meta.read_text())
        old = dict(info.get("settings") or {})
        # format 2 (no `confidence`) is still good: tiers only need methods, types, PubMed ids
        if 2 <= int(old.pop("format", 0)) <= SNAPSHOT_FORMAT and old == \
                {k: v for k, v in settings.items() if k != "format"} and not info.get("failed"):
            return _read_tsv(tsv)

    psicquic = icfg.get("psicquic") or {}
    unknown = [d for d in settings["databases"] if d not in BUILTIN and d not in psicquic]
    if unknown:
        raise ValueError(f"interactors {name!r}: unknown database(s) {unknown} "
                         f"(built-in: {list(BUILTIN)}; PSICQUIC: {sorted(psicquic)})")
    taxa = {a: md.get("taxon_id") for a, md in fetch_metadata(settings["of"], cfg).items()}
    rows, counts, failed = [], {}, []
    for acc in settings["of"]:
        for db in settings["databases"]:
            try:
                if db == "uniprot":
                    hits = _uniprot(acc, cfg, icfg, taxa.get(acc))
                elif db == "string":
                    hits = _string(acc, cfg, icfg, taxa.get(acc))
                elif db == "complexportal":
                    hits = _complexportal(acc, cfg, icfg, taxa.get(acc))
                else:
                    hits = _psicquic(psicquic[db], acc, icfg)
            except Exception as e:                # noqa: BLE001 -- one dead service != no run
                print(f"[fetch_interactors] WARNING {acc} / {db}: {e}", file=sys.stderr)
                failed.append(f"{acc}/{db}")
                continue
            for h in hits:
                p, sc, tx, method, itype, pubs, conf = (*h, "", "", "", "")[:7]
                rows.append({"query": acc, "partner": p, "database": db, "score": sc,
                             "partner_taxon": tx, "detection_method": method,
                             "interaction_type": itype, "publications": pubs,
                             "confidence": conf})
            hits = {h[0] for h in hits}
            counts[f"{acc}/{db}"] = len(hits)
            print(f"[fetch_interactors] {acc} / {db}: {len(hits)} partner(s)", file=sys.stderr)
    # one current primary accession per protein: databases lag behind UniProt
    # (secondary / merged accessions), and the same protein must count once
    primary = _idmapping(sorted({r["partner"] for r in rows}), "UniProtKB_AC-ID", icfg, cfg)
    gone = sorted({r["partner"] for r in rows} - set(primary))
    if gone:
        print(f"[fetch_interactors] {name}: {len(gone)} obsolete accession(s) dropped: "
              f"{', '.join(gone[:10])}{' ...' if len(gone) > 10 else ''}", file=sys.stderr)
    rows = [{**r, "partner": primary[r["partner"]]} for r in rows if r["partner"] in primary]
    atomic_write(tsv, "\t".join(COLUMNS) + "\n" + "".join(
        "\t".join("" if r[c] is None else str(r[c]) for c in COLUMNS) + "\n" for r in rows))
    atomic_write(meta, json.dumps({"name": name, "settings": settings, "counts": counts,
                                   "failed": failed, "query_taxa": taxa,
                                   "retrieved": date.today().isoformat()}, indent=1))
    if failed:
        print(f"[fetch_interactors] {name}: {len(failed)} database call(s) failed -- the "
              f"snapshot will be re-fetched next time", file=sys.stderr)
    return rows


def _read_tsv(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    head = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        r = {c: "" for c in COLUMNS} | dict(zip(head, ln.split("\t")))
        for k in ("score", "partner_taxon"):
            r[k] = int(float(r[k])) if r.get(k) not in (None, "") else None
        rows.append(r)
    return rows


def psi_mi_descendants(terms: list[str], cfg: dict) -> set[str]:
    """`terms` and every PSI-MI term below them (is_a / part_of), from OLS;
    cached per term in <interactors.snapshot_dir>/psi_mi_<term>.json. OLS
    down and nothing cached: the terms alone, with a warning."""
    icfg = cfg.get("interactors") or {}
    sdir = Path(icfg.get("snapshot_dir", "data/interactors"))
    ols = (icfg.get("ols_url") or "https://www.ebi.ac.uk/ols4/api").rstrip("/")
    out: set[str] = set()
    for t in terms:
        t = str(t).upper()
        out.add(t)
        cache = sdir / f"psi_mi_{t.replace(':', '_')}.json"
        if cache.exists():
            out |= set(json.loads(cache.read_text())["descendants"])
            continue
        iri = f"http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252F{t.replace(':', '_')}"
        try:
            d = _get(icfg, f"{ols}/ontologies/mi/terms/{iri}/hierarchicalDescendants",
                     {"size": 1000}).json()
        except RuntimeError as e:
            print(f"[fetch_interactors] WARNING: PSI-MI descendants of {t} unavailable ({e}); "
                  "using the term alone", file=sys.stderr)
            continue
        terms_ = (d.get("_embedded") or {}).get("terms") or []
        desc = {x["obo_id"]: x.get("label") for x in terms_ if x.get("obo_id")}
        sdir.mkdir(parents=True, exist_ok=True)
        atomic_write(cache, json.dumps({"term": t, "descendants": desc,
                                        "retrieved": date.today().isoformat()}, indent=1))
        out |= set(desc)
    return out


def evidence_terms(src: dict, cfg: dict) -> dict:
    """The PSI-MI term sets the evidence is judged with (each with its
    descendants): `types` / `methods` (direct), `proximity` (never direct
    alone), `structure` (high tier), and `miscore` -- {"method" | "type":
    {listed MIscore term: its descendants}}."""
    keys = {**DIRECT_DEFAULTS, **TIER_DEFAULTS}
    d = {**keys, **{k: v for k, v in src.items() if k in keys}}
    return {"types": psi_mi_descendants(d["direct_types"], cfg),
            "methods": psi_mi_descendants(d["direct_methods"], cfg),
            "proximity": psi_mi_descendants(d["proximity_methods"], cfg),
            "structure": psi_mi_descendants(d["structure_methods"], cfg),
            "miscore": {kind: {t: psi_mi_descendants([t], cfg) for t in table if t != "unknown"}
                        for kind, table in (("method", MISCORE_METHODS), ("type", MISCORE_TYPES))}}


def _mi(field: str | None) -> str | None:
    m = _MI.search(field or "")
    return m.group(0) if m else None


def is_direct(row: dict, terms: dict) -> bool:
    t, m = _mi(row.get("interaction_type")), _mi(row.get("detection_method"))
    if m and m in terms["methods"]:
        return True
    return bool(t and t in terms["types"] and not (m and m in terms["proximity"]))


def _label(term: str) -> str:
    """'MI:0018(two hybrid)' -> 'two hybrid'."""
    m = re.search(r"\(([^)]*)\)", term or "")
    return m.group(1) if m else term


# ── MIscore (port of github.com/EBI-IntAct/miscore, normalised score) ──────

def _miscore_term(term: str | None, parents: dict[str, set]) -> str:
    """The MIscore category term of a PSI-MI term: its most specific listed
    ancestor (BiFC is below both biophysical and complementation assay:
    complementation), `unknown` when none."""
    hits = [p for p, desc in parents.items() if term and term in desc]
    return min(hits, key=lambda p: len(parents[p])) if hits else "unknown"


def _category_score(terms: list[str], table: dict[str, tuple]) -> float:
    """miscore CategoryScore: log_b(a). a = 1 + the evidence terms' scores;
    b = 1 + per category, (count + 1) x score for a one-term category, else
    its best score + count x score of each term. `unknown` adds to a only,
    as in IntAct (it has no parent term there)."""
    a = 1.0 + sum(table[t][0] for t in terms)
    n = Counter(t for t in terms if t != "unknown")
    cats: dict[int, list[str]] = {}
    for t, (_, c) in table.items():
        cats.setdefault(c, []).append(t)
    b = 1.0
    for ts in cats.values():
        if len(ts) == 1:
            b += (n[ts[0]] + 1) * table[ts[0]][0]
        else:
            b += max(table[t][0] for t in ts) + sum(n[t] * table[t][0] for t in ts)
    return min(1.0, math.log(a) / math.log(b))


def miscore(units: dict[tuple, str], terms: dict) -> float:
    """MIscore of one interaction from its evidence units {(publication,
    method term): type term}: mean of the method, type and publication
    scores (weights 1, 1, 1), 0-1."""
    if not units:
        return 0.0
    par = terms["miscore"]
    methods = [_miscore_term(m, par["method"]) for _, m in units]
    types = [_miscore_term(t, par["type"]) for t in units.values()]
    n_pubs = len({p for p, _ in units})
    pub = 1.0 if n_pubs > MISCORE_MAX_PUBLICATIONS else \
        math.log(n_pubs + 1) / math.log(MISCORE_MAX_PUBLICATIONS + 1)
    return (_category_score(methods, MISCORE_METHODS) + _category_score(types, MISCORE_TYPES) + pub) / 3


def _type_rank(term: str | None, terms: dict) -> float:
    return MISCORE_TYPES[_miscore_term(term, terms["miscore"]["type"])][0]


def _conf(field: str | None, key: str) -> float | None:
    """'intact-miscore:0.44|...' -> 0.44 (key 'intact-miscore'); None if absent."""
    vals = [float(tok.split(":", 1)[1]) for tok in (field or "").split("|")
            if tok.split(":", 1)[0] == key and re.fullmatch(r"[\d.]+", tok.split(":", 1)[1] or "")]
    return max(vals) if vals else None


# ── tiers + summary ─────────────────────────────────────────────────────────

def min_tier_of(src: dict) -> str:
    """`min_tier` of a source; the former `direct_only` maps to medium (true)
    or low (false)."""
    t = src.get("min_tier") or ("medium" if src.get("direct_only") else "low")
    if t not in TIERS:
        raise ValueError(f"min_tier {t!r}: expected one of {list(TIERS)}")
    return t


def summarize(rows: list[dict], src: dict, query_taxa: dict | None = None,
              terms: dict | None = None, queries: list[str] | None = None) -> list[dict]:
    """Long table -> one record per unique partner, filtered and ranked:
    {partner, tier, tier_of, miscore, intact_miscore, databases, n_databases,
     evidence, string_score, string_escore, string_dscore, string_tscore, of,
     direct_databases, direct_methods, n_direct_publications, direct_of,
     curated_pmids, curated_note}.
    `of` lists every query the partner is reported with (any evidence);
    `tier_of` its tier with each (`<query>:<tier>`, see the module doc);
    `direct_of` the queries whose tier is at least `min_tier` -- the baits it
    is a positive control for. `tier` / `miscore` are the best over queries.

      min_tier: high         keep partners at least this tier with one query
                             (curated > high > medium > low; default low)
      curated: {Q: [{partner: P, pmids: [..], note: ..}]}
                             hand-picked partners: tier curated with query Q
      min_databases: 2       reported by at least N databases (curated exempt)
      same_organism: true    drop partners from another species than their query
      include_self: true     keep the query itself (homo-oligomer) when reported
      max_partners: 50       keep the N best (tier, MIscore, direct
                             publications, databases, STRING score, evidence)
      + the DIRECT_DEFAULTS / TIER_DEFAULTS knobs (evidence_terms)
    """
    query_taxa = query_taxa or {}
    if terms is None:
        raise ValueError("summarize: needs the PSI-MI evidence terms (evidence_terms)")
    same = src.get("same_organism", True)
    min_tier = min_tier_of(src)
    knobs = {**DIRECT_DEFAULTS, **TIER_DEFAULTS, **{k: v for k, v in src.items()
                                                    if k in DIRECT_DEFAULTS or k in TIER_DEFAULTS}}
    curated: dict[str, dict[str, dict]] = {}       # partner -> {query: entry}
    known_queries = set(queries or []) | set(query_taxa) | {r["query"] for r in rows}
    for q, entries in (src.get("curated") or {}).items():
        if q not in known_queries:
            continue
        for e in entries or []:
            e = e if isinstance(e, dict) else {"partner": e}
            curated.setdefault(str(e["partner"]), {})[q] = e

    def new(p: str) -> dict:
        return {"partner": p, "databases": set(), "of": set(), "evidence": 0,
                "string_score": None, "string": {}, "intact_miscore": None,
                "direct_databases": set(), "direct_methods": set(), "direct_pubs": set(),
                "direct_pubs_by": {}, "units_by": {}, "structure_of": set()}

    agg: dict[str, dict] = {}
    for r in rows:
        if same and r["partner_taxon"] is not None and query_taxa.get(r["query"]) \
                and r["partner_taxon"] != query_taxa[r["query"]]:
            continue
        if not src.get("include_self", True) and r["partner"] == r["query"]:
            continue
        a = agg.setdefault(r["partner"], new(r["partner"]))
        a["databases"].add(r["database"])
        a["of"].add(r["query"])
        if r["database"] == "string":
            a["string_score"] = max(a["string_score"] or 0, r["score"] or 0)
            for c in ("escore", "dscore", "tscore"):
                v = _conf(r.get("confidence"), c)
                if v is not None:
                    a["string"][c] = max(a["string"].get(c, 0), int(v))
            continue
        if r["score"]:
            a["evidence"] += r["score"]
        ms = _conf(r.get("confidence"), "intact-miscore")
        if ms is not None:
            a["intact_miscore"] = max(a["intact_miscore"] or 0, ms)
        if not (r.get("detection_method") or r.get("interaction_type")):
            continue                               # uniprot / complexportal: no evidence detail
        pubs = [x for x in (r.get("publications") or "").split("|") if x]
        pubs = set(pubs) or {f"{r['database']}:{r['query']}:{r.get('detection_method')}"}
        method, itype = _mi(r.get("detection_method")), _mi(r.get("interaction_type"))
        units = a["units_by"].setdefault(r["query"], {})
        for pub in pubs:                           # one unit per (paper, method), best type
            k = (pub, method)
            if k not in units or _type_rank(itype, terms) > _type_rank(units[k], terms):
                units[k] = itype
        if method and method in terms["structure"]:
            a["structure_of"].add(r["query"])
        if is_direct(r, terms):
            a["direct_databases"].add(r["database"])
            a["direct_methods"].add(_label(r.get("detection_method") or r.get("interaction_type")))
            a["direct_pubs"] |= pubs
            a["direct_pubs_by"].setdefault(r["query"], set()).update(pubs)
    for p, by_q in curated.items():                # curated partners need no database
        a = agg.setdefault(p, new(p))
        a["of"] |= set(by_q)

    min_pubs = int(knobs["min_direct_publications"])
    rank = {t: i for i, t in enumerate(TIERS)}
    out = []
    for a in agg.values():
        tier_of, ms_of = {}, {}
        for q in sorted(a["of"]):
            dp = len(a["direct_pubs_by"].get(q, ()))
            ms_of[q] = miscore(a["units_by"].get(q, {}), terms)
            if q in curated.get(a["partner"], {}):
                tier_of[q] = "curated"
            elif q in a["structure_of"] or (dp >= min_pubs and (
                    dp >= int(knobs["high_min_publications"]) or ms_of[q] >= float(knobs["high_miscore"]))):
                tier_of[q] = "high"
            elif dp >= min_pubs:
                tier_of[q] = "medium"
            else:
                tier_of[q] = "low"
        cur = curated.get(a["partner"], {})
        rec = {"partner": a["partner"],
               "tier": min(tier_of.values(), key=rank.get),
               "tier_of": ";".join(f"{q}:{t}" for q, t in tier_of.items()),
               "miscore": round(max(ms_of.values()), 3) if ms_of else 0.0,
               "intact_miscore": a["intact_miscore"],
               "databases": ";".join(sorted(a["databases"] | ({"curated"} if cur else set()))),
               "n_databases": len(a["databases"]),
               "evidence": a["evidence"],
               "string_score": a["string_score"],
               **{f"string_{c}": a["string"].get(c) for c in ("escore", "dscore", "tscore")},
               "of": ";".join(sorted(a["of"])),
               "direct_databases": ";".join(sorted(a["direct_databases"])),
               "direct_methods": ";".join(sorted(a["direct_methods"])),
               "n_direct_publications": len(a["direct_pubs"]),
               "direct_of": ";".join(q for q, t in tier_of.items() if rank[t] <= rank[min_tier]),
               "curated_pmids": ";".join(sorted({str(x) for e in cur.values() for x in e.get("pmids") or []})),
               "curated_note": "; ".join(sorted({str(e["note"]) for e in cur.values() if e.get("note")}))}
        if not cur and rec["n_databases"] < int(src.get("min_databases", 1)):
            continue
        if not rec["direct_of"]:                   # below min_tier with every query
            continue
        out.append(rec)
    out.sort(key=lambda a: (rank[a["tier"]], -a["miscore"], -a["n_direct_publications"],
                            -a["n_databases"], -(a["string_score"] or 0), -a["evidence"], a["partner"]))
    if src.get("max_partners"):
        out = out[:int(src["max_partners"])]
    return out


def interactors(name: str, src: dict, of: list[str], cfg: dict,
                refresh: bool = False) -> list[dict]:
    """fetch_interactions + summarize: the unique partners a source selects,
    best first (only those at least src `min_tier`)."""
    rows = fetch_interactions(name, src, of, cfg, refresh=refresh)
    sdir = Path((cfg.get("interactors") or {}).get("snapshot_dir", "data/interactors"))
    taxa = json.loads((sdir / f"{name}.json").read_text()).get("query_taxa") or {}
    return summarize(rows, src, taxa, evidence_terms(src, cfg), queries=of)


def main() -> int:
    ap = argparse.ArgumentParser(description="preview the known interactors of protein(s)")
    ap.add_argument("accessions", nargs="+")
    ap.add_argument("--name", help="snapshot name (default: interactors_<accessions>)")
    ap.add_argument("--databases", nargs="+", help=f"default: {DEFAULT_DATABASES}")
    ap.add_argument("--min-databases", type=int, default=1)
    ap.add_argument("--max-partners", type=int)
    ap.add_argument("--all-organisms", action="store_true")
    ap.add_argument("--min-tier", choices=TIERS[1:], default="low",
                    help="keep partners at least this tier (high: the positive-control default)")
    ap.add_argument("--direct", action="store_true", help="same as --min-tier medium")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    src = {"databases": a.databases, "min_databases": a.min_databases,
           "max_partners": a.max_partners, "same_organism": not a.all_organisms,
           "min_tier": "medium" if a.direct and a.min_tier == "low" else a.min_tier}
    name = a.name or "interactors_" + "_".join(a.accessions)
    hits = interactors(name, src, a.accessions, cfg, refresh=a.refresh)
    print("partner\ttier\tmiscore\tintact_miscore\tn_direct_publications\tdirect_methods\t"
          "n_databases\tdatabases\tstring_score\tstring_escore\ttier_of")
    for h in hits:
        print(f"{h['partner']}\t{h['tier']}\t{h['miscore']:.2f}\t{h['intact_miscore'] or ''}\t"
              f"{h['n_direct_publications']}\t{h['direct_methods']}\t{h['n_databases']}\t"
              f"{h['databases']}\t{h['string_score'] or ''}\t{h['string_escore'] or ''}\t{h['tier_of']}")
    print(f"[fetch_interactors] {len(hits)} unique partner(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
