#!/usr/bin/env python3
"""
scripts/fetch_interactors.py — Stage 1a
========================================
Known interaction partners of one or more proteins, gathered from the
protein-protein interaction databases UniProt cross-references, as UniProt
accessions -- the `interactors:` input source of scripts/resolve_inputs.py:

* uniprot       -- the entry's own INTERACTION comments (binary interactions
                   curated from IntAct; `numberOfExperiments` as score)
* intact, mint, biogrid, imex, ...
                -- any PSICQUIC service named in `interactors.psicquic`
                   (MITAB 2.5; score = number of evidence rows, kept per
                   detection method x interaction type, with their PubMed
                   ids). MINT is now curated inside IntAct, and IMEx is the
                   union of the IMEx members (IntAct, MINT, DIP-IMEx ...), so
                   `imex` is opt-in.
* string        -- STRING interaction_partners (physical subnetwork by default;
                   combined score 0-1000 as score). Non-UniProt STRING ids
                   (e.g. 9606.ENSP...) are mapped through UniProt ID mapping.
* complexportal -- the other protein members of every Complex Portal complex
                   the protein is part of (score = number of shared complexes)

CORUM (mammals, download only), DIP (service offline) and ELM (motifs, not
binary interactions) have no usable per-protein API and are not queried.

Direct interactions (`direct_only`, the positive controls of
scripts/controls.py): only the PSICQUIC databases say how an interaction was
seen. An evidence row is direct when its PSI-MI interaction type is one of
`direct_types` (default MI:0407 direct interaction) or its detection method
one of `direct_methods` (default MI:0018 two hybrid), each with every PSI-MI
descendant (OLS, cached in <snapshot_dir>/psi_mi_*.json): covalent binding,
enzymatic reactions (ubiquitination, phosphorylation ...), two-hybrid
variants. Co-IP / affinity capture-MS ("association", "physical
association"), Complex Portal co-membership and STRING (no per-evidence
detail in its API) are not direct; STRING's score is still reported.

Isoform / chain ids (`Q9LMA8-2`, `P04637-PRO_0000...`) collapse to the
canonical accession; non-protein interactors (ChEBI, sub-complexes, RNA)
are dropped. The long table -- one row per (query, partner, database) -- is
snapshotted to `<interactors.snapshot_dir>/<name>.tsv` + `<name>.json` and
reused while the query proteins, databases and STRING settings are unchanged,
so a run stays reproducible as the databases grow (`refresh=True` re-fetches;
a snapshot with a failed database is never reused). Filtering (min number of
databases, same organism, top-N) works on the snapshot and never re-fetches.

Preview from the command line:
    python scripts/fetch_interactors.py A1L4W5 --min-databases 2
    python scripts/fetch_interactors.py Q9M548 --direct      # the positive controls
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_uniprot import fetch_metadata  # noqa: E402
from lib_pipeline import atomic_write, load_config  # noqa: E402

BUILTIN = ("uniprot", "string", "complexportal")
DEFAULT_DATABASES = ["uniprot", "intact", "mint", "biogrid", "string", "complexportal"]
COLUMNS = ("query", "partner", "database", "score", "partner_taxon",
           "detection_method", "interaction_type", "publications")
SNAPSHOT_FORMAT = 2                  # 2: + detection_method / interaction_type / publications
DIRECT_DEFAULTS = {"direct_types": ["MI:0407"],     # direct interaction (+ descendants)
                   "direct_methods": ["MI:0018"],   # two hybrid (+ descendants)
                   "min_direct_publications": 1}
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
    # (partner, detection method, interaction type) -> [n rows, taxon, pubmed ids]
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
        for p in partners:
            rec = counts.setdefault((p, method, itype), [0, tax, set()])
            rec[0] += 1
            rec[2] |= _mitab_pubs(f[8])
    unmapped = sum(n for (p, _, _), (n, _, _) in counts.items() if p is None)
    if unmapped:
        print(f"[fetch_interactors]   {acc}: {unmapped} evidence row(s) with a partner "
              f"that has no UniProt id (skipped)", file=sys.stderr)
    return [(p, n, tax, method, itype, "|".join(sorted(pubs)))
            for (p, method, itype), (n, tax, pubs) in counts.items() if p is not None]


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
        best[sid] = (round(float(h["score"]) * 1000), int(tax) if tax.isdigit() else taxon,
                     canonical(local))
    mapped = _idmapping([sid for sid, v in best.items() if not v[2]], "STRING", icfg, cfg)
    out = []
    for sid, (score, tax, local) in best.items():
        p = local or canonical(mapped.get(sid))
        if p:
            out.append((p, score, tax))
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
        if info.get("settings") == settings and not info.get("failed"):
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
                p, sc, tx, method, itype, pubs = (*h, "", "", "")[:6]
                rows.append({"query": acc, "partner": p, "database": db, "score": sc,
                             "partner_taxon": tx, "detection_method": method,
                             "interaction_type": itype, "publications": pubs})
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


def direct_terms(src: dict, cfg: dict) -> tuple[set[str], set[str]]:
    """(interaction types, detection methods) whose evidence counts as direct."""
    d = {**DIRECT_DEFAULTS, **{k: v for k, v in src.items() if k in DIRECT_DEFAULTS}}
    return psi_mi_descendants(d["direct_types"], cfg), psi_mi_descendants(d["direct_methods"], cfg)


def is_direct(row: dict, direct: tuple[set[str], set[str]]) -> bool:
    types, methods = direct
    t, m = _MI.search(row.get("interaction_type") or ""), _MI.search(row.get("detection_method") or "")
    return bool((t and t.group(0) in types) or (m and m.group(0) in methods))


def _label(term: str) -> str:
    """'MI:0018(two hybrid)' -> 'two hybrid'."""
    m = re.search(r"\(([^)]*)\)", term or "")
    return m.group(1) if m else term


def summarize(rows: list[dict], src: dict, query_taxa: dict | None = None,
              direct: tuple[set[str], set[str]] | None = None) -> list[dict]:
    """Long table -> one record per unique partner, filtered and ranked:
    {partner, databases, n_databases, evidence, string_score, of,
     direct_databases, direct_methods, n_direct_publications, direct_of}.
    `of` lists every query the partner is reported with (any evidence);
    `direct_of` only the queries it has direct evidence with in
    >= min_direct_publications publications -- the baits it is a positive
    control for.

      min_databases: 2       reported by at least N databases
      same_organism: true    drop partners from another species than their query
      include_self: true     keep the query itself (homo-oligomer) when reported
      direct_only: true      keep partners with direct evidence (`direct`, see
                             direct_terms) in >= min_direct_publications
                             publications (PubMed ids; a row without one
                             counts as one)
      max_partners: 50       keep the N best (direct publications when
                             direct_only, then most databases, STRING score,
                             evidence count)
    """
    query_taxa = query_taxa or {}
    same = src.get("same_organism", True)
    direct_only = bool(src.get("direct_only"))
    if direct_only and direct is None:
        raise ValueError("summarize: direct_only needs the direct PSI-MI terms (direct_terms)")
    agg: dict[str, dict] = {}
    for r in rows:
        if same and r["partner_taxon"] is not None and query_taxa.get(r["query"]) \
                and r["partner_taxon"] != query_taxa[r["query"]]:
            continue
        if not src.get("include_self", True) and r["partner"] == r["query"]:
            continue
        a = agg.setdefault(r["partner"], {"partner": r["partner"], "databases": set(),
                                          "of": set(), "evidence": 0, "string_score": None,
                                          "direct_databases": set(), "direct_methods": set(),
                                          "direct_pubs": set(), "direct_pubs_by": {}})
        a["databases"].add(r["database"])
        a["of"].add(r["query"])
        if r["database"] == "string":
            a["string_score"] = max(a["string_score"] or 0, r["score"] or 0)
        elif r["score"]:
            a["evidence"] += r["score"]
        if direct is not None and is_direct(r, direct):
            a["direct_databases"].add(r["database"])
            a["direct_methods"].add(_label(r.get("detection_method") or r.get("interaction_type")))
            pubs = [x for x in (r.get("publications") or "").split("|") if x]
            pubs = set(pubs) or {f"{r['database']}:{r['query']}:{r.get('detection_method')}"}
            a["direct_pubs"] |= pubs
            a["direct_pubs_by"].setdefault(r["query"], set()).update(pubs)
    min_pubs = int(src.get("min_direct_publications", DIRECT_DEFAULTS["min_direct_publications"]))
    out = []
    for a in agg.values():
        a["n_databases"] = len(a["databases"])
        a["n_direct_publications"] = len(a.pop("direct_pubs"))
        a["direct_of"] = ";".join(sorted(q for q, ps in a.pop("direct_pubs_by").items() if len(ps) >= min_pubs))
        for k in ("databases", "of", "direct_databases", "direct_methods"):
            a[k] = ";".join(sorted(a[k]))
        if a["n_databases"] < int(src.get("min_databases", 1)):
            continue
        if direct_only and not a["direct_of"]:      # direct with no single query often enough
            continue
        out.append(a)
    out.sort(key=lambda a: (-(a["n_direct_publications"] if direct_only else 0), -a["n_databases"],
                            -(a["string_score"] or 0), -a["evidence"], a["partner"]))
    if src.get("max_partners"):
        out = out[:int(src["max_partners"])]
    return out


def interactors(name: str, src: dict, of: list[str], cfg: dict,
                refresh: bool = False) -> list[dict]:
    """fetch_interactions + summarize: the unique partners a source selects,
    best first (only the direct ones when src `direct_only`)."""
    rows = fetch_interactions(name, src, of, cfg, refresh=refresh)
    sdir = Path((cfg.get("interactors") or {}).get("snapshot_dir", "data/interactors"))
    taxa = json.loads((sdir / f"{name}.json").read_text()).get("query_taxa") or {}
    return summarize(rows, src, taxa, direct_terms(src, cfg))


def main() -> int:
    ap = argparse.ArgumentParser(description="preview the known interactors of protein(s)")
    ap.add_argument("accessions", nargs="+")
    ap.add_argument("--name", help="snapshot name (default: interactors_<accessions>)")
    ap.add_argument("--databases", nargs="+", help=f"default: {DEFAULT_DATABASES}")
    ap.add_argument("--min-databases", type=int, default=1)
    ap.add_argument("--max-partners", type=int)
    ap.add_argument("--all-organisms", action="store_true")
    ap.add_argument("--direct", action="store_true",
                    help="only partners with direct evidence (two hybrid, direct interaction ...)")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    src = {"databases": a.databases, "min_databases": a.min_databases,
           "max_partners": a.max_partners, "same_organism": not a.all_organisms,
           "direct_only": a.direct}
    name = a.name or "interactors_" + "_".join(a.accessions)
    hits = interactors(name, src, a.accessions, cfg, refresh=a.refresh)
    print("partner\tn_databases\tdatabases\tstring_score\tevidence\tn_direct_publications\t"
          "direct_methods\tof")
    for h in hits:
        print(f"{h['partner']}\t{h['n_databases']}\t{h['databases']}\t"
              f"{h['string_score'] or ''}\t{h['evidence']}\t{h['n_direct_publications']}\t"
              f"{h['direct_methods']}\t{h['of']}")
    print(f"[fetch_interactors] {len(hits)} unique partner(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
