#!/usr/bin/env python3
"""
nanopulldown.annotate.domains — Stage 1b
=======================================
Per protein chain of a pair, query two web services and dump every domain
/ family / motif hit to a flat TSV:

  * ScanProsite   (PROSITE patterns + profiles)   — prosite.expasy.org REST
  * InterPro (Pfam / SMART / PROSITE profiles / Gene3D / ...) — precomputed matches by
    UniProt accession from the InterPro API (fast path; see interpro_by_accession),
    falling back to InterProScan v6 submission of the sequence itself

Adapted from ab_initio_pipeline's scripts/annotate_domains.py: the
InterProScan call now uses v6's MD5-hash match/caching mechanism instead of
v5's submit-and-poll loop. Since InterProScan has already analysed most
UniProt sequences at least once, submitting MD5(sequence) first usually
returns cached matches immediately with no wait; only a genuine cache miss
(e.g. a novel fragment/mutant) falls back to a full submit-and-poll run.

No local databases. Per sequence, the cache (data/annotation/_cache/
domains_v2/<md5>.json) holds {"version": 2, "hits": [...], "features": {...}}:

  hits      every signature hit, plus the InterPro entries themselves
            (db "InterPro"), each with: source db accession name start end
            score description type (member-DB / entry type) ipr ipr_name
            (the InterPro entry it is integrated in) representative
            (InterPro's representative-domain/family flag; None when unknown,
            i.e. hits from the InterProScan fallback or ScanProsite)
  features  {disorder, coils, signal_peptide, tm}: [[start, end], ...] from
            InterPro's extra features (MobiDB-lite, COILS, SignalP/Phobius,
            TMbed/Phobius) -- the tracks InterPro's protein viewer shows

Region chains (`ID:start-stop` specs, and every fragment-mode chain -- see
nanopulldown.inputs.fragments) are not looked up on their own: InterPro has no
precomputed matches for a fragment, so they would each fall back to an
InterProScan submission. They get the FULL-LENGTH entry's cached record
instead, projected onto fragment numbering (project_record): hits clipped to
the regions and shifted, a hit straddling a fragment edge kept as the part
inside it.

TSV output columns:
  chain  source  db  accession  name  start  end  score  description  type  ipr  representative

The raw hits are NOT authoritative — nanopulldown.annotate.merge turns them
into a *proposed* `domains:` block for configs/curated_domains.yaml that a human
reviews and curates.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import requests

from nanopulldown.core.cache import annotation_cache_dir
from nanopulldown.core.config import LazyConfig
from nanopulldown.core.io import atomic_write, seq_md5
from nanopulldown.core.specs import apply_regions, load_pair, protein_chains
from nanopulldown.services.http import STOP, RateLimitAbort, RateLimiter, http_get  # noqa: F401 (STOP: re-exported)
from nanopulldown.services.uniprot import fetch_sequence

CFG = LazyConfig()


@lru_cache(maxsize=None)
def _ua() -> dict:
    email = (CFG["annotate"]["interproscan"].get("email") or "").strip()
    return {"User-Agent": "nanopulldown/annotate_domains" + (f" (contact: {email})" if email else "")}


@lru_cache(maxsize=None)
def _lim(service: str) -> RateLimiter:
    """The process-wide limiter of one service ("interpro" | "prosite"), from annotate.rate_limits."""
    rl = CFG["annotate"].get("rate_limits", {})
    key = {"interpro": "interpro_per_s", "prosite": "scanprosite_per_s"}[service]
    default = {"interpro": 1.0, "prosite": 0.5}[service]
    return RateLimiter(float(rl.get(key, default)), float(rl.get("ramp_s", 120)),
                       int(rl.get("max_throttle_events", 10)))


# ── ScanProsite ────────────────────────────────────────────────────────────

def scanprosite(seq: str, url: str, skip_profiles: bool) -> list[dict]:
    """ScanProsite REST — GET with output=xml, parse <match> elements
    (namespace-agnostic):
      <match><sequence_ac>..</sequence_ac><start>..</start><stop>..</stop>
             <signature_ac>PS#####</signature_ac><score>..</score>...</match>
    """
    params = {"seq": seq, "output": "xml"}
    if skip_profiles:
        params["skip"] = "on"          # patterns only
    r = http_get(_lim("prosite"), url, params=params, headers=_ua(), timeout=120)
    r.raise_for_status()
    root = ET.fromstring(r.text)

    def _txt(el, tag):
        for c in el.iter():
            if c.tag.rsplit("}", 1)[-1] == tag and c.text:
                return c.text.strip()
        return None

    hits = []
    for m in root.iter():
        if m.tag.rsplit("}", 1)[-1] != "match":
            continue
        start, stop = _txt(m, "start"), _txt(m, "stop")
        sig = _txt(m, "signature_ac")
        if not (start and stop and sig):
            continue
        hits.append({
            "source": "ScanProsite", "db": "PROSITE",
            "accession": sig, "name": sig,
            "start": int(start), "end": int(stop),
            "score": _txt(m, "score") or "",
            "description": "",
        })
    return hits


# ── InterProScan v6 (EBI MATCH API, MD5-cached) ────────────────────────────

_md5_of = seq_md5


def _ips6_match_lookup(md5: str, cfg: dict) -> list[dict] | None:
    """Query the MATCH API by MD5. Returns the parsed hit list on a cache
    hit, or None on a cache miss (or any error -- callers fall back to a
    full submission rather than fail the run over the cache path alone)."""
    base = cfg["base_url"]
    endpoint = cfg.get("match_endpoint", "matches")
    try:
        r = http_get(_lim("interpro"), f"{base}/{endpoint}/{md5}", headers=_ua(), timeout=60)
        if r.status_code == 404:
            return None                # documented "not analysed yet" response
        r.raise_for_status()
        return _parse_ips_tsv(r.text)
    except Exception as e:             # noqa: BLE001
        print(f"  [interproscan] MATCH API lookup failed ({e}), falling back to full submission",
              file=sys.stderr)
        return None


def _ips6_submit_and_poll(seq: str, cfg: dict) -> list[dict]:
    """Cache miss: same submit/poll shape as v5, against the v6 endpoint."""
    base = cfg["base_url"]
    payload = {"email": cfg["email"], "sequence": seq, "stype": "p"}
    appl = ",".join(cfg.get("applications", []))
    if appl:
        payload["appl"] = appl
    _lim("interpro").wait()
    r = requests.post(f"{base}/run", data=payload, headers=_ua(), timeout=120)
    if r.status_code == 400 and "appl" in r.text and appl:
        print(f"  [interproscan] server rejected appl='{appl}', retrying with defaults",
              file=sys.stderr)
        payload.pop("appl")
        _lim("interpro").wait()
    r = requests.post(f"{base}/run", data=payload, headers=_ua(), timeout=120)
    r.raise_for_status()
    job_id = r.text.strip()
    poll = int(cfg.get("poll_s", 15))
    while True:
        s = requests.get(f"{base}/status/{job_id}", headers=_ua(), timeout=60).text.strip()
        if s == "FINISHED":
            break
        if s in ("ERROR", "FAILURE", "NOT_FOUND"):
            raise RuntimeError(f"InterProScan job {job_id} status={s}")
        time.sleep(poll)
    out = requests.get(f"{base}/result/{job_id}/tsv", headers=_ua(), timeout=120)
    out.raise_for_status()
    return _parse_ips_tsv(out.text)


# InterProScan analyses that are feature tracks, not domain signatures
_IPS_FEATURES = {"mobidblite": "disorder", "mobidb-lite": "disorder", "coils": "coils",
                 "signalp_euk": "signal_peptide", "signalp": "signal_peptide",
                 "tmhmm": "tm", "deeptmhmm": "tm", "tmbed": "tm"}


def _parse_ips_tsv(txt: str) -> list[dict]:
    """InterProScan TSV -> hits. No representative flags or entry types in
    TSV (`representative`/`type` None: merge_annotation selects its own
    representative set); feature analyses come back as db "feature:<track>"."""
    hits = []
    for line in txt.splitlines():
        p = line.rstrip("\n").split("\t")
        # IPRScan tsv: 0 md5 1 md5hex 2 len 3 analysis 4 sig_acc 5 sig_desc
        # 6 start 7 stop 8 score 9 status 10 date 11 ipr_acc 12 ipr_desc ...
        if len(p) < 8:
            continue
        try:
            start, stop = int(p[6]), int(p[7])
        except ValueError:
            continue
        ipr = p[11] if len(p) > 11 and p[11] not in ("", "-") else None
        feat = _IPS_FEATURES.get(p[3].lower())
        hits.append({
            "source": "InterProScan", "db": f"feature:{feat}" if feat else p[3],
            "accession": p[4],
            "name": next((x for x in (p[5] if len(p) > 5 else "", p[4]) if x not in ("", "-")), p[4]),
            "start": start, "end": stop,
            "score": p[8] if len(p) > 8 else "",
            "description": (p[12] if len(p) > 12 else p[5] if len(p) > 5 else ""),
            "type": None, "ipr": ipr,
            "ipr_name": p[12] if ipr and len(p) > 12 and p[12] not in ("", "-") else None,
            "representative": None,
        })
    return hits


# ── InterPro API by UniProt accession (precomputed, ~2 s, no email needed) ─

# InterPro API source_database -> the db labels InterProScan's TSV uses, so
# merge_annotation.py (DOMAIN_DBS_PREFERRED) sees identical values either way.
_IPR_DB = {"interpro": "InterPro", "pfam": "Pfam", "smart": "SMART", "profile": "PROSITE profiles",
           "prosite": "PROSITE patterns", "cathgene3d": "CATH-Gene3D",
           "cath-gene3d": "CATH-Gene3D", "ssf": "SUPERFAMILY", "cdd": "CDD",
           "panther": "PANTHER", "prints": "PRINTS", "pirsf": "PIRSF",
           "hamap": "HAMAP", "ncbifam": "NCBIfam", "sfld": "SFLD",
           "mobidblt": "MobiDB-lite", "mobidb-lite": "MobiDB-lite",
           "antifam": "AntiFam", "funfam": "CATH-FunFam"}


_SITE_TYPES = {"conserved_site", "active_site", "binding_site", "ptm", "site"}


def _norm_type(t: str | None) -> str | None:
    t = (t or "").lower() or None
    return "site" if t in _SITE_TYPES else t


def interpro_by_accession(accession: str, seq_len: int, cfg: dict) -> list[dict] | None:
    """Precomputed InterPro matches for a UniProt accession (one request; the
    API has no multi-accession form -- a comma list returns 204): every member
    signature hit AND every InterPro entry (db "InterPro"), with InterPro's
    `representative` flag and the entry each signature is integrated in.
    Returns None when the API can't answer for THIS sequence (unknown
    accession / 204 / 404, or protein_length != len(seq), i.e. the pair's
    sequence is not the UniProt entry the matches were computed on) so the
    caller falls back to sequence-based InterProScan."""
    base = cfg.get("api_url", "https://www.ebi.ac.uk/interpro/api")
    url = f"{base}/entry/all/protein/uniprot/{accession}?page_size=200"
    hits: list[dict] = []
    while url:
        r = http_get(_lim("interpro"), url, headers={**_ua(), "Accept": "application/json"}, timeout=60)
        if r.status_code in (204, 404):
            return None
        r.raise_for_status()
        data = r.json()
        for e in data.get("results", []):
            meta = e["metadata"]
            db = meta["source_database"].lower()
            ipr = meta.get("integrated")
            for prot in e.get("proteins", []):
                if int(prot.get("protein_length", seq_len)) != seq_len:
                    return None
                for loc in prot.get("entry_protein_locations", []):
                    frs = loc.get("fragments", [])
                    if not frs:
                        continue
                    hits.append({
                        "source": "InterPro", "db": _IPR_DB.get(db, meta["source_database"]),
                        "accession": meta["accession"],
                        "name": meta.get("name") or meta["accession"],
                        # discontinuous domains: one span, first to last fragment
                        "start": int(frs[0]["start"]), "end": int(frs[-1]["end"]),
                        "score": "" if loc.get("score") is None else loc["score"],
                        "description": meta.get("name") or "",
                        "type": _norm_type(meta.get("type")),
                        "ipr": ipr, "ipr_name": None,
                        "representative": bool(loc.get("representative")),
                    })
        url = data.get("next")
    names = {h["accession"]: h["name"] for h in hits if h["db"] == "InterPro"}
    for h in hits:
        if h["ipr"]:
            h["ipr_name"] = names.get(h["ipr"])
    return hits


# InterPro extra-feature source -> track; first source listed wins per track
_FEATURE_SOURCES = {
    "disorder": [("mobidblt", None)],
    "coils": [("coils", None)],
    "signal_peptide": [("signalp_e", None), ("signalp_g+", None), ("signalp_g-", None),
                       ("phobius", "SIGNAL_PEPTIDE")],
    "tm": [("tmbed", "TMhelix"), ("phobius", "TRANSMEMBRANE")],
}


def interpro_features(accession: str, cfg: dict) -> dict[str, list[list[int]]]:
    """InterPro's extra features of a UniProt entry (the protein viewer's
    disorder / coiled-coil / signal-peptide / TM tracks) -> {track: [[s, e]]}."""
    base = cfg.get("api_url", "https://www.ebi.ac.uk/interpro/api")
    r = http_get(_lim("interpro"), f"{base}/protein/uniprot/{accession}?extra_features",
                 headers={**_ua(), "Accept": "application/json"}, timeout=60)
    if r.status_code in (204, 404):
        return {}
    r.raise_for_status()
    found: dict[tuple[str, str], list[list[int]]] = {}
    for key, v in (r.json() or {}).items():
        db = (v.get("source_database") or "").lower()
        for loc in v.get("locations", []):
            for fr in loc.get("fragments", []):
                found.setdefault((db, key), []).append([int(fr["start"]), int(fr["end"])])
    out = {}
    for track, sources in _FEATURE_SOURCES.items():
        for db, prefix in sources:
            locs = [l for (d, k), ls in found.items() if d == db and (prefix is None or k.startswith(prefix))
                    for l in ls]
            if locs:
                out[track] = sorted(map(list, {tuple(l) for l in locs}))
                break
    return out


def interproscan(seq: str, cfg: dict) -> tuple[list[dict], bool]:
    """(hits, cache_hit) -- MD5 match lookup first, full submission on miss."""
    md5 = _md5_of(seq)
    cached = _ips6_match_lookup(md5, cfg)
    if cached is not None:
        return cached, True
    return _ips6_submit_and_poll(seq, cfg), False


# ── per-sequence annotation (cached) ──────────────────────────────────────

def _retry(fn, *args, tries: int = 3, backoff: float = 5.0):
    """Web services throttle and hiccup; retry with linear backoff."""
    for i in range(tries):
        try:
            return fn(*args)
        except RateLimitAbort:
            raise
        except Exception:                            # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(backoff * (i + 1))


DOMAIN_CACHE = "domains_v2"          # bump when the cached record's shape changes


def _features_from_hits(hits: list[dict]) -> dict[str, list[list[int]]]:
    out: dict[str, list[list[int]]] = {}
    for h in hits:
        if str(h["db"]).startswith("feature:"):
            out.setdefault(h["db"].split(":", 1)[1], []).append([int(h["start"]), int(h["end"])])
    return out


def annotate_sequence(seq: str, acfg: dict, email: str, log=print,
                      accession: str | None = None) -> dict:
    """{"version": 2, "hits": [...], "features": {...}} for ONE sequence (no
    chain id): ScanProsite + InterPro (by accession, else InterProScan),
    cached on disk by md5(sequence). Thread-safe (used by the batch pool); an
    entry is only cached when every service that was queried succeeded, so a
    transient failure is retried on the next run instead of frozen in."""
    cache = annotation_cache_dir(CFG, DOMAIN_CACHE) / f"{seq_md5(seq)}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    ok = True
    try:
        ps = _retry(scanprosite, seq, acfg["prosite"]["scan_url"],
                    acfg["prosite"].get("skip_profiles", False))
    except Exception as e:                           # noqa: BLE001
        log(f"  ScanProsite FAILED: {e}"); ps, ok = [], False
    ip, features = None, {}
    if accession:                                    # fast path: precomputed by UniProt accession
        try:
            ip = _retry(interpro_by_accession, accession, len(seq), acfg["interproscan"])
            if ip is not None:
                features = _retry(interpro_features, accession, acfg["interproscan"])
        except Exception as e:                       # noqa: BLE001
            log(f"  InterPro API FAILED for {accession}: {e}"); ok = False
    if ip is None and email and ok:                  # slow path: submit the sequence itself
        try:
            ip, _ = _retry(interproscan, seq, acfg["interproscan"])
            features = _features_from_hits(ip)
            ip = [h for h in ip if not str(h["db"]).startswith("feature:")]
        except Exception as e:                       # noqa: BLE001
            log(f"  InterProScan FAILED: {e}"); ok = False
    rec = {"version": 2, "hits": ps + (ip or []), "features": features}
    if ok:
        atomic_write(cache, json.dumps(rec))
    return rec


def seq_accessions(spec: dict) -> dict[str, str]:
    """{sequence: UniProt accession} for chains that ARE the full-length entry
    (no `regions` fragment, plain accession -- no isoform suffix), so the
    InterPro API's precomputed matches apply directly."""
    out = {}
    for ch in protein_chains(spec):
        ent = spec.get(ch.get("role") or "", {}) or {}
        acc = ent.get("uniprot") or ""
        if acc and not ent.get("regions") and "-" not in acc:
            out[ch["sequence"]] = acc
    return out


def region_source(spec: dict, ch: dict) -> tuple[str, str, list] | None:
    """(full-length sequence, accession, regions) when chain `ch` is a region
    fragment of a plain UniProt entry -- its annotation is then the
    full-length record's, projected (project_record) -- else None."""
    ent = spec.get(ch.get("role") or "", {}) or {}
    acc, regions = ent.get("uniprot") or "", ent.get("regions")
    if not (acc and regions) or "-" in acc:
        return None
    full = fetch_sequence(acc, CFG)
    if apply_regions(full, regions) != ch["sequence"]:   # UniProt sequence changed since expansion
        return None
    return full, acc, [tuple(r) for r in regions]


def project_span(start: int, end: int, regions: list) -> list[list[int]]:
    """Full-length span [start, end] -> its pieces in the concatenated
    regions' numbering (1-based, inclusive); [] when outside every region."""
    out, off = [], 0
    for a, b in regions:
        s, e = max(start, a), min(end, b)
        if s <= e:
            ps, pe = s - a + 1 + off, e - a + 1 + off
            if out and ps == out[-1][1] + 1:
                out[-1][1] = pe
            else:
                out.append([ps, pe])
        off += b - a + 1
    return out


def project_record(rec: dict, regions: list) -> dict:
    hits = [{**h, "start": s, "end": e} for h in rec.get("hits", [])
            for s, e in project_span(int(h["start"]), int(h["end"]), regions)]
    feats = {k: [p for a, b in v for p in project_span(int(a), int(b), regions)]
             for k, v in (rec.get("features") or {}).items()}
    return {**rec, "hits": hits, "features": {k: v for k, v in feats.items() if v}}


def annotation_targets(spec: dict) -> dict[str, str | None]:
    """{sequence to annotate: UniProt accession or None} for a pair -- the
    full-length entry in place of each region chain -- for the batch rule."""
    accs, out = seq_accessions(spec), {}
    for ch in protein_chains(spec):
        src = region_source(spec, ch)
        if src:
            out[src[0]] = src[1]
        else:
            out.setdefault(ch["sequence"], accs.get(ch["sequence"]))
    return out


def chain_record(spec: dict, ch: dict, acfg: dict, email: str, accs: dict, log=print) -> dict:
    """One chain's annotation record: its own, or for a region chain the
    full-length entry's, projected onto the fragment."""
    src = region_source(spec, ch)
    if src:
        full, acc, regions = src
        return project_record(annotate_sequence(full, acfg, email, log, accession=acc), regions)
    return annotate_sequence(ch["sequence"], acfg, email, log, accession=accs.get(ch["sequence"]))


def interproscan_email(acfg: dict) -> str:
    email = (acfg["interproscan"].get("email") or "").strip()
    if not email:
        print("[annotate_domains] no annotate.interproscan.email set (config.local.yaml) "
              "-- InterProScan submission fallback disabled; InterPro API (by UniProt "
              "accession) + ScanProsite still run", file=sys.stderr)
    return email


# ── main ────────────────────────────────────────────────────────────────

DOMAIN_COLUMNS = ["chain", "source", "db", "accession", "name", "start", "end", "score", "description",
                  "type", "ipr", "representative"]


def pair_domain_rows(spec: dict) -> list[dict]:
    """Every domain/motif hit of every chain of a pair (DOMAIN_COLUMNS keys),
    from the per-sequence cache, falling back to a live lookup on a miss."""
    acfg = CFG["annotate"]
    rows: list[dict] = []
    email = interproscan_email(acfg)

    # identical chains (homodimers etc.) share one lookup; the on-disk cache
    # (pre-filled by nanopulldown.annotate.batch) makes the bait, which
    # is in every pair, a free lookup after the first.
    by_seq: dict[str, list[dict]] = {}
    for ch in protein_chains(spec):
        by_seq.setdefault(ch["sequence"], []).append(ch)

    accs = seq_accessions(spec)
    for seq, chs in by_seq.items():
        cids = [ch["id"] for ch in chs]
        hits = chain_record(spec, chs[0], acfg, email, accs,
                            log=lambda m: print(m, file=sys.stderr))["hits"]
        print(f"[annotate_domains] chains {cids} ({len(seq)} aa): {len(hits)} hit(s)", flush=True)
        for cid in cids:
            for h in hits:
                rows.append({**h, "chain": cid})
    return sorted(rows, key=lambda x: (x["chain"], x["start"], x["end"]))


def pair_domain_features(spec: dict) -> dict[str, dict[str, list[list[int]]]]:
    """{chain id: {track: [[start, end], ...]}} (see the module docstring),
    from the same per-sequence cache as pair_domain_rows."""
    acfg, accs, out = CFG["annotate"], seq_accessions(spec), {}
    email = (acfg["interproscan"].get("email") or "").strip()
    for ch in protein_chains(spec):
        rec = chain_record(spec, ch, acfg, email, accs, log=lambda m: print(m, file=sys.stderr))
        out[ch["id"]] = rec.get("features") or {}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = pair_domain_rows(load_pair(args.pair))
    cols = DOMAIN_COLUMNS
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")).replace("\t", " ") for c in cols) + "\n")
    print(f"[annotate_domains] wrote {len(rows)} row(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
