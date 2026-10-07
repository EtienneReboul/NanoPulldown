#!/usr/bin/env python3
"""
scripts/annotate_domains.py — Stage 1b
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

No local databases. Output columns:
  chain  source  db  accession  name  start  end  score  description

The raw hits are NOT authoritative — scripts/merge_annotation.py turns them
into a *proposed* `domains:` block for configs/<pair>.yaml that a human
reviews and curates.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import (annotation_cache_dir, atomic_write, load_config, load_pair,  # noqa: E402
                          protein_chains, seq_md5)

CFG = load_config()
_email = (CFG["annotate"]["interproscan"].get("email") or "").strip()
UA = {"User-Agent": "nanopulldown/annotate_domains" + (f" (contact: {_email})" if _email else "")}


# ── polite access to the public web services ───────────────────────────────
# Neither InterPro nor ScanProsite publishes a numeric rate limit. InterPro's
# only guidance (NAR 2025, D444): tens of thousands of requests in a very short
# period can make the API unavailable for everyone; "start querying our API
# gradually, then ramp up". So every outgoing request goes through a shared,
# ramped rate limiter; 429/5xx honour Retry-After and slow the limiter down;
# repeated throttling trips a circuit breaker that aborts the run (the
# per-sequence cache keeps all progress, rerun later).

class RateLimitAbort(Exception):
    """Circuit breaker tripped: the service keeps throttling us."""


STOP = threading.Event()
_throttle_events = 0
_throttle_lock = threading.Lock()


class RateLimiter:
    """Thread-safe minimum spacing between requests, ramping from 10% to 100%
    of the target rate over `ramp_s` seconds after the first request."""

    def __init__(self, per_s: float, ramp_s: float = 0.0):
        self.base = 1.0 / max(per_s, 1e-6)
        self.ramp_s, self.slow, self.next, self.t0 = ramp_s, 1.0, 0.0, None
        self.lock = threading.Lock()

    def wait(self) -> None:
        if STOP.is_set():
            raise RateLimitAbort("aborted after repeated throttling")
        with self.lock:
            now = time.monotonic()
            self.t0 = self.t0 if self.t0 is not None else now
            ramp = min(1.0, (now - self.t0) / self.ramp_s) if self.ramp_s else 1.0
            interval = self.base * self.slow / (0.1 + 0.9 * ramp)
            at = max(now, self.next)
            self.next = at + interval
        time.sleep(max(0.0, at - now))

    def slow_down(self, factor: float = 2.0, cap: float = 8.0) -> None:
        with self.lock:
            self.slow = min(cap, self.slow * factor)


_RL = CFG["annotate"].get("rate_limits", {})
LIM_INTERPRO = RateLimiter(float(_RL.get("interpro_per_s", 1.0)), float(_RL.get("ramp_s", 120)))
LIM_PROSITE = RateLimiter(float(_RL.get("scanprosite_per_s", 0.5)), float(_RL.get("ramp_s", 120)))
MAX_THROTTLE_EVENTS = int(_RL.get("max_throttle_events", 10))


def http_get(limiter: RateLimiter, url: str, tries: int = 3, **kw):
    """requests.get through `limiter`. 429 / 502 / 503 / 504 => wait Retry-After
    (default 60 s x attempt), slow the limiter 2x, count a throttle event; too
    many events trips STOP for the whole process."""
    global _throttle_events
    for attempt in range(1, tries + 1):
        limiter.wait()
        r = requests.get(url, **kw)
        if r.status_code not in (429, 502, 503, 504):
            return r
        with _throttle_lock:
            _throttle_events += 1
            if _throttle_events >= MAX_THROTTLE_EVENTS:
                STOP.set()
        limiter.slow_down()
        if STOP.is_set():
            raise RateLimitAbort(f"HTTP {r.status_code} from {url.split('?')[0]}: too many throttle events")
        try:
            wait_s = float(r.headers.get("Retry-After", ""))
        except ValueError:
            wait_s = 60.0 * attempt
        print(f"  [throttle] HTTP {r.status_code} from {url.split('?')[0]} -- sleeping {wait_s:.0f}s, "
              f"rate now /{limiter.slow:g} (event {_throttle_events}/{MAX_THROTTLE_EVENTS})",
              file=sys.stderr, flush=True)
        time.sleep(min(wait_s, 900))
    r.raise_for_status()
    return r


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
    r = http_get(LIM_PROSITE, url, params=params, headers=UA, timeout=120)
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
        r = http_get(LIM_INTERPRO, f"{base}/{endpoint}/{md5}", headers=UA, timeout=60)
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
    LIM_INTERPRO.wait()
    r = requests.post(f"{base}/run", data=payload, headers=UA, timeout=120)
    if r.status_code == 400 and "appl" in r.text and appl:
        print(f"  [interproscan] server rejected appl='{appl}', retrying with defaults",
              file=sys.stderr)
        payload.pop("appl")
        LIM_INTERPRO.wait()
    r = requests.post(f"{base}/run", data=payload, headers=UA, timeout=120)
    r.raise_for_status()
    job_id = r.text.strip()
    poll = int(cfg.get("poll_s", 15))
    while True:
        s = requests.get(f"{base}/status/{job_id}", headers=UA, timeout=60).text.strip()
        if s == "FINISHED":
            break
        if s in ("ERROR", "FAILURE", "NOT_FOUND"):
            raise RuntimeError(f"InterProScan job {job_id} status={s}")
        time.sleep(poll)
    out = requests.get(f"{base}/result/{job_id}/tsv", headers=UA, timeout=120)
    out.raise_for_status()
    return _parse_ips_tsv(out.text)


def _parse_ips_tsv(txt: str) -> list[dict]:
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
        hits.append({
            "source": "InterProScan", "db": p[3],
            "accession": p[4],
            "name": next((x for x in (p[12] if len(p) > 12 else "", p[5] if len(p) > 5 else "", p[4])
                          if x not in ("", "-")), p[4]),
            "start": start, "end": stop,
            "score": p[8] if len(p) > 8 else "",
            "description": (p[12] if len(p) > 12 else p[5] if len(p) > 5 else ""),
        })
    return hits


# ── InterPro API by UniProt accession (precomputed, ~2 s, no email needed) ─

# InterPro API source_database -> the db labels InterProScan's TSV uses, so
# merge_annotation.py (DOMAIN_DBS_PREFERRED) sees identical values either way.
_IPR_DB = {"pfam": "Pfam", "smart": "SMART", "profile": "PROSITE profiles",
           "prosite": "PROSITE patterns", "cathgene3d": "CATH-Gene3D",
           "cath-gene3d": "CATH-Gene3D", "ssf": "SUPERFAMILY", "cdd": "CDD",
           "panther": "PANTHER", "prints": "PRINTS", "pirsf": "PIRSF",
           "hamap": "HAMAP", "ncbifam": "NCBIfam", "sfld": "SFLD",
           "mobidblt": "MobiDB-lite", "mobidb-lite": "MobiDB-lite",
           "antifam": "AntiFam", "funfam": "CATH-FunFam"}


def interpro_by_accession(accession: str, seq_len: int, cfg: dict) -> list[dict] | None:
    """Precomputed InterPro matches for a UniProt accession (one request; the
    API has no multi-accession form -- a comma list returns 204). Returns the
    hit list, or None when the API can't answer for THIS sequence (unknown
    accession / 204 / 404, or protein_length != len(seq), i.e. the pair's
    sequence is not the UniProt entry the matches were computed on) so the
    caller falls back to sequence-based InterProScan."""
    base = cfg.get("api_url", "https://www.ebi.ac.uk/interpro/api")
    url = f"{base}/entry/all/protein/uniprot/{accession}?page_size=200"
    hits: list[dict] = []
    while url:
        r = http_get(LIM_INTERPRO, url, headers={**UA, "Accept": "application/json"}, timeout=60)
        if r.status_code in (204, 404):
            return None
        r.raise_for_status()
        data = r.json()
        for e in data.get("results", []):
            meta = e["metadata"]
            db = meta["source_database"].lower()
            if db == "interpro":           # aggregate of member signatures, not a signature itself
                continue
            for prot in e.get("proteins", []):
                if int(prot.get("protein_length", seq_len)) != seq_len:
                    return None
                for loc in prot.get("entry_protein_locations", []):
                    for fr in loc.get("fragments", []):
                        hits.append({
                            "source": "InterPro", "db": _IPR_DB.get(db, meta["source_database"]),
                            "accession": meta["accession"],
                            "name": meta.get("name") or meta["accession"],
                            "start": int(fr["start"]), "end": int(fr["end"]),
                            "score": "" if loc.get("score") is None else loc["score"],
                            "description": meta.get("name") or "",
                        })
        url = data.get("next")
    return hits


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


def annotate_sequence(seq: str, acfg: dict, email: str, log=print,
                      accession: str | None = None) -> list[dict]:
    """ScanProsite + InterProScan hits for ONE sequence (no chain id), cached
    on disk by md5(sequence). Thread-safe (used by the batch pool); an entry
    is only cached when every service that was queried succeeded, so a
    transient failure is retried on the next run instead of frozen in."""
    cache = annotation_cache_dir(CFG, "domains") / f"{seq_md5(seq)}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    ok = True
    try:
        ps = _retry(scanprosite, seq, acfg["prosite"]["scan_url"],
                    acfg["prosite"].get("skip_profiles", False))
    except Exception as e:                           # noqa: BLE001
        log(f"  ScanProsite FAILED: {e}"); ps, ok = [], False
    ip = None
    if accession:                                    # fast path: precomputed by UniProt accession
        try:
            ip = _retry(interpro_by_accession, accession, len(seq), acfg["interproscan"])
        except Exception as e:                       # noqa: BLE001
            log(f"  InterPro API FAILED for {accession}: {e}"); ok = False
    if ip is None and email and ok:                  # slow path: submit the sequence itself
        try:
            ip, _ = _retry(interproscan, seq, acfg["interproscan"])
        except Exception as e:                       # noqa: BLE001
            log(f"  InterProScan FAILED: {e}"); ok = False
    ip = ip or []
    hits = ps + ip
    if ok:
        atomic_write(cache, json.dumps(hits))
    return hits


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


def interproscan_email(acfg: dict) -> str:
    email = (acfg["interproscan"].get("email") or "").strip()
    if not email:
        print("[annotate_domains] no annotate.interproscan.email set (config.local.yaml) "
              "-- InterProScan submission fallback disabled; InterPro API (by UniProt "
              "accession) + ScanProsite still run", file=sys.stderr)
    return email


# ── main ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = load_pair(args.pair, Path(args.spec).resolve().parent.parent)
    acfg = CFG["annotate"]
    rows: list[dict] = []
    email = interproscan_email(acfg)

    # identical chains (homodimers etc.) share one lookup; the on-disk cache
    # (pre-filled by scripts/annotate_domains_batch.py) makes the bait, which
    # is in every pair, a free lookup after the first.
    by_seq: dict[str, list[str]] = {}
    for ch in protein_chains(spec):
        by_seq.setdefault(ch["sequence"], []).append(ch["id"])

    accs = seq_accessions(spec)
    for seq, cids in by_seq.items():
        hits = annotate_sequence(seq, acfg, email, log=lambda m: print(m, file=sys.stderr),
                                 accession=accs.get(seq))
        print(f"[annotate_domains] chains {cids} ({len(seq)} aa): {len(hits)} hit(s)", flush=True)
        for cid in cids:
            for h in hits:
                rows.append({**h, "chain": cid})

    cols = ["chain", "source", "db", "accession", "name",
            "start", "end", "score", "description"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in sorted(rows, key=lambda x: (x["chain"], x["start"], x["end"])):
            fh.write("\t".join(str(r.get(c, "")).replace("\t", " ") for c in cols) + "\n")
    print(f"[annotate_domains] wrote {len(rows)} row(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
