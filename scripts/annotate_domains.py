#!/usr/bin/env python3
"""
scripts/annotate_domains.py — Stage 1b
=======================================
Per protein chain of a pair, query two web services and dump every domain
/ family / motif hit to a flat TSV:

  * ScanProsite   (PROSITE patterns + profiles)   — prosite.expasy.org REST
  * InterProScan v6 (Pfam / SMART / PROSITE profiles / Gene3D) — EBI MATCH API

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
import hashlib
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import load_config, load_pair, protein_chains  # noqa: E402

CFG = load_config()
UA = {"User-Agent": "nanopulldown/annotate_domains"}


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
    r = requests.get(url, params=params, headers=UA, timeout=120)
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

def _md5_of(seq: str) -> str:
    return hashlib.md5(seq.encode("ascii")).hexdigest()


def _ips6_match_lookup(md5: str, cfg: dict) -> list[dict] | None:
    """Query the MATCH API by MD5. Returns the parsed hit list on a cache
    hit, or None on a cache miss (or any error -- callers fall back to a
    full submission rather than fail the run over the cache path alone)."""
    base = cfg["base_url"]
    endpoint = cfg.get("match_endpoint", "matches")
    try:
        r = requests.get(f"{base}/{endpoint}/{md5}", headers=UA, timeout=60)
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
    r = requests.post(f"{base}/run", data=payload, headers=UA, timeout=120)
    if r.status_code == 400 and "appl" in r.text and appl:
        print(f"  [interproscan] server rejected appl='{appl}', retrying with defaults",
              file=sys.stderr)
        payload.pop("appl")
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
            "name": (p[11] if len(p) > 11 and p[11] not in ("", "-") else p[4]),
            "start": start, "end": stop,
            "score": p[8] if len(p) > 8 else "",
            "description": (p[12] if len(p) > 12 else p[5] if len(p) > 5 else ""),
        })
    return hits


def interproscan(seq: str, cfg: dict) -> tuple[list[dict], bool]:
    """(hits, cache_hit) -- MD5 match lookup first, full submission on miss."""
    md5 = _md5_of(seq)
    cached = _ips6_match_lookup(md5, cfg)
    if cached is not None:
        return cached, True
    return _ips6_submit_and_poll(seq, cfg), False


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

    email = (acfg["interproscan"].get("email") or "").strip()
    if not email:
        print("[annotate_domains] no annotate.interproscan.email set (config.local.yaml) "
              "— skipping InterProScan, running ScanProsite only", file=sys.stderr)

    # identical chains (homodimers etc.) — query the web services once per
    # unique sequence, then replicate the hits onto every chain that shares it.
    by_seq: dict[str, list[str]] = {}
    for ch in protein_chains(spec):
        by_seq.setdefault(ch["sequence"], []).append(ch["id"])

    for seq, cids in by_seq.items():
        print(f"[annotate_domains] chains {cids} ({len(seq)} aa)", flush=True)
        try:
            ps = scanprosite(seq, acfg["prosite"]["scan_url"],
                             acfg["prosite"].get("skip_profiles", False))
            print(f"  ScanProsite: {len(ps)} hit(s)")
        except Exception as e:                       # noqa: BLE001
            print(f"  ScanProsite FAILED: {e}", file=sys.stderr)
            ps = []
        ip = []
        if email:
            try:
                ip, cache_hit = interproscan(seq, acfg["interproscan"])
                print(f"  InterProScan v6: {len(ip)} hit(s) ({'MD5 cache hit' if cache_hit else 'fresh submission'})")
            except Exception as e:                   # noqa: BLE001
                print(f"  InterProScan FAILED: {e}", file=sys.stderr)
                ip = []
        for cid in cids:
            for h in ps + ip:
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
