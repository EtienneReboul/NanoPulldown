#!/usr/bin/env python3
"""
nanopulldown.annotate.merge — Stage 1b (optional human review)
======================================================================
Fuse each chain's domain hits (nanopulldown.annotate.domains) and AIUPred
disorder/binding tracks (nanopulldown.annotate.disorder), both read
straight from the per-sequence annotation cache, into ONE proposed `domains:`
block per protein chain, written to data/annotation/<pair>/annotation.yaml
for a human to curate into configs/curated_domains.yaml whenever convenient --
folding does not wait on this. Also emits a plain-text diff against
whatever `domains:` the config already has.

Adapted from ab_initio_pipeline's scripts/merge_annotation.py: disorder now
comes from a single AIUPred "disorder" track (no multi-track consensus
needed -- there's only one predictor now, so `disorder_cutoff` alone decides
the mask), and MoRF segments are extracted here directly from AIUPred's
"binding" track (contiguous runs >= `morfchibi.score_threshold`, at least
`morfchibi.min_len` residues) rather than read from a separate MoRFchibi2
segments file.

Segment logic, per chain, left to right (select_domains):
  * folded domains = InterPro's representative domain set -- the regions its
    protein viewer summarises (`representative` flag from the InterPro API;
    for InterProScan-fallback hits, which carry no flag, the same selection
    is recomputed here: overlapping candidates grouped, the combination with
    no pair overlapping >= 30% of the shorter that covers most residues wins,
    ties to more Pfam -- InterProScan 6's ProcessReprLocations) ...
  * ... plus InterPro domain / repeat ENTRIES no selected region covers (e.g.
    MYC2's N-terminal JAZ-interaction domain, which Pfam types as a family
    and so never enters the representative set)
  * each region is NAMED after the InterPro domain/repeat entry that best
    matches it (overlap >= half the shorter; best Jaccard x member-database
    consensus), so a Gene3D superfamily region reads "BTB/POZ domain", not
    "SKP1/BTB/POZ domain superfamily"; else its own entry or signature name
    (a site-type entry keeps the signature's name: "Jas motif")
  * no InterPro data at all -> ScanProsite hits, merged as before
  * disorder mask           = residues where the "disorder" track >= disorder_cutoff
  * gaps between domains    -> kind: disordered if mostly in the mask, else kind: linker
  * binding-track segments  -> added as separate kind: morf entries

annotation.yaml also gets an `interpro:` block per chain (family, entries,
unintegrated signatures, feature tracks) for nanopulldown.report.plots.domain_bar.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from nanopulldown.annotate.disorder import DISORDER_COLUMNS, pair_disorder_rows
from nanopulldown.annotate.domains import pair_domain_features, pair_domain_rows
from nanopulldown.core.config import load_config
from nanopulldown.core.specs import load_pair, protein_chains

CFG = load_config()
# member DBs whose domain/repeat/homologous-superfamily signatures are
# representative-domain candidates (InterPro docs: Representative Domain and
# Family selection), as annotate_domains labels them (API) / InterProScan TSV
DOMAIN_DBS = {"Pfam", "CDD", "PROSITE profiles", "ProSiteProfiles", "SMART", "NCBIfam", "NCBIFAM",
              "CATH-Gene3D", "Gene3D", "SUPERFAMILY"}
FAMILY_DBS = {"PANTHER", "PIRSF", "HAMAP", "NCBIfam", "NCBIFAM", "SFLD", "Pfam"}
DOMAIN_TYPES = {"domain", "repeat", "homologous_superfamily"}
OVERLAP = 0.3                 # InterProScan 6: max overlap (fraction of the shorter) inside a set
MAX_PER_GROUP = 20


_ACCESSION_RE = re.compile(r"^(IPR\d+|PF\d+|SM\d+|PS\d+|PTHR\d+|G3DSA:.*|cd\d+)$")


def _label(h: dict) -> str:
    """Human-readable hit name. Older cached/raw hits carry an accession
    (IPRxxxxxx, PFxxxxx, ...) in `name`; the readable text is in `description`."""
    name = (h.get("name") or "").strip()
    if name and not _ACCESSION_RE.match(name):
        return name
    return (h.get("description") or "").strip() or name or h["accession"]


def _merge_intervals(ivals: list[tuple[int, int, str]], gap: int = 10) -> list[tuple[int, int, str]]:
    if not ivals:
        return []
    ivals = sorted(ivals)
    out = [list(ivals[0])]
    for s, e, name in ivals[1:]:
        if s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, name])
    return [(s, e, n) for s, e, n in out]


def _binding_segments(dis_rows: list[dict], threshold: float, min_len: int) -> list[tuple[int, int]]:
    """Contiguous runs of the "binding" track >= threshold, length >= min_len."""
    by_res = {int(r["resi"]): float(r["score"]) for r in dis_rows if r["track"] == "binding"}
    if not by_res:
        return []
    residues = sorted(by_res)
    segs: list[tuple[int, int]] = []
    run_start = None
    prev = None
    for i in residues:
        hit = by_res[i] >= threshold
        if hit and run_start is None:
            run_start = i
        if not hit and run_start is not None:
            if prev - run_start + 1 >= min_len:
                segs.append((run_start, prev))
            run_start = None
        prev = i
    if run_start is not None and prev - run_start + 1 >= min_len:
        segs.append((run_start, prev))
    return segs



def _ov(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _frac_shorter(a, b) -> float:
    return _ov(a, b) / max(1, min(a[1] - a[0] + 1, b[1] - b[0] + 1))


def _jaccard(a, b) -> float:
    o = _ov(a, b)
    return o / max(1, (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - o)


def representative_set(cands: list[dict]) -> list[dict]:
    """InterProScan 6's representative-location selection (ProcessReprLocations):
    group overlapping candidates; per group keep the 20 longest, then the
    combination whose members pairwise overlap < 30% of the shorter and that
    covers the most residues (ties: more Pfam)."""
    cands = sorted(cands, key=lambda h: (h["start"], h["end"]))
    groups, cur, stop = [], [], -1
    for h in cands:
        if cur and h["start"] > stop:
            groups.append(cur)
            cur, stop = [], -1
        cur.append(h)
        stop = max(stop, h["end"])
    if cur:
        groups.append(cur)
    chosen = []
    for g in groups:
        g = sorted(g, key=lambda h: (-(h["end"] - h["start"]), h["db"] != "Pfam"))[:MAX_PER_GROUP]
        best, best_key = [], (-1, -1)

        def rec(i, picked):
            nonlocal best, best_key
            if i == len(g):
                cov = len({r for h in picked for r in range(h["start"], h["end"] + 1)})
                key = (cov, sum(h["db"] == "Pfam" for h in picked))
                if key > best_key:
                    best, best_key = list(picked), key
                return
            h = g[i]
            if all(_frac_shorter((h["start"], h["end"]), (o["start"], o["end"])) < OVERLAP for o in picked):
                rec(i + 1, picked + [h])
            rec(i + 1, picked)
        rec(0, [])
        chosen += best
    return chosen


def _entry_members(hits: list[dict]) -> dict[str, set[str]]:
    m: dict[str, set[str]] = defaultdict(set)
    for h in hits:
        if h.get("ipr") and h["db"] != "InterPro":
            m[h["ipr"]].add(h["db"])
    return m


def _region_name(region: tuple[int, int], sig: dict | None, entries: list[dict],
                 members: dict[str, set[str]]) -> str:
    """Best-matching InterPro domain/repeat entry; else the region's own entry
    (sites keep the signature's name, e.g. 'Jas motif'), else the signature."""
    best, best_score = None, 0.0
    for e in entries:
        if e.get("type") not in ("domain", "repeat"):
            continue
        span = (e["start"], e["end"])
        if _frac_shorter(region, span) < 0.5:
            continue
        score = _jaccard(region, span) * (1 + 0.1 * len(members.get(e["accession"], ())))
        if score > best_score:
            best, best_score = e, score
    if best:
        return best["name"]
    if sig:
        own = next((e for e in entries if e["accession"] == sig.get("ipr")), None)
        if own and own.get("type") != "site":
            return own["name"]
        if sig.get("ipr_name") and not own and sig.get("type") != "site":
            return sig["ipr_name"]
        return _label(sig)
    return "domain"


def select_domains(dom_hits: list[dict]) -> list[tuple[int, int, str]]:
    """(start, end, name) folded-domain regions of one chain -- see the
    module docstring."""
    ip = [h for h in dom_hits if h.get("source") in ("InterPro", "InterProScan")]
    if not ip:                                   # ScanProsite only: old behaviour
        allh = [(int(h["start"]), int(h["end"]), _label(h)) for h in dom_hits]
        return _merge_intervals(allh)
    entries = [h for h in ip if h["db"] == "InterPro"]
    sigs = [h for h in ip if h["db"] != "InterPro"]
    members = _entry_members(sigs)
    if any(h.get("representative") is not None for h in sigs):
        rep = [h for h in sigs if h.get("representative") and h.get("type") != "family"]
    else:                                        # InterProScan TSV: no flags -> recompute
        rep = representative_set([h for h in sigs if h["db"] in DOMAIN_DBS
                                  and (h.get("type") in DOMAIN_TYPES or h.get("type") is None)])
    regions = [((int(h["start"]), int(h["end"])), h) for h in rep]
    # InterPro domain/repeat entries nothing selected covers
    for e in sorted(entries, key=lambda e: -(e["end"] - e["start"])):
        span = (int(e["start"]), int(e["end"]))
        if e.get("type") in ("domain", "repeat") and all(
                _frac_shorter(span, r) < OVERLAP for r, _ in regions):
            regions.append((span, None))
    named = [(s, e, _region_name((s, e), sig, entries, members)) for (s, e), sig in regions]
    return sorted(named)


def interpro_tracks(dom_hits: list[dict], features: dict) -> dict:
    """The InterPro-viewer view of one chain, for plot_domain_bar.py:
    family / representative / entries / unintegrated / features."""
    ip = [h for h in dom_hits if h.get("source") in ("InterPro", "InterProScan")]
    entries = [h for h in ip if h["db"] == "InterPro"]
    sigs = [h for h in ip if h["db"] != "InterPro"]
    members = _entry_members(sigs)

    def span(h, **kw):
        return {"start": int(h["start"]), "end": int(h["end"]), **kw}
    fam_entries = [span(e, name=e["name"], accession=e["accession"]) for e in entries if e.get("type") == "family"]
    fam_sigs = [h for h in sigs if h.get("type") == "family" and h["db"] in FAMILY_DBS]
    rep_fam = [h for h in fam_sigs if h.get("representative")] or fam_sigs
    by_entry: dict[str, dict] = {}
    for e in entries:
        if e.get("type") == "family":
            continue
        d = by_entry.setdefault(e["accession"], {"accession": e["accession"], "name": e["name"],
                                                 "type": e.get("type"),
                                                 "members": sorted(members.get(e["accession"], ())),
                                                 "locations": []})
        d["locations"].append([int(e["start"]), int(e["end"])])
    unint = [span(h, name=_label(h), accession=h["accession"], db=h["db"], type=h.get("type"))
             for h in sigs if not h.get("ipr") and h.get("type") != "family"]
    return {
        "family": fam_entries or [span(h, name=_label(h), accession=h["accession"]) for h in rep_fam],
        "family_sources": sorted({f"{h['db']} {h['accession']}" for h in rep_fam}),
        "entries": sorted(by_entry.values(), key=lambda d: min(l[0] for l in d["locations"])),
        "unintegrated": sorted(unint, key=lambda d: d["start"]),
        "features": {k: [list(map(int, l)) for l in v] for k, v in (features or {}).items()},
    }


def _chain_segments(seqlen, dom_hits, dis_rows, acfg):
    domains = select_domains(dom_hits)

    # disorder mask (single AIUPred "disorder" track)
    by_res = defaultdict(dict)
    for r in dis_rows:
        by_res[int(r["resi"])][r["track"]] = float(r["score"])
    cutoff = acfg["consensus"]["disorder_cutoff"]
    mask = {i for i in range(1, seqlen + 1) if by_res.get(i, {}).get("disorder", 0.0) >= cutoff}

    segs: list[dict] = []
    cursor = 1
    di = 1
    for s, e, name in domains:
        s, e = max(s, 1, cursor), min(e, seqlen)    # trim the small overlaps a set may keep
        if e < s:
            continue
        if s > cursor:                       # gap before this domain
            gap_res = range(cursor, s)
            frac = sum(1 for i in gap_res if i in mask) / max(1, len(gap_res))
            segs.append({"name": f"region{di}", "start": cursor, "end": s - 1,
                         "kind": "disordered" if frac >= 0.5 else "linker"})
            di += 1
        segs.append({"name": name[:60] or f"dom{di}", "start": s, "end": e, "kind": "domain"})
        cursor = e + 1
    if cursor <= seqlen:
        gap_res = range(cursor, seqlen + 1)
        frac = sum(1 for i in gap_res if i in mask) / max(1, len(gap_res))
        segs.append({"name": f"region{di}", "start": cursor, "end": seqlen,
                     "kind": "disordered" if frac >= 0.5 else "linker"})

    morf_cfg = acfg["morfchibi"]
    for s, e in _binding_segments(dis_rows, morf_cfg["score_threshold"], morf_cfg["min_len"]):
        segs.append({"name": f"morf_{s}_{e}", "start": s, "end": e, "kind": "morf"})

    disfrac = len(mask) / max(1, seqlen)
    return segs, {"disordered_fraction": round(disfrac, 3),
                  "tracks": sorted({r["track"] for r in dis_rows})}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = load_pair(args.pair)
    acfg = CFG["annotate"]

    dom_all = pair_domain_rows(spec)
    feats = pair_domain_features(spec)
    dis_all = [dict(zip(DISORDER_COLUMNS, r)) for r in pair_disorder_rows(spec)]

    proposed, summary, tracks = {}, {}, {}
    for ch in protein_chains(spec):
        cid = ch["id"]
        tracks[cid] = interpro_tracks([h for h in dom_all if h["chain"] == cid], feats.get(cid))
        segs, summ = _chain_segments(
            len(ch["sequence"]),
            [h for h in dom_all if h["chain"] == cid],
            [r for r in dis_all if r["chain"] == cid],
            acfg,
        )
        proposed[cid] = segs
        summary[cid] = summ

    # diff vs current config
    current = spec.get("domains") or {}
    diff_lines = []
    for cid in proposed:
        cur = current.get(cid)
        if not cur:
            diff_lines.append(f"  {cid}: NEW ({len(proposed[cid])} segments proposed)")
        else:
            cur_set = {(d["start"], d["end"], d["kind"]) for d in cur}
            new_set = {(d["start"], d["end"], d["kind"]) for d in proposed[cid]}
            if cur_set != new_set:
                diff_lines.append(f"  {cid}: CHANGED  (config has {len(cur)}, proposal {len(proposed[cid])})")
            else:
                diff_lines.append(f"  {cid}: unchanged")

    doc = {
        "pair": args.pair,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "domains": proposed,
        "disorder_summary": summary,
        "interpro": tracks,
        "review": (
            "Curate these segments, paste the `domains:` block under "
            f"`{args.pair}:` in configs/curated_domains.yaml, with `annotation_reviewed: true`.\n"
            "Diff vs current config:\n" + "\n".join(diff_lines)
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False, width=100))
    print(f"[merge_annotation] -> {out}")
    print("\n".join(diff_lines))
    if not spec.get("annotation_reviewed"):
        print(f"\n[merge_annotation] {args.pair} is not reviewed in configs/curated_domains.yaml "
              "— folding will proceed anyway; curate `domains:` whenever convenient.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
