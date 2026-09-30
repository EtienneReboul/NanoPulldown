#!/usr/bin/env python3
"""
scripts/merge_annotation.py — Stage 1b (optional human review)
======================================================================
Fuse domains.raw.tsv + disorder_binding.tsv into ONE proposed `domains:`
block per protein chain, written to data/annotation/<pair>/annotation.yaml
for a human to curate into configs/<pair>.yaml whenever convenient --
folding does not wait on this. Also emits a plain-text diff against
whatever `domains:` the config already has.

Adapted from ab_initio_pipeline's scripts/merge_annotation.py: disorder now
comes from a single AIUPred "disorder" track (no multi-track consensus
needed -- there's only one predictor now, so `disorder_cutoff` alone decides
the mask), and MoRF segments are extracted here directly from AIUPred's
"binding" track (contiguous runs >= `morfchibi.score_threshold`, at least
`morfchibi.min_len` residues) rather than read from a separate MoRFchibi2
segments file.

Segment logic, per chain, left to right:
  * folded-domain intervals = merged InterProScan/PROSITE hits (profiles
    preferred over patterns; overlapping same-DB hits merged)
  * disorder mask           = residues where the "disorder" track >= disorder_cutoff
  * gaps between domains    -> kind: disordered if mostly in the mask, else kind: linker
  * binding-track segments  -> added as separate kind: morf entries
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import load_config, load_pair, protein_chains  # noqa: E402

CFG = load_config()
DOMAIN_DBS_PREFERRED = ("PfamA", "Pfam", "PROSITEProfiles", "PROSITE_profiles", "SMART", "Gene3D")


def _read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


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


def _chain_segments(seqlen, dom_hits, dis_rows, acfg):
    # folded domains from hits
    pref = [(int(h["start"]), int(h["end"]), h["name"] or h["accession"])
            for h in dom_hits if h["db"] in DOMAIN_DBS_PREFERRED]
    allh = [(int(h["start"]), int(h["end"]), h["name"] or h["accession"]) for h in dom_hits]
    domains = _merge_intervals(pref or allh)

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
        s, e = max(s, 1), min(e, seqlen)
        if s > cursor:                       # gap before this domain
            gap_res = range(cursor, s)
            frac = sum(1 for i in gap_res if i in mask) / max(1, len(gap_res))
            segs.append({"name": f"region{di}", "start": cursor, "end": s - 1,
                         "kind": "disordered" if frac >= 0.5 else "linker"})
            di += 1
        segs.append({"name": name[:24] or f"dom{di}", "start": s, "end": e, "kind": "domain"})
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
    ap.add_argument("--spec", required=True)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--disorder", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path(args.spec).resolve().parent.parent
    spec = load_pair(args.pair, repo)
    acfg = CFG["annotate"]

    dom_all = _read_tsv(Path(args.domains))
    dis_all = _read_tsv(Path(args.disorder))

    proposed, summary = {}, {}
    for ch in protein_chains(spec):
        cid = ch["id"]
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
        "review": (
            "Curate these segments, paste the `domains:` block into "
            f"configs/{args.pair}.yaml, then set `annotation_reviewed: true`.\n"
            "Diff vs current config:\n" + "\n".join(diff_lines)
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False, width=100))
    print(f"[merge_annotation] -> {out}")
    print("\n".join(diff_lines))
    if not spec.get("annotation_reviewed"):
        print(f"\n[merge_annotation] configs/{args.pair}.yaml still has annotation_reviewed: "
              "false — folding will proceed anyway; curate `domains:` whenever convenient.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
