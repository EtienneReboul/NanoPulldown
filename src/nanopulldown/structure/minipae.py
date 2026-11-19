#!/usr/bin/env python3
"""
nanopulldown.structure.minipae — Stage 2d (third SLiM evidence)
===============================================================
Short linear motifs located by the fold itself: the residues of one side of a
pair whose position the model places confidently on the other side. A
per-residue minimum inter-chain PAE (min-iPAE, Omidi et al. 2024 PNAS
121:e2406407121; AF2_MiniPAE, github.com/martinovein/AF2_MiniPAE). For a
residue j of the motif side and the partner side's residues i,

    min-iPAE(j) = min_i PAE(i, j)        PAE(i, j): error at j when aligned on i

In Omidi 2024 this pinpoints the binding IDR of a full-length protein on its
receptor with AUROC 0.91 (AF-Multimer, Rg set). Unlike ELM (a regex, blind to
the partner) and SPARROW (IDR x IDR physicochemistry, no motif), it is
specific to the partner and to the class. It is only as good as the model,
though: it carries the fold's own false positives, which is what the decoy
p-value is for.

Per pair, for every chain of each side against all chains of the other side,
from results/<pair>/raw/fold.parquet (no new folding; both engines):
  1. min-iPAE per residue per sample; the profile is the MEDIAN over samples,
     `support` the fraction of samples below minipae.pae_threshold.
  2. Segments: residues with median < threshold AND support >=
     min_sample_fraction, merged over gaps <= max_gap, >= min_len long.
     Each is annotated with AIUPred disorder (full-length track, projected
     onto fragment chains), MoRF overlap, the partner residues it is placed on
     (argmin i of the best sample) and the partner's Pfam domains there, and
     the ELM hits it overlaps. `slim_like`: <= max_slim_len and
     disorder_fraction >= min_disorder_fraction (a long confident stretch of
     ordered residues is a domain-domain interface, not a SLiM).
  3. Per chain: `mini_pae` (min of the median profile), AF2_MiniPAE's own
     numbers on the best sample (min, and the greedy first-to-last confident
     residue span), and a conformal p-value against the bait's decoy pairs
     (same bait, same motif side): p = (1 + #{decoys with mini_pae <= x}) / (n + 1).
  4. ELM support: every ELM motif x partner-domain match (annotate_elm) gets
     the min-iPAE of the motif onto THAT domain (PAE rows restricted to the
     domain span) and onto the whole partner -- is the regex hit where the
     model docks the partner?

Best sample = highest 0.8 ipTM + 0.2 pTM (AF-Multimer's ranking confidence,
ColabFold's rank_001). The PAE threshold (default 4 A, AF2_MiniPAE's) was set
on AlphaFold-Multimer; ESMFold2's PAE is on a similar scale, but check it on
the decoy p-values before trusting it there.

Outputs:
  <results>/_minipae/minipae_profiles.parquet   one row per (pair, motif chain, residue)
  <results>/_minipae/minipae_segments.parquet   one row per confident segment
  <results>/_minipae/minipae_chains.parquet     one row per (pair, motif chain)
  <reports>/folding/minipae_segments.csv        segments, ranked run-wide, display names
  <reports>/folding/minipae_chains.csv          chains, with decoy p
  <reports>/folding/minipae_elm_support.csv     ELM partner matches + min-iPAE (if ELM on)

Usage:
  nanopulldown minipae --pairs <pair> [...] --results-root <results> --out-dir <results>/_minipae \\
      --csv-dir <reports>/folding [--decoy-pairs <pair> ...] [--elm-motifs <pq> --elm-matches <pq>]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from nanopulldown.core.config import LazyConfig

CFG = LazyConfig()

REFERENCES = ("Omidi A et al. (2024) AlphaFold-Multimer accurately captures interactions and dynamics "
              "of intrinsically disordered protein regions. PNAS 121:e2406407121 (min-iPAE); "
              "Veinstein M, AF2_MiniPAE (github.com/martinovein/AF2_MiniPAE, MIT)")

DEFAULTS = {
    "pae_threshold": 4.0,         # A; AF2_MiniPAE's default
    "min_sample_fraction": 0.5,   # of the pair's samples below the threshold
    "max_gap": 2,                 # residues above the threshold bridged inside a segment
    "min_len": 3,
    "max_slim_len": 30,
    "min_disorder_fraction": 0.5,
}


def mcfg() -> dict:
    return {**DEFAULTS, **(CFG.get("minipae") or {})}


# ── arithmetic (numpy only; tested in tests/test_minipae.py) ─────────────

def min_ipae(pae: np.ndarray, motif_idx, partner_idx) -> tuple[np.ndarray, np.ndarray]:
    """(min_i PAE(i, j) for j in motif_idx, the argmin i as an index into
    partner_idx). Rows of PAE are the aligned residue, so the partner is the
    row: "where is j, given the partner's frame"."""
    block = pae[np.ix_(np.asarray(partner_idx), np.asarray(motif_idx))]
    return block.min(axis=0), block.argmin(axis=0)


def call_segments(profile, support, threshold: float, min_support: float,
                  max_gap: int, min_len: int) -> list[tuple[int, int]]:
    """1-based inclusive segments of residues with profile < threshold and
    support >= min_support; runs separated by <= max_gap other residues are
    merged; segments shorter than min_len dropped."""
    ok = (np.asarray(profile) < threshold) & (np.asarray(support) >= min_support)
    runs, start = [], None
    for i, v in enumerate(list(ok) + [False], start=1):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i - 1])
            start = None
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s + 1 >= min_len]


def greedy_span(values, seq: str, threshold: float) -> tuple[int | None, int | None, str | None]:
    """AF2_MiniPAE's motif: confident residues upper-cased, then the first to
    the last upper-case residue (1-based inclusive); None with < 2 of them."""
    s = "".join(a.upper() if v < threshold else a.lower() for a, v in zip(seq, values))
    m = re.search(r"[A-Z].*[A-Z]", s)
    if not m:
        return None, None, None
    return m.start() + 1, m.end(), s[m.start():m.end()]


def conformal_p(x: float, null) -> float:
    """p of a LOWER-is-better score against a null sample."""
    null = np.asarray([v for v in null if np.isfinite(v)], float)
    return float((1 + (null <= x).sum()) / (len(null) + 1))


def full_positions(length: int, regions) -> list[int]:
    """Full-length residue number of each chain residue (identity without regions)."""
    if not regions:
        return list(range(1, length + 1))
    pos = [p for a, b in regions for p in range(int(a), int(b) + 1)]
    return pos if len(pos) == length else list(range(1, length + 1))


def segments_overlap(s: int, e: int, spans) -> int:
    return sum(max(0, min(e, b) - max(s, a) + 1) for a, b in spans)


# ── tables ────────────────────────────────────────────────────────────────

PROFILE_COLUMNS = {
    "pair": "pair id",
    "motif_role": "bait | prey: the side the residue is on",
    "motif_chain": "chain id",
    "residue": "residue number, chain numbering (1-based)",
    "full_residue": "residue number, full-length numbering",
    "aa": "amino acid",
    "min_ipae": "min-iPAE: min over the other side's residues i of PAE(i, j), median over samples (A)",
    "min_ipae_best": "min-iPAE in the best sample (highest 0.8 ipTM + 0.2 pTM) (A)",
    "support": "fraction of samples with min-iPAE < minipae.pae_threshold",
    "partner_chain": "chain of the argmin partner residue (best sample)",
    "partner_residue": "argmin partner residue, chain numbering (best sample)",
    "disorder": "AIUPred disorder of the residue (full-length prediction); null if not cached",
}

SEGMENT_COLUMNS = {
    "pair": "pair id",
    "rank": "rank within the pair: slim_like first, then lower min_pae, then higher support",
    "motif_role": PROFILE_COLUMNS["motif_role"],
    "motif_chain": "chain carrying the segment",
    "motif_accession": "UniProt accession of the segment's protein",
    "start": "first residue, chain numbering",
    "end": "last residue, chain numbering (inclusive)",
    "full_start": "first residue, full-length numbering",
    "full_end": "last residue, full-length numbering",
    "segment": "sequence; upper case = min-iPAE < threshold",
    "length": "residues",
    "min_pae": "min over the segment of the median min-iPAE profile (A)",
    "mean_pae": "mean over the segment of the median min-iPAE profile (A)",
    "support": "fraction of samples in which a segment residue has min-iPAE < threshold",
    "disorder_fraction": "fraction of residues with AIUPred disorder >= annotate.consensus.disorder_cutoff",
    "morf_overlap": "residues inside an AIUPred MoRF segment (annotate.morfchibi)",
    "slim_like": "length <= max_slim_len and disorder_fraction >= min_disorder_fraction",
    "partner_chain": "partner chain the segment is placed on (best sample; ';' if several)",
    "partner_start": "first partner residue the segment's residues are placed on (argmin, best sample)",
    "partner_end": "last such partner residue",
    "partner_domains": "partner Pfam domains overlapping partner_start-end, ';'-joined",
    "elm_hits": "disordered ELM hits (annotate_elm) overlapping the segment, ';'-joined",
    "elm_partner_match": "an ELM motif x partner-domain match of this pair overlaps the segment",
}

CHAIN_COLUMNS = {
    "pair": "pair id",
    "motif_role": PROFILE_COLUMNS["motif_role"],
    "motif_chain": "chain id",
    "motif_accession": "UniProt accession",
    "length": "chain length",
    "n_samples": "samples with a PAE matrix",
    "mini_pae": "min over the chain of the median min-iPAE profile (A)",
    "mini_pae_best": "AF2_MiniPAE's miniPAE: min min-iPAE of the best sample (A)",
    "span_start": "AF2_MiniPAE motif: first confident residue of the best sample",
    "span_end": "AF2_MiniPAE motif: last confident residue (the span includes the non-confident ones between)",
    "span_sequence": "AF2_MiniPAE motif sequence (upper = confident)",
    "n_segments": "confident segments (step 2)",
    "n_slim_like": "of which slim_like",
    "is_decoy": "the prey is a decoy (controls.decoys)",
    "n_decoys": "decoy pairs of the same bait, same motif side, in the null",
    "decoy_p": "conformal p of mini_pae against those decoys (lower mini_pae = better); null without decoys",
}

ELM_EXTRA_COLUMNS = {
    "minipae_to_domain": "min over the motif residues of min-iPAE onto the partner domain span, median over samples (A)",
    "minipae_to_partner": "min over the motif residues of min-iPAE onto the whole partner, median over samples (A)",
    "domain_support": "fraction of samples with minipae_to_domain < threshold",
    "supported": "minipae_to_domain < threshold",
}


# ── per pair ──────────────────────────────────────────────────────────────

def read_fold_pae(path: Path) -> list[dict]:
    """The fold.parquet columns minipae needs, as read_fold() returns them,
    but columnar: the PAE lists go to numpy in one Arrow call instead of
    through Python lists (~10x faster on 1000-token pairs)."""
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["sample_index", "ptm", "iptm", "n_tokens", "token_chain", "pae"])
    meta = t.select(["sample_index", "ptm", "iptm", "n_tokens", "token_chain"]).to_pylist()
    pae_col = t.column("pae").combine_chunks()
    flat = pae_col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    offs = pae_col.offsets.to_numpy()
    rows = []
    for k, r in enumerate(meta):
        n = r["n_tokens"]
        block = flat[offs[k]:offs[k + 1]]
        r["pae"] = block.reshape(n, n) if block.size == n * n else None
        r["token_chain"] = np.asarray(r["token_chain"])
        rows.append(r)
    return sorted(rows, key=lambda r: r["sample_index"])


def _ranking(r: dict) -> float:
    iptm, ptm = r.get("iptm"), r.get("ptm")
    if iptm is None:
        return -np.inf
    return 0.8 * float(iptm) + 0.2 * float(ptm or 0.0)


def pair_profiles(rows: list[dict], spec: dict) -> dict:
    """{chain: {"role", "values" (S, L), "best" (L,), "partner_chain", "partner_res"}}
    from one pair's fold rows; chains whose token count differs from the
    spec's sequence are skipped."""
    from nanopulldown.core.specs import bait_chain_ids, prey_chain_ids

    roles = {c: "bait" for c in bait_chain_ids(spec)} | {c: "prey" for c in prey_chain_ids(spec)}
    seqs = {ch["id"]: ch["sequence"] for ch in spec.get("sequences", [])}
    rows = [r for r in rows if r["pae"] is not None and len(r["token_chain"]) == r["pae"].shape[0]]
    if not rows:
        return {}
    best = max(range(len(rows)), key=lambda k: _ranking(rows[k]))
    out = {}
    for c, role in roles.items():
        per_sample, best_vals, pch, pres = [], None, None, None
        for k, r in enumerate(rows):
            tc = r["token_chain"]
            motif = np.flatnonzero(tc == c)
            partner = np.flatnonzero(np.isin(tc, [x for x, ro in roles.items() if ro != role]))
            if len(motif) != len(seqs.get(c, "")) or not len(partner):
                break
            v, am = min_ipae(r["pae"], motif, partner)
            per_sample.append(v)
            if k == best:
                best_vals = v
                ptok = partner[am]
                pch = tc[ptok]
                # chain numbering of the partner token = its rank within its chain
                first = {x: np.flatnonzero(tc == x)[0] for x in np.unique(pch)}
                pres = np.array([t - first[ch] + 1 for t, ch in zip(ptok, pch)])
        else:
            out[c] = {"role": role, "values": np.vstack(per_sample), "best": best_vals,
                      "partner_chain": pch, "partner_res": pres}
    return out


def domain_min_ipae(rows: list[dict], motif_chain: str, start: int, end: int,
                    partner_chain: str, dstart: int | None, dend: int | None) -> np.ndarray:
    """Per sample: min over motif residues start..end of min over partner
    residues dstart..dend (all of partner_chain when None) of PAE(i, j)."""
    out = []
    for r in rows:
        if r["pae"] is None or len(r["token_chain"]) != r["pae"].shape[0]:
            continue
        tc = r["token_chain"]
        m = np.flatnonzero(tc == motif_chain)[start - 1:end]
        p = np.flatnonzero(tc == partner_chain)
        if dstart is not None:
            p = p[dstart - 1:dend]
        if len(m) and len(p):
            out.append(float(min_ipae(r["pae"], m, p)[0].min()))
    return np.asarray(out)


def prefetch(fn, items, workers: int):
    """(item, fn(item)) in order, with at most 2 * workers calls in flight:
    the float16 PAE decode (~1 s per pair) runs in parallel while memory
    stays bounded to a few pairs' matrices."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max(1, workers)) as ex:
        q, it = deque(), iter(items)
        for x in it:
            q.append((x, ex.submit(fn, x)))
            if len(q) >= 2 * max(1, workers):
                break
        while q:
            x, fut = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append((nxt, ex.submit(fn, nxt)))
            yield x, fut.result()


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    import pandas as pd

    from nanopulldown.annotate.domains import (chain_record, interproscan_email, region_source,
                                               seq_accessions)
    from nanopulldown.annotate.elm import morf_segments
    from nanopulldown.core.cache import cached_disorder_tracks
    from nanopulldown.core.fold_table import FOLD_FILE
    from nanopulldown.core.io import seq_md5, write_parquet_with_metadata
    from nanopulldown.core.proteins import load_proteins, protein_name
    from nanopulldown.core.specs import load_pairs

    ap = argparse.ArgumentParser(description="min-iPAE short linear motifs (Omidi 2024 / AF2_MiniPAE)")
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--decoy-pairs", nargs="*", default=[])
    ap.add_argument("--elm-motifs", default=None)
    ap.add_argument("--elm-matches", default=None)
    ap.add_argument("--threads", type=int, default=4, help="fold.parquet reads in flight (Arrow decodes off the GIL)")
    a = ap.parse_args()

    c = mcfg()
    thr, min_sup = float(c["pae_threshold"]), float(c["min_sample_fraction"])
    acfg = CFG["annotate"]
    dcut = float(acfg["consensus"]["disorder_cutoff"])
    morf = acfg["morfchibi"]
    email = interproscan_email(acfg)
    specs = load_pairs()
    decoys = set(a.decoy_pairs)

    elm_by_md5: dict[str, list[tuple[int, int, str]]] = {}
    if a.elm_motifs and Path(a.elm_motifs).exists():
        for r in pd.read_parquet(a.elm_motifs).itertuples():
            elm_by_md5.setdefault(r.seq_md5, []).append((int(r.start), int(r.end), r.elm_identifier))
    elm_matches = (pd.read_parquet(a.elm_matches) if a.elm_matches and Path(a.elm_matches).exists()
                   else None)
    elm_spans: dict[tuple[str, str], list[tuple[int, int]]] = {}
    if elm_matches is not None:
        for r in elm_matches.itertuples():
            elm_spans.setdefault((r.pair, r.motif_chain), []).append((int(r.start), int(r.end)))

    prof_rows, seg_rows, chain_rows, elm_rows = [], [], [], []
    def load(p):
        fold_path = Path(a.results_root) / p / "raw" / FOLD_FILE
        return read_fold_pae(fold_path) if fold_path.exists() else None

    for p, rows in prefetch(load, a.pairs, a.threads):
        spec = specs[p]
        if rows is None:
            print(f"[minipae] {p}: no fold.parquet, skipped", file=sys.stderr)
            continue
        prof = pair_profiles(rows, spec)
        if not prof:
            print(f"[minipae] {p}: no usable PAE matrix, skipped", file=sys.stderr)
            continue
        accs = seq_accessions(spec)
        chains = {ch["id"]: ch for ch in spec.get("sequences", [])}
        pfam: dict[str, list[dict]] = {}
        for cid, ch in chains.items():
            try:
                rec = chain_record(spec, ch, acfg, email, accs, log=lambda m: print(m, file=sys.stderr))
                pfam[cid] = [h for h in rec["hits"] if h.get("db") == "Pfam"]
            except Exception as e:          # cache miss offline: domains are an annotation only
                print(f"[minipae] {p} {cid}: no domain record ({e})", file=sys.stderr)
                pfam[cid] = []

        for cid, d in prof.items():
            ch = chains[cid]
            seq, role = ch["sequence"], d["role"]
            acc = (spec.get(role) or {}).get("uniprot")
            src = region_source(spec, ch)
            full, regions = (src[0], src[2]) if src else (seq, None)
            fpos = full_positions(len(seq), regions)
            tracks = cached_disorder_tracks(CFG, full)
            fdis = tracks.get("disorder") or []
            dis = [fdis[q - 1] for q in fpos] if len(fdis) == len(full) else None
            fbind = tracks.get("binding") or []
            fmorf = (morf_segments(fbind, float(morf["score_threshold"]), int(morf["min_len"]))
                     if len(fbind) == len(full) else [])
            med = np.median(d["values"], axis=0)
            sup = (d["values"] < thr).mean(axis=0)
            for k in range(len(seq)):
                prof_rows.append({
                    "pair": p, "motif_role": role, "motif_chain": cid, "residue": k + 1,
                    "full_residue": fpos[k], "aa": seq[k], "min_ipae": float(med[k]),
                    "min_ipae_best": float(d["best"][k]), "support": float(sup[k]),
                    "partner_chain": str(d["partner_chain"][k]), "partner_residue": int(d["partner_res"][k]),
                    "disorder": None if dis is None else float(dis[k])})

            segs = call_segments(med, sup, thr, min_sup, int(c["max_gap"]), int(c["min_len"]))
            seq_elm = elm_by_md5.get(seq_md5(full), [])
            rows_here = []
            for s, e in segs:
                sl = slice(s - 1, e)
                dfrac = (None if dis is None else
                         round(float(np.mean([x >= dcut for x in dis[sl]])), 3))
                fs, fe = fpos[s - 1], fpos[e - 1]
                pch = sorted(set(d["partner_chain"][sl].tolist()))
                pres = d["partner_res"][sl]
                ps, pe = int(pres.min()), int(pres.max())
                pdoms = sorted({h.get("name") or h.get("accession") for pc in pch for h in pfam.get(pc, [])
                                if int(h["start"]) <= pe and int(h["end"]) >= ps})
                hits = sorted({eid for a_, b_, eid in seq_elm if a_ <= fe and b_ >= fs})
                rows_here.append({
                    "pair": p, "motif_role": role, "motif_chain": cid, "motif_accession": acc,
                    "start": s, "end": e, "full_start": fs, "full_end": fe,
                    "segment": "".join(x.upper() if v < thr else x.lower()
                                       for x, v in zip(seq[sl], med[sl])),
                    "length": e - s + 1,
                    "min_pae": round(float(med[sl].min()), 2), "mean_pae": round(float(med[sl].mean()), 2),
                    "support": round(float((d["values"][:, sl] < thr).any(axis=1).mean()), 3),
                    "disorder_fraction": dfrac,
                    "morf_overlap": segments_overlap(fs, fe, fmorf),
                    "slim_like": bool(e - s + 1 <= int(c["max_slim_len"]) and dfrac is not None
                                      and dfrac >= float(c["min_disorder_fraction"])),
                    "partner_chain": ";".join(pch), "partner_start": ps, "partner_end": pe,
                    "partner_domains": ";".join(pdoms) or None,
                    "elm_hits": ";".join(hits) or None,
                    "elm_partner_match": segments_overlap(s, e, elm_spans.get((p, cid), [])) > 0,
                })
            seg_rows += rows_here

            bs, be, bseq = greedy_span(d["best"], seq, thr)
            chain_rows.append({
                "pair": p, "motif_role": role, "motif_chain": cid, "motif_accession": acc,
                "length": len(seq), "n_samples": int(d["values"].shape[0]),
                "mini_pae": round(float(med.min()), 2), "mini_pae_best": round(float(d["best"].min()), 2),
                "span_start": bs, "span_end": be, "span_sequence": bseq,
                "n_segments": len(segs), "n_slim_like": sum(r["slim_like"] for r in rows_here),
                "is_decoy": p in decoys, "n_decoys": 0, "decoy_p": None})

        if elm_matches is not None:
            for r in elm_matches[elm_matches.pair == p].to_dict("records"):
                dom = domain_min_ipae(rows, r["motif_chain"], int(r["start"]), int(r["end"]),
                                      r["partner_chain"], int(r["partner_domain_start"]),
                                      int(r["partner_domain_end"]))
                whole = domain_min_ipae(rows, r["motif_chain"], int(r["start"]), int(r["end"]),
                                        r["partner_chain"], None, None)
                md = float(np.median(dom)) if len(dom) else None
                elm_rows.append({**r,
                                 "minipae_to_domain": None if md is None else round(md, 2),
                                 "minipae_to_partner": round(float(np.median(whole)), 2) if len(whole) else None,
                                 "domain_support": round(float((dom < thr).mean()), 3) if len(dom) else None,
                                 "supported": bool(md is not None and md < thr)})
        print(f"[minipae] {p}: {sum(1 for r in seg_rows if r['pair'] == p)} segment(s)", flush=True)

    # decoy null: same bait spec, same motif side
    chains_df = pd.DataFrame(chain_rows, columns=list(CHAIN_COLUMNS))
    if len(chains_df) and chains_df.is_decoy.any():
        bait = chains_df.pair.str.split("__").str[0]
        for (b, role), g in chains_df.groupby([bait, chains_df.motif_role]):
            null = g.loc[g.is_decoy, "mini_pae"].to_numpy()
            if not len(null):
                continue
            for i in g.index:
                own = null if not chains_df.at[i, "is_decoy"] else g.loc[g.is_decoy & (g.index != i), "mini_pae"]
                chains_df.at[i, "n_decoys"] = len(own)
                chains_df.at[i, "decoy_p"] = round(conformal_p(chains_df.at[i, "mini_pae"], own), 4)

    segs_df = pd.DataFrame(seg_rows, columns=[k for k in SEGMENT_COLUMNS if k != "rank"])
    if len(segs_df):
        segs_df = segs_df.sort_values(["pair", "slim_like", "min_pae", "support"],
                                      ascending=[True, False, True, False], kind="stable")
        segs_df.insert(1, "rank", segs_df.groupby("pair").cumcount() + 1)
    else:
        segs_df.insert(1, "rank", pd.Series(dtype=int))
    prof_df = pd.DataFrame(prof_rows, columns=list(PROFILE_COLUMNS))

    out, csvd = Path(a.out_dir), Path(a.csv_dir)
    out.mkdir(parents=True, exist_ok=True)
    csvd.mkdir(parents=True, exist_ok=True)
    params = (f"pae_threshold={thr} A; min_sample_fraction={min_sup}; max_gap={c['max_gap']}; "
              f"min_len={c['min_len']}; max_slim_len={c['max_slim_len']}; "
              f"min_disorder_fraction={c['min_disorder_fraction']}")
    for df, name, cols, what in (
            (prof_df, "minipae_profiles", PROFILE_COLUMNS, "per-residue min-iPAE profiles"),
            (segs_df, "minipae_segments", SEGMENT_COLUMNS, "confident min-iPAE segments (candidate SLiMs)"),
            (chains_df, "minipae_chains", CHAIN_COLUMNS, "per-chain min-iPAE summary + decoy p")):
        write_parquet_with_metadata(df, out / f"{name}.parquet", column_descriptions=cols,
                                    table_references=REFERENCES,
                                    table_description=f"nanopulldown.structure.minipae: {what} ({params})")

    proteins = load_proteins()

    def named(df):
        v = df.copy()
        v.insert(1, "protein", [protein_name(x, proteins) if x else "" for x in v["motif_accession"]])
        return v

    view = named(segs_df)
    if len(view):
        view = view.sort_values(["slim_like", "min_pae", "support"], ascending=[False, True, False],
                                kind="stable")
    view.to_csv(csvd / "minipae_segments.csv", index=False)
    named(chains_df).sort_values(["decoy_p", "mini_pae"], na_position="last", kind="stable").to_csv(
        csvd / "minipae_chains.csv", index=False)
    if elm_matches is None:          # ELM off: an empty table keeps the report config valid
        from nanopulldown.annotate.elm import MATCH_COLUMNS
        cols = list(MATCH_COLUMNS)
        pd.DataFrame(columns=[cols[0], "motif_protein", "partner_protein", *cols[1:],
                              *ELM_EXTRA_COLUMNS]).to_csv(
            csvd / "minipae_elm_support.csv", index=False)
    else:
        ev = pd.DataFrame(elm_rows, columns=list(elm_matches.columns) + list(ELM_EXTRA_COLUMNS))
        ev.insert(1, "motif_protein", [protein_name(x, proteins) if x else "" for x in ev["motif_accession"]])
        ev.insert(2, "partner_protein", [protein_name(x, proteins) if x else "" for x in ev["partner_accession"]])
        ev.sort_values(["supported", "minipae_to_domain"], ascending=[False, True], na_position="last",
                       kind="stable").to_csv(csvd / "minipae_elm_support.csv", index=False)
        print(f"[minipae] ELM support: {int(ev.supported.sum())}/{len(ev)} motif-domain match(es) "
              f"with min-iPAE onto the domain < {thr} A", flush=True)
    print(f"[minipae] {len(segs_df)} segment(s) ({int(segs_df.slim_like.sum()) if len(segs_df) else 0} "
          f"SLiM-like) in {chains_df.pair.nunique()} pair(s); {params}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
