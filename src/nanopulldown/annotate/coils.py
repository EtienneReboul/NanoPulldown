#!/usr/bin/env python3
"""
nanopulldown.annotate.coils — Stage 1b (coiled coils)
=====================================================
DeepCoil 2 coiled-coil segments and heptad register, which take precedence
over InterPro's COILS (the `coils` extra feature) wherever DeepCoil has a
prediction for the sequence.

Why: COILS (Lupas 1991) is a sequence-profile window scan, a single 21-residue
window above threshold is enough for a call, and its false-positive rate is
high; in the 2026-11-10 test on 20 DNE1 proteins, 8 of the 12 COILS calls
scored ~0 with DeepCoil. DeepCoil also gives a/d heptad positions, which
nanopulldown.structure.coiled_coil checks against the knobs it finds in the
models.

Pieces:
  fasta     (rule deepcoil_fasta) every unique sequence of the run's pairs --
            chain sequences and, for region chains, the full-length entry, as
            for the domain cache -- as a multi-FASTA with md5 headers
  weights   (rule fetch_deepcoil_weights) the SeqVec ELMo weights DeepCoil
            embeds with (~370 MB, once; login node on offline clusters)
  (the prediction itself is nanopulldown.annotate.deepcoil_runner, run in
   envs/deepcoil.yaml by rule annotate_coils_batch)
  chain_coils()   per chain, used by `annotate merge`: segments (cc >=
            annotate.deepcoil.cc_threshold for >= min_len residues) and their
            a/d residues, computed on the sequence DeepCoil saw and projected
            onto the chain's numbering for region chains

Register: per segment, the heptad phase maximising sum(prob_a at a) +
sum(prob_d at d) (d = a + 3); one phase per segment, so a stutter or stammer
inside a long coil is smoothed over.

Usage:
  nanopulldown annotate coils-fasta   --pairs <pair> [<pair> ...] --out <fasta>
  nanopulldown annotate coils-weights --out-dir <dir>
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import requests

from nanopulldown.core.cache import annotation_cache_dir
from nanopulldown.core.config import LazyConfig
from nanopulldown.core.io import atomic_write, seq_md5

CFG = LazyConfig()
CACHE_KIND = "deepcoil"
WEIGHT_FILES = ("uniref50_v2/options.json", "uniref50_v2/weights.hdf5")
DEFAULT_URLS = ["https://lbs.cent.uw.edu.pl/static/files/seqvec.zip",
                "https://rostlab.org/~deepppi/seqvec.zip"]


def deepcoil_cfg() -> dict:
    return CFG["annotate"].get("deepcoil") or {}


def deepcoil_record(seq: str) -> dict | None:
    """The cached DeepCoil prediction of a sequence ({cc, raw_cc, prob_a,
    prob_d} per residue), or None when absent / skipped / malformed."""
    try:
        rec = json.loads((annotation_cache_dir(CFG, CACHE_KIND) / f"{seq_md5(seq)}.json").read_text())
    except (OSError, ValueError):
        return None
    return rec if len(rec.get("cc") or ()) == len(seq) else None


def segments(cc: list[float], threshold: float, min_len: int) -> list[tuple[int, int]]:
    """1-based inclusive runs of cc >= threshold at least min_len long."""
    out, st = [], None
    for i, v in enumerate(list(cc) + [-1.0]):
        if v >= threshold and st is None:
            st = i
        elif v < threshold and st is not None:
            if i - st >= min_len:
                out.append((st + 1, i))
            st = None
    return out


def register(prob_a: list[float], prob_d: list[float], start: int, end: int) -> dict:
    """Best single heptad phase of segment [start, end] (1-based): its a and
    d residues and the mean probability DeepCoil gives them."""
    best = None
    for ph in range(7):
        a = [r for r in range(start, end + 1) if (r - start - ph) % 7 == 0]
        d = [r for r in range(start, end + 1) if (r - start - ph) % 7 == 3]
        sc = sum(prob_a[r - 1] for r in a) + sum(prob_d[r - 1] for r in d)
        if best is None or sc > best[0]:
            best = (sc, a, d)
    sc, a, d = best
    return {"a": a, "d": d, "register_prob": round(sc / max(1, len(a) + len(d)), 3)}


def coil_calls(seq: str, dcfg: dict | None = None) -> list[dict] | None:
    """[{start, end, mean_cc, a, d, register_prob}] of a sequence in its own
    numbering, or None when DeepCoil has no prediction for it."""
    dcfg = dcfg if dcfg is not None else deepcoil_cfg()
    rec = deepcoil_record(seq)
    if rec is None:
        return None
    cc = rec["cc"]
    out = []
    for s, e in segments(cc, float(dcfg.get("cc_threshold", 0.5)), int(dcfg.get("min_len", 7))):
        out.append({"start": s, "end": e, "mean_cc": round(sum(cc[s - 1:e]) / (e - s + 1), 3),
                    **register(rec["prob_a"], rec["prob_d"], s, e)})
    return out


def _project_residues(res: list[int], regions: list) -> list[int]:
    """Full-length residue numbers -> the concatenated regions' numbering."""
    out, off = [], 0
    for a, b in regions:
        out += [r - a + 1 + off for r in res if a <= r <= b]
        off += b - a + 1
    return sorted(out)


def chain_coils(spec: dict, ch: dict, dcfg: dict | None = None) -> list[dict] | None:
    """DeepCoil coils of one chain in CHAIN numbering (region chains: called
    on the full-length entry, then projected; a coil cut by a fragment edge
    keeps its inside part), or None without a DeepCoil prediction."""
    from nanopulldown.annotate.domains import project_span, region_source

    src = region_source(spec, ch)
    calls = coil_calls(src[0] if src else ch["sequence"], dcfg)
    if calls is None or not src:
        return calls
    regions = src[2]
    out = []
    for c in calls:
        a, d = _project_residues(c["a"], regions), _project_residues(c["d"], regions)
        for s, e in project_span(c["start"], c["end"], regions):
            out.append({**c, "start": s, "end": e,
                        "a": [r for r in a if s <= r <= e], "d": [r for r in d if s <= r <= e]})
    return out


def overlay_tracks(tracks: dict, coils: list[dict] | None) -> dict:
    """annotation.yaml `interpro.<chain>` with DeepCoil's coils in place of
    COILS': features.coils <- DeepCoil, COILS kept as features.coils_interpro,
    coils_source names which one `coils` is, coil_register holds a/d."""
    feats = tracks.setdefault("features", {})
    if coils is None:
        tracks["coils_source"] = "COILS" if feats.get("coils") else None
        return tracks
    if feats.get("coils"):
        feats["coils_interpro"] = feats["coils"]
    feats["coils"] = [[c["start"], c["end"]] for c in coils]
    if not feats["coils"]:
        del feats["coils"]
    tracks["coils_source"] = "DeepCoil"
    tracks["coil_register"] = [{k: c[k] for k in ("start", "end", "mean_cc", "a", "d", "register_prob")}
                               for c in coils]
    return tracks


# ── CLI ─────────────────────────────────────────────────────────────────

def fasta_main() -> int:
    ap = argparse.ArgumentParser(description="unique sequences of the pairs, md5 headers, for DeepCoil")
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from nanopulldown.annotate.domains import annotation_targets
    from nanopulldown.core.specs import load_pairs

    specs, seqs = load_pairs(), set()
    for p in a.pairs:
        seqs.update(ch["sequence"] for ch in specs[p]["sequences"])
        seqs.update(annotation_targets(specs[p]))
    by_md5 = {seq_md5(s): s for s in seqs}
    atomic_write(Path(a.out), "".join(f">{m}\n{by_md5[m]}\n" for m in sorted(by_md5)))
    print(f"[coils] {len(by_md5)} unique sequence(s) -> {a.out}")
    return 0


def weights_main() -> int:
    ap = argparse.ArgumentParser(description="download DeepCoil's SeqVec weights (once)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    if all((out / f).is_file() for f in WEIGHT_FILES):
        print(f"[coils] SeqVec weights cached in {out}")
        return 0
    urls = deepcoil_cfg().get("seqvec_urls") or DEFAULT_URLS
    for url in urls:
        for attempt in range(1, 4):
            try:
                r = requests.get(url, timeout=1800, headers={"User-Agent": "nanopulldown/annotate_coils"})
                r.raise_for_status()
                zf = zipfile.ZipFile(io.BytesIO(r.content))
                out.mkdir(parents=True, exist_ok=True)
                for f in WEIGHT_FILES:
                    _write_bytes(out / f, zf.read(f))
                print(f"[coils] SeqVec weights from {url} -> {out}")
                return 0
            except (requests.RequestException, zipfile.BadZipFile, KeyError) as e:
                print(f"[coils] {url}: {e} (attempt {attempt}/3)", file=sys.stderr, flush=True)
                if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 404:
                    break
                time.sleep(20 * attempt)
    print("[coils] could not download the SeqVec weights", file=sys.stderr)
    return 1


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


if __name__ == "__main__":
    sys.exit(fasta_main())
