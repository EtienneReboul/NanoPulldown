"""
nanopulldown.annotate.batch — Stage 1b (batch)
===============================================
Fills the per-sequence annotation caches for every UNIQUE sequence across the
given pairs; the per-pair `annotate merge` then assembles from the cache.

`annotate domains` fills the per-sequence domain cache (data/annotation/_cache/domains_v2/) for
every UNIQUE sequence across all pair specs, with a small thread pool for the
web calls (ScanProsite + InterPro API by UniProt accession). The per-pair `annotate_domains` rule
then assembles its TSV from the cache with no network access.

Incremental: cached sequences are skipped, so a crash / rate-limit stall
resumes where it stopped. Pool size: annotate.web_workers (keep it polite --
EBI and ExPASy throttle).

`annotate disorder` fills the per-sequence AIUPred cache
(data/annotation/_cache/disorder/) using a few multi-FASTA container
invocations instead of one `docker run` per chain. CPU only: AIUPred's
networks are tiny (measured 0.12 s/sequence natively, 0.26 s emulated in a
container), so a GPU buys nothing here and stays free for ESMFold2. Sequences
are sent in chunks (annotate.aiupred.batch_size, default 200) so a failure
costs one chunk, not the run. Incremental: cached sequences are skipped.

Usage:
  nanopulldown annotate domains  --pairs <pair> [<pair> ...] --done <sentinel>
  nanopulldown annotate disorder --pairs <pair> [<pair> ...] --done <sentinel>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nanopulldown.annotate.disorder import cache_is_current, run_aiupred_batch
from nanopulldown.annotate.domains import (
    CFG,
    DOMAIN_CACHE,
    STOP,
    annotate_sequence,
    annotation_targets,
    interproscan_email,
)
from nanopulldown.core.cache import annotation_cache_dir
from nanopulldown.core.io import atomic_write, seq_md5
from nanopulldown.core.specs import load_pairs


def domains_main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--done", required=True)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    acfg = CFG["annotate"]
    workers = a.workers or int(acfg.get("web_workers", 4))
    email = interproscan_email(acfg)

    # region chains are annotated through their full-length entry
    # (annotate_domains.region_source), so that is what gets cached
    seqs: set[str] = set()
    accs: dict[str, str] = {}
    specs = load_pairs()
    for p in a.pairs:
        for seq, acc in annotation_targets(specs[p]).items():
            seqs.add(seq)
            if acc:
                accs[seq] = acc
    cache = annotation_cache_dir(CFG, DOMAIN_CACHE)
    todo = sorted((s for s in seqs if not (cache / f"{seq_md5(s)}.json").exists()), key=len)
    print(f"[domains_batch] {len(seqs)} unique sequence(s), {len(todo)} to annotate, "
          f"{workers} worker(s)", flush=True)

    t0, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(annotate_sequence, s, acfg, email,
                            lambda m: print(m, file=sys.stderr, flush=True),
                            accs.get(s)): s for s in todo}
        for fut in as_completed(futs):
            done += 1
            try:
                fut.result()
            except Exception as e:                   # noqa: BLE001
                failed += 1
                print(f"[domains_batch] {seq_md5(futs[fut])}: {e}", file=sys.stderr, flush=True)
            if done % 25 == 0 or done == len(todo):
                print(f"[domains_batch] {done}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    if STOP.is_set():
        print("[domains_batch] ABORTED: the web services kept throttling (circuit breaker). Progress is "
              "cached; wait a while (or lower annotate.rate_limits.*) and rerun.", file=sys.stderr)
        return 2

    missing = sum(not (cache / f"{seq_md5(s)}.json").exists() for s in seqs)
    if missing:
        # not fatal: the per-pair rule falls back to a live lookup for these
        print(f"[domains_batch] WARNING: {missing} sequence(s) not cached (service errors); "
              "per-pair jobs will retry them live", file=sys.stderr)
    Path(a.done).parent.mkdir(parents=True, exist_ok=True)
    Path(a.done).write_text(f"{len(seqs) - missing}/{len(seqs)} cached\n")
    return 0



def disorder_main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--done", required=True)
    a = ap.parse_args()

    acfg = CFG["annotate"]["aiupred"]
    chunk = int(acfg.get("batch_size", 200))
    seqs: set[str] = set()
    specs = load_pairs()
    for p in a.pairs:
        for ch in specs[p]["sequences"]:
            seqs.add(ch["sequence"])
    cache = annotation_cache_dir(CFG, "disorder")
    todo = sorted((s for s in seqs if not cache_is_current(cache / f"{seq_md5(s)}.json")), key=len)
    print(f"[disorder_batch] {len(seqs)} unique sequence(s), {len(todo)} to run, chunks of {chunk}",
          flush=True)

    t0 = time.time()
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        for md5, tracks in run_aiupred_batch(part, acfg).items():
            atomic_write(cache / f"{md5}.json", json.dumps(tracks))
        print(f"[disorder_batch] {min(i + chunk, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s)",
              flush=True)

    missing = sum(not cache_is_current(cache / f"{seq_md5(s)}.json") for s in seqs)
    if missing:
        print(f"[disorder_batch] WARNING: {missing} sequence(s) not cached; per-pair jobs will "
              "fall back to single-sequence runs", file=sys.stderr)
    Path(a.done).parent.mkdir(parents=True, exist_ok=True)
    Path(a.done).write_text(f"{len(seqs) - missing}/{len(seqs)} cached\n")
    return 0
