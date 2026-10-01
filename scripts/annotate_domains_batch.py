#!/usr/bin/env python3
"""
scripts/annotate_domains_batch.py — Stage 1b (batch)
=====================================================
Fills the per-sequence domain cache (data/annotation/_cache/domains/) for
every UNIQUE sequence across all pair specs, with a small thread pool for the
web calls (ScanProsite + InterPro API by UniProt accession). The per-pair `annotate_domains` rule
then assembles its TSV from the cache with no network access.

Incremental: cached sequences are skipped, so a crash / rate-limit stall
resumes where it stopped. Pool size: annotate.web_workers (keep it polite --
EBI and ExPASy throttle).

Usage: python scripts/annotate_domains_batch.py --specs configs/*.yaml --done <sentinel>
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import annotation_cache_dir, seq_md5  # noqa: E402
from annotate_domains import (CFG, STOP, annotate_sequence, interproscan_email,  # noqa: E402
                              seq_accessions)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--done", required=True)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    acfg = CFG["annotate"]
    workers = a.workers or int(acfg.get("web_workers", 4))
    email = interproscan_email(acfg)

    seqs: set[str] = set()
    accs: dict[str, str] = {}
    for f in a.specs:
        spec = yaml.safe_load(Path(f).read_text())
        accs.update(seq_accessions(spec))
        for ch in spec["sequences"]:
            seqs.add(ch["sequence"])
    cache = annotation_cache_dir(CFG, "domains")
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


if __name__ == "__main__":
    sys.exit(main())
