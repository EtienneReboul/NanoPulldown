#!/usr/bin/env python3
"""
scripts/annotate_disorder_batch.py — Stage 1b (batch)
======================================================
Fills the per-sequence AIUPred cache (data/annotation/_cache/disorder/) for
every UNIQUE sequence across all pair specs using a few multi-FASTA container
invocations instead of one `docker run` per chain. CPU only: AIUPred's
networks are tiny (measured 0.12 s/sequence natively, 0.26 s emulated in a
container), so a GPU buys nothing here and stays free for ESMFold2.

Sequences are sent in chunks (annotate.aiupred.batch_size, default 200) so a
failure costs one chunk, not the run. Incremental: cached sequences are skipped.

Usage: python scripts/annotate_disorder_batch.py --specs configs/*.yaml --done <sentinel>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import annotation_cache_dir, atomic_write, seq_md5  # noqa: E402
from annotate_disorder_aiupred import CFG, run_aiupred_batch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--done", required=True)
    a = ap.parse_args()

    acfg = CFG["annotate"]["aiupred"]
    chunk = int(acfg.get("batch_size", 200))
    seqs: set[str] = set()
    for f in a.specs:
        for ch in yaml.safe_load(Path(f).read_text())["sequences"]:
            seqs.add(ch["sequence"])
    cache = annotation_cache_dir(CFG, "disorder")
    todo = sorted((s for s in seqs if not (cache / f"{seq_md5(s)}.json").exists()), key=len)
    print(f"[disorder_batch] {len(seqs)} unique sequence(s), {len(todo)} to run, chunks of {chunk}",
          flush=True)

    t0 = time.time()
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        for md5, tracks in run_aiupred_batch(part, acfg).items():
            atomic_write(cache / f"{md5}.json", json.dumps(tracks))
        print(f"[disorder_batch] {min(i + chunk, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s)",
              flush=True)

    missing = sum(not (cache / f"{seq_md5(s)}.json").exists() for s in seqs)
    if missing:
        print(f"[disorder_batch] WARNING: {missing} sequence(s) not cached; per-pair jobs will "
              "fall back to single-sequence runs", file=sys.stderr)
    Path(a.done).parent.mkdir(parents=True, exist_ok=True)
    Path(a.done).write_text(f"{len(seqs) - missing}/{len(seqs)} cached\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
