#!/usr/bin/env python3
"""
scripts/run_esmfold2_batch.py — Stage 2a (batched)
====================================================
Loads ESMFold2-Fast ONCE and folds a list of pairs (one Snakemake "chunk", see
esmfold2.pairs_per_load) in a loop, instead of one process -- and one ~25 GB
transient model load, with its swap spike -- per pair. Output per pair is
identical to scripts/run_esmfold2.py (raw/sample_<i>.cif, *_confidences.json,
prediction.done).

ONE-TIME MIGRATION: Snakemake deletes the declared outputs of any job it is about
to (re)run BEFORE the job starts, so per-pair results from earlier runs (raw/
only, no cache yet) would be wiped before this script could adopt them. Run
`python scripts/run_esmfold2_batch.py --adopt-only --pairs $(pairs...)` once,
before the first chunked Snakemake run, to link them into the cache first.

Crash safety: results are written to <out-root>/_fold_cache/<pair>/ first and
hard-linked into <out-root>/<pair>/raw/. Snakemake deletes a failed job's
declared outputs (the links) but not the cache, so a rerun after a crash --
e.g. the Metal allocation abort, a C++ abort no `except` can catch -- relinks
the finished pairs instantly and only folds the rest. Pairs whose raw/ output
already exists (earlier per-pair runs) are adopted, not refolded.

Same one-model-at-a-time rule as run_esmfold2.py: the Snakefile gives each
chunk gpu=1 and the whole docker_heavy pool.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import REPO_ROOT, load_config, load_pair  # noqa: E402
from run_esmfold2 import _load_model_and_builder, fold_pair  # noqa: E402


def _link_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        t = dst / f.name
        t.unlink(missing_ok=True)
        try:
            os.link(f, t)
        except OSError:
            shutil.copy2(f, t)


def _complete(d: Path) -> bool:
    return (d / "prediction.done").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--adopt-only", action="store_true",
                    help="only link finished raw/ outputs into the cache, then exit (no model)")
    a = ap.parse_args()

    root = Path(a.out_root)
    cache_root = root / "_fold_cache"

    # adopt finished per-pair outputs from earlier runs into the cache
    adopted = 0
    for p in a.pairs:
        raw, cache = root / p / "raw", cache_root / p
        if not _complete(cache) and _complete(raw):
            _link_tree(raw, cache)
            adopted += 1
    if a.adopt_only:
        print(f"[fold_batch] adopted {adopted} finished pair(s) into {cache_root}")
        return 0

    import mlx.core as mx
    clear = getattr(mx, "clear_cache", None) or mx.metal.clear_cache
    peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    reset = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
    active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
    cfg = load_config()

    todo = [p for p in a.pairs if not _complete(cache_root / p)]
    print(f"[fold_batch] {len(a.pairs)} pair(s) in chunk, {len(todo)} to fold, "
          f"{len(a.pairs) - len(todo)} already done", flush=True)

    loaded, failed = None, []
    if todo:
        t0 = time.time()
        loaded = _load_model_and_builder(cfg["esmfold2"]["checkpoint"])
        print(f"[fold_batch] model loaded once in {time.time() - t0:.0f}s, "
              f"{active() / 2**30:.1f} GiB resident", flush=True)

    for i, p in enumerate(todo, 1):
        cache = cache_root / p
        shutil.rmtree(cache, ignore_errors=True)
        t0 = time.time()
        reset()
        try:
            fold_pair(load_pair(p, REPO_ROOT), cfg, cache, loaded)
            (cache / "prediction.done").write_text("ok\n")
            print(f"[fold_batch] {i}/{len(todo)} {p}: {time.time() - t0:.0f}s, "
                  f"peak {peak() / 2**30:.1f} GiB", flush=True)
        except Exception:                                  # noqa: BLE001
            failed.append(p)
            print(f"[fold_batch] {i}/{len(todo)} {p} FAILED:\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)
        finally:
            clear()                                        # return the fold's buffers; keep the weights

    for p in a.pairs:
        if _complete(cache_root / p):
            _link_tree(cache_root / p, root / p / "raw")

    if failed:
        print(f"[fold_batch] {len(failed)} pair(s) failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
