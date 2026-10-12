#!/usr/bin/env python3
"""
scripts/run_esmfold2_batch.py — Stage 2a (batched)
====================================================
Loads ESMFold2-Fast ONCE and folds a list of pairs (one Snakemake "chunk", see
esmfold2.pairs_per_load) in a loop, instead of one process -- and one ~25 GB
transient model load, with its swap spike -- per pair. Output per pair is
identical to scripts/run_esmfold2.py (raw/sample_<i>.cif.gz + raw/fold.parquet).

Crash safety: results are written to <out-root>/_fold_cache/<pair>/ first and
hard-linked into <out-root>/<pair>/raw/. Snakemake deletes a failed job's
declared outputs (the links) but not the cache, so a rerun after a crash --
e.g. the Metal allocation abort, a C++ abort no `except` can catch -- relinks
the finished pairs instantly and only folds the rest. The cache holds exactly
the final outputs and nothing else (hard links: no extra disk), so it costs
nothing to keep. Results from before the fold.parquet layout are converted by
scripts/migrate_results_layout.py, not here.

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
from fold_io import FOLD_FILE  # noqa: E402
from lib_pipeline import REPO_ROOT, load_config, load_pair  # noqa: E402
from run_esmfold2 import _load_model_and_builder, fold_pair  # noqa: E402


def _link_tree(src: Path, dst: Path) -> None:
    """Make dst hold exactly src's files (hard links): stray files in dst -- e.g.
    sample_1/2 left by an earlier 3-sample run -- are removed first."""
    dst.mkdir(parents=True, exist_ok=True)
    keep = {f.name for f in src.iterdir()}
    for f in dst.iterdir():
        if f.is_file() and f.name not in keep:
            f.unlink()
    for f in src.iterdir():
        t = dst / f.name
        t.unlink(missing_ok=True)
        try:
            os.link(f, t)
        except OSError:
            shutil.copy2(f, t)


def _complete(d: Path, n_samples: int) -> bool:
    """fold.parquet (written last, atomically) AND every sample's structure."""
    return (d / FOLD_FILE).exists() and all(
        (d / f"sample_{i}.cif.gz").exists() for i in range(n_samples))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out-root", default="results")
    a = ap.parse_args()

    root = Path(a.out_root)
    cache_root = root / "_fold_cache"

    import mlx.core as mx
    clear = getattr(mx, "clear_cache", None) or mx.metal.clear_cache
    peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    reset = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
    active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
    cfg = load_config()
    n_samples = int(cfg["esmfold2"]["num_diffusion_samples"])

    todo = [p for p in a.pairs if not _complete(cache_root / p, n_samples)]
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
            print(f"[fold_batch] {i}/{len(todo)} {p}: {time.time() - t0:.0f}s, "
                  f"peak {peak() / 2**30:.1f} GiB", flush=True)
        except Exception:                                  # noqa: BLE001
            failed.append(p)
            print(f"[fold_batch] {i}/{len(todo)} {p} FAILED:\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)
        finally:
            clear()                                        # return the fold's buffers; keep the weights

    for p in a.pairs:
        if _complete(cache_root / p, n_samples):
            _link_tree(cache_root / p, root / p / "raw")

    if failed:
        print(f"[fold_batch] {len(failed)} pair(s) failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
