#!/usr/bin/env python3
"""
scripts/annotate_disorder_aiupred.py — Stage 1b
=================================================
Per-residue disorder AND MoRF/binding-propensity tracks for every protein
chain, via AIUPred's docker image
(ghcr.io/doszilab/aiupred:cpu on this Mac -- no CUDA, so :gpu is not usable
locally; left configurable for a future CUDA host).

Replaces ab_initio_pipeline's three-way metapredict/IUPred3+ANCHOR2/AIUPred
-optional split AND its separate MoRFchibi2 step (scripts/annotate_morf.py,
tools/MC2) with a single tool run twice: AIUPred's "disorder" mode and its
"binding" mode (its ANCHOR2-like MoRF/binding-region predictor). "linker" is
never a raw predictor output here (or in ab_initio_pipeline) -- it stays a
human call made in configs/<pair>.yaml's `domains:` during curation.

Output: long-format TSV  chain  resi  restype  track  score
        track is one of: disorder, binding

*** IMPLEMENTATION NOTE ***
This script assumes AIUPred's container exposes a CLI of the shape
`aiupred -i <fasta> -t {disorder,binding} -o <out>` writing a whitespace/TSV
table whose last column is the per-residue score, one row per residue in
input order (mirroring the real AIUPred2 CLI's documented --type flag).
Confirm the actual contract with
`docker run --rm ghcr.io/doszilab/aiupred:cpu --help` before relying on this
in production and adjust `_run_aiupred_container()` / `_parse_aiupred_output()`
if it differs -- the rest of the pipeline (merge_annotation.py, the domain
schema, the report) only depends on this script's own TSV output contract
above, not on AIUPred's exact CLI.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import load_config, load_pair, protein_chains  # noqa: E402

CFG = load_config()


def _run_aiupred_container(seq: str, track: str, cfg: dict) -> list[float] | None:
    image = cfg["image_gpu"] if cfg.get("use_gpu") else cfg["image_cpu"]
    platform_args = ["--platform", cfg["docker_platform"]] if cfg.get("docker_platform") else []
    gpu_args = ["--gpus", "all"] if cfg.get("use_gpu") else []

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fasta_path = td / "input.fasta"
        out_path = td / f"{track}.tsv"
        fasta_path.write_text(f">query\n{seq}\n")

        cmd = [
            "docker", "run", "--rm", *platform_args, *gpu_args,
            "-v", f"{td}:/work",
            image,
            "-i", "/work/input.fasta", "-t", track, "-o", f"/work/{track}.tsv",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[disorder] AIUPred ({track}) docker invocation failed: {e}", file=sys.stderr)
            return None
        if proc.returncode != 0:
            print(f"[disorder] AIUPred ({track}) exited {proc.returncode}: "
                  f"{proc.stderr[:300]}", file=sys.stderr)
            return None
        if not out_path.exists():
            print(f"[disorder] AIUPred ({track}) produced no output file", file=sys.stderr)
            return None
        return _parse_aiupred_output(out_path.read_text(), len(seq))


def _parse_aiupred_output(text: str, expected_len: int) -> list[float] | None:
    scores: list[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            scores.append(float(parts[-1]))
        except (ValueError, IndexError):
            continue
    if len(scores) != expected_len:
        print(f"[disorder] AIUPred output length {len(scores)} != sequence length "
              f"{expected_len}; keeping what parsed", file=sys.stderr)
    return scores or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = load_pair(args.pair, Path(args.spec).resolve().parent.parent)
    acfg = CFG["annotate"]["aiupred"]
    rows: list[tuple] = []

    for ch in protein_chains(spec):
        cid, seq = ch["id"], ch["sequence"]
        print(f"[disorder] chain {cid} ({len(seq)} aa)", flush=True)
        tracks: dict[str, list[float]] = {}
        for track in ("disorder", "binding"):
            vals = _run_aiupred_container(seq, track, acfg)
            if vals:
                tracks[track] = vals
        for i, aa in enumerate(seq, start=1):
            for tname, vals in tracks.items():
                if i - 1 < len(vals):
                    rows.append((cid, i, aa, tname, round(float(vals[i - 1]), 5)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("chain\tresi\trestype\ttrack\tscore\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    seen = sorted({r[3] for r in rows})
    print(f"[disorder] wrote {len(rows)} rows, tracks={seen} -> {out}")
    if not rows:
        print("[disorder] WARNING: no tracks produced — is the AIUPred docker "
              "image reachable (docker pull ghcr.io/doszilab/aiupred:cpu)?", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
