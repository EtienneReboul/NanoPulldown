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

Container contract (confirmed against ghcr.io/doszilab/aiupred:cpu v2.1.0,
2026-09-30 -- the image sets no ENTRYPOINT/CMD, so it must be invoked as
`python3 -m aiupred.cli`):
  python3 -m aiupred.cli -i <in.fasta> -o <out.tsv> -b [--force-cpu]
`-b/--binding` adds a 4th column to the SAME output file rather than
needing a second run -- one docker invocation per chain covers both
tracks. Output: `#`-prefixed header/citation lines, then one row per
residue: `resi<TAB>restype<TAB>disorder_score[<TAB>binding_score]`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import (annotation_cache_dir, atomic_write, load_config, load_pair,  # noqa: E402
                          protein_chains, seq_md5)

CFG = load_config()


def _run_aiupred_container(seq: str, cfg: dict) -> dict[str, list[float]] | None:
    image = cfg["image_gpu"] if cfg.get("use_gpu") else cfg["image_cpu"]
    platform_args = ["--platform", cfg["docker_platform"]] if cfg.get("docker_platform") else []
    gpu_args = ["--gpus", "all"] if cfg.get("use_gpu") else []
    cpu_args = [] if cfg.get("use_gpu") else ["--force-cpu"]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fasta_path = td / "input.fasta"
        out_path = td / "aiupred.tsv"
        fasta_path.write_text(f">query\n{seq}\n")

        cmd = [
            "docker", "run", "--rm", *platform_args, *gpu_args,
            "-v", f"{td}:/work",
            "--entrypoint", "python3",
            image,
            "-m", "aiupred.cli",
            "-i", "/work/input.fasta", "-o", "/work/aiupred.tsv",
            "-b", *cpu_args,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[disorder] AIUPred docker invocation failed: {e}", file=sys.stderr)
            return None
        if proc.returncode != 0:
            print(f"[disorder] AIUPred exited {proc.returncode}: "
                  f"{proc.stderr[:300]}", file=sys.stderr)
            return None
        if not out_path.exists():
            print("[disorder] AIUPred produced no output file", file=sys.stderr)
            return None
        return _parse_aiupred_output(out_path.read_text(), len(seq))


def _parse_aiupred_output(text: str, expected_len: int) -> dict[str, list[float]] | None:
    disorder: list[float] = []
    binding: list[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            disorder.append(float(parts[2]))
            if len(parts) >= 4:
                binding.append(float(parts[3]))
        except (ValueError, IndexError):
            continue
    if len(disorder) != expected_len:
        print(f"[disorder] AIUPred disorder length {len(disorder)} != sequence length "
              f"{expected_len}; keeping what parsed", file=sys.stderr)
    tracks = {}
    if disorder:
        tracks["disorder"] = disorder
    if binding:
        tracks["binding"] = binding
    return tracks or None


def run_aiupred_batch(seqs: list[str], cfg: dict, timeout: int = 3600) -> dict[str, dict[str, list[float]]]:
    """ONE container invocation over many sequences (the CLI takes multi-FASTA
    and loads the networks once). Records are keyed by md5(sequence); returns
    {md5: tracks} for the sequences that parsed. This is the efficient path:
    per-chain `docker run` costs ~2.6 s each on Apple Silicon (amd64 image,
    emulated) vs ~0.26 s/sequence inside a single container."""
    image = cfg["image_gpu"] if cfg.get("use_gpu") else cfg["image_cpu"]
    platform_args = ["--platform", cfg["docker_platform"]] if cfg.get("docker_platform") else []
    gpu_args = ["--gpus", "all"] if cfg.get("use_gpu") else []
    cpu_args = [] if cfg.get("use_gpu") else ["--force-cpu"]
    by_md5 = {seq_md5(q): q for q in seqs}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "input.fasta").write_text("".join(f">{m}\n{q}\n" for m, q in by_md5.items()))
        cmd = ["docker", "run", "--rm", *platform_args, *gpu_args, "-v", f"{td}:/work",
               "--entrypoint", "python3", image, "-m", "aiupred.cli",
               "-i", "/work/input.fasta", "-o", "/work/aiupred.tsv", "-b", *cpu_args]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[disorder] AIUPred batch invocation failed: {e}", file=sys.stderr)
            return {}
        out = td / "aiupred.tsv"
        if proc.returncode != 0 or not out.exists():
            print(f"[disorder] AIUPred batch exited {proc.returncode}: {proc.stderr[:300]}",
                  file=sys.stderr)
            return {}
        # records: "#>md5" line, then residue rows, blank line between records
        results, cur, buf = {}, None, []
        for line in out.read_text().splitlines() + ["#>"]:
            if line.startswith("#>"):
                if cur in by_md5:
                    tracks = _parse_aiupred_output("\n".join(buf), len(by_md5[cur]))
                    if tracks:
                        results[cur] = tracks
                cur, buf = line[2:].strip(), []
            elif cur is not None:
                buf.append(line)
        return results


def get_tracks(seq: str, acfg: dict) -> dict[str, list[float]]:
    """Per-sequence tracks, cached on disk by md5(sequence); falls back to a
    single-sequence container run on a miss (and caches that)."""
    cache = annotation_cache_dir(CFG, "disorder") / f"{seq_md5(seq)}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    tracks = _run_aiupred_container(seq, acfg) or {}
    if tracks:
        atomic_write(cache, json.dumps(tracks))
    return tracks


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
        tracks = get_tracks(seq, acfg)
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
