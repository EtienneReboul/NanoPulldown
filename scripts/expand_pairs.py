#!/usr/bin/env python3
"""
scripts/expand_pairs.py — Stage 1a
====================================
AlphaPulldown's bait-file x prey-file combinatorial screen, adapted:
reads `configs/baits.txt` and `configs/preys.txt` (one protein spec per
line, AlphaPulldown syntax: `ID`, `ID:N`, `ID:start-stop`, `ID:N:start-stop`,
blank lines and `#` comments ignored), fetches every unique accession via
scripts/fetch_uniprot.py, and writes one `configs/<bait>__<prey>.yaml` stub
per combination (skipping any that already exist, so hand-curated `domains:`
blocks are never clobbered by a re-run).

Pairs whose total token count (sum of all chain lengths) exceeds
`--max-tokens` (default: config's `esmfold2.max_tokens`, 0 = no limit) are NOT
written: ESMFold2 memory grows ~quadratically with tokens, and past the Metal
buffer cap the fold aborts the whole process. They are listed, with their
token counts, in `configs/skipped_too_long.tsv` instead. Measure your
machine's limit with notebooks/esmfold2_memory_model.ipynb.

Usage:
    python scripts/expand_pairs.py --baits configs/baits.txt --preys configs/preys.txt
    python scripts/expand_pairs.py --max-tokens 1500    # override the config limit
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import REPO_ROOT, load_config, pair_tokens  # noqa: E402
from fetch_uniprot import fetch_sequence, parse_protein_spec  # noqa: E402


def read_specs(path: Path) -> list[str]:
    if not path.exists():
        return []
    specs = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            specs.append(line)
    return specs


def slug(spec: str) -> str:
    """Filesystem/YAML-key-safe stem for a protein spec, e.g.
    'Q8I2G6:2:1-100' -> 'Q8I2G6x2x1-100'."""
    return re.sub(r"[^A-Za-z0-9_.-]", "x", spec)


def build_pair_spec(bait_spec: str, prey_spec: str, cfg: dict) -> dict:
    cache_dir = Path(cfg["dirs"]["sequences"])
    bait_ps = parse_protein_spec(bait_spec)
    prey_ps = parse_protein_spec(prey_spec)
    bait_seq = fetch_sequence(bait_spec, cfg, cache_dir)
    prey_seq = fetch_sequence(prey_spec, cfg, cache_dir)

    sequences = []
    chain_ids = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    bait_chains, prey_chains = [], []
    for role, ps, seq, prefix in (
        ("bait", bait_ps, bait_seq, "bait"),
        ("prey", prey_ps, prey_seq, "prey"),
    ):
        for copy_i in range(ps.copies):
            cid = next(chain_ids)
            name = f"{prefix}_{ps.uniprot}" + (f"_{copy_i + 1}" if ps.copies > 1 else "")
            sequences.append({"id": cid, "name": name, "sequence": seq, "role": role})
            (bait_chains if role == "bait" else prey_chains).append(cid)

    return {
        "name": f"{slug(bait_spec)}__{slug(prey_spec)}",
        "bait": {"uniprot": bait_ps.uniprot, "copies": bait_ps.copies,
                 "regions": bait_ps.regions},
        "prey": {"uniprot": prey_ps.uniprot, "copies": prey_ps.copies,
                 "regions": prey_ps.regions},
        "sequences": sequences,
        "bait_chains": bait_chains,
        "prey_chains": prey_chains,
        "annotation_reviewed": False,
        "domains": {},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baits", default="configs/baits.txt")
    ap.add_argument("--preys", default="configs/preys.txt")
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="skip pairs with more total tokens than this "
                         "(default: config esmfold2.max_tokens; 0 = no limit)")
    a = ap.parse_args()

    cfg = load_config(REPO_ROOT)
    max_tokens = a.max_tokens if a.max_tokens is not None else int(cfg["esmfold2"].get("max_tokens", 0))
    baits = read_specs(Path(a.baits))
    preys = read_specs(Path(a.preys))
    if not baits or not preys:
        print(f"[expand_pairs] nothing to expand ({len(baits)} bait(s), {len(preys)} prey(s))")
        return 0

    out_dir = Path(a.configs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped, too_long = 0, 0, []
    for bait_spec in baits:
        for prey_spec in preys:
            pair = build_pair_spec(bait_spec, prey_spec, cfg)
            n_tok = pair_tokens(pair)
            if max_tokens and n_tok > max_tokens:
                too_long.append((pair["name"], n_tok))
                continue
            out_path = out_dir / f"{pair['name']}.yaml"
            if out_path.exists():
                skipped += 1
                continue
            out_path.write_text(yaml.safe_dump(pair, sort_keys=False))
            written += 1
            print(f"[expand_pairs] wrote {out_path}")
    skip_tsv = out_dir / "skipped_too_long.tsv"
    if too_long:
        skip_tsv.write_text("pair\ttokens\n" + "".join(f"{n}\t{t}\n" for n, t in too_long))
    elif skip_tsv.exists():
        skip_tsv.unlink()
    if too_long:
        print(f"[expand_pairs] {len(too_long)} pair(s) over max_tokens={max_tokens} NOT written "
              f"-> {skip_tsv}")
    print(f"[expand_pairs] {written} new pair(s), {skipped} already existed "
          f"({len(baits)} bait(s) x {len(preys)} prey(s) = {len(baits) * len(preys)} combination(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
