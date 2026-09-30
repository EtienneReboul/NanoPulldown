#!/usr/bin/env python3
"""
scripts/fetch_uniprot.py — Stage 1a
====================================
Resolve a UniProt accession to its sequence via the UniProt REST API,
caching the raw FASTA under `uniprot.cache_dir` (config.yaml) so repeated
pairs sharing a bait/prey only hit the network once. Deliberately small and
dependency-light (stdlib `urllib` + `requests`, no `alphapulldown-input-
parser`): nanopulldown only needs "accession -> sequence", not that
package's full feature/MSA object model.

Also parses AlphaPulldown's compact protein-spec syntax
(`ID`, `ID:N` for N copies, `ID:start-stop[:start2-stop2...]` for one or more
regions, `ID:N:start-stop` for both) so bait/prey entries in
configs/baits.txt / configs/preys.txt can use the same conventions
AlphaPulldown's sample sheet does.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import load_config  # noqa: E402


@dataclass
class ProteinSpec:
    uniprot: str
    copies: int = 1
    regions: list[tuple[int, int]] | None = None


def parse_protein_spec(spec: str) -> ProteinSpec:
    """Parse AlphaPulldown-style `ID`, `ID:N`, `ID:start-stop[:start2-stop2]`,
    or `ID:N:start-stop[:start2-stop2]` into a ProteinSpec."""
    parts = spec.strip().split(":")
    uniprot = parts[0]
    rest = parts[1:]
    copies = 1
    regions: list[tuple[int, int]] | None = None

    if rest and "-" not in rest[0]:
        copies = int(rest[0])
        rest = rest[1:]

    if rest:
        regions = []
        for token in rest:
            start, stop = token.split("-")
            regions.append((int(start), int(stop)))

    return ProteinSpec(uniprot=uniprot, copies=copies, regions=regions)


def apply_regions(sequence: str, regions: list[tuple[int, int]] | None) -> str:
    """1-based, inclusive AlphaPulldown-style region slicing, concatenating
    discontinuous regions (residue numbering gaps are not preserved here --
    nanopulldown has no mmCIF author-numbering step to carry them into)."""
    if not regions:
        return sequence
    return "".join(sequence[start - 1:stop] for start, stop in regions)


def fasta_to_sequence(fasta_text: str) -> str:
    lines = [ln.strip() for ln in fasta_text.splitlines() if ln.strip()]
    return "".join(ln for ln in lines if not ln.startswith(">"))


def fetch_uniprot_fasta(accession: str, cfg: dict, cache_dir: Path) -> str:
    """Fetch (or reuse a cached) raw FASTA for one UniProt accession."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{accession}.fasta"
    if cache_path.exists():
        return cache_path.read_text()

    ucfg = cfg["uniprot"]
    url = f"{ucfg['base_url']}/{accession}.fasta"
    last_err = None
    for attempt in range(int(ucfg.get("retries", 3))):
        try:
            r = requests.get(url, timeout=float(ucfg.get("timeout_s", 30)))
            r.raise_for_status()
            cache_path.write_text(r.text)
            return r.text
        except Exception as e:                    # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"UniProt fetch failed for {accession} ({url}): {last_err}")


def fetch_sequence(spec: str, cfg: dict, cache_dir: Path | None = None) -> str:
    """Full resolve: parse spec -> fetch FASTA (cached) -> apply regions."""
    if cache_dir is None:
        cache_dir = Path(cfg["dirs"]["sequences"])
    ps = parse_protein_spec(spec)
    fasta = fetch_uniprot_fasta(ps.uniprot, cfg, cache_dir)
    return apply_regions(fasta_to_sequence(fasta), ps.regions)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="UniProt accession or AlphaPulldown-style ID:N:start-stop")
    ap.add_argument("--out", help="write the resolved sequence (FASTA) here; default: print")
    a = ap.parse_args()

    cfg = load_config()
    seq = fetch_sequence(a.spec, cfg)
    record = f">{a.spec}\n{seq}\n"
    if a.out:
        Path(a.out).write_text(record)
        print(f"[fetch_uniprot] {a.spec}: {len(seq)} aa -> {a.out}")
    else:
        print(record, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
