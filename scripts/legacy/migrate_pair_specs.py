#!/usr/bin/env python3
"""
scripts/legacy/migrate_pair_specs.py — ONE-OFF
=========================================
Folds the per-pair configs/<bait>__<prey>.yaml specs into configs/pairs.parquet
(+ configs/curated_domains.yaml for any pair with `annotation_reviewed: true`
or a non-empty `domains:` block). Pairs already in pairs.parquet are kept;
a YAML spec for the same pair replaces the table row.

DRY RUN by default; --apply writes the table, checks that load_pair() returns
exactly the old spec for EVERY pair, and only then deletes the YAML files.

Env: any with pyyaml + pyarrow (e.g. the controller env).
Usage:
  python scripts/legacy/migrate_pair_specs.py            # dry run
  python scripts/legacy/migrate_pair_specs.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # standalone: find the package
from nanopulldown.core.config import REPO_ROOT
from nanopulldown.core.specs import CURATED_DOMAINS, PAIRS_TABLE, load_pairs, write_pairs


def _normalised(spec: dict) -> dict:
    """The old load_pair()'s view of a YAML spec."""
    spec = dict(spec)
    spec["bait_chains"] = list(spec.get("bait_chains", []))
    spec["prey_chains"] = list(spec.get("prey_chains", []))
    spec["annotation_reviewed"] = bool(spec.get("annotation_reviewed", False))
    spec["domains"] = spec.get("domains") or {}
    for role in ("bait", "prey"):
        r = (spec.get(role) or {}).get("regions")
        if r is not None:
            spec[role] = {**spec[role], "regions": [[int(a), int(b)] for a, b in r]}
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO_ROOT))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    root = Path(a.repo)
    files = sorted((root / "configs").glob("*__*.yaml"))
    old = {}
    for f in files:
        spec = _normalised(yaml.safe_load(f.read_text()))
        if spec.get("name") != f.stem:
            print(f"[migrate_specs] {f.name}: name {spec.get('name')!r} != file stem -- skipped",
                  file=sys.stderr)
            continue
        old[f.stem] = spec

    specs = load_pairs(root)              # whatever is already in the table
    specs.update(old)
    curated = (yaml.safe_load((root / CURATED_DOMAINS).read_text()) or {}) \
        if (root / CURATED_DOMAINS).exists() else {}
    for name, spec in old.items():
        if spec["annotation_reviewed"] or spec["domains"]:
            curated[name] = {"annotation_reviewed": spec["annotation_reviewed"],
                             "domains": spec["domains"]}

    print(f"[migrate_specs] {len(files)} YAML spec(s) -> {root / PAIRS_TABLE} "
          f"({len(specs)} pair(s) total), {len(curated)} curated -> {root / CURATED_DOMAINS}")
    if not a.apply:
        print("[migrate_specs] dry run only -- rerun with --apply")
        return 0

    write_pairs(list(specs.values()), root)
    if curated:
        (root / CURATED_DOMAINS).write_text(
            "# Hand-curated domain maps, merged into a pair's spec by core.specs.load_pair().\n"
            "# {pair: {annotation_reviewed: bool, domains: {chain: [{name, start, end, kind}]}}}\n"
            "# -- see configs/_schema.md. Start from data/annotation/<pair>/annotation.yaml.\n"
            + yaml.safe_dump(dict(sorted(curated.items())), sort_keys=False, width=100))

    new = load_pairs(root)
    bad = [n for n, spec in old.items() if new.get(n) != spec]
    if bad:
        print(f"[migrate_specs] VERIFY FAILED for {len(bad)} pair(s), e.g. {bad[:3]} -- "
              "YAML files kept", file=sys.stderr)
        return 1
    for f in files:
        if f.stem in old:
            f.unlink()
    size = (root / PAIRS_TABLE).stat().st_size
    print(f"[migrate_specs] verified {len(old)} pair(s) round-trip exactly; deleted their YAML; "
          f"pairs.parquet = {size / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
