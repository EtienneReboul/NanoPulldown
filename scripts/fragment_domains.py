#!/usr/bin/env python3
"""
scripts/fragment_domains.py — Stage 1a (fragment mode)
=======================================================
Fragment mode cuts baits and preys into their annotated domains before
pairing, so the screen folds domain x domain instead of full-length x
full-length. This mirrors yeast two-hybrid practice, where constructs are
domain fragments. It also lets proteins that exceed `esmfold2.max_tokens`
at full length be screened through their domains.

scripts/expand_pairs.py calls protein_fragments() for every bait/prey spec
when the config's `fragments.enabled` is true. Each fragment becomes an
ordinary AlphaPulldown region spec (`ID:start-stop`, `ID:N:start-stop`), so
every later step treats it like any other region pair. Specs that already
give regions are explicit constructs and are kept as they are.

Where the domains come from, per accession:
  1. `fragments.overrides` (configs/fragment_domains.yaml, hand-edited,
     tracked): {accession: [{name, start, end}, ...]}, used as given
     (only padding is applied) -- the place for constructs from the
     literature or a curated annotation.
  2. otherwise the folded domains the reports show: merge_annotation's
     select_domains() over the full-length entry's InterPro / ScanProsite
     hits (cached per sequence, see scripts/annotate_domains.py), then
       * domains <= `merge_gap` residues apart merged into one fragment
         (tandem repeats, a domain split across two signatures)
       * blocks shorter than `min_length` dropped
       * `padding` residues added on each side, clipped to the protein
  3. no domain left: `without_domains: full_length` keeps the whole protein
     (named "full length"), `skip` drops the protein from the screen.

`include_full_length: true` also keeps each fragmented protein whole, as a
full-length control next to its fragments.

Preview before expanding (prints the fragments, writes nothing):
    NANOPULLDOWN_RUN_CONFIG=config_drm2_fragments.yaml python scripts/fragment_domains.py Q9M548 O22607
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import REPO_ROOT, load_config  # noqa: E402
from fetch_uniprot import fetch_sequence, parse_protein_spec  # noqa: E402

DEFAULTS = {"enabled": False, "sides": ["bait", "prey"], "min_length": 30, "merge_gap": 20,
            "padding": 10, "without_domains": "full_length", "include_full_length": False,
            "overrides": "configs/fragment_domains.yaml"}


@dataclass
class Fragment:
    spec: str                   # AlphaPulldown spec expand_pairs pairs, e.g. Q9M548:120-345
    accession: str
    role: str
    name: str | None            # domain name(s); None = the protein as given (no fragmenting)
    start: int
    end: int
    source: str                 # interpro | override | full_length | as_given
    domains: list[str]


def fragment_config(cfg: dict) -> dict:
    fcfg = {**DEFAULTS, **(cfg.get("fragments") or {})}
    if fcfg["without_domains"] not in ("full_length", "skip"):
        raise ValueError("fragments.without_domains must be full_length or skip, "
                         f"not {fcfg['without_domains']!r}")
    bad = set(fcfg["sides"]) - {"bait", "prey"}
    if bad:
        raise ValueError(f"fragments.sides: unknown side(s) {sorted(bad)}")
    return fcfg


def load_overrides(fcfg: dict) -> dict[str, list[dict]]:
    path = REPO_ROOT / fcfg["overrides"] if fcfg.get("overrides") else None
    if not path or not path.exists():
        return {}
    return {str(acc): list(doms or []) for acc, doms in (yaml.safe_load(path.read_text()) or {}).items()}


def annotated_domains(accession: str, sequence: str) -> list[tuple[int, int, str]]:
    """(start, end, name) folded domains of a full-length entry, as
    merge_annotation proposes them (InterPro representative set + uncovered
    InterPro domain entries; ScanProsite when InterPro has nothing)."""
    from annotate_domains import CFG, annotate_sequence, interproscan_email
    from merge_annotation import select_domains

    acfg = CFG["annotate"]
    rec = annotate_sequence(sequence, acfg, interproscan_email(acfg),
                            log=lambda m: print(m, file=sys.stderr), accession=accession)
    out = []
    for s, e, name in select_domains(rec["hits"]):
        s, e = max(1, int(s)), min(len(sequence), int(e))
        if s <= e:
            out.append((s, e, name))
    return sorted(out)


def _join_names(names: list[str]) -> str:
    """'WD40 repeat' x 4 -> 'WD40 repeat ×4'; distinct names joined by ' + '."""
    out: list[tuple[str, int]] = []
    for n in names:
        if out and out[-1][0] == n:
            out[-1] = (n, out[-1][1] + 1)
        else:
            out.append((n, 1))
    return " + ".join(n if k == 1 else f"{n} ×{k}" for n, k in out)


def merge_domains(domains: list[tuple[int, int, str]], merge_gap: int) -> list[tuple[int, int, list[str]]]:
    blocks: list[list] = []
    for s, e, name in sorted(domains):
        if blocks and s - blocks[-1][1] - 1 <= merge_gap:
            blocks[-1][1] = max(blocks[-1][1], e)
            blocks[-1][2].append(name)
        else:
            blocks.append([s, e, [name]])
    return [(s, e, names) for s, e, names in blocks]


def _fragment_spec(acc: str, copies: int, start: int, end: int, length: int) -> str:
    region = "" if (start, end) == (1, length) else f":{start}-{end}"
    return f"{acc}{f':{copies}' if copies > 1 else ''}{region}"


def protein_fragments(spec: str, role: str, cfg: dict, fcfg: dict | None = None,
                      overrides: dict | None = None) -> list[Fragment]:
    """The fragments one bait/prey spec contributes to the screen -- see the
    module docstring. [] when the protein is skipped."""
    fcfg = fcfg or fragment_config(cfg)
    overrides = load_overrides(fcfg) if overrides is None else overrides
    ps = parse_protein_spec(spec)
    seq = fetch_sequence(ps.uniprot, cfg)
    n = len(seq)
    whole = Fragment(spec, ps.uniprot, role, None, 1, n, "as_given", [])
    if role not in fcfg["sides"] or ps.regions or "-" in ps.uniprot:
        # side not fragmented / explicit construct / isoform (no InterPro by accession)
        return [whole]

    pad = int(fcfg["padding"])
    if ps.uniprot in overrides:
        source = "override"
        blocks = [(int(d["start"]), int(d["end"]), [str(d.get("name") or f"{d['start']}-{d['end']}")])
                  for d in overrides[ps.uniprot]]
    else:
        source = "interpro"
        blocks = [b for b in merge_domains(annotated_domains(ps.uniprot, seq), int(fcfg["merge_gap"]))
                  if b[1] - b[0] + 1 >= int(fcfg["min_length"])]

    frags = []
    for s, e, names in blocks:
        s, e = max(1, s - pad), min(n, e + pad)
        frags.append(Fragment(_fragment_spec(ps.uniprot, ps.copies, s, e, n), ps.uniprot, role,
                              _join_names(names), s, e, source, names))
    if not frags:
        if fcfg["without_domains"] == "skip":
            return []
        return [Fragment(spec, ps.uniprot, role, "full length", 1, n, "full_length", [])]
    if fcfg["include_full_length"] and all(f.spec != spec for f in frags):
        frags.insert(0, whole)
    # two blocks padded into the same span (or an override listed twice) -> one fragment
    uniq: dict[str, Fragment] = {}
    for f in frags:
        uniq.setdefault(f.spec, f)
    return list(uniq.values())


FRAGMENT_COLUMNS = {
    "accession": "UniProt accession",
    "role": "bait | prey",
    "spec": "AlphaPulldown spec the fragment is paired as (ID[:N][:start-stop])",
    "fragment": "Domain name(s) of the fragment; null = the protein as given, not fragmented",
    "start": "First residue, 1-based, full-length numbering (after padding)",
    "end": "Last residue, inclusive, full-length numbering (after padding)",
    "length": "Fragment length (residues)",
    "source": "interpro (annotated domains) | override (configs/fragment_domains.yaml) | "
              "full_length (no domain found) | as_given (side not fragmented, explicit regions, "
              "or include_full_length control)",
    "domains": "Names of the domains merged into the fragment",
}


def write_fragments_table(frags: list[Fragment], path: Path) -> Path:
    import pandas as pd

    from parquet_utils import write_parquet_with_metadata

    df = pd.DataFrame([{"accession": f.accession, "role": f.role, "spec": f.spec, "fragment": f.name,
                        "start": f.start, "end": f.end, "length": f.end - f.start + 1,
                        "source": f.source, "domains": f.domains} for f in frags],
                      columns=list(FRAGMENT_COLUMNS))
    write_parquet_with_metadata(
        df, path,
        table_description="nanopulldown fragment mode: one row per bait/prey fragment the run "
                          "pairs (scripts/fragment_domains.py). Written by scripts/expand_pairs.py.",
        column_descriptions=FRAGMENT_COLUMNS)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="preview the fragments of some proteins (writes nothing)")
    ap.add_argument("specs", nargs="+", help="protein specs (ID, ID:N, ...)")
    ap.add_argument("--role", default="prey", choices=["bait", "prey"])
    a = ap.parse_args()
    cfg = load_config()
    fcfg = fragment_config(cfg)
    if a.role not in fcfg["sides"]:
        print(f"[fragment_domains] note: fragments.sides = {fcfg['sides']} -- {a.role}s are not cut")
    overrides = load_overrides(fcfg)
    for spec in a.specs:
        frags = protein_fragments(spec, a.role, cfg, fcfg, overrides)
        print(f"{spec}: {len(frags)} fragment(s)" + ("" if frags else " -- skipped (no domain)"))
        for f in frags:
            print(f"  {f.spec:<22} {f.end - f.start + 1:>5} aa  {f.source:<11}  {f.name or '(as given)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
