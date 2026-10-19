#!/usr/bin/env python3
"""
scripts/controls.py — Stage 1a (positive + negative controls)
===============================================================
Every run's screen gets two control sets added to its preys by
scripts/expand_pairs.py (config.yaml `controls:`; a run overlay can turn
either off):

* positives (`controls.positives`, source tag `known`): the baits' known
  DIRECT partners in the PPI databases (scripts/fetch_interactors.py: UniProt,
  IntAct, MINT, BioGRID, STRING, Complex Portal). Direct = at least
  `min_direct_publications` publications with a PSI-MI "direct interaction"
  (or descendant) or two-hybrid (or descendant) evidence row; co-IP / affinity
  capture-MS, complex co-membership and STRING alone are not enough.
  `direct_only: false` takes every partner the databases report (with
  `min_databases`), as the former `interactors:` input source did.
  Positives are cut like any prey in fragment mode. A screen prey that is a
  known partner (direct or not) gets the `known_*` columns too; only the
  direct ones get the `known` tag.
* negatives (`controls.decoys`, source tag `decoy`): the matched decoy set of
  scripts/build_decoys.py, drawn after the screen's preys are cut.

Both are paired with every bait. scripts/decoy_calibration.py (Snakefile rule
decoy_calibration) scores the screen against the decoys and reports how many
positives it recovers.

One interactor snapshot per run serves both: data/interactors/interactors_<run>.tsv
(every partner of the baits, every evidence row), from which the positives
are the direct ones and the decoy exclusions are all of them.

Columns carried into configs/inputs_<run>.parquet / the results table
(prefix = `controls.positives.name`, default `known`):
  known_databases, known_n_databases, known_string_score,
  known_direct_databases, known_direct_methods, known_n_direct_publications,
  known_direct_of (the baits a partner has direct evidence with: it is a
  positive control for those baits only, scripts/decoy_calibration.py)

Preview the positives of a run (fetches, writes only the snapshot):
    NANOPULLDOWN_RUN_CONFIG=config_drm2_ppi.yaml python scripts/controls.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_interactors import DEFAULT_DATABASES, direct_terms, fetch_interactions, summarize  # noqa: E402
from fetch_uniprot import fetch_metadata, parse_protein_spec  # noqa: E402

POSITIVE_DEFAULTS = {
    "enabled": True,
    "name": "known",                # input-source tag in the inputs / results tables
    "databases": DEFAULT_DATABASES,
    "direct_only": True,
    "direct_types": ["MI:0407"],    # direct interaction (+ descendants)
    "direct_methods": ["MI:0018"],  # two hybrid (+ descendants)
    "min_direct_publications": 1,
    "min_databases": 1,
    "same_organism": True,
    "include_self": True,           # the bait itself, when a homo-oligomer is reported
    "max_partners": None,
}
COLUMNS = ("databases", "n_databases", "string_score", "direct_databases", "direct_methods",
           "n_direct_publications", "direct_of")


def positives_config(cfg: dict) -> dict:
    return {**POSITIVE_DEFAULTS, **((cfg.get("controls") or {}).get("positives") or {})}


def known_partners(cfg: dict, bait_accs: list[str], refresh: bool = False) -> tuple[list[dict], list[dict]]:
    """(positives, every partner) of the baits, from the run's interactor
    snapshot. Both are fetch_interactors.summarize records."""
    import json

    pcfg = positives_config(cfg)
    run = cfg.get("run_name") or "main"
    name = f"interactors_{run}"
    rows = fetch_interactions(name, pcfg, bait_accs, cfg, refresh=refresh)
    sdir = Path((cfg.get("interactors") or {}).get("snapshot_dir", "data/interactors"))
    taxa = json.loads((sdir / f"{name}.json").read_text()).get("query_taxa") or {}
    direct = direct_terms(pcfg, cfg)
    everyone = summarize(rows, {"same_organism": False}, taxa, direct)
    positives = summarize(rows, pcfg, taxa, direct)
    return positives, everyone


def loci_of(accessions, cfg: dict) -> dict[str, str]:
    """{accession: gene locus (AGI, upper case)} for the accessions UniProt
    gives one for (metadata cached in data/uniprot_meta/). Several UniProt
    entries of one gene (Swiss-Prot canonical + TrEMBL gene models) share it."""
    md = fetch_metadata(sorted(set(accessions)), cfg)
    return {a: str(m["locus"]).upper() for a, m in md.items() if m.get("locus")}


def add_positives(res, cfg: dict, refresh: bool = False, verbose: bool = True) -> list[dict]:
    """Add the baits' known (direct) partners to `res` (resolve_inputs.Resolved)
    as preys tagged `positives.name`; annotate screen preys that are partners.
    Returns every partner of the baits (for the decoy exclusions).

    Partners are matched to the screen by gene LOCUS, then accession: the
    databases name the Swiss-Prot entry, an IP list often a TrEMBL entry of
    the same gene (AT5G43900: Q9LKB9 vs F4K7C5). A partner whose gene is in the
    screen tags the screen's own entry (`<tag>_partner_accession` keeps the
    database's accession) instead of adding -- and folding -- the gene twice."""
    pcfg = positives_config(cfg)
    baits = sorted({parse_protein_spec(s).uniprot for s in res.specs["bait"]})
    if not baits:
        return []
    positives, everyone = known_partners(cfg, baits, refresh=refresh)
    tag = pcfg["name"]
    extra = {p["partner"]: {f"{tag}_{c}": p[c] for c in COLUMNS} for p in everyone}
    direct_of = {p["partner"]: p["direct_of"] for p in positives}    # under this run's thresholds
    for acc, ex in extra.items():
        ex[f"{tag}_direct_of"] = direct_of.get(acc, "")
        ex[f"{tag}_partner_accession"] = acc
    if not pcfg["enabled"]:
        positives = []
    excluded = res.excluded.get("prey", set())
    screen_specs = {parse_protein_spec(s).uniprot: s for s in res.specs["prey"]}
    locus = loci_of(list(extra) + list(screen_specs) + list(excluded), cfg)
    screen_by_locus: dict[str, list[str]] = {}
    for acc in screen_specs:
        if acc in locus:
            screen_by_locus.setdefault(locus[acc], []).append(acc)
    excluded_loci = {locus[a] for a in excluded if a in locus}

    def in_screen(acc: str) -> list[str]:
        """The screen entries of a partner: same accession, or same gene."""
        hits = [acc] if acc in screen_specs else []
        return hits + [s for s in screen_by_locus.get(locus.get(acc, ""), []) if s != acc]

    added, by_locus = 0, []
    for p in positives:
        acc = p["partner"]
        if acc in excluded or locus.get(acc) in excluded_loci:
            continue
        targets = in_screen(acc)
        if targets:
            for t in targets:
                res.add("prey", screen_specs[t], tag, extra[acc])
                if t != acc:
                    by_locus.append(f"{acc}->{t} ({locus[acc]})")
        else:
            added += 1
            res.add("prey", acc, tag, extra[acc])
    for acc, ex in extra.items():                  # non-direct partners already in the screen
        for t in in_screen(acc):
            if (t, "prey") in res.provenance and tag not in res.provenance[(t, "prey")]["sources"]:
                res.provenance[(t, "prey")].update(ex)
    if verbose and by_locus:
        print(f"[controls] matched to the screen by gene locus: {', '.join(by_locus)}", file=sys.stderr)
    if verbose and pcfg["enabled"]:
        crit = (f"direct evidence in >= {pcfg['min_direct_publications']} publication(s)"
                if pcfg["direct_only"] else f">= {pcfg['min_databases']} database(s)")
        print(f"[controls] positives: {len(positives)} known partner(s) of {len(baits)} bait(s) with "
              f"{crit} (of {len(everyone)} reported), {added} not already in the screen", file=sys.stderr)
    return everyone


def screen_prey_fragments(sides: dict, res, cfg: dict) -> list:
    """The prey fragments decoys are matched to: everything but the positive
    controls (unless the run has no other preys)."""
    tag = positives_config(cfg)["name"]
    screen = [f for f in sides["prey"]
              if set((res.provenance.get((f.accession, "prey")) or {}).get("sources", [])) - {tag}]
    return screen or list(sides["prey"])


def main() -> int:
    from lib_pipeline import load_config
    from resolve_inputs import resolve_inputs

    cfg = load_config()
    res = resolve_inputs(cfg)
    baits = sorted({parse_protein_spec(s).uniprot for s in res.specs["bait"]})
    if not baits:
        print("[controls] the config resolves to no baits")
        return 1
    positives, everyone = known_partners(cfg, baits)
    print("partner\tn_direct_publications\tdirect_methods\tdirect_databases\tdatabases\tstring_score\tof")
    for p in positives:
        print(f"{p['partner']}\t{p['n_direct_publications']}\t{p['direct_methods']}\t"
              f"{p['direct_databases']}\t{p['databases']}\t{p['string_score'] or ''}\t{p['of']}")
    print(f"[controls] {len(positives)} positive(s) of {len(everyone)} known partner(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
