#!/usr/bin/env python3
"""
nanopulldown.inputs.controls — Stage 1a (positive + negative controls)
===============================================================
Every run's screen gets two control sets added to its preys by
nanopulldown.inputs.expand (config.yaml `controls:`; a run overlay can turn
either off):

* positives (`controls.positives`, source tag `known`): the baits' known
  partners in the PPI databases (nanopulldown.services.interactors: UniProt,
  IntAct, MINT, BioGRID, STRING, Complex Portal) whose evidence tier with the
  bait is at least `min_tier` (default high), plus the hand-picked
  `curated:` partners. Tiers (module doc of nanopulldown.services.interactors):
  curated > high (direct evidence in >= 2 publications, or direct + MIscore
  >= 0.6, or a structure) > medium (direct evidence in 1 publication: two
  hybrid, or "direct interaction" by anything but FRET / BRET / BiFC) > low
  (co-IP, AP-MS, proximity assays alone, complex co-membership, STRING).
  `min_tier: low` takes every partner the databases report (with
  `min_databases`), as the former `interactors:` input source did.
  Positives are cut like any prey in fragment mode. A screen prey that is a
  known partner (any tier) gets the `known_*` columns too; only the positives
  get the `known` tag.
* negatives (`controls.decoys`, source tag `decoy`): the matched decoy set of
  nanopulldown.inputs.decoys, drawn after the screen's preys are cut.

Both are paired with every bait. nanopulldown.report.calibration (Snakefile rule
decoy_calibration) scores the screen against the decoys and reports how many
positives it recovers.

One interactor snapshot per run serves both: data/interactors/interactors_<run>.tsv
(every partner of the baits, every evidence row), from which the positives
are the direct ones and the decoy exclusions are all of them.

Columns carried into configs/inputs_<run>.parquet / the results table
(prefix = `controls.positives.name`, default `known`):
  known_tier, known_tier_of (`<bait>:<tier>`), known_miscore (recomputed over
  every database), known_intact_miscore (IntAct's own), known_databases,
  known_n_databases, known_string_score, known_string_escore / _dscore /
  _tscore (experiments / curated databases / text-mining, annotation only),
  known_direct_databases, known_direct_methods, known_n_direct_publications,
  known_curated_pmids, known_curated_note, known_direct_of (the baits the
  partner reaches `min_tier` with: it is a positive control for those baits
  only, nanopulldown.report.calibration)

Preview the positives of a run (fetches, writes only the snapshot):
    nanopulldown --run-config config_drm2_ppi.yaml inputs positives
"""
from __future__ import annotations

import sys
from pathlib import Path

from nanopulldown.core.specs import parse_protein_spec
from nanopulldown.services.interactors import (DEFAULT_DATABASES, DIRECT_DEFAULTS, TIER_DEFAULTS, TIERS,
                                               evidence_terms, fetch_interactions, summarize)
from nanopulldown.services.uniprot import fetch_metadata

POSITIVE_DEFAULTS = {
    "enabled": True,
    "name": "known",                # input-source tag in the inputs / results tables
    "databases": DEFAULT_DATABASES,
    "min_tier": "high",             # curated > high > medium > low
    "curated": {},                  # {bait: [{partner, pmids, note}]}
    **DIRECT_DEFAULTS,
    **TIER_DEFAULTS,
    "min_databases": 1,
    "same_organism": True,
    "include_self": True,           # the bait itself, when a homo-oligomer is reported
    "max_partners": None,
}
COLUMNS = ("tier", "tier_of", "miscore", "intact_miscore", "databases", "n_databases",
           "string_score", "string_escore", "string_dscore", "string_tscore",
           "direct_databases", "direct_methods", "n_direct_publications", "direct_of",
           "curated_pmids", "curated_note")


def positives_config(cfg: dict) -> dict:
    pcfg = {**POSITIVE_DEFAULTS, **((cfg.get("controls") or {}).get("positives") or {})}
    if "direct_only" in pcfg:                  # replaced by min_tier
        pcfg["min_tier"] = "medium" if pcfg.pop("direct_only") else "low"
        print(f"[controls] positives: `direct_only` is replaced by `min_tier` "
              f"(taken as min_tier: {pcfg['min_tier']})", file=sys.stderr)
    if pcfg["min_tier"] not in TIERS:
        raise ValueError(f"controls.positives.min_tier {pcfg['min_tier']!r}: expected one of {list(TIERS)}")
    return pcfg


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
    terms = evidence_terms(pcfg, cfg)
    everyone = summarize(rows, {**pcfg, "same_organism": False, "min_tier": "low", "min_databases": 1,
                                "max_partners": None}, taxa, terms, queries=bait_accs)
    positives = summarize(rows, pcfg, taxa, terms, queries=bait_accs)
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
        n = {t: sum(p["tier"] == t for p in everyone) for t in TIERS}
        print(f"[controls] positives: {len(positives)} known partner(s) of {len(baits)} bait(s) at tier "
              f">= {pcfg['min_tier']} (of {len(everyone)} reported: "
              f"{', '.join(f'{v} {t}' for t, v in n.items())}), {added} not already in the screen",
              file=sys.stderr)
    return everyone


def screen_prey_fragments(sides: dict, res, cfg: dict) -> list:
    """The prey fragments decoys are matched to: everything but the positive
    controls (unless the run has no other preys)."""
    tag = positives_config(cfg)["name"]
    screen = [f for f in sides["prey"]
              if set((res.provenance.get((f.accession, "prey")) or {}).get("sources", [])) - {tag}]
    return screen or list(sides["prey"])


def main() -> int:
    import argparse

    from nanopulldown.core.config import load_config
    from nanopulldown.inputs.resolve import resolve_inputs

    argparse.ArgumentParser(description="preview the run's positive controls: the baits' known "
                            "direct partners (TSV on stdout; set the run with --run-config)").parse_args()
    cfg = load_config()
    res = resolve_inputs(cfg)
    baits = sorted({parse_protein_spec(s).uniprot for s in res.specs["bait"]})
    if not baits:
        print("[controls] the config resolves to no baits")
        return 1
    positives, everyone = known_partners(cfg, baits)
    print("partner\ttier\tmiscore\tintact_miscore\tn_direct_publications\tdirect_methods\t"
          "databases\tstring_score\tstring_escore\tcurated_pmids\ttier_of")
    for p in everyone:
        print(f"{p['partner']}\t{p['tier']}\t{p['miscore']:.2f}\t{p['intact_miscore'] or ''}\t"
              f"{p['n_direct_publications']}\t{p['direct_methods']}\t{p['databases']}\t"
              f"{p['string_score'] or ''}\t{p['string_escore'] or ''}\t{p['curated_pmids']}\t{p['tier_of']}")
    print(f"[controls] {len(positives)} positive(s) at tier >= {positives_config(cfg)['min_tier']} "
          f"of {len(everyone)} known partner(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
