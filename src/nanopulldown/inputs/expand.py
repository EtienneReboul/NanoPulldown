#!/usr/bin/env python3
"""
nanopulldown.inputs.expand — Stage 1a
====================================
AlphaPulldown's bait-file x prey-file combinatorial screen, adapted. Baits
and preys come from the run config's `inputs:` block (list files, inline
accessions, UniProt GO/InterPro queries, IP-MS tables -- see
nanopulldown.inputs.resolve), or, without one / with --baits/--preys, from two
AlphaPulldown-style list files (`ID`, `ID:N`, `ID:start-stop`,
`ID:N:start-stop`, blank lines and `#` comments ignored). Every unique
accession is fetched via nanopulldown.services.uniprot, and one row per
combination is added to configs/pairs.parquet (see configs/_schema.md).
Pairs already in the table are left untouched; hand-curated domains live in
configs/curated_domains.yaml and are never touched here.

Alongside the pairs it writes, for the reports:
  * configs/proteins.parquet      -- UniProt names/gene/locus per accession
  * configs/inputs_<run>.parquet  -- which source picked each protein (+ IP-MS columns)
  * the run's pair list           -- config `pairs_file:` (default
                                     configs/pairs_<run>.txt), which the
                                     Snakefile runs when `pairs_file:` is set

Controls (config `controls:`, nanopulldown.inputs.controls), added to every run's
preys and paired with every bait:
  * positives -- the baits' known direct partners in the PPI databases
                 (source tag `known`)
  * decoys    -- a matched decoy set (nanopulldown.inputs.decoys, source tag
                 `decoy`), drawn once per run and reused while its settings
                 are unchanged (--rebuild-decoys redraws it)
Turn either off in a run overlay (`controls: {decoys: {enabled: false}}`).

Fragment mode (config `fragments.enabled: true`, or --fragments): every bait
and prey is first cut into its annotated domains (nanopulldown.inputs.fragments),
and the pairs are bait fragment x prey fragment, each an ordinary region pair
(`Q9M548x120-345__O22607x10-160`) carrying its domain names for the reports.
The fragments are listed in configs/fragments_<run>.parquet.

Pairs whose total token count (sum of all chain lengths) exceeds
`--max-tokens` (default: config's `esmfold2.max_tokens`, 0 = no limit) are NOT
added: ESMFold2 memory grows ~quadratically with tokens, and past the Metal
buffer cap the fold aborts the whole process. They are listed, with their
token counts, in `configs/skipped_too_long.tsv` instead. Measure your
machine's limit with notebooks/esmfold2_memory_model.ipynb.

Usage:
    nanopulldown --run-config config_bpm6_curated.yaml inputs expand
    nanopulldown inputs expand --baits configs/baits.txt --preys configs/preys.txt
    nanopulldown inputs expand --max-tokens 1500    # override the config limit
    nanopulldown inputs expand --refresh-queries    # re-run UniProt queries
    nanopulldown --run-config config_drm2_fragments.yaml inputs expand   # fragment mode
    nanopulldown --run-config config_<run>.yaml inputs expand --dry-run   # counts only
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from nanopulldown.core.config import REPO_ROOT, load_config
from nanopulldown.core.proteins import update_proteins
from nanopulldown.core.specs import (
    PAIRS_TABLE,
    inputs_table_path,
    load_pairs,
    pair_tokens,
    parse_protein_spec,
    write_pairs,
)
from nanopulldown.inputs.controls import add_positives, loci_of, positives_config, screen_prey_fragments
from nanopulldown.inputs.decoys import decoy_config, decoy_fragments
from nanopulldown.inputs.fragments import (
    Fragment,
    fragment_config,
    load_overrides,
    prefetch_annotation,
    protein_fragments,
    write_fragments_table,
)
from nanopulldown.inputs.resolve import Resolved, collapse_same_gene, read_specs, resolve_inputs, write_inputs_table
from nanopulldown.services.uniprot import fetch_metadata, fetch_sequence


def slug(spec: str) -> str:
    """Filesystem/YAML-key-safe stem for a protein spec, e.g.
    'Q8I2G6:2:1-100' -> 'Q8I2G6x2x1-100'."""
    return re.sub(r"[^A-Za-z0-9_.-]", "x", spec)


def build_pair_spec(bait_spec: str, prey_spec: str, cfg: dict,
                    fragments: tuple[str | None, str | None] = (None, None)) -> dict:
    """One pair's spec. `fragments` = (bait, prey) fragment-mode domain names
    (None = not a fragment); a fragment's chains are named after its region."""
    cache_dir = Path(cfg["dirs"]["sequences"])
    bait_ps = parse_protein_spec(bait_spec)
    prey_ps = parse_protein_spec(prey_spec)
    bait_seq = fetch_sequence(bait_spec, cfg, cache_dir)
    prey_seq = fetch_sequence(prey_spec, cfg, cache_dir)

    sequences = []
    chain_ids = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    bait_chains, prey_chains = [], []
    for role, ps, seq, prefix, frag in (
        ("bait", bait_ps, bait_seq, "bait", fragments[0]),
        ("prey", prey_ps, prey_seq, "prey", fragments[1]),
    ):
        stem = f"{prefix}_{ps.uniprot}"
        if frag and ps.regions:
            stem += "_" + "_".join(f"{a}-{b}" for a, b in ps.regions)
        for copy_i in range(ps.copies):
            cid = next(chain_ids)
            name = stem + (f"_{copy_i + 1}" if ps.copies > 1 else "")
            sequences.append({"id": cid, "name": name, "sequence": seq, "role": role})
            (bait_chains if role == "bait" else prey_chains).append(cid)

    return {
        "name": f"{slug(bait_spec)}__{slug(prey_spec)}",
        "bait": {"uniprot": bait_ps.uniprot, "copies": bait_ps.copies,
                 "regions": bait_ps.regions, "fragment": fragments[0]},
        "prey": {"uniprot": prey_ps.uniprot, "copies": prey_ps.copies,
                 "regions": prey_ps.regions, "fragment": fragments[1]},
        "sequences": sequences,
        "bait_chains": bait_chains,
        "prey_chains": prey_chains,
    }


@dataclass
class Prepared:
    res: Resolved
    sides: dict[str, list[Fragment]]   # role -> fragments paired (decoys included)
    fragment_mode: bool
    n_positives: int                   # positive-control preys not in the screen
    n_decoys: int


def cut_sides(baits: list[str], preys: list[str], cfg: dict, fragment_mode: bool) -> dict | None:
    """One Fragment per protein as given, or its domain fragments in fragment
    mode. None when a protein's domain annotation is incomplete."""
    if not fragment_mode:
        return {"bait": [Fragment(s, parse_protein_spec(s).uniprot, "bait", None, 0, 0, "as_given", [])
                         for s in baits],
                "prey": [Fragment(s, parse_protein_spec(s).uniprot, "prey", None, 0, 0, "as_given", [])
                         for s in preys]}
    fcfg = fragment_config(cfg)
    overrides = load_overrides(fcfg)
    cut = [s for role, rs in (("bait", baits), ("prey", preys)) if role in fcfg["sides"]
           for s in rs if parse_protein_spec(s).uniprot not in overrides]
    failed = prefetch_annotation(cut, cfg)
    if failed:
        print(f"[expand_pairs] ERROR: domain annotation incomplete for {len(failed)} protein(s) "
              f"(web service errors), e.g. {failed[:5]}. Nothing written; progress is cached, "
              "rerun to retry them.", file=sys.stderr)
        return None
    sides = {}
    for role, role_specs in (("bait", baits), ("prey", preys)):
        per_spec = [protein_fragments(s, role, cfg, fcfg, overrides) for s in role_specs]
        sides[role] = [f for frags in per_spec for f in frags]
        skipped_specs = [s for s, frags in zip(role_specs, per_spec) if not frags]
        print(f"[expand_pairs] fragment mode: {len(role_specs)} {role}(s) -> {len(sides[role])} fragment(s)"
              + (f"; no domain, skipped: {', '.join(skipped_specs)}" if skipped_specs else ""))
    return sides


def prepare(cfg: dict, baits_file: str | None = None, preys_file: str | None = None,
            refresh: bool = False, fragments: bool = False, rebuild_decoys: bool = False,
            dry_run: bool = False) -> Prepared | None:
    """Everything the run pairs: the screen's baits and preys (config `inputs:`,
    or two list files), plus the controls (positives before cutting, decoys
    matched to the cut screen preys). None when nothing can be expanded."""
    if cfg.get("inputs") and not (baits_file or preys_file):
        res = resolve_inputs(cfg, refresh=refresh)
    else:                                   # legacy: two list files
        res = Resolved()
        for role, path in (("bait", baits_file or "configs/baits.txt"),
                           ("prey", preys_file or "configs/preys.txt")):
            if Path(path).exists():
                for spec in read_specs(Path(path)):
                    res.add(role, spec, Path(path).stem)
    if not res.specs["bait"]:
        print("[expand_pairs] nothing to expand (no bait)")
        return None

    pcfg, dcfg = positives_config(cfg), decoy_config(cfg)
    n_screen = len({parse_protein_spec(s).uniprot for s in res.specs["prey"]})
    partners = []
    if pcfg["enabled"] or (dcfg["enabled"] and dcfg["exclude_interactors"]):
        partners = add_positives(res, cfg, refresh=refresh)
    # one spec per gene: several UniProt entries of one gene are folded once
    accs = {parse_protein_spec(s).uniprot for r in ("bait", "prey") for s in res.specs[r]}
    meta = fetch_metadata(sorted(accs), cfg)
    locus = {a: str(m["locus"]).upper() for a, m in meta.items() if m.get("locus")}
    collapsed = collapse_same_gene(res, locus, {pcfg["name"], dcfg["name"]},
                                   {a: bool(m.get("reviewed")) for a, m in meta.items()})
    if collapsed:
        print(f"[expand_pairs] same gene, folded once: {', '.join(collapsed)}")
    n_pos = len({parse_protein_spec(s).uniprot for s in res.specs["prey"]}) - n_screen
    baits, preys = res.specs["bait"], res.specs["prey"]
    if not preys:
        print(f"[expand_pairs] nothing to expand ({len(baits)} bait(s), no prey and no known partner)")
        return None

    fragment_mode = fragments or bool(fragment_config(cfg)["enabled"])
    sides = cut_sides(baits, preys, cfg, fragment_mode)
    if sides is None:
        return None

    decoys = []
    if dcfg["enabled"]:
        exclude = ({parse_protein_spec(s).uniprot for s in baits + preys} | res.excluded["bait"]
                   | res.excluded["prey"] | {p["partner"] for p in partners})
        # and their genes: another UniProt entry of an excluded gene is no decoy either
        exclude |= {f"locus:{l}" for l in loci_of(exclude, cfg).values()}
        decoys = decoy_fragments(cfg, screen_prey_fragments(sides, res, cfg), sides["bait"], exclude,
                                 dcfg, rebuild=rebuild_decoys, dry_run=dry_run)
        for f in decoys:
            res.add("prey", f.spec, dcfg["name"])
        sides["prey"] += decoys
    nb, npy = len(sides["bait"]), len(sides["prey"])
    print(f"[expand_pairs] controls: +{n_pos} positive prey protein(s) ({pcfg['name']}), "
          f"+{len(decoys)} decoy prey(s) ({dcfg['name']}) -> {nb} bait x {npy} prey "
          f"{'fragment' if fragment_mode else 'protein'}(s), of which {nb * len(decoys)} decoy pair(s)")
    return Prepared(res, sides, fragment_mode, n_pos, len(decoys))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baits", help="bait list file (overrides the config's inputs.baits)")
    ap.add_argument("--preys", help="prey list file (overrides the config's inputs.preys)")
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="skip pairs with more total tokens than this "
                         "(default: config esmfold2.max_tokens; 0 = no limit)")
    ap.add_argument("--refresh-queries", action="store_true",
                    help="re-run uniprot_query sources and the interactor lookup instead of "
                         "reusing their snapshots")
    ap.add_argument("--pairs-out", default=None,
                    help="write this run's pair list here (default: config pairs_file, "
                         "else configs/pairs_<run_name>.txt)")
    ap.add_argument("--fragments", action="store_true",
                    help="fragment mode: pair domain fragments (config `fragments:`), "
                         "even when fragments.enabled is false")
    ap.add_argument("--rebuild-decoys", action="store_true",
                    help="redraw the decoy set even if its settings are unchanged")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve, cut and draw the controls, print the counts, write nothing "
                         "(snapshots and annotation caches excepted)")
    a = ap.parse_args()

    cfg = load_config(REPO_ROOT)
    max_tokens = a.max_tokens if a.max_tokens is not None else int(cfg["esmfold2"].get("max_tokens", 0))
    prep = prepare(cfg, a.baits, a.preys, refresh=a.refresh_queries, fragments=a.fragments,
                   rebuild_decoys=a.rebuild_decoys, dry_run=a.dry_run)
    if prep is None:
        return 0 if not a.dry_run else 1
    res, sides, fragment_mode = prep.res, prep.sides, prep.fragment_mode
    baits, preys = res.specs["bait"], res.specs["prey"]

    out_dir = Path(a.configs_dir)
    run = cfg.get("run_name") or "main"
    if a.dry_run:
        n = len(sides["bait"]) * len(sides["prey"])
        print(f"[expand_pairs] dry run: {len(sides['bait'])} x {len(sides['prey'])} = {n} combination(s) "
              f"before the max_tokens={max_tokens} filter; nothing written")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    if fragment_mode:
        frag_table = write_fragments_table(sides["bait"] + sides["prey"], out_dir / f"fragments_{run}.parquet")
        print(f"[expand_pairs] fragments -> {frag_table}")

    specs = load_pairs(REPO_ROOT)
    written, skipped, too_long, run_pairs = 0, 0, [], []
    for bf in sides["bait"]:
        for pf in sides["prey"]:
            pair = build_pair_spec(bf.spec, pf.spec, cfg, (bf.name, pf.name))
            n_tok = pair_tokens(pair)
            if max_tokens and n_tok > max_tokens:
                too_long.append((pair["name"], n_tok))
                continue
            run_pairs.append(pair["name"])
            if pair["name"] in specs:
                skipped += 1
                continue
            specs[pair["name"]] = pair
            written += 1
            print(f"[expand_pairs] added {pair['name']} ({n_tok} tokens)")
    if written:
        print(f"[expand_pairs] -> {write_pairs(list(specs.values()), REPO_ROOT)}")

    # names for the reports + where each protein came from
    accs = sorted({parse_protein_spec(s).uniprot for s in baits + preys})
    meta = fetch_metadata(accs, cfg)
    for acc, name in res.manual_names.items():
        if acc in meta:
            meta[acc] = {**meta[acc], "manual_name": name}
    print(f"[expand_pairs] {len(accs)} protein(s) -> {update_proteins(meta.values(), REPO_ROOT)}")
    print(f"[expand_pairs] inputs -> {write_inputs_table(res, inputs_table_path(cfg, REPO_ROOT))}")

    pairs_out = Path(a.pairs_out or cfg.get("pairs_file") or out_dir / f"pairs_{run}.txt")
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    mode = (f" = {len(sides['bait'])} x {len(sides['prey'])} domain fragment(s)"
            if fragment_mode else "")
    controls = f" (incl. {prep.n_positives} positive control(s), {prep.n_decoys} decoy(s))"
    pairs_out.write_text(f"# {run}: {len(baits)} bait(s) x {len(preys)} prey(s){controls}{mode}, "
                         f"written by nanopulldown.inputs.expand\n" + "".join(f"{p}\n" for p in run_pairs))
    print(f"[expand_pairs] {len(run_pairs)} pair(s) for this run -> {pairs_out}"
          + ("" if cfg.get("pairs_file") else f"  (set `pairs_file: {pairs_out}` to run them)"))

    skip_tsv = out_dir / (f"skipped_too_long_{run}.tsv" if cfg.get("run_name") else "skipped_too_long.tsv")
    if too_long:
        skip_tsv.write_text("pair\ttokens\n" + "".join(f"{n}\t{t}\n" for n, t in too_long))
    elif skip_tsv.exists():
        skip_tsv.unlink()
    if too_long:
        print(f"[expand_pairs] {len(too_long)} pair(s) over max_tokens={max_tokens} NOT written "
              f"-> {skip_tsv}")
    print(f"[expand_pairs] {written} new pair(s), {skipped} already in {PAIRS_TABLE} "
          f"({len(sides['bait'])} bait x {len(sides['prey'])} prey "
          f"{'fragment' if fragment_mode else 'protein'}(s) = "
          f"{len(sides['bait']) * len(sides['prey'])} combination(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
