#!/usr/bin/env python3
"""
nanopulldown.inputs.decoys — Stage 1a (negative controls)
========================================================
Builds a DUD-E-style decoy prey set for a run: real, folded Arabidopsis
proteins that cannot meet a nuclear bait in the cell, cut and matched so that
they look like the run's preys to the scoring metrics. Folded against the
same baits, their scores are the per-bait null distribution that
nanopulldown.report.calibration turns into conformal p-values + BH FDR.

nanopulldown.inputs.expand calls decoy_fragments() for every run with
`controls.decoys.enabled` (config.yaml), after the screen's preys are cut,
and pairs the decoys with every bait like any prey. They are tagged with the
input source `controls.decoys.name` (default `decoy`), which is how
decoy_calibration.py finds them in the results table. The positive controls
(the baits' known direct partners, nanopulldown.inputs.controls) are the other half.

How decoys are picked (all knobs in `controls.decoys`, see DEFAULTS; CLI
flags override):

  1. pool       -- a UniProt query, snapshotted like any `uniprot_query:`
                   source (data/uniprot_retrieval/decoy_pool_<run>.tsv):
                   organism, reviewed entries localised to a compartment a
                   nuclear bait never reaches (`compartments`, ORed: plastid
                   stroma, thylakoid lumen, mitochondrial matrix, apoplast,
                   vacuolar lumen, ER lumen, peroxisomal matrix,
                   extracellular region) and to none of `exclude_go` (nucleus,
                   cytosol). Dual-targeted proteins are therefore dropped.
  2. exclusions -- the run's own baits and preys (controls included), its
                   `exclude:` entries, and every known partner of the baits in
                   any PPI database, direct or not (the controls' interactor
                   snapshot, data/interactors/interactors_<run>.tsv).
  3. fragments  -- in fragment mode, pool proteins are cut by the run's own
                   `fragments:` rules (nanopulldown.inputs.fragments), so decoy
                   fragments are domain blocks of the same kind as the preys';
                   otherwise whole proteins. Fragments too long to fold with the
                   longest bait chain (esmfold2.max_tokens) are dropped, as are
                   duplicate sequences and more than `max_per_protein` per protein.
  4. matching   -- stratified on the run's screen prey fragments (positive
                   controls left out, unless the run has no other preys):
                   `length_bins` length quantile bins x Cys+His-rich (fraction
                   >= `cys_his_cutoff`: zinc-binding RING / U-box / zinc-finger
                   blocks, which score differently from large globular domains).
                   Each stratum gets its share of `n`; a short stratum is filled
                   from the nearest length bins of the same Cys+His class.

Outputs (reused while the decoy settings are unchanged, so adding preys to a
run never re-draws -- and re-folds -- its decoys; --rebuild forces a new draw):
  * configs/decoys_<run>.txt      -- one explicit region spec per decoy
    (`ACC:start-stop`), for reading
  * data/decoys/decoys_<run>.parquet (+ .json: settings, prey-reference
    digest) -- provenance: fragment, domains, length, Cys+His fraction,
    stratum, UniProt name + subcellular location
  * a match report (prey vs decoy strata, length KS distance) on stdout

Preview / redraw outside expand_pairs (prepares the run exactly as
expand_pairs does, pairs nothing):
    nanopulldown --run-config config_drm2_e3_fbox_fragments.yaml inputs decoys --dry-run
    ... --n 300 --seed 1          # override the config, redraw and write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nanopulldown.core.config import REPO_ROOT
from nanopulldown.core.fold_table import fold_max_tokens
from nanopulldown.core.io import atomic_write
from nanopulldown.core.specs import parse_protein_spec
from nanopulldown.inputs.fragments import (
    AnnotationIncomplete,
    Fragment,
    fragment_config,
    load_overrides,
    prefetch_annotation,
    protein_fragments,
)
from nanopulldown.services.uniprot import build_query, fetch_metadata, fetch_sequence, run_query

DEFAULTS = {
    "enabled": True,
    "n": 500,                       # decoy preys per run (each is paired with every bait)
    "name": "decoy",                # input-source tag in the inputs / results tables
    "seed": 0,
    "organism": 3702,
    "reviewed": True,
    # compartments a nuclear bait never reaches (ORed, child terms included)
    "compartments": ["GO:0009570",  # chloroplast stroma
                     "GO:0031977",  # thylakoid lumen
                     "GO:0005759",  # mitochondrial matrix
                     "GO:0048046",  # apoplast
                     "GO:0005775",  # vacuolar lumen
                     "GO:0005788",  # ER lumen
                     "GO:0005782",  # peroxisomal matrix
                     "GO:0005576"],  # extracellular region
    # (the first five alone gave 800 reviewed proteins: too few Cys+His-rich
    # blocks to match a RING/U-box-heavy E3 screen at n=500)
    "exclude_go": ["GO:0005634",    # nucleus
                   "GO:0005829"],   # cytosol
    "exclude_transmembrane": False,  # True: drop KW-0812 (keep only soluble decoys)
    "exclude_interactors": True,    # drop the baits' known PPI partners (any evidence)
    "length_bins": 8,
    "cys_his_cutoff": 0.08,
    "max_per_protein": 2,
    "oversample": 3,                # candidates collected per needed decoy, per stratum
    "batch": 100,                   # pool proteins annotated + cut per round
}
FORMAT = 1                          # bump when the draw changes for the same settings


def decoy_config(cfg: dict, **overrides) -> dict:
    """DEFAULTS < config `controls.decoys` < non-None overrides (CLI)."""
    dcfg = {**DEFAULTS, **((cfg.get("controls") or {}).get("decoys") or {})}
    dcfg.update({k: v for k, v in overrides.items() if v is not None})
    return dcfg


def decoy_paths(cfg: dict) -> tuple[Path, Path, Path]:
    run = cfg.get("run_name") or "main"
    prov = REPO_ROOT / "data" / "decoys" / f"decoys_{run}.parquet"
    return REPO_ROOT / "configs" / f"decoys_{run}.txt", prov, prov.with_suffix(".json")


def decoy_settings(cfg: dict, dcfg: dict) -> dict:
    """Everything that changes the draw (the reuse key). The prey reference is
    left out on purpose: growing a screen keeps its decoys (and their folds)."""
    fcfg = fragment_config(cfg)
    return {"format": FORMAT, "decoys": {k: v for k, v in dcfg.items() if k not in ("enabled", "name")},
            "fragments": {k: fcfg[k] for k in ("enabled", "sides", "merge_gap", "repeat_gap", "min_length",
                                               "padding", "without_domains")},
            "max_tokens": fold_max_tokens(cfg)}


def pool_query(dcfg: dict) -> str:
    q = build_query({"name": "decoy_pool", "go_any": dcfg["compartments"],
                     "organism": dcfg["organism"], "reviewed": dcfg["reviewed"]})
    for t in dcfg["exclude_go"]:
        q += f" NOT (go:{str(t).upper().removeprefix('GO:')})"
    if dcfg["exclude_transmembrane"]:
        q += " NOT (keyword:KW-0812)"
    return q


def cys_his(seq: str) -> float:
    return (seq.count("C") + seq.count("H")) / max(len(seq), 1)


def ks_distance(a, b) -> float:
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / len(a)
                               - np.searchsorted(b, grid, "right") / len(b))))


def reference_table(frags: list[Fragment], cfg: dict) -> pd.DataFrame:
    """[spec, accession, length, cys_his] of the prey fragments to match."""
    rows = []
    for f in frags:
        seq = fetch_sequence(f.spec, cfg)
        rows.append({"spec": f.spec, "accession": f.accession, "length": len(seq), "cys_his": cys_his(seq)})
    return pd.DataFrame(rows, columns=["spec", "accession", "length", "cys_his"])


def candidate_fragments(accs: list[str], cfg: dict, fcfg: dict, overrides: dict) -> list[Fragment]:
    if not fcfg["enabled"] or "prey" not in fcfg["sides"]:
        return [Fragment(a, a, "prey", None, 1, len(fetch_sequence(a, cfg)), "as_given", []) for a in accs]
    failed = set(prefetch_annotation(accs, cfg))
    out = []
    for a in accs:
        if a in failed:
            continue
        try:
            out += protein_fragments(a, "prey", cfg, fcfg, overrides)
        except AnnotationIncomplete:
            continue
    return out


def stratify(lengths: pd.Series, ch: pd.Series, edges: np.ndarray, cutoff: float) -> pd.Series:
    b = np.clip(np.searchsorted(edges, lengths, side="right") - 1, 0, len(edges) - 2)
    return pd.Series([f"L{i}|{'CH' if c >= cutoff else 'std'}" for i, c in zip(b, ch)], index=lengths.index)


def allocate(shares: pd.Series, n: int) -> pd.Series:
    """Largest-remainder split of n over strata by their prey share."""
    raw = shares * n
    out = np.floor(raw).astype(int)
    out[(raw - out).sort_values(ascending=False).index[:n - out.sum()]] += 1
    return out


def select_decoys(cfg: dict, dcfg: dict, ref: pd.DataFrame, bait_len: int,
                  exclude: set[str]) -> tuple[pd.DataFrame, str]:
    """The stratified decoy draw (module docstring, steps 1, 3, 4): (decoys
    [spec, accession, fragment, domains, start, end, length, cys_his, source,
    stratum], pool query). Prints the match report."""
    run = cfg.get("run_name") or "main"
    rng = np.random.default_rng(int(dcfg["seed"]))
    max_tokens = fold_max_tokens(cfg)
    ref = ref.copy()

    query = pool_query(dcfg)
    pool = [acc for acc in run_query(f"decoy_pool_{run}", query, cfg) if acc not in exclude]
    pool = _drop_excluded_genes(pool, exclude, cfg)
    rng.shuffle(pool)
    print(f"[build_decoys] pool: {len(pool)} protein(s) after exclusions\n  query: {query}")

    # strata from the preys
    edges = np.unique(np.quantile(ref.length, np.linspace(0, 1, int(dcfg["length_bins"]) + 1)))
    edges[0], edges[-1] = 0, np.inf
    ref["stratum"] = stratify(ref.length, ref.cys_his, edges, float(dcfg["cys_his_cutoff"]))
    target = allocate(ref.stratum.value_counts(normalize=True), int(dcfg["n"]))

    # collect candidate fragments round by round until every stratum is covered
    fcfg, overrides = fragment_config(cfg), load_overrides(fragment_config(cfg))
    limit = max_tokens - bait_len if max_tokens else np.inf
    cand, seen = [], set()
    for i in range(0, len(pool), int(dcfg["batch"])):
        for f in candidate_fragments(pool[i:i + int(dcfg["batch"])], cfg, fcfg, overrides):
            seq = fetch_sequence(f.spec, cfg)
            if len(seq) > limit or seq in seen:
                continue
            seen.add(seq)
            cand.append({"spec": f.spec, "accession": f.accession, "fragment": f.name,
                         "domains": list(f.domains), "start": f.start, "end": f.end,
                         "length": len(seq), "cys_his": cys_his(seq), "source": f.source})
        if not cand:
            continue
        c = pd.DataFrame(cand)
        have = stratify(c.length, c.cys_his, edges, float(dcfg["cys_his_cutoff"])).value_counts()
        short = (target * int(dcfg["oversample"]) - have.reindex(target.index, fill_value=0)).clip(lower=0)
        print(f"[build_decoys] {min(i + int(dcfg['batch']), len(pool))}/{len(pool)} pool protein(s) -> "
              f"{len(c)} candidate fragment(s); strata still short: {int((short > 0).sum())}", flush=True)
        if not short.any():
            break
    if not cand:
        raise RuntimeError(f"no decoy candidate fits next to the longest bait chain ({bait_len} aa, "
                           f"max_tokens {max_tokens}) -- check controls.decoys / esmfold2.max_tokens")
    c = pd.DataFrame(cand)
    c["stratum"] = stratify(c.length, c.cys_his, edges, float(dcfg["cys_his_cutoff"]))

    # stratified draw, at most max_per_protein fragments per protein
    per_prot: dict[str, int] = {}
    picked: list[int] = []

    def draw(rows: pd.DataFrame, k: int) -> int:
        got = 0
        for idx in rng.permutation(rows.index.to_numpy()):
            if got == k:
                break
            acc = c.at[idx, "accession"]
            if idx in picked or per_prot.get(acc, 0) >= int(dcfg["max_per_protein"]):
                continue
            picked.append(idx)
            per_prot[acc] = per_prot.get(acc, 0) + 1
            got += 1
        return got

    shortfall = {}
    for stratum, k in target.items():
        got = draw(c[c.stratum == stratum], int(k))
        if got < k:                                  # fill from the nearest length bins, same class
            b, cls = int(stratum[1:].split("|")[0]), stratum.split("|")[1]
            for d in range(1, len(edges)):
                for nb in (b - d, b + d):
                    if got < k and 0 <= nb < len(edges) - 1:
                        got += draw(c[c.stratum == f"L{nb}|{cls}"], int(k) - got)
            if got < k:
                shortfall[stratum] = int(k - got)
    dec = c.loc[picked].copy()

    # match report
    rep = pd.DataFrame({"preys": ref.stratum.value_counts(normalize=True),
                        "decoys": dec.stratum.value_counts(normalize=True)}).fillna(0)
    rep["length range"] = [f"{edges[int(s[1:].split('|')[0])]:.0f}-{edges[int(s[1:].split('|')[0]) + 1]:.0f}"
                           for s in rep.index]
    print("\n[build_decoys] match report (fraction of fragments per stratum)")
    print((rep[["length range", "preys", "decoys"]]).round(3).sort_index().to_string())
    print(f"  length median: preys {ref.length.median():.0f} / decoys {dec.length.median():.0f} aa; "
          f"KS distance {ks_distance(ref.length, dec.length):.3f}")
    print(f"  Cys+His-rich: preys {(ref.cys_his >= dcfg['cys_his_cutoff']).mean():.1%} / "
          f"decoys {(dec.cys_his >= dcfg['cys_his_cutoff']).mean():.1%}")
    print(f"  {len(dec)} decoy fragment(s) from {dec.accession.nunique()} protein(s)"
          + (f"; SHORT by {sum(shortfall.values())} in {shortfall}" if shortfall else ""))
    return dec, query


def write_decoys(dec: pd.DataFrame, cfg: dict, dcfg: dict, query: str, meta: dict) -> pd.DataFrame:
    """Decoy list + provenance parquet + settings sidecar (decoy_paths)."""
    run = cfg.get("run_name") or "main"
    out_list, out_prov, out_meta = decoy_paths(cfg)
    names = fetch_metadata(sorted(dec.accession.unique()), cfg)
    dec = dec.copy()
    dec["protein_name"] = dec.accession.map(lambda x: names.get(x, {}).get("protein_name"))
    dec["subcellular_location"] = dec.accession.map(
        lambda x: "; ".join(names.get(x, {}).get("subcellular_location") or []))
    dec = dec.sort_values(["stratum", "length"]).reset_index(drop=True)

    out_list.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# {run}: {len(dec)} decoy prey(s), written by nanopulldown.inputs.decoys (seed {dcfg['seed']})\n"
              f"# pool query: {query}\n"
              f"# paired with every bait by nanopulldown.inputs.expand (controls.decoys); for reading only\n")
    out_list.write_text(header + "".join(
        f"{r.spec}  # {r.length} aa  {r.fragment or 'full length'}  [{r.protein_name or r.accession}]\n"
        for r in dec.itertuples()))

    from nanopulldown.core.io import write_parquet_with_metadata
    out_prov.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_with_metadata(
        dec, out_prov,
        table_description=f"nanopulldown decoy preys of run {run} (nanopulldown.inputs.decoys): "
                          f"pool query `{query}`, seed {dcfg['seed']}, matched to the run's prey "
                          "fragments on length quantile bin x Cys+His-rich stratum.",
        column_descriptions={
            "spec": "AlphaPulldown region spec the decoy is paired as",
            "accession": "UniProt accession", "fragment": "Domain name(s); null = whole protein",
            "domains": "Domains merged into the fragment", "start": "First residue (1-based)",
            "end": "Last residue (inclusive)", "length": "Fragment length (residues)",
            "cys_his": "Fraction of Cys + His residues",
            "source": "interpro | full_length | as_given (nanopulldown.inputs.fragments)",
            "stratum": "Matching stratum: L<length bin>|CH (Cys+His-rich) or |std",
            "protein_name": "UniProt protein name",
            "subcellular_location": "UniProt subcellular location(s)"})
    atomic_write(out_meta, json.dumps({**meta, "query": query, "n": len(dec)}, indent=1))
    print(f"[build_decoys] -> {out_list}\n[build_decoys] -> {out_prov}")
    return dec


def _drop_excluded_genes(accs: list[str], exclude: set[str], cfg: dict) -> list[str]:
    """Drop accessions whose gene locus is excluded (`exclude` holds
    "locus:<AGI>" entries next to accessions): another UniProt entry of a
    screen protein or known partner is no decoy."""
    loci = {e[6:] for e in exclude if e.startswith("locus:")}
    if not loci or not accs:
        return accs
    md = fetch_metadata(accs, cfg)
    return [a for a in accs if str((md.get(a) or {}).get("locus") or "").upper() not in loci]


def _digest(ref: pd.DataFrame) -> str:
    return hashlib.sha1("\n".join(sorted(ref.spec)).encode()).hexdigest()[:12]


def decoy_fragments(cfg: dict, ref_frags: list[Fragment], bait_frags: list[Fragment], exclude: set[str],
                    dcfg: dict | None = None, rebuild: bool = False, dry_run: bool = False) -> list[Fragment]:
    """The run's decoy preys, as Fragments (source `decoy`): the stored set
    when its settings match, else a new draw matched to `ref_frags` (written
    unless dry_run)."""
    dcfg = dcfg or decoy_config(cfg)
    _, out_prov, out_meta = decoy_paths(cfg)
    settings = decoy_settings(cfg, dcfg)
    ref = reference_table(ref_frags, cfg)
    max_tokens = fold_max_tokens(cfg)
    if max_tokens:                  # only preys some bait is folded with
        shortest = min(len(fetch_sequence(f.spec, cfg)) * parse_protein_spec(f.spec).copies
                       for f in bait_frags)
        ref = ref[ref.length <= max_tokens - shortest]
    if ref.empty:
        raise ValueError("no prey fragment fits under esmfold2.max_tokens next to a bait: "
                         "nothing to match decoys to")
    stored = json.loads(out_meta.read_text()) if out_meta.exists() else None
    if not rebuild and stored and stored.get("settings") == settings and out_prov.exists():
        dec = pd.read_parquet(out_prov)
        if stored.get("reference") != _digest(ref):
            print(f"[build_decoys] the run's preys changed since its {len(dec)} decoys were drawn; "
                  "keeping them (their folds stay valid). Redraw with build_decoys.py / "
                  "expand_pairs.py --rebuild-decoys", file=sys.stderr)
        keep = set(_drop_excluded_genes(sorted(set(dec.accession)), exclude, cfg))
        if len(keep) < dec.accession.nunique():
            gone = sorted(set(dec.accession) - keep)
            print(f"[build_decoys] dropping {len(gone)} stored decoy protein(s) whose gene is a screen "
                  f"protein or a known partner: {', '.join(gone)}", file=sys.stderr)
            dec = dec[dec.accession.isin(keep)]
        print(f"[build_decoys] reusing {len(dec)} decoy(s) from {out_prov}")
    else:
        if stored and not rebuild:
            print("[build_decoys] decoy settings changed: drawing a new decoy set", file=sys.stderr)
        bait_len = max(len(fetch_sequence(f.spec, cfg)) * parse_protein_spec(f.spec).copies
                       for f in bait_frags)
        print(f"[build_decoys] {len(ref)} prey fragment(s) to match, longest bait chain {bait_len} aa")
        dec, query = select_decoys(cfg, dcfg, ref, bait_len, exclude)
        if not dry_run:
            dec = write_decoys(dec, cfg, dcfg, query, {"settings": settings, "reference": _digest(ref)})
    return [Fragment(r.spec, r.accession, "prey", r.fragment if isinstance(r.fragment, str) else None,
                     int(r.start), int(r.end), "decoy", [str(d) for d in (r.domains if r.domains is not None
                                                                           else [])])
            for r in dec.itertuples()]


def main() -> int:
    ap = argparse.ArgumentParser(description="draw (or preview) the run's matched decoy prey set")
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--dry-run", action="store_true", help="match report only, write nothing")
    a = ap.parse_args()

    from nanopulldown.core.config import load_config
    from nanopulldown.inputs.expand import prepare

    cfg = load_config()
    cfg.setdefault("controls", {}).setdefault("decoys", {}).update(
        {k: v for k, v in (("n", a.n), ("seed", a.seed)) if v is not None})
    cfg["controls"]["decoys"]["enabled"] = True
    prep = prepare(cfg, rebuild_decoys=True, dry_run=a.dry_run)
    return 0 if prep is not None else 2


if __name__ == "__main__":
    sys.exit(main())
