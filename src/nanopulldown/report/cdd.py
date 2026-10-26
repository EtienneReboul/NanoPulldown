"""
nanopulldown.report.cdd
=======================
PLIP contacts read against each chain's CDD annotation (<annotation dir>/
cdd_sites.parquet, nanopulldown.annotate.cdd), shared by the results table's
CDD columns and the plip-cdd / bait-cdd-map figures.

A residue is "contacted" in a sample when it appears in any PLIP bait-prey
interaction of that sample. Per sample and side (bait / prey, all of that
side's chains together):

  <side>_cdd_sites          CDD site residues on the side (pair constant;
                            0 = CDD annotates no site there)
  <side>_cdd_sites_hit      how many of them PLIP contacts (NaN: no sites)
  <side>_cdd_sites_enrich   fraction of site residues contacted / fraction of
                            all residues contacted -- > 1 when the interface
                            lands on the sites more than its size predicts.
                            Whole chain as background (no solvent
                            accessibility), so buried sites pull it down;
                            NaN with no sites or no contacts
  <side>_contacts_in_cdd    fraction of the contacted residues inside a CDD
                            domain location (NaN: no CDD domain, or no contacts)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

SIDES = ("bait", "prey")
STATS = ("cdd_sites", "cdd_sites_hit", "cdd_sites_enrich", "contacts_in_cdd")
CDD_COLS = [f"{side}_{m}" for side in SIDES for m in STATS]
NAN = float("nan")


def read_cdd(path) -> pd.DataFrame | None:
    """The run's cdd_sites.parquet, or None when there is none."""
    return pd.read_parquet(path) if path and Path(path).exists() else None


def cdd_residues(cdd: pd.DataFrame | None, chains) -> tuple[set, set]:
    """(site residues, CDD-domain residues) of some chains of one pair's rows,
    as (chain, resnr)."""
    sites, spans = set(), set()
    if cdd is None:
        return sites, spans
    for c, site, s, e, res in zip(cdd["chain"], cdd["site"], cdd["start"], cdd["end"], cdd["residues"]):
        if c not in chains:
            continue
        if site is None or pd.isna(site):
            spans |= {(c, r) for r in range(int(s), int(e) + 1)}
        else:
            sites |= {(c, int(r)) for r in res}
    return sites, spans


def contacts_by_sample(pl: pd.DataFrame, chains) -> dict[int, set]:
    """{sample: {(chain, resnr)}} of the PLIP-contacted residues on `chains`."""
    out: dict[int, set] = defaultdict(set)
    for smp, c1, r1, c2, r2 in zip(pl["sample"], pl["reschain"].astype(str), pl["resnr"],
                                   pl["reschain_lig"].astype(str), pl["resnr_lig"]):
        if c1 in chains:
            out[int(smp)].add((c1, int(r1)))
        if c2 in chains:
            out[int(smp)].add((c2, int(r2)))
    return out


def cdd_stats(contacts: set, sites: set, spans: set, n_res: int) -> dict:
    """STATS of one side of one sample -- see the module docstring."""
    hit = len(contacts & sites)
    return {
        "cdd_sites": len(sites),
        "cdd_sites_hit": hit if sites else NAN,
        "cdd_sites_enrich": (hit / len(sites)) / (len(contacts) / n_res) if sites and contacts else NAN,
        "contacts_in_cdd": len(contacts & spans) / len(contacts) if spans and contacts else NAN,
    }


def cdd_columns(plip_path: Path, cdd: pd.DataFrame | None, spec: dict, samples) -> pd.DataFrame:
    """CDD_COLS per sample_index of one pair; all NaN when PLIP never ran for
    it or the run has no CDD table."""
    out = pd.DataFrame({"sample_index": list(samples)})
    if cdd is None or not Path(plip_path).exists():
        return out.assign(**{c: NAN for c in CDD_COLS})
    pl = pd.read_parquet(plip_path, columns=["sample", "resnr", "reschain", "resnr_lig", "reschain_lig"])
    lengths = {c["id"]: len(c["sequence"]) for c in spec.get("sequences", [])}
    vals = {c: [] for c in CDD_COLS}
    for side in SIDES:
        chains = set(spec.get(f"{side}_chains", []))
        sites, spans = cdd_residues(cdd, chains)
        by_smp = contacts_by_sample(pl, chains)
        n_res = sum(lengths.get(c, 0) for c in chains)
        for smp in out["sample_index"]:
            for m, v in cdd_stats(by_smp.get(int(smp), set()), sites, spans, n_res).items():
                vals[f"{side}_{m}"].append(v)
    return out.assign(**vals)


def to_full(pos: int, regions) -> int:
    """Chain position -> full-length UniProt position, through the chain's
    `regions` (1-based, inclusive; None = the chain is the full-length entry)."""
    if not regions:
        return pos
    off = 0
    for a, b in regions:
        n = int(b) - int(a) + 1
        if pos <= off + n:
            return int(a) + pos - off - 1
        off += n
    raise ValueError(f"position {pos} is past the chain's regions {regions}")
