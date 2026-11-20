"""Canonical AIM ([WFY]xx[LIV]) check of AGO1 against the nine ATG8 isoforms, from PLIP contacts + CDD peptide-binding sites.

  * AIM windows: every [WFY]xx[LIV] in AGO1 (O04379); aromatic = i, aliphatic = i+3.
  * A window is "supported" in a sample when the aromatic AND the aliphatic residue each make a PLIP contact with an
    ATG8 residue listed in a CDD "peptide binding site" (1 or 2) of that ATG8.
  * CDD finds no domain in ATG8H (Q8S925), so its site is SYNTHETIC: the CDD sites of the other ATG8s are transferred
    through pairwise alignments and kept when >= half of the donors that carry the site agree (flagged synthetic=True).

Usage: python notebooks/ago1_aim_atg8.py     (writes reports_ago1_autophagy_colabfold/aim_atg8/*)
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

ROOT = Path(__file__).resolve().parents[1]
RES, ANNOT = ROOT / "results_ago1_autophagy_colabfold", ROOT / "data/annotation_ago1_autophagy"
OUT = ROOT / "reports_ago1_autophagy_colabfold/aim_atg8"
BAIT = "O04379"
ATG8 = {"ATG8A": "Q8LEM4", "ATG8B": "Q9XEB5", "ATG8C": "Q8S927", "ATG8D": "Q9SL04", "ATG8E": "Q8S926",
        "ATG8F": "Q8VYK7", "ATG8G": "Q9LZZ9", "ATG8H": "Q8S925", "ATG8I": "Q9LRP7"}
SITES = ("peptide binding site 1", "peptide binding site 2")
SYNTH = "ATG8H"
DONORS = [n for n in ATG8 if n not in (SYNTH, "ATG8I")]  # ATG8I (cd16108) carries no peptide sites


def seq(acc: str) -> str:
    return "".join(l.strip() for l in open(ROOT / f"data/sequences/{acc}.fasta") if not l.startswith(">"))


def cdd_sites() -> pd.DataFrame:
    """One row per (ATG8, site): residues (UniProt numbering), synthetic flag."""
    c = pd.read_parquet(ANNOT / "cdd_sites.parquet")
    c = c[(c.role == "prey") & c.site.isin(SITES)]
    rows = [dict(atg8=n, site=s, residues=sorted(int(x) for x in r), synthetic=False)
            for n, a in ATG8.items() for s in SITES
            for r in c[(c.pair == f"{BAIT}__{a}") & (c.site == s)].residues]
    al = Align.PairwiseAligner(mode="global", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                               open_gap_score=-10, extend_gap_score=-0.5)
    tgt, votes, donors_with = seq(ATG8[SYNTH]), defaultdict(lambda: defaultdict(set)), defaultdict(set)
    ident = {}
    for d in DONORS:
        dseq = seq(ATG8[d]); aln = al.align(dseq, tgt)[0]
        m = {i + 1: j + 1 for (a0, a1), (b0, b1) in zip(*aln.aligned) for i, j in zip(range(a0, a1), range(b0, b1))}
        ident[d] = sum(dseq[i - 1] == tgt[j - 1] for i, j in m.items()) / len(tgt)
        for r in rows:
            if r["atg8"] == d:
                donors_with[r["site"]].add(d)
                for x in r["residues"]:
                    if x in m: votes[r["site"]][m[x]].add(d)
    for s in SITES:
        keep = sorted(p for p, v in votes[s].items() if len(v) >= len(donors_with[s]) / 2)
        rows.append(dict(atg8=SYNTH, site=s, residues=keep, synthetic=True))
    print("ATG8H identity to donors:", {k: round(v, 2) for k, v in ident.items()})
    return pd.DataFrame(rows)


def aim_windows(s: str) -> pd.DataFrame:
    return pd.DataFrame([dict(arom_pos=m.start(1) + 1, aim=m.group(1), arom=m.group(1)[0], aliph_pos=m.start(1) + 4)
                         for m in re.finditer(r"(?=([WFY]..[LIV]))", s)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sites = cdd_sites(); sites.to_parquet(OUT / "atg8_sites.parquet")
    site_res = {(r.atg8, r.site): set(r.residues) for r in sites.itertuples()}
    win = aim_windows(seq(BAIT))
    rows = []
    for name, acc in ATG8.items():
        p = pd.read_parquet(RES / f"{BAIT}__{acc}/plip_contacts.parquet")
        for s in sorted(p["sample"].unique()):
            ps = p[p["sample"] == s]
            for w in win.itertuples():
                def contacts(pos):
                    q = ps[ps.resnr == pos]
                    return {int(r): t for r, t in zip(q.resnr_lig, q.interaction_type)}, set(q.resnr_lig)
                ca, ra = contacts(w.arom_pos); cl, rl = contacts(w.aliph_pos)
                if not (ra or rl): continue
                hit = lambda r: [i + 1 for i, k in enumerate(SITES) if r & site_res.get((name, k), set())]
                rows.append(dict(atg8=name, uniprot=acc, sample=s, aim=w.aim, arom_pos=w.arom_pos, aliph_pos=w.aliph_pos,
                                 arom_atg8_res=sorted(ra), aliph_atg8_res=sorted(rl),
                                 arom_sites=hit(ra), aliph_sites=hit(rl),
                                 arom_in_site=bool(hit(ra)), aliph_in_site=bool(hit(rl)),
                                 synthetic_sites=name == SYNTH))
    d = pd.DataFrame(rows)
    d["supported"] = d.arom_in_site & d.aliph_in_site
    d.to_parquet(OUT / "aim_contacts_per_sample.parquet")
    summ = (d.groupby(["atg8", "aim", "arom_pos"], as_index=False)
            .agg(n_samples_contact=("sample", "nunique"), n_samples_supported=("supported", "sum"),
                 n_arom_in_site=("arom_in_site", "sum"), n_aliph_in_site=("aliph_in_site", "sum"),
                 synthetic_sites=("synthetic_sites", "first")))
    summ["arom_pos"] = summ.arom_pos.astype(int)
    summ = summ.sort_values(["atg8", "n_samples_supported", "n_samples_contact"], ascending=[True, False, False])
    summ.to_parquet(OUT / "aim_summary.parquet"); summ.to_csv(OUT / "aim_summary.csv", index=False)
    print(f"{len(win)} [WFY]xx[LIV] windows in AGO1")
    print(summ[summ.n_samples_contact > 0].to_string(index=False))
    print("\nsites:\n", sites.assign(n=sites.residues.map(len)).to_string(index=False))
    # all windows x ATG8: supported samples (matrix)
    mat = summ.pivot_table(index=["arom_pos", "aim"], columns="atg8", values="n_samples_supported", fill_value=0)
    mat.to_csv(OUT / "aim_supported_matrix.csv"); print("\n", mat.to_string())


if __name__ == "__main__":
    main()
