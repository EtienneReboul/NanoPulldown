"""
nanopulldown.core.metric_refs
=============================
The one place every reported score is defined and cited: what it measures,
whether it scores the whole predicted fold or the bait-prey interface, and
the publication(s) behind it. Read by the Parquet writers (each score column
carries its references in its field metadata, see
core.io.write_parquet_with_metadata), the metric-distribution violins (one
section per scope, references under each) and the Snakefile (the report
caption of those pages), so the three can't cite differently.

Every DOI below was resolved against Crossref on 2026-10-27; where a preprint
has since been published (Pinc), the journal version is cited.

Scopes:
  fold        confidence in the predicted complex as a whole (pTM, pLDDT)
  interface   scores of the bait-prey interface: PAE-derived confidence
              (ipTM, ipSAE, LIS family, Pinc, pDockQ), PLIP contacts and the
              pyDock energy / combined z-score
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ref:
    short: str        # "Jumper et al. 2021"
    full: str         # authors. title. journal volume:pages (year).
    doi: str

    @property
    def url(self) -> str:
        return f"https://doi.org/{self.doi}"

    def __str__(self) -> str:
        return f"{self.full} {self.url}"


REFS = {
    "jumper2021": Ref("Jumper et al. 2021",
                      "Jumper J, et al. Highly accurate protein structure prediction with AlphaFold. "
                      "Nature 596:583-589 (2021).", "10.1038/s41586-021-03819-2"),
    "mariani2013": Ref("Mariani et al. 2013",
                       "Mariani V, Biasini M, Barbato A, Schwede T. lDDT: a local superposition-free score "
                       "for comparing protein structures and models using distance difference tests. "
                       "Bioinformatics 29:2722-2728 (2013).", "10.1093/bioinformatics/btt473"),
    "zhang2004": Ref("Zhang & Skolnick 2004",
                     "Zhang Y, Skolnick J. Scoring function for automated assessment of protein structure "
                     "template quality. Proteins 57:702-710 (2004).", "10.1002/prot.20264"),
    "evans2021": Ref("Evans et al. 2021",
                     "Evans R, et al. Protein complex prediction with AlphaFold-Multimer. bioRxiv (2021).",
                     "10.1101/2021.10.04.463034"),
    "abramson2024": Ref("Abramson et al. 2024",
                        "Abramson J, et al. Accurate structure prediction of biomolecular interactions with "
                        "AlphaFold 3. Nature 630:493-500 (2024).", "10.1038/s41586-024-07487-w"),
    "dunbrack2025": Ref("Dunbrack 2025",
                        "Dunbrack RL. Rēs ipSAE loquuntur: What's wrong with AlphaFold's ipTM score and how "
                        "to fix it. bioRxiv (2025).", "10.1101/2025.02.10.637595"),
    "kim2024": Ref("Kim et al. 2024",
                   "Kim AR, Hu Y, Comjean A, Rodiger J, Mohr SE, Perrimon N. Enhanced protein-protein "
                   "interaction discovery via AlphaFold-Multimer. bioRxiv (2024).",
                   "10.1101/2024.02.19.580970"),
    "kim2026": Ref("Kim et al. 2026",
                   "Kim AR, Comjean A, Veal A, Rodiger J, Han M, Hu Y, Perrimon N. FlyPredictome: a "
                   "structural atlas of predicted protein-protein interactions in Drosophila. "
                   "bioRxiv (2026).", "10.64898/2026.04.14.718529"),
    "badonyi2026": Ref("Badonyi & Toth-Petroczy 2026",
                       "Badonyi M, Toth-Petroczy A. A simple probabilistic AlphaFold interaction score. "
                       "Protein Science 35:e70760 (2026).", "10.1002/pro.70760"),
    "bryant2022": Ref("Bryant et al. 2022",
                      "Bryant P, Pozzati G, Elofsson A. Improved prediction of protein-protein interactions "
                      "using AlphaFold2. Nature Communications 13:1265 (2022).", "10.1038/s41467-022-28865-w"),
    "zhu2023": Ref("Zhu et al. 2023",
                   "Zhu W, Shenoy A, Kundrotas P, Elofsson A. Evaluation of AlphaFold-Multimer prediction "
                   "on multi-chain protein complexes. Bioinformatics 39:btad424 (2023).",
                   "10.1093/bioinformatics/btad424"),
    "salentin2015": Ref("Salentin et al. 2015",
                        "Salentin S, Schreiber S, Haupt VJ, Adasme MF, Schroeder M. PLIP: fully automated "
                        "protein-ligand interaction profiler. Nucleic Acids Research 43:W443-W447 (2015).",
                        "10.1093/nar/gkv315"),
    "adasme2021": Ref("Adasme et al. 2021",
                      "Adasme MF, et al. PLIP 2021: expanding the scope of the protein-ligand interaction "
                      "profiler to DNA and RNA. Nucleic Acids Research 49:W530-W534 (2021).",
                      "10.1093/nar/gkab294"),
    "cheng2007": Ref("Cheng et al. 2007",
                     "Cheng TMK, Blundell TL, Fernandez-Recio J. pyDock: electrostatics and desolvation for "
                     "effective scoring of rigid-body protein-protein docking. Proteins 68:503-515 (2007).",
                     "10.1002/prot.21419"),
    "jimenezgarcia2018": Ref("Jiménez-García et al. 2018",
                             "Jiménez-García B, et al. LightDock: a new multi-scale approach to "
                             "protein-protein docking. Bioinformatics 34:49-55 (2018).",
                             "10.1093/bioinformatics/btx555"),
    "rodriguezlumbreras2026": Ref("Rodríguez-Lumbreras et al. 2026",
                                  "Rodríguez-Lumbreras LA, et al. Performance of pyDock in 8th CAPRI: "
                                  "energy-based scoring applied to docking and AlphaFold models. "
                                  "Proteins 94:1827-1835 (2026).", "10.1002/prot.70038"),
}


@dataclass(frozen=True)
class Metric:
    label: str
    scope: str        # a SCOPES key
    definition: str
    refs: tuple[str, ...]


SCOPES = {
    "fold": ("Global fold confidence",
             "Confidence in the predicted complex as a whole: high values mean a well-predicted "
             "structure, not necessarily an interaction."),
    "interface": ("Bait-prey interface",
                  "Scores of the bait-prey interface only: PAE-derived interface confidence, PLIP "
                  "contacts and pyDock energy of the minimized model."),
}

_PLIP = ("salentin2015", "adasme2021")
_PYDOCK = ("cheng2007", "jimenezgarcia2018")

METRICS = {
    "ptm": Metric("pTM", "fold",
                  "predicted TM-score of the whole complex (0-1, higher = better).",
                  ("jumper2021", "zhang2004")),
    "mean_plddt": Metric("mean pLDDT", "fold",
                         "mean per-residue predicted lDDT over the complex (0-1, higher = better).",
                         ("jumper2021", "mariani2013")),
    "iptm": Metric("ipTM", "interface",
                   "interface pTM over all inter-chain residue pairs of the complex (0-1).",
                   ("evans2021",)),
    "pair_iptm": Metric("pair ipTM", "interface",
                        "ipTM of the bait-prey chain pair alone, from the engine's chain-pair ipTM matrix (0-1).",
                        ("evans2021", "abramson2024")),
    "ipsae": Metric("ipSAE", "interface",
                    "ipTM-like score restricted to residues with inter-chain PAE under the cutoff, "
                    "max of the two directions (ipsae.py v4; 0-1).",
                    ("dunbrack2025",)),
    "lis": Metric("LIS", "interface",
                  "Local Interaction Score: mean (cutoff - PAE) / cutoff over inter-chain residue pairs "
                  "with PAE <= 12 A (0-1).",
                  ("kim2024",)),
    "clis": Metric("cLIS", "interface",
                   "contact LIS: LIS over the residue pairs that are also within 8 A (Cb-Cb) (0-1).",
                   ("kim2026",)),
    "ilis": Metric("iLIS", "interface",
                   "integrated LIS, sqrt(LIS x cLIS) (0-1).",
                   ("kim2026", "kim2024")),
    "pinc": Metric("Pinc", "interface",
                   "probability of interface native contacts: mean PAE-derived contact probability over "
                   "inter-chain residue pairs within 12 A (0-1).",
                   ("badonyi2026",)),
    "pdockq": Metric("pDockQ", "interface",
                     "predicted DockQ from interface pLDDT and contact count (0-1).",
                     ("bryant2022",)),
    "pdockq2": Metric("pDockQ2", "interface",
                      "predicted DockQ from interface pLDDT and PAE (0-1).",
                      ("zhu2023",)),
    "plip_contacts": Metric("PLIP contacts", "interface",
                            "bait-prey non-covalent interactions PLIP detects in the minimized model (count).",
                            _PLIP),
    **{col: Metric(label, "interface",
                   f"bait-prey {label} PLIP detects in the minimized model (count).", _PLIP)
       for col, label in [("plip_hbond", "hydrogen bonds"), ("plip_hydrophobic", "hydrophobic contacts"),
                          ("plip_salt_bridge", "salt bridges"), ("plip_water_bridge", "water bridges"),
                          ("plip_pi_stacking", "π-stacking interactions"),
                          ("plip_pi_cation", "π-cation interactions"),
                          ("plip_halogen", "halogen bonds"), ("plip_metal", "metal complexes")]},
    "pydock_elec": Metric("pyDock elec", "interface",
                          "pyDock electrostatics of the minimized model (kcal/mol).", _PYDOCK),
    "pydock_vdw": Metric("pyDock vdW", "interface",
                         "pyDock van der Waals of the minimized model (kcal/mol).", _PYDOCK),
    "pydock_desolv": Metric("pyDock desolv", "interface",
                            "pyDock ASA-based desolvation of the minimized model (kcal/mol).", _PYDOCK),
    "pydock": Metric("pyDock", "interface",
                     "pyDock binding energy elec + desolv + 0.1 vdW (kcal/mol, lower = better).", _PYDOCK),
    "pydock_1vdw": Metric("pyDock-1VDW", "interface",
                          "pyDock binding energy elec + desolv + 1.0 vdW, the weighting used for AlphaFold "
                          "models (kcal/mol, lower = better).",
                          (*_PYDOCK, "rodriguezlumbreras2026")),
    "af_mc": Metric("AF model confidence", "interface",
                    "0.8 ipTM + 0.2 pTM (0-1).", ("evans2021", "rodriguezlumbreras2026")),
    "z_af_pydock_run": Metric("Z(AF) - Z(pyDock-1VDW), run", "interface",
                              "Z(0.8 ipTM + 0.2 pTM) - Z(pyDock-1VDW), z-scores over every sample of the run "
                              "(higher = better).",
                              ("rodriguezlumbreras2026",)),
    "z_af_pydock_pair": Metric("Z(AF) - Z(pyDock-1VDW), pair", "interface",
                               "Z(0.8 ipTM + 0.2 pTM) - Z(pyDock-1VDW), z-scores over the samples of one pair "
                               "(higher = better).",
                               ("rodriguezlumbreras2026",)),
}


# the metric-distribution page's panels (report.plots.violins default, and its caption)
REPORT_METRICS = ["ptm", "mean_plddt", "iptm", "pair_iptm", "ipsae", "lis", "clis", "ilis", "pinc",
                  "pdockq", "pdockq2", "plip_contacts", "pydock_1vdw", "z_af_pydock_run"]


def refs_of(column: str) -> list[Ref]:
    m = METRICS.get(column)
    return [REFS[k] for k in m.refs] if m else []


def references_text(column: str) -> str:
    """The column's references as one plain string (Parquet field metadata)."""
    return " | ".join(str(r) for r in refs_of(column))


def column_references(columns) -> dict[str, str]:
    """{column: references_text} for the cited columns among `columns`."""
    return {c: references_text(c) for c in columns if c in METRICS}


def by_scope(columns) -> list[tuple[str, list[str]]]:
    """[(scope, columns)] in SCOPES order, then ("other", uncited columns)."""
    out = [(s, [c for c in columns if c in METRICS and METRICS[c].scope == s]) for s in SCOPES]
    other = [c for c in columns if c not in METRICS]
    return [(s, cs) for s, cs in out + [("other", other)] if cs]


def scope_title(scope: str) -> str:
    return SCOPES[scope][0] if scope in SCOPES else "Other"


def caption_rst(columns, intro: str = "") -> str:
    """reStructuredText caption for a page of `columns`: one section per scope
    with each score's definition and citation, then the full references."""
    lines = [intro, ""] if intro else []
    cited: list[str] = []
    for scope, cols in by_scope(columns):
        title = scope_title(scope)
        lines += [f"**{title}**" + (f" — {SCOPES[scope][1]}" if scope in SCOPES else ""), ""]
        for c in cols:
            m = METRICS.get(c)
            if m is None:
                lines.append(f"* ``{c}``")
                continue
            cites = "; ".join(f"`{REFS[k].short} <{REFS[k].url}>`__" for k in m.refs)
            lines.append(f"* **{m.label}** (``{c}``) — {m.definition} {cites}")
            cited += [k for k in m.refs if k not in cited]
        lines.append("")
    if cited:
        lines += ["**References**", ""]
        lines += [f"#. {REFS[k].full} `doi:{REFS[k].doi} <{REFS[k].url}>`__" for k in cited]
    return "\n".join(lines) + "\n"


def references_html(columns) -> str:
    """HTML list of each cited column's label, definition and DOI links."""
    import html
    items = []
    for c in columns:
        m = METRICS.get(c)
        if m is None:
            continue
        cites = "; ".join(f'<a href="{REFS[k].url}">{html.escape(REFS[k].short)}</a>' for k in m.refs)
        items.append(f"<li><b>{html.escape(m.label)}</b> — {html.escape(m.definition)} {cites}</li>")
    return f"<ul class=\"refs\">{''.join(items)}</ul>" if items else ""
