#!/usr/bin/env python3
"""
nanopulldown.annotate.idr_pairs — Stage 1f
==========================================
Are the disordered regions of bait and prey physicochemically complementary?
Motif-independent IDR-IDR binding (fuzzy complexes, electrostatic coupling,
co-condensation; e.g. histone H1 (+) x prothymosin alpha (-), Borgia et al.
2018 Nature 555:61) leaves no ELM regex and no domain to dock: this ranks
pairs by opposite charge blocks, aromatic / cation-pi stickers and
hydrophobic patches, against two null models. It complements annotate.elm
(motif x domain) and the AIUPred MoRFs. Ranks candidates; it is not a
binding energy.

  reference  (rule fetch_idr_reference, web; login node on CAIUS) a background
             sample of `background.n` proteins of `background.taxon`
             (UniProt query snapshot + FASTA) and the control proteins' sequences.
  main       (rule idr_compatibility, envs/sparrow.yaml, offline)
    1. IDRs: AIUPred disorder >= cutoff, ordered gaps <= max_gap merged,
       >= min_len aa, called on the FULL-LENGTH protein and projected onto
       fragment chains (pieces >= min_len kept). metapredict (SPARROW's
       predictor) is run on the same proteins: `metapredict_agreement` is the
       fraction of each IDR's residues it also calls disordered.
    2. Per-IDR SPARROW features (charge, patterning, hydropathy, stickers,
       PScore) + ALBATROSS ensemble dimensions (batch_predict).
    3. Per pair, every bait-side IDR x prey-side IDR: the scores of
       nanopulldown.annotate.idr_scores, the best complementary segments.
    4. Nulls: n_shuffles composition-preserving shuffles of each IDR -> z / p
       of the LOCAL scores (does the patterning matter?); every score of
       IDR_a x background and background x IDR_b -> percentile (is the pair
       more complementary than a random partner of either?).
    5. Phosphorylation (optional): sites from a TSV and/or SPARROW's S/T/Y
       predictors, pS/pT/pY = -2: the charge scores recomputed.
    6. Controls: listed protein pairs scored the same way (best IDR pair
       each), with pass/fail against their expectation.

Outputs:
  <annotation>/idr_features.parquet     one row per IDR (run, background, control)
  <annotation>/idr_pair_scores.parquet  one row per (pair, bait IDR, prey IDR)
  <reports>/preprocessing/idr_pair_scores.csv  the same, ranked run-wide, display names
  <reports>/preprocessing/idr_controls.csv     the controls, with pass/fail
  <reports>/preprocessing/idr_pairs/    mechanism heatmap + profile figure of the top pairs

Usage:
  nanopulldown annotate idr-reference --background <fasta> --controls <fasta>
  nanopulldown annotate idr-pairs --pairs <pair> [...] --background <fasta> --controls <fasta> \\
      --features <parquet> --scores <parquet> --csv <csv> --controls-csv <csv> --fig-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from nanopulldown.annotate import idr_scores as S
from nanopulldown.core.config import LazyConfig
from nanopulldown.core.io import atomic_write, seq_md5

CFG = LazyConfig()

REFERENCES = ("Emenecker RJ et al. (2026) SPARROW / ALBATROSS (idptools, Holehouse lab); "
              "Lotthammer JM et al. (2024) Nat Methods 21:465 (ALBATROSS); "
              "Erdos G & Dosztanyi Z (2024) NAR 52:W176 (AIUPred); "
              "Borgia A et al. (2018) Nature 555:61 (H1-ProTalpha positive control)")

DEFAULT_CONTROLS = [
    # Borgia 2018 used H1.0; H1.4 is the somatic H1 the original spec named
    {"name": "H1.4 x ProTalpha", "a": "P10412", "b": "P06454", "expect": "high"},
    {"name": "H1.0 x ProTalpha", "a": "P07305", "b": "P06454", "expect": "high"},
    {"name": "H1.4 x H1.0 (both basic)", "a": "P10412", "b": "P07305", "expect": "low"},
    {"name": "ProTalpha x parathymosin (both acidic)", "a": "P06454", "b": "P20962", "expect": "low"},
]


def icfg() -> dict:
    return CFG["annotate"].get("idr_pairs") or {}


def _read_fasta(path: Path) -> dict[str, str]:
    out, name = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            out[name] = ""
        elif name:
            out[name] += line.strip()
    return out


# ── reference: background sample + control sequences (web) ───────────────

def reference_main() -> int:
    from nanopulldown.services.uniprot import _get, fetch_sequence, run_query

    ap = argparse.ArgumentParser(description="background IDR proteome sample + control sequences")
    ap.add_argument("--background", required=True)
    ap.add_argument("--controls", required=True)
    a = ap.parse_args()
    c = icfg()
    bg = c.get("background") or {}
    taxon, n, seed = int(bg.get("taxon", 3702)), int(bg.get("n", 500)), int(bg.get("seed", 0))
    max_len = int(bg.get("max_len", 2000))
    query = f"(organism_id:{taxon})" + (" AND (reviewed:true)" if bg.get("reviewed", True) else "")
    accs = sorted(run_query(f"idr_background_{taxon}", query, CFG))
    rng = random.Random(seed)
    order = rng.sample(accs, len(accs))         # deterministic shuffle; take the first n usable
    # ONE stream of the whole query as FASTA (accession OR-lists draw 429s from UniProt)
    ucfg = {**CFG["uniprot"], "timeout_s": 600}
    r = _get(ucfg, f"{CFG['uniprot']['base_url'].rstrip('/')}/stream", {"query": query, "format": "fasta"})
    seqs = {k: s for k, s in _parse_uniprot_fasta(r.text).items()
            if 30 <= len(s) <= max_len and set(s) <= set(S.KD)}
    picked = [x for x in order if x in seqs][:n]
    Path(a.background).parent.mkdir(parents=True, exist_ok=True)
    atomic_write(Path(a.background), "".join(f">{x}\n{seqs[x]}\n" for x in picked))
    print(f"[idr] background: {len(picked)} proteins of taxon {taxon} (of {len(accs)} matching {query!r}, "
          f"seed {seed}, standard residues, <= {max_len} aa)", flush=True)

    ctl_accs = sorted({x for ctl in (c.get("controls") or DEFAULT_CONTROLS) for x in (ctl["a"], ctl["b"])})
    atomic_write(Path(a.controls), "".join(f">{x}\n{fetch_sequence(x, CFG)}\n" for x in ctl_accs))
    print(f"[idr] controls: {', '.join(ctl_accs)}", flush=True)
    return 0


def _parse_uniprot_fasta(text: str) -> dict[str, str]:
    out, name = {}, None
    for line in text.splitlines():
        if line.startswith(">"):
            name = line.split("|")[1] if "|" in line else line[1:].split()[0]
            out[name] = ""
        elif name:
            out[name] += line.strip()
    return out


# ── proteins, chains, IDRs ────────────────────────────────────────────────

class Protein:
    """A full-length sequence with its AIUPred tracks, IDRs and MoRFs."""

    def __init__(self, seq: str, acc: str | None, tracks: dict, c: dict, cutoff: float, morf: dict):
        from nanopulldown.annotate.elm import morf_segments
        self.seq, self.acc, self.md5 = seq, acc, seq_md5(seq)
        self.disorder = np.asarray(tracks.get("disorder") or [], dtype=float)
        bind = tracks.get("binding") or []
        self.morfs = morf_segments(bind, float(morf["score_threshold"]), int(morf["min_len"])) if bind else []
        ok = len(self.disorder) == len(seq)
        self.idrs = S.define_idrs(self.disorder, cutoff, int(c.get("max_gap", 10)),
                                  int(c.get("min_len", 30))) if ok else []
        self.metapredict = None
        self.phospho: set[int] = set()            # full-length 1-based positions


def chain_idrs(prot: Protein, regions, chain_id: str, role: str, min_len: int) -> list[dict]:
    """The protein's IDRs on one chain: full-length IDRs restricted to the
    chain's regions (None = the whole protein), pieces >= min_len."""
    full_pos = (np.concatenate([np.arange(a, b + 1) for a, b in regions]) if regions
                else np.arange(1, len(prot.seq) + 1))
    chain_seq = "".join(prot.seq[p - 1] for p in full_pos)
    out = []
    for s, e in prot.idrs:
        inside = np.nonzero((full_pos >= s) & (full_pos <= e))[0]
        if not len(inside):
            continue
        # contiguous runs of chain positions
        for run in np.split(inside, np.nonzero(np.diff(inside) != 1)[0] + 1):
            if len(run) < min_len:
                continue
            cs, ce = int(run[0]) + 1, int(run[-1]) + 1
            fp = full_pos[run]
            out.append({
                "chain": chain_id, "role": role, "accession": prot.acc, "protein": prot,
                "start": cs, "end": ce, "full_start": int(fp[0]), "full_end": int(fp[-1]),
                "full_pos": fp, "sequence": chain_seq[cs - 1:ce],
                "phospho_idx": [k for k, p in enumerate(fp) if int(p) in prot.phospho],
                "morfs": [(m0, m1) for m0, m1 in _project(prot.morfs, full_pos)],
            })
    return out


def _project(segs, full_pos) -> list[tuple[int, int]]:
    """Full-length segments -> chain-numbered pieces."""
    out = []
    for a, b in segs:
        idx = np.nonzero((full_pos >= a) & (full_pos <= b))[0]
        for run in np.split(idx, np.nonzero(np.diff(idx) != 1)[0] + 1) if len(idx) else []:
            out.append((int(run[0]) + 1, int(run[-1]) + 1))
    return out


# ── SPARROW features ──────────────────────────────────────────────────────

FEATURE_COLUMNS = {
    "set": "run (a protein of the screened pairs) | background (random proteome sample) | control",
    "idr_id": "md5 of the IDR sequence, first 12 hex (key of idr_pair_scores a_/b_idr_id)",
    "accession": "UniProt accession of the full-length protein (null: no plain accession)",
    "protein_md5": "md5 of the full-length sequence (annotation cache key)",
    "full_start": "first IDR residue, full-length numbering (1-based)",
    "full_end": "last IDR residue, full-length numbering (inclusive)",
    "length": "IDR length (aa)",
    "sequence": "IDR sequence",
    "mean_disorder": "mean AIUPred disorder over the IDR",
    "metapredict_agreement": "fraction of the IDR's residues metapredict v3 (SPARROW predictor.disorder) also "
                             "scores >= 0.5; low values flag an AIUPred / metapredict disagreement",
    "morf_residues": "IDR residues inside an AIUPred MoRF (binding track, annotate.morfchibi)",
    "fcr": "SPARROW FCR: fraction of charged residues (K R D E)",
    "ncpr": "SPARROW NCPR: net charge per residue",
    "f_positive": "SPARROW fraction_positive (K R)",
    "f_negative": "SPARROW fraction_negative (D E)",
    "kappa": "SPARROW kappa (Das & Pappu 2013): charge segregation, 0 well mixed .. 1 two blocks; "
             "-1 when one charge type is absent",
    "scd": "SPARROW SCD (Sawle & Ghosh 2015): sequence charge decoration; more negative = blockier "
           "opposite charges (intramolecular attraction)",
    "hydrophobicity": "SPARROW mean Kyte-Doolittle hydrophobicity, shifted 0..9 scale",
    "shd": "SPARROW SHD (Zheng et al. 2020): sequence hydropathy decoration (clustered hydrophobics)",
    "complexity": "SPARROW Wootton-Federhen (SEG) compositional complexity",
    "f_aromatic": "SPARROW fraction_aromatic (F Y W)",
    "f_R": "fraction R", "f_K": "fraction K", "f_QN": "fraction Q + N", "f_GS": "fraction G + S",
    "n_Y": "count of Y", "n_R": "count of R",
    "pscore_mean": "mean of SPARROW predictor.pscore (PScore, Vernon et al. 2018: pi-pi contact "
                   "phase-separation propensity, normalized)",
    "rg": "ALBATROSS radius of gyration (A; scaled network below 35 aa)",
    "re": "ALBATROSS end-to-end distance (A)",
    "nu": "ALBATROSS scaling exponent: ~0.5 theta / compact (strong intramolecular interactions), "
          "~0.6 self-avoiding expanded",
    "asphericity": "ALBATROSS asphericity (0 sphere .. 1 rod)",
    "prefactor": "ALBATROSS polymer scaling prefactor (A)",
    "n_phospho": "phosphosites inside the IDR (annotate.idr_pairs.phosphosites / predict_phosphosites)",
}


def _window_feature_columns(windows) -> dict[str, str]:
    out = {}
    for w in windows:
        out[f"ncpr_max_w{w}"] = f"max NCPR over {w}-residue windows of the IDR (most basic block)"
        out[f"ncpr_min_w{w}"] = f"min NCPR over {w}-residue windows (most acidic block)"
        out[f"arom_max_w{w}"] = f"max aromatic (F Y W) fraction over {w}-residue windows"
        out[f"cat_max_w{w}"] = f"max R + K fraction over {w}-residue windows"
        out[f"hyd_max_w{w}"] = f"max mean hydrophobicity (Kyte-Doolittle scaled 0..1) over {w}-residue windows"
    return out


def sparrow_features(idr_seqs: dict[str, str]) -> dict[str, dict]:
    """{idr_id: SPARROW + ALBATROSS features} for unique IDR sequences."""
    from sparrow import Protein as SP
    from sparrow.predictors.batch_predict import batch_predict

    out = {}
    for k, s in idr_seqs.items():
        p = SP(s)
        fr = p.amino_acid_fractions
        out[k] = {"fcr": p.FCR, "ncpr": p.NCPR, "f_positive": p.fraction_positive,
                  "f_negative": p.fraction_negative, "kappa": p.kappa, "scd": p.SCD,
                  "hydrophobicity": p.hydrophobicity, "shd": p.SHD, "complexity": p.complexity,
                  "f_aromatic": p.fraction_aromatic, "f_R": fr.get("R", 0.0), "f_K": fr.get("K", 0.0),
                  "f_QN": fr.get("Q", 0.0) + fr.get("N", 0.0), "f_GS": fr.get("G", 0.0) + fr.get("S", 0.0),
                  "n_Y": s.count("Y"), "n_R": s.count("R"),
                  "pscore_mean": float(np.mean(p.predictor.pscore()))}
    for net, col in (("rg", "rg"), ("re", "re"), ("scaling_exponent", "nu"),
                     ("asphericity", "asphericity"), ("prefactor", "prefactor")):
        for k, (_, v) in batch_predict(idr_seqs, network=net, show_progress_bar=False).items():
            out[k][col] = float(v)
    return {k: {c: (float(v) if isinstance(v, (float, np.floating)) else v) for c, v in d.items()}
            for k, d in out.items()}


def predicted_phosphosites(seq: str) -> set[int]:
    """1-based S/T/Y positions SPARROW's phosphorylation predictors call."""
    from sparrow import Protein as SP
    pr = SP(seq).predictor
    hits = set()
    for fn in (pr.serine_phosphorylation, pr.threonine_phosphorylation, pr.tyrosine_phosphorylation):
        hits |= {i + 1 for i, v in enumerate(fn()) if v}
    return hits


# ── pair scoring ──────────────────────────────────────────────────────────

def score_columns(windows) -> dict[str, str]:
    w0 = windows[0]
    cols = {
        "pair": "pair id (control name for idr_controls)",
        "rank": "rank of this IDR pair within the pair (1 = best): max_mechanism_pct, then local_charge",
        "dominant_mechanism": "charge | cation_pi | pi | hydrophobic: the mechanism with the highest background "
                              "percentile when >= annotate.idr_pairs.label_percentile (raw score > 0), else none",
        "max_mechanism_pct": "highest mechanism background percentile (ranking key)",
    }
    for side, role in (("a", "bait"), ("b", "prey")):
        cols |= {
            f"{side}_chain": f"{role}-side chain id", f"{side}_accession": f"UniProt accession of the {role}-side protein",
            f"{side}_idr_id": f"{role}-side IDR (idr_features.idr_id)",
            f"{side}_start": f"first {role}-side IDR residue, chain numbering",
            f"{side}_end": f"last {role}-side IDR residue, chain numbering",
            f"{side}_full_start": f"first {role}-side IDR residue, full-length numbering",
            f"{side}_full_end": f"last {role}-side IDR residue, full-length numbering",
            f"{side}_length": f"{role}-side IDR length",
            f"ncpr_{side}": f"NCPR of the {role}-side IDR", f"kappa_{side}": f"kappa of the {role}-side IDR",
            f"scd_{side}": f"SCD of the {role}-side IDR", f"shd_{side}": f"SHD of the {role}-side IDR",
            f"hydrophobicity_{side}": f"mean KD hydrophobicity (0..9) of the {role}-side IDR",
            f"nu_{side}": f"ALBATROSS scaling exponent of the {role}-side IDR",
        }
    cols |= {
        "s_charge": "global charge complementarity -NCPR_a * NCPR_b (> 0: opposite net charges)",
        "fpos_a_x_fneg_b": "f+ (bait IDR) x f- (prey IDR)", "fneg_a_x_fpos_b": "f- (bait IDR) x f+ (prey IDR)",
        "s_charge_pct_bg": "background percentile of s_charge (IDR_a x background and background x IDR_b)",
    }
    for w in windows:
        cols |= {
            f"local_charge_w{w}": f"best local charge complementarity, {w}-residue windows: "
                                  f"max_ij -NCPR_win_a[i] * NCPR_win_b[j]",
            f"a_seg_start_w{w}": f"first residue (chain numbering) of the bait-side window of local_charge_w{w}",
            f"a_seg_end_w{w}": "last residue of that bait-side window",
            f"a_seg_w{w}": "sequence of that bait-side window",
            f"b_seg_start_w{w}": f"first residue (chain numbering) of the prey-side window of local_charge_w{w}",
            f"b_seg_end_w{w}": "last residue of that prey-side window",
            f"b_seg_w{w}": "sequence of that prey-side window",
            f"local_charge_w{w}_z_shuffle": f"z-score of local_charge_w{w} against the shuffle null "
                                            f"(both IDRs shuffled, n_shuffles): > 0 = patterning helps",
            f"local_charge_w{w}_p_shuffle": f"one-sided empirical p of local_charge_w{w} under the shuffle null",
            f"local_charge_w{w}_pct_bg": f"background percentile of local_charge_w{w}",
        }
    cols |= {
        "a_seg_morf": f"bait-side local_charge_w{w0} window residues inside an AIUPred MoRF",
        "b_seg_morf": f"prey-side local_charge_w{w0} window residues inside an AIUPred MoRF",
        "blocky_both": "both IDRs have kappa >= annotate.idr_pairs.kappa_blocky (blocky charges bind more "
                       "strongly than evenly spread ones)",
        "s_catpi": "cation-pi: f_arom_a * f_RK_b + f_arom_b * f_RK_a (arom = F Y W)",
        "s_catpi_pct_bg": "background percentile of s_catpi",
        "local_catpi": f"max over {w0}-residue windows: aromatic block of one IDR x R/K block of the other",
        "local_catpi_z_shuffle": "z-score of local_catpi, shuffle null",
        "local_catpi_pct_bg": "background percentile of local_catpi",
        "s_pi": "pi-pi: f_arom_a * f_arom_b", "s_pi_pct_bg": "background percentile of s_pi",
        "local_pi": f"max {w0}-window aromatic fraction of a x that of b",
        "local_pi_z_shuffle": "z-score of local_pi, shuffle null",
        "local_pi_pct_bg": "background percentile of local_pi",
        "hydro_local": f"max {w0}-window hydrophobicity (KD 0..1) of a x that of b",
        "hydro_local_z_shuffle": "z-score of hydro_local, shuffle null",
        "hydro_local_pct_bg": "background percentile of hydro_local",
        "hydro_patch_both": "both IDRs have a window at or above the annotate.idr_pairs.hydro_patch_percentile "
                            "quantile of the background IDRs' max window hydrophobicity",
        **{f"pct_{m}": f"background percentile of the {m} mechanism (max of its scores' percentiles, "
                       f"scores > 0 only)" for m in S.MECHANISMS},
        "a_mech_start": "first residue (chain numbering) of the bait-side window behind dominant_mechanism",
        "a_mech_end": "last residue of that window",
        "b_mech_start": "first residue (chain numbering) of the prey-side window behind dominant_mechanism",
        "b_mech_end": "last residue of that window",
        "n_phospho_a": "phosphosites in the bait-side IDR (null: phosphorylation not configured)",
        "n_phospho_b": "phosphosites in the prey-side IDR",
        "s_charge_phospho": "s_charge with every phosphosite -2",
        f"local_charge_w{w0}_phospho": f"local_charge_w{w0} with every phosphosite -2",
        "d_s_charge_phospho": "s_charge_phospho - s_charge (> 0: phosphorylation strengthens the complementarity)",
        f"d_local_charge_w{w0}_phospho": f"local_charge_w{w0}_phospho - local_charge_w{w0}",
    }
    return cols


class Scorer:
    def __init__(self, c: dict, bg_idrs: list[dict], feats: dict):
        self.windows = [int(w) for w in c.get("windows", [15, 25])]
        self.w0 = self.windows[0]
        self.n_shuf = int(c.get("n_shuffles", 1000))
        self.min_pct = float(c.get("label_percentile", 0.95))
        self.kappa_blocky = float(c.get("kappa_blocky", 0.3))
        self.feats = feats
        self._sum, self._shuf = {}, {}
        self.bg = S.stack_summaries([self.summary(d["sequence"]) for d in bg_idrs], self.windows)
        self.hyd_patch = float(np.quantile(self.bg[f"hyd_max_{self.w0}"],
                                           float(c.get("hydro_patch_percentile", 0.9))))

    def summary(self, seq: str, phospho=()) -> dict:
        key = (seq, tuple(phospho))
        if key not in self._sum:
            self._sum[key] = S.summary(S.property_vectors(seq, phospho), self.windows)
        return self._sum[key]

    def shuffled(self, seq: str) -> dict:
        if seq not in self._shuf:
            self._shuf[seq] = S.shuffle_summary(seq, self.n_shuf, self.windows)
        return self._shuf[seq]

    def score(self, a: dict, b: dict) -> dict:
        """One IDR pair (chain_idrs() records): every column of score_columns
        except pair / rank and the a_/b_ identity columns."""
        sa, sb = self.summary(a["sequence"]), self.summary(b["sequence"])
        ha, hb = self.shuffled(a["sequence"]), self.shuffled(b["sequence"])
        if a["sequence"] == b["sequence"]:          # same IDR on both sides: decorrelate the shuffles
            hb = {k: np.roll(v, 1) for k, v in hb.items()}
        fa, fb = self.feats[seq_md5(a["sequence"])[:12]], self.feats[seq_md5(b["sequence"])[:12]]
        row = {"ncpr_a": sa["ncpr"], "ncpr_b": sb["ncpr"],
               "fpos_a_x_fneg_b": sa["fpos"] * sb["fneg"], "fneg_a_x_fpos_b": sa["fneg"] * sb["fpos"]}
        for side, f in (("a", fa), ("b", fb)):
            row |= {f"kappa_{side}": f["kappa"], f"scd_{side}": f["scd"], f"shd_{side}": f["shd"],
                    f"hydrophobicity_{side}": f["hydrophobicity"], f"nu_{side}": f["nu"]}
        pcts, raw = {}, {}
        for w in self.windows:
            obs = S.pair_scores(sa, sb, w)
            null_bg = {k: np.concatenate([v, S.pair_scores(self.bg, sb, w)[k]])
                       for k, v in S.pair_scores(sa, self.bg, w).items()}
            null_sh = S.pair_scores(ha, hb, w)
            i, j = S.local_charge_windows(sa, sb, w)
            z, p = S.shuffle_null(obs["local_charge"], null_sh["local_charge"])
            row |= {f"local_charge_w{w}": obs["local_charge"],
                    f"a_seg_start_w{w}": a["start"] + i, f"a_seg_end_w{w}": a["start"] + i + w - 1,
                    f"a_seg_w{w}": a["sequence"][i:i + w],
                    f"b_seg_start_w{w}": b["start"] + j, f"b_seg_end_w{w}": b["start"] + j + w - 1,
                    f"b_seg_w{w}": b["sequence"][j:j + w],
                    f"local_charge_w{w}_z_shuffle": z, f"local_charge_w{w}_p_shuffle": p,
                    f"local_charge_w{w}_pct_bg": S.percentile(obs["local_charge"], null_bg["local_charge"])}
            if w != self.w0:
                continue
            for k in S.GLOBAL_SCORES + S.LOCAL_SCORES:
                raw[k], pcts[k] = float(obs[k]), S.percentile(obs[k], null_bg[k])
            row |= {"s_charge": raw["s_charge"], "s_charge_pct_bg": pcts["s_charge"],
                    "a_seg_morf": S.segments_overlap(a["start"] + i, a["start"] + i + w - 1, a["morfs"]),
                    "b_seg_morf": S.segments_overlap(b["start"] + j, b["start"] + j + w - 1, b["morfs"])}
            for k in ("s_catpi", "s_pi"):
                row |= {k: raw[k], f"{k}_pct_bg": pcts[k]}
            for k in ("local_catpi", "local_pi", "hydro_local"):
                row |= {k: raw[k], f"{k}_z_shuffle": S.shuffle_null(raw[k], null_sh[k])[0],
                        f"{k}_pct_bg": pcts[k]}
        row["blocky_both"] = bool(fa["kappa"] >= self.kappa_blocky and fb["kappa"] >= self.kappa_blocky)
        row["hydro_patch_both"] = bool(sa[f"hyd_max_{self.w0}"] >= self.hyd_patch
                                       and sb[f"hyd_max_{self.w0}"] >= self.hyd_patch)
        label, mp = S.dominant_mechanism(pcts, raw, self.min_pct)
        row |= {f"pct_{m}": v for m, v in mp.items()}
        row["dominant_mechanism"], row["max_mechanism_pct"] = label, max(mp.values())
        mi, mj = S.mechanism_windows(label if label != "none" else max(mp, key=mp.get), sa, sb, self.w0)
        row |= {"a_mech_start": a["start"] + mi, "a_mech_end": a["start"] + mi + self.w0 - 1,
                "b_mech_start": b["start"] + mj, "b_mech_end": b["start"] + mj + self.w0 - 1}
        # phosphorylation
        if a.get("phospho_on"):
            pa, pb = self.summary(a["sequence"], a["phospho_idx"]), self.summary(b["sequence"], b["phospho_idx"])
            ph = S.pair_scores(pa, pb, self.w0)
            row |= {"n_phospho_a": len(a["phospho_idx"]), "n_phospho_b": len(b["phospho_idx"]),
                    "s_charge_phospho": float(ph["s_charge"]),
                    f"local_charge_w{self.w0}_phospho": float(ph["local_charge"]),
                    "d_s_charge_phospho": float(ph["s_charge"]) - raw["s_charge"],
                    f"d_local_charge_w{self.w0}_phospho": float(ph["local_charge"]) - raw["local_charge"]}
        return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in row.items()}


def identity(side: str, d: dict) -> dict:
    return {f"{side}_chain": d["chain"], f"{side}_accession": d["accession"],
            f"{side}_idr_id": seq_md5(d["sequence"])[:12], f"{side}_start": d["start"],
            f"{side}_end": d["end"], f"{side}_full_start": d["full_start"],
            f"{side}_full_end": d["full_end"], f"{side}_length": len(d["sequence"])}


def _rank_key(r: dict) -> tuple:
    return (-r["max_mechanism_pct"], -r[next(k for k in r if k.startswith("local_charge_w"))])


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    import pandas as pd

    from nanopulldown.annotate.disorder import cache_is_current, run_aiupred_batch
    from nanopulldown.annotate.domains import region_source
    from nanopulldown.core.cache import annotation_cache_dir, cached_disorder_tracks
    from nanopulldown.core.io import write_parquet_with_metadata
    from nanopulldown.core.proteins import load_proteins, protein_name
    from nanopulldown.core.specs import load_pairs, protein_chains

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--background", required=True, help="FASTA of background proteins (fetch_idr_reference)")
    ap.add_argument("--controls", required=True, help="FASTA of the control proteins (fetch_idr_reference)")
    ap.add_argument("--features", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--controls-csv", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--formats", default="svg")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()

    import sparrow
    import torch
    torch.set_num_threads(a.threads)

    acfg, c = CFG["annotate"], icfg()
    cutoff = float(c.get("disorder_cutoff") or acfg["consensus"]["disorder_cutoff"])
    min_len = int(c.get("min_len", 30))
    windows = [int(w) for w in c.get("windows", [15, 25])]
    controls = c.get("controls") or DEFAULT_CONTROLS

    # full-length proteins: {seq: accession}
    specs = load_pairs()
    chains: dict[str, list] = {}                 # pair -> [(chain, full seq, regions)]
    targets: dict[str, str | None] = {}
    for p in a.pairs:
        rows = []
        for ch in protein_chains(specs[p]):
            src = region_source(specs[p], ch)
            if src:                                  # fragment of a UniProt entry
                full, acc, regions = src
            else:                                    # the chain is the whole protein
                ent = specs[p].get(ch["role"]) or {}
                full, regions = ch["sequence"], None
                acc = ent.get("uniprot") if not ent.get("regions") else None
                acc = None if acc and "-" in acc else acc
            rows.append((ch, full, regions))
            if acc or full not in targets:
                targets[full] = acc
        chains[p] = rows
    bg_fa, ctl_fa = _read_fasta(Path(a.background)), _read_fasta(Path(a.controls))
    every = {**{s: acc for acc, s in bg_fa.items()}, **{s: acc for acc, s in ctl_fa.items()}, **targets}

    # AIUPred for sequences the disorder batch did not cover (background, controls)
    dcache = annotation_cache_dir(CFG, "disorder")
    todo = [s for s in every if not cache_is_current(dcache / f"{seq_md5(s)}.json")]
    if todo:
        print(f"[idr] AIUPred for {len(todo)} uncached sequence(s)", flush=True)
        for md5, tracks in run_aiupred_batch(todo, acfg["aiupred"]).items():
            atomic_write(dcache / f"{md5}.json", json.dumps(tracks))
    prots = {s: Protein(s, acc, cached_disorder_tracks(CFG, s), c, cutoff, acfg["morfchibi"])
             for s, acc in every.items()}
    missing = [p for p in prots.values() if len(p.disorder) != len(p.seq)]
    if missing:
        print(f"[idr] WARNING: {len(missing)} protein(s) without AIUPred tracks have no IDRs", file=sys.stderr)

    # phosphosites (optional)
    ph_on = bool(c.get("phosphosites") or c.get("predict_phosphosites"))
    if c.get("phosphosites"):
        tab = pd.read_csv(c["phosphosites"], sep="\t")
        sites: dict[str, set[int]] = {}
        for acc, pos in zip(tab["accession"], tab["position"]):
            sites.setdefault(str(acc), set()).add(int(pos))
        for pr in prots.values():
            pr.phospho |= sites.get(pr.acc or "", set())
    if c.get("predict_phosphosites"):
        for pr in prots.values():
            if pr.idrs:
                pr.phospho |= predicted_phosphosites(pr.seq)

    # metapredict cross-check (SPARROW's disorder predictor)
    from sparrow import Protein as SP
    for pr in prots.values():
        if pr.idrs:
            pr.metapredict = np.asarray(SP(pr.seq).predictor.disorder(), dtype=float)

    def idrs_of(pr, regions, cid, role):
        out = chain_idrs(pr, regions, cid, role, min_len)
        for d in out:
            d["phospho_on"] = ph_on
        return out

    bg_idrs = [d for s in bg_fa.values() for d in idrs_of(prots[s], None, "-", "background")]
    ctl_idrs = {acc: idrs_of(prots[s], None, "-", "control") for acc, s in ctl_fa.items()}
    run_idrs = {p: [(ch, idrs_of(prots[full], regions, ch["id"], ch["role"])) for ch, full, regions in rows]
                for p, rows in chains.items()}
    print(f"[idr] IDRs: {len(bg_idrs)} background (from {len(bg_fa)} proteins), "
          f"{sum(len(v) for v in ctl_idrs.values())} control, "
          f"{len({d['sequence'] for v in run_idrs.values() for _, ds in v for d in ds})} unique run IDRs", flush=True)

    # per-IDR features
    records = ([("background", d) for d in bg_idrs] + [("control", d) for v in ctl_idrs.values() for d in v]
               + [("run", d) for v in run_idrs.values() for _, ds in v for d in ds])
    uniq = {seq_md5(d["sequence"])[:12]: d["sequence"] for _, d in records}
    feats = sparrow_features(uniq)
    print(f"[idr] SPARROW features of {len(uniq)} unique IDR(s) ({time.time() - t0:.0f} s)", flush=True)
    scorer = Scorer(c, bg_idrs, feats)

    feat_cols = {**FEATURE_COLUMNS, **_window_feature_columns(windows)}
    feat_rows, seen = [], set()
    for set_name, d in records:
        pr, k = d["protein"], seq_md5(d["sequence"])[:12]
        key = (set_name, pr.md5, d["full_start"], d["full_end"], k)
        if key in seen:
            continue
        seen.add(key)
        sm = scorer.summary(d["sequence"])
        fp = d["full_pos"] - 1
        feat_rows.append({
            "set": set_name, "idr_id": k, "accession": pr.acc, "protein_md5": pr.md5,
            "full_start": d["full_start"], "full_end": d["full_end"], "length": len(d["sequence"]),
            "sequence": d["sequence"], "mean_disorder": float(pr.disorder[fp].mean()),
            "metapredict_agreement": float((pr.metapredict[fp] >= 0.5).mean()) if pr.metapredict is not None else None,
            "morf_residues": sum(e - s + 1 for s, e in d["morfs"]), **feats[k],
            "n_phospho": len(d["phospho_idx"]) if ph_on else None,
            **{f"{s}_w{w}": float(sm[f"{s}_{w}"]) for w in windows
               for s in ("ncpr_max", "ncpr_min", "arom_max", "cat_max", "hyd_max")},
        })
    features = pd.DataFrame(feat_rows, columns=list(feat_cols))
    low = features[(features["set"] == "run") & (features["metapredict_agreement"] < 0.5)]
    if len(low):
        print(f"[idr] {len(low)} run IDR(s) where metapredict calls < 50% of the residues disordered "
              f"(AIUPred/metapredict disagreement; metapredict_agreement column)", flush=True)

    # pairs
    cols = score_columns(windows)
    rows = []
    for p, chs in run_idrs.items():
        bait = {ch["sequence"]: ds for ch, ds in chs if ch["role"] == "bait"}   # copies collapse
        prey = {ch["sequence"]: ds for ch, ds in chs if ch["role"] == "prey"}
        prow = []
        for da in [d for ds in bait.values() for d in ds]:
            for db in [d for ds in prey.values() for d in ds]:
                prow.append({"pair": p, **identity("a", da), **identity("b", db), **scorer.score(da, db)})
        prow.sort(key=_rank_key)
        for i, r in enumerate(prow, start=1):
            r["rank"] = i
        rows += prow
    scores = pd.DataFrame(rows, columns=list(cols))
    print(f"[idr] {len(scores)} IDR pair(s) in {scores['pair'].nunique() if len(scores) else 0}/{len(a.pairs)} "
          f"pair(s) ({time.time() - t0:.0f} s)", flush=True)

    # controls
    run_max = scores["max_mechanism_pct"].to_numpy() if len(scores) else np.array([])
    ctl_rows = []
    for ctl in controls:
        # the IDR pair with the strongest charge complementarity: a positive
        # passes when charge is its dominant mechanism; a negative when even
        # that pair stays below the label percentile for charge
        best = None
        for da in ctl_idrs.get(ctl["a"], []):
            for db in ctl_idrs.get(ctl["b"], []):
                r = {"pair": ctl["name"], **identity("a", da), **identity("b", db), **scorer.score(da, db)}
                if best is None or r["pct_charge"] > best["pct_charge"]:
                    best = r
        if best is None:
            ctl_rows.append({"control": ctl["name"], "expect": ctl["expect"], "passed": False,
                             "note": "no IDR >= min_len on one side"})
            continue
        ok = (best["dominant_mechanism"] == "charge" if ctl["expect"] == "high"
              else best["pct_charge"] < scorer.min_pct)
        ctl_rows.append({"control": ctl["name"], "expect": ctl["expect"], "passed": bool(ok),
                         "run_percentile": S.percentile(best["max_mechanism_pct"], run_max) if len(run_max) else None,
                         **{k: best[k] for k in ("a_accession", "a_full_start", "a_full_end", "b_accession",
                                                 "b_full_start", "b_full_end", "dominant_mechanism", "pct_charge",
                                                 "s_charge", "s_charge_pct_bg", f"local_charge_w{windows[0]}",
                                                 f"local_charge_w{windows[0]}_pct_bg",
                                                 f"local_charge_w{windows[0]}_z_shuffle", "max_mechanism_pct")}})
    ctl_df = pd.DataFrame(ctl_rows)
    for r in ctl_rows:
        print(f"[idr] control {r['control']!r} (expect {r['expect']}): "
              f"{'PASS' if r['passed'] else 'FAIL'} -- {r.get('dominant_mechanism')}, "
              f"charge pct {r.get('pct_charge', float('nan')):.3f}", flush=True)
    if not ctl_df.empty and not ctl_df["passed"].all():
        print("[idr] WARNING: a control misbehaved -- read the scores with caution", file=sys.stderr)

    # write
    meta = (f"SPARROW {sparrow.__version__}; IDR = AIUPred >= {cutoff}, gaps <= {c.get('max_gap', 10)} merged, "
            f">= {min_len} aa; windows {windows}; {scorer.n_shuf} shuffles; background {len(bg_idrs)} IDRs "
            f"of {len(bg_fa)} proteins (taxon {(c.get('background') or {}).get('taxon', 3702)}); "
            f"label percentile {scorer.min_pct}; phosphorylation "
            f"{'on' if ph_on else 'off'}")
    for path in (a.features, a.scores, a.csv, a.controls_csv):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_parquet_with_metadata(features, Path(a.features), column_descriptions=feat_cols,
                                table_references=REFERENCES,
                                table_description=f"Per-IDR sequence features ({meta})")
    write_parquet_with_metadata(scores, Path(a.scores), column_descriptions=cols, table_references=REFERENCES,
                                table_description=f"Bait IDR x prey IDR complementarity, per pair ({meta}). "
                                                  "Heuristic ranking, not binding energies")
    proteins = load_proteins()
    view = scores.copy()
    view.insert(1, "bait_protein", [protein_name(x, proteins) if x else "" for x in view["a_accession"]])
    view.insert(2, "prey_protein", [protein_name(x, proteins) if x else "" for x in view["b_accession"]])
    if len(view):
        view = view.sort_values(["max_mechanism_pct", f"local_charge_w{windows[0]}"],
                                ascending=False, kind="stable")
    view.to_csv(a.csv, index=False)
    ctl_df.to_csv(a.controls_csv, index=False)

    from nanopulldown.report.plots.idr_pairs import draw_all
    draw_all(view, chains, run_idrs, prots, scorer, Path(a.fig_dir), int(c.get("top_n_plots", 20)), a.formats, meta)
    print(f"[idr] done ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
