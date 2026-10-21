#!/usr/bin/env python3
"""
nanopulldown.report.calibration — decoy-calibrated hit calling
============================================================
Turns a run's folded controls (nanopulldown.inputs.expand adds both, see
nanopulldown.inputs.controls) into calibrated hit calls for the real preys, per bait
and per metric, and checks them against the known partners:

  * negatives -- the decoy preys (nanopulldown.inputs.decoys; input-source tag
    `decoy` by default): the per-bait null
  * positives -- the baits' known direct partners (tag `known`): how many the
    hit calls recover (recall), and how well each metric separates them from
    the decoys (AUC)

Per bait and metric:

  * conformal p-value of a prey's score s against the bait's n decoys
        p = (1 + #{decoys >= s}) / (n + 1)
    valid in finite samples as long as decoys and non-interacting preys are
    exchangeable (Bates et al. 2023, "Testing for outliers with conformal
    p-values") -- what build_decoys' matching is for. The smallest p is
    1/(n+1), so BH can only reach a hit with n decoys >~ m / (k * alpha)
    for the k-th hit among m preys.
  * Benjamini-Hochberg q-values over the bait's real preys; hit = q <= alpha.
  * a null check: does the bulk of the screen look like the decoys? AUC =
    P(prey score > decoy score) and the KS distance. AUC ~0.5 means most
    preys behave like decoys (the median/MAD and adjusted-boxplot fences on
    the screen alone are then justified); AUC well above 0.5 with few hits
    means the screen is shifted as a whole (a matching or bait problem).
  * protein-level AUCs per bait PROTEIN (protein_auc.csv): one value per
    prey protein over all its fragment pairs, as the best score (biased up for
    proteins with more fragments) and as a Sidak-adjusted min p (slightly
    conservative) -- see protein_auc(). In fragment mode these are the AUCs to
    read: a partner binding through one domain is not diluted by the others.
  * positives are PER BAIT: a known partner counts only for the baits it has
    direct evidence with (`known_direct_of`, nanopulldown.inputs.controls); a
    control-only partner of another bait is dropped for this one. Each
    positive carries the fraction of the bait and of the partner folded in
    the run; below 0.9 it is flagged `partial` (positives.csv, index.html).
  * positive controls: AUC = P(positive score > decoy score) per bait
    (fragment), and recall per bait PROTEIN: a known partner is recovered when
    any of its pairs (any bait fragment x any of its fragments) is a hit. In
    fragment mode most fragment pairs of a true partner miss the interface,
    so the pair-level AUC understates a metric; recall is the number to read.
    Positives are real preys, so they are in the BH family too.

One value per pair (median over samples and bait-prey chain pairs), and
preys with an identical sequence against the same bait (isoforms sharing a
fragment) are collapsed to one, so duplicates neither count twice in BH nor
in the decoy null. "Higher is better" for every metric (all defaults are).

Outputs (--outdir): calibration.csv (one row per bait x prey: score, p, q,
hit per metric), summary.csv (per bait x metric: n decoys/preys/positives,
AUCs, KS, hits), positives.csv (per bait protein x known partner: best q and
recovered per metric), recall.csv (per bait protein x metric),
protein_auc.csv (per bait protein x metric: best-fragment and fair AUCs), index.html
(decoy / prey / positive ECDFs per bait x metric + the recall and hit tables).

Usage (report env: envs/report.yaml):
    nanopulldown calibrate \
        --table reports_<run>/folding/results_table.csv \
        --outdir reports_<run>/folding/decoy_calibration
    # decoys folded as a separate run: pass both results tables
    ... --table reports_<run>/folding/results_table.csv reports_<run>_decoys/folding/results_table.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nanopulldown.core.proteins import load_proteins
from nanopulldown.core.specs import load_pairs

# higher = better for all (conformal_p counts decoys >= score); the per-pair
# z_af_pydock_pair is left out -- it isn't comparable between pairs
METRICS = ["pinc", "ipsae", "pair_iptm", "ilis", "pdockq", "z_af_pydock_run"]


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def conformal_p(scores: np.ndarray, null: np.ndarray) -> np.ndarray:
    null = np.sort(null)
    n_ge = len(null) - np.searchsorted(null, scores, side="left")
    return (1 + n_ge) / (len(null) + 1)


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) + 0.5 P(X = Y) (Mann-Whitney)."""
    r = pd.Series(np.concatenate([x, y])).rank().to_numpy()
    return float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def ks(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.sort(x), np.sort(y)
    g = np.concatenate([x, y])
    return float(np.max(np.abs(np.searchsorted(x, g, "right") / len(x) - np.searchsorted(y, g, "right") / len(y))))


def prey_sequence(pair: str, specs: dict) -> str | None:
    spec = specs.get(pair)
    if not spec:
        return None
    ids = set(spec.get("prey_chains") or [])
    return "|".join(c["sequence"] for c in spec.get("sequences") or [] if c["id"] in ids) or None


def main() -> int:
    ap = argparse.ArgumentParser(description="decoy-calibrated conformal p-values + BH per bait")
    ap.add_argument("--table", nargs="+", required=True, help="results_table.csv (one or more)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    ap.add_argument("--decoy-tag", default="decoy", help="input-source tag of the decoys")
    ap.add_argument("--positive-tag", default="known", help="input-source tag of the positive controls")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    t = pd.concat([pd.read_csv(p) for p in a.table], ignore_index=True)
    metrics = [m for m in a.metrics if m in t.columns]
    t["bait_spec"] = t.pair.str.split("__").str[0]
    src = t.get("prey_sources", pd.Series("", index=t.index)).fillna("").astype(str)
    tags = src.str.split(r",\s*")
    t["is_decoy"] = tags.map(lambda s: a.decoy_tag in s)
    tagged = tags.map(lambda s: a.positive_tag in s) & ~t["is_decoy"]
    of_col = f"{a.positive_tag}_direct_of"
    if of_col in t:
        # a known partner is a positive for the baits it has direct evidence with only;
        # a control-only partner (no other source) means nothing for the other baits: dropped
        direct_of = t[of_col].fillna("").astype(str).str.split(";")
        own = [str(b) in d for b, d in zip(t.get("bait_uniprot", pd.Series("", index=t.index)), direct_of)]
        t["is_positive"] = tagged & pd.Series(own, index=t.index)
        control_only = tags.map(lambda s: set(x for x in s if x) == {a.positive_tag})
        n_drop = int((control_only & tagged & ~t["is_positive"]).sum())
        t = t[~(control_only & tagged & ~t["is_positive"])].copy()
        if n_drop:
            print(f"[decoy_calibration] dropped {n_drop} row(s): positive controls of another bait")
    else:
        print(f"[decoy_calibration] WARNING: no {of_col} column (expanded before per-bait positives): "
              "every known partner counts as a positive for every bait -- re-run expand_pairs.py",
              file=sys.stderr)
        t["is_positive"] = tagged
    first = {c: (c, "first") for c in ("bait", "prey", "prey_name", "prey_sources", "bait_uniprot",
                                       "prey_uniprot") if c in t}
    g = t.groupby(["bait_spec", "pair"], as_index=False).agg(
        is_decoy=("is_decoy", "max"), is_positive=("is_positive", "max"), **first,
        **{m: (m, "median") for m in metrics})

    specs = load_pairs()
    g["prey_seq"] = g.pair.map(lambda p: prey_sequence(p, specs))
    n0 = len(g)
    g = g[g.prey_seq.isna() | ~g.duplicated(["bait_spec", "prey_seq"])]
    if len(g) < n0:
        print(f"[decoy_calibration] collapsed {n0 - len(g)} pair(s) with a duplicate prey sequence")

    summary, parts = [], []
    for bait, d in g.groupby("bait_spec"):
        d = d.copy()
        dec, scr = d[d.is_decoy], d[~d.is_decoy]
        if dec.empty:
            print(f"[decoy_calibration] {bait}: no decoys -- skipped", file=sys.stderr)
            continue
        for m in metrics:
            null, s = dec[m].dropna().to_numpy(), scr[m]
            ok = s.notna()
            p = pd.Series(np.nan, index=scr.index)
            p[ok] = conformal_p(s[ok].to_numpy(), null)
            q = pd.Series(np.nan, index=scr.index)
            q[ok] = bh(p[ok].to_numpy())
            d.loc[scr.index, f"{m}_p"], d.loc[scr.index, f"{m}_q"] = p, q
            d.loc[scr.index, f"{m}_hit"] = q <= a.alpha
            pos = scr.loc[scr.is_positive, m].dropna().to_numpy()
            summary.append({"bait_spec": bait, "metric": m, "n_decoys": len(null), "n_preys": int(ok.sum()),
                            "n_positives": len(pos),
                            "min_p": 1 / (len(null) + 1), "auc_prey_vs_decoy": auc(s[ok].to_numpy(), null),
                            "auc_positive_vs_decoy": auc(pos, null) if len(pos) else np.nan,
                            "ks": ks(s[ok].to_numpy(), null), "decoy_median": float(np.median(null)),
                            "prey_median": float(s[ok].median()), f"hits_q{a.alpha:g}": int((q <= a.alpha).sum())})
        parts.append(d)
    if not parts:
        print("[decoy_calibration] no bait has decoys -- add build_decoys' list to the run and fold it",
              file=sys.stderr)
        return 1
    g = pd.concat(parts)
    summ = pd.DataFrame(summary)
    print(summ.round(3).to_string(index=False))
    positives, recall = positive_recovery(g, metrics)
    if not recall.empty:
        print(recall.round(3).to_string(index=False))
    prot_auc = protein_auc(g, metrics)
    if not prot_auc.empty:
        print(prot_auc.round(3).to_string(index=False))

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    g.drop(columns="prey_seq").to_csv(out / "calibration.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    positives.to_csv(out / "positives.csv", index=False)
    recall.to_csv(out / "recall.csv", index=False)
    prot_auc.to_csv(out / "protein_auc.csv", index=False)
    write_html(g, summ, recall, prot_auc, positives, metrics, a.alpha, out / "index.html")
    print(f"[decoy_calibration] -> {out}/calibration.csv, summary.csv, positives.csv, recall.csv, "
          "protein_auc.csv, index.html")
    return 0


def protein_auc(g: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Protein-level AUCs per bait PROTEIN x metric: one value per bait protein
    x prey protein over all its fragment pairs, so a partner that binds through
    one domain is not diluted by its other fragments.

      *_best  -- best score over the protein's fragment pairs. Biased upward
                 for proteins with more fragment pairs (more draws).
      *_fair  -- min conformal p over its fragment pairs, Sidak-adjusted for
                 their number k: 1 - (1 - p_min)^k, decoys alike (each decoy's
                 p is leave-one-out against its bait fragment's other decoys).
                 Exact for independent fragments, slightly conservative since
                 fragments of one protein are correlated: the truth lies
                 between _best and _fair.
    """
    if "bait_uniprot" not in g or "prey_uniprot" not in g:
        return pd.DataFrame()
    g = g.copy()
    for m in metrics:
        p = pd.Series(np.nan, index=g.index)
        for _, d in g.groupby("bait_spec"):
            null = d.loc[d.is_decoy, m].dropna().to_numpy()
            if not len(null):
                continue
            ok = d[m].notna()
            scr, dec = ok & ~d.is_decoy, ok & d.is_decoy
            p[scr[scr].index] = conformal_p(d.loc[scr, m].to_numpy(), null)
            srt = np.sort(null)                    # leave-one-out: n_ge includes the decoy itself
            n_ge = len(srt) - np.searchsorted(srt, d.loc[dec, m].to_numpy(), side="left")
            p[dec[dec].index] = n_ge / len(srt)
        g[f"{m}__p"] = p
    g["cls"] = np.where(g.is_decoy, "decoy", np.where(g.is_positive, "positive", "prey"))
    rows = []
    for bait, d in g.groupby("bait_uniprot"):
        for m in metrics:
            prot = d.groupby(["prey_uniprot", "cls"], as_index=False).agg(
                best=(m, "max"), pmin=(f"{m}__p", "min"), k=(f"{m}__p", "count"))
            prot = prot[prot.k > 0]
            prot["padj"] = 1 - (1 - prot.pmin) ** prot.k
            dec, pos, pr = (prot[prot.cls == c] for c in ("decoy", "positive", "prey"))
            if dec.empty:
                continue
            row = {"bait_uniprot": bait, "metric": m, "n_decoy_proteins": len(dec),
                   "n_prey_proteins": len(pr), "n_positive_proteins": len(pos),
                   "frag_pairs_per_positive": pos.k.mean() if len(pos) else np.nan,
                   "frag_pairs_per_decoy": dec.k.mean()}
            for lab, x in (("positive", pos), ("prey", pr)):
                row[f"auc_{lab}_vs_decoy_best"] = (auc(x.best.dropna().to_numpy(), dec.best.dropna().to_numpy())
                                                    if x.best.notna().any() else np.nan)
                row[f"auc_{lab}_vs_decoy_fair"] = auc(-x.padj.to_numpy(), -dec.padj.to_numpy()) if len(x) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _coverage(specs: set[str], length: float) -> float:
    """Fraction of a protein covered by the union of its folded specs
    (ACC = whole protein, ACCxSTART-END = a fragment)."""
    spans = []
    for sp in specs:
        if "x" not in sp:
            return 1.0
        lo, hi = sp.split("x", 1)[1].split("-")
        spans.append((int(lo), int(hi)))
    if not spans or not length or np.isnan(length):
        return np.nan
    covered, end = 0, 0
    for lo, hi in sorted(spans):
        lo = max(lo, end + 1)
        if hi >= lo:
            covered += hi - lo + 1
        end = max(end, hi)
    return covered / length


def positive_recovery(g: pd.DataFrame, metrics: list[str],
                      partial_below: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per bait protein x known partner: n pairs, best q + recovered per
    metric, and how much of the bait / the partner was folded; per bait
    protein x metric: n positives, recovered, recall -- baits with no known
    direct partner get one row saying so).

    bait_coverage / prey_coverage: fraction of the protein covered by the
    fragments folded in this run (1 = folded whole). Below `partial_below`
    the pair is flagged `partial`: the known interface may lie in a region
    that was never folded, so a miss there says little about the metric."""
    if "bait_uniprot" not in g or "prey_uniprot" not in g:
        return (pd.DataFrame(columns=["bait_uniprot", "prey_uniprot"]),
                pd.DataFrame(columns=["bait_uniprot", "metric", "n_positives", "recovered", "recall"]))
    lengths = {acc: float(r.get("length") or np.nan) for acc, r in load_proteins().items()}
    bait_cov = {b: _coverage(set(d.bait_spec), lengths.get(b, np.nan)) for b, d in g.groupby("bait_uniprot")}
    pos = g[g.is_positive]
    rec = []
    if not pos.empty:
        agg = {"n_pairs": ("pair", "size")}
        agg |= {c: (c, "first") for c in ("bait", "prey", "prey_name") if c in pos}
        for m in metrics:
            agg[f"{m}_best_q"] = (f"{m}_q", "min")
            agg[f"{m}_recovered"] = (f"{m}_hit", lambda h: bool(h.fillna(False).astype(bool).any()))
        per = pos.groupby(["bait_uniprot", "prey_uniprot"], as_index=False).agg(**agg)
        prey_specs = pos.assign(prey_spec=pos.pair.str.split("__").str[1]).groupby(
            ["bait_uniprot", "prey_uniprot"]).prey_spec.agg(set)
        per["bait_coverage"] = per.bait_uniprot.map(bait_cov)
        per["prey_coverage"] = [_coverage(prey_specs[(b, p)], lengths.get(p, np.nan))
                                for b, p in zip(per.bait_uniprot, per.prey_uniprot)]
        per["partial"] = (per.bait_coverage < partial_below) | (per.prey_coverage < partial_below)
    else:
        per = pd.DataFrame(columns=["bait_uniprot", "prey_uniprot"])
    for bait in sorted(g.bait_uniprot.dropna().unique()):
        d = per[per.bait_uniprot == bait]
        name = g.loc[g.bait_uniprot == bait, "bait"].iloc[0] if "bait" in g else bait
        if d.empty:
            rec.append({"bait_uniprot": bait, "bait": name, "metric": "", "n_positives": 0,
                        "recovered": np.nan, "recall": np.nan, "note": "no known direct partner"})
            continue
        for m in metrics:
            k = int(d[f"{m}_recovered"].sum())
            rec.append({"bait_uniprot": bait, "bait": name, "metric": m, "n_positives": len(d),
                        "n_partial": int(d.partial.sum()), "recovered": k, "recall": k / len(d),
                        "note": ""})
    return per, pd.DataFrame(rec)


def write_html(g: pd.DataFrame, summ: pd.DataFrame, recall: pd.DataFrame, prot_auc: pd.DataFrame,
               positives: pd.DataFrame, metrics: list[str], alpha: float, path: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Fragment mode: one curve point per bait PROTEIN x prey protein (its best
    # fragment pair), rows = bait proteins, titles = protein-level AUC best (fair).
    # A partner binds through one domain; its other fragment pairs only bury it.
    frag = bool(g.pair.str.contains("x", regex=False).any()) and not prot_auc.empty
    if frag:
        cls = np.where(g.is_decoy, "decoys", np.where(g.is_positive, "known partners", "preys"))
        plot = (g.assign(cls=cls).groupby(["bait_uniprot", "prey_uniprot", "cls"], as_index=False)
                [metrics].max().rename(columns={"bait_uniprot": "row"}))
        pa = prot_auc.set_index(["bait_uniprot", "metric"])
        has_pos = prot_auc.groupby("bait_uniprot").n_positive_proteins.max() > 0
        label = g.drop_duplicates("bait_uniprot").set_index("bait_uniprot")["bait"] if "bait" in g \
            else pd.Series(dtype=str)

        def title(b: str, m: str) -> str:
            r = pa.loc[(b, m)]
            t = f"{m} · AUC prey {r.auc_prey_vs_decoy_best:.2f} ({r.auc_prey_vs_decoy_fair:.2f})"
            if has_pos.get(b, False):
                t += f" · known {r.auc_positive_vs_decoy_best:.2f} ({r.auc_positive_vs_decoy_fair:.2f})"
            return t

        def row_name(b: str) -> str:
            return f"{str(label.get(b, b)).split(' ')[0]} · best fragment"
        what = ("bait protein (rows; best fragment pair per prey protein) and metric (columns)",
                "AUC = best fragment pair per protein vs decoys, (Sidak-fair) in brackets: the best-of view "
                "is biased up for proteins with more fragments")
    else:
        si = summ.set_index(["bait_spec", "metric"])
        cls = np.where(g.is_decoy, "decoys", np.where(g.is_positive, "known partners", "preys"))
        plot = g.assign(cls=cls).rename(columns={"bait_spec": "row"})
        has_pos = summ.groupby("bait_spec").n_positives.max() > 0
        label = g.drop_duplicates("bait_spec").set_index("bait_spec")["bait"] if "bait" in g \
            else pd.Series(dtype=str)

        def title(b: str, m: str) -> str:
            t = f"{m} · AUC prey {si.at[(b, m), 'auc_prey_vs_decoy']:.2f}"
            if has_pos.get(b, False):
                t += f" · known {si.at[(b, m), 'auc_positive_vs_decoy']:.2f}"
            return t + f" · hits {si.at[(b, m), f'hits_q{alpha:g}']}"

        def row_name(b: str) -> str:
            region = b.split("x", 1)[1] if "x" in b else "full length"
            return f"{str(label.get(b, b)).split(' ')[0]} · {region}"
        what = ("bait fragment (rows) and metric (columns)",
                f"AUC prey / known = screen preys / known direct partners vs decoys, per fragment pair; "
                f"hits: BH q ≤ {alpha:g} on conformal p against the decoys")
    # rows with known partners first (stable order otherwise): they carry the green line
    rows = sorted(plot.row.unique(), key=lambda b: not has_pos.get(b, False))
    # short subplot titles; the row labels itself once, on the first column's y axis;
    # x titles only on the bottom row. Spacing is fixed in pixels so the grid scales.
    titles = [title(b, m) for b in rows for m in metrics]
    row_h, gap_px, top_px = 260, 70, 120
    height = top_px + len(rows) * row_h + (len(rows) - 1) * gap_px + 60
    fig = make_subplots(rows=len(rows), cols=len(metrics), subplot_titles=titles,
                        horizontal_spacing=0.035, vertical_spacing=gap_px / height)
    fig.update_annotations(font_size=11 if not frag else 10)
    colors = {"decoys": "#d95926", "preys": "#3987e5", "known partners": "#199e70"}   # dataviz categorical slots 1-3, dark steps
    shown: set[str] = set()
    for i, b in enumerate(rows, start=1):
        d = plot[plot.row == b]
        for j, m in enumerate(metrics, start=1):
            for lab in ("decoys", "preys", "known partners"):
                x = np.sort(d.loc[d.cls == lab, m].dropna().to_numpy())
                if not len(x):
                    continue
                fig.add_trace(go.Scatter(x=x, y=np.arange(1, len(x) + 1) / len(x), mode="lines",
                                         line=dict(color=colors[lab], width=2), name=lab, legendgroup=lab,
                                         showlegend=(lab not in shown and not shown.add(lab)),
                                         hovertemplate=f"{lab}: {m}=%{{x:.3g}}, ECDF=%{{y:.2f}}<extra></extra>"),
                              row=i, col=j)
            if i == len(rows):
                fig.update_xaxes(title_text=m, row=i, col=j)
        note = "" if has_pos.get(b, False) else "<br><i>no known partner</i>"
        fig.update_yaxes(title_text=f"{row_name(b)}{note}<br>ECDF", title_font_size=11, row=i, col=1)
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(t=top_px, l=110, r=30, b=60),
        title=dict(text=f"Controls — score ECDFs per {what[0]}<br><sup>{what[1]}. Rows with no known "
                        "direct partner (no green line) come last.</sup>", y=1 - 20 / height, yanchor="top"),
        legend=dict(orientation="h", x=0, xanchor="left", y=1 + 32 / (height - top_px - 60), yanchor="bottom"))
    hit_cols = [f"{m}_hit" for m in metrics]
    hits = g[g[hit_cols].fillna(False).any(axis=1)].copy()
    show = ["bait_spec", "prey"] + [c for m in metrics for c in (m, f"{m}_q")]
    hits = hits[[c for c in show if c in hits]].round(4).sort_values(["bait_spec", show[3]])
    def tbl(df: pd.DataFrame, digits: int = 3) -> str:
        return df.round(digits).to_html(index=False, border=0, classes="t", na_rep="—")

    if "note" in recall:
        recall = recall.assign(note=recall.note.fillna(""))
    html = fig.to_html(include_plotlyjs=True, full_html=True)
    html = html.replace("</body>", (
        "<style>body{background:#111;color:#f0efec;font-family:sans-serif}.t{border-collapse:collapse;"
        "margin:24px;font-size:13px}.t th,.t td{padding:4px 10px;border-bottom:1px solid #333;text-align:left}"
        "</style><h3 style='margin-left:24px'>Known partners recovered (any of their pairs a hit)</h3>"
        + (tbl(recall) if not recall.empty
           else "<p style='margin-left:24px'>no positive control in this run</p>")
        + "<h3 style='margin-left:24px'>Known direct partners per bait (positives)</h3>"
          "<p style='margin-left:24px;max-width:900px'>A known partner is a positive only for the "
          "bait(s) it has direct evidence with. Coverage = fraction of the protein folded in this "
          "run (1 = whole); <b>partial</b> pairs may bind through a region that was never folded, "
          "so a miss there says little about the metric.</p>"
        + (positives[[c for c in ("bait_uniprot", "prey_uniprot", "prey_name", "n_pairs", "bait_coverage",
                                  "prey_coverage", "partial") if c in positives]]
           .pipe(tbl, 2) if not positives.empty
           else "<p style='margin-left:24px'>no positive control in this run</p>")
        + "<h3 style='margin-left:24px'>Protein-level AUC per bait protein (best fragment pair; "
          "_fair = Sidak-adjusted min p, the truth lies between _best and _fair)</h3>"
          "<p style='margin-left:24px'>— = not applicable: the bait has no known direct partner.</p>"
        + (tbl(prot_auc) if not prot_auc.empty
           else "<p style='margin-left:24px'>no protein ids in the table</p>")
        + "<h3 style='margin-left:24px'>Per bait (fragment) and metric</h3>"
          "<p style='margin-left:24px'>— = not applicable: the bait has no known direct partner.</p>"
        + tbl(summ)
        + f"<h3 style='margin-left:24px'>Hits on any metric ({len(hits)})</h3>"
        + tbl(hits, 4) + "</body>"))
    path.write_text(html)


if __name__ == "__main__":
    sys.exit(main())
