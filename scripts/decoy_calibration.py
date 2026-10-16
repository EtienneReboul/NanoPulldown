#!/usr/bin/env python3
"""
scripts/decoy_calibration.py — decoy-calibrated hit calling
============================================================
Turns a run's folded controls (scripts/expand_pairs.py adds both, see
scripts/controls.py) into calibrated hit calls for the real preys, per bait
and per metric, and checks them against the known partners:

  * negatives -- the decoy preys (scripts/build_decoys.py; input-source tag
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
recovered per metric), recall.csv (per bait protein x metric), index.html
(decoy / prey / positive ECDFs per bait x metric + the recall and hit tables).

Usage (report env: envs/report.yaml):
    python scripts/decoy_calibration.py \
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_pipeline import load_pairs  # noqa: E402

METRICS = ["pinc", "ipsae", "pair_iptm", "ilis", "pdockq"]


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
    t["is_positive"] = tags.map(lambda s: a.positive_tag in s) & ~t["is_decoy"]
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

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    g.drop(columns="prey_seq").to_csv(out / "calibration.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    positives.to_csv(out / "positives.csv", index=False)
    recall.to_csv(out / "recall.csv", index=False)
    write_html(g, summ, recall, metrics, a.alpha, out / "index.html")
    print(f"[decoy_calibration] -> {out}/calibration.csv, summary.csv, positives.csv, recall.csv, index.html")
    return 0


def positive_recovery(g: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per bait protein x known partner: n pairs, best q + recovered per
    metric; per bait protein x metric: n positives, recovered, recall)."""
    pos = g[g.is_positive]
    if pos.empty or "bait_uniprot" not in pos or "prey_uniprot" not in pos:
        return (pd.DataFrame(columns=["bait_uniprot", "prey_uniprot"]),
                pd.DataFrame(columns=["bait_uniprot", "metric", "n_positives", "recovered", "recall"]))
    agg = {"n_pairs": ("pair", "size")}
    agg |= {c: (c, "first") for c in ("bait", "prey", "prey_name") if c in pos}
    for m in metrics:
        agg[f"{m}_best_q"] = (f"{m}_q", "min")
        agg[f"{m}_recovered"] = (f"{m}_hit", lambda h: bool(h.fillna(False).astype(bool).any()))
    per = pos.groupby(["bait_uniprot", "prey_uniprot"], as_index=False).agg(**agg)
    rec = []
    for bait, d in per.groupby("bait_uniprot"):
        for m in metrics:
            k = int(d[f"{m}_recovered"].sum())
            rec.append({"bait_uniprot": bait, "bait": d["bait"].iloc[0] if "bait" in d else bait,
                        "metric": m, "n_positives": len(d), "recovered": k, "recall": k / len(d)})
    return per, pd.DataFrame(rec)


def write_html(g: pd.DataFrame, summ: pd.DataFrame, recall: pd.DataFrame, metrics: list[str], alpha: float,
               path: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    baits = list(summ.bait_spec.unique())
    si = summ.set_index(["bait_spec", "metric"])
    titles = [f"{b} · {m} (AUC {si.at[(b, m), 'auc_prey_vs_decoy']:.2f}, "
              f"hits {si.at[(b, m), f'hits_q{alpha:g}']})" for b in baits for m in metrics]
    fig = make_subplots(rows=len(baits), cols=len(metrics), subplot_titles=titles,
                        horizontal_spacing=0.04, vertical_spacing=0.12 / max(len(baits), 1))
    colors = {"decoys": "#d95926", "preys": "#3987e5", "known partners": "#199e70"}   # dataviz categorical slots 1-3, dark steps
    shown: set[str] = set()
    for i, b in enumerate(baits, start=1):
        d = g[g.bait_spec == b]
        for j, m in enumerate(metrics, start=1):
            for lab, sub in (("decoys", d[d.is_decoy]), ("preys", d[~d.is_decoy & ~d.is_positive]),
                             ("known partners", d[d.is_positive])):
                x = np.sort(sub[m].dropna().to_numpy())
                if not len(x):
                    continue
                fig.add_trace(go.Scatter(x=x, y=np.arange(1, len(x) + 1) / len(x), mode="lines",
                                         line=dict(color=colors[lab], width=2), name=lab, legendgroup=lab,
                                         showlegend=(lab not in shown and not shown.add(lab)),
                                         hovertemplate=f"{lab}: {m}=%{{x:.3g}}, ECDF=%{{y:.2f}}<extra></extra>"),
                              row=i, col=j)
            fig.update_xaxes(title_text=m, row=i, col=j)
    fig.update_layout(template="plotly_dark", height=380 * len(baits), title=(
        f"Controls — decoy / prey / known-partner score ECDFs per bait (hits: BH q ≤ {alpha:g} "
        "on conformal p against the decoys)"))
    hit_cols = [f"{m}_hit" for m in metrics]
    hits = g[g[hit_cols].fillna(False).any(axis=1)].copy()
    show = ["bait_spec", "prey"] + [c for m in metrics for c in (m, f"{m}_q")]
    hits = hits[[c for c in show if c in hits]].round(4).sort_values(["bait_spec", show[3]])
    html = fig.to_html(include_plotlyjs=True, full_html=True)
    html = html.replace("</body>", (
        "<style>body{background:#111;color:#f0efec;font-family:sans-serif}.t{border-collapse:collapse;"
        "margin:24px;font-size:13px}.t th,.t td{padding:4px 10px;border-bottom:1px solid #333;text-align:left}"
        "</style><h3 style='margin-left:24px'>Known partners recovered (any of their pairs a hit)</h3>"
        + (recall.round(3).to_html(index=False, border=0, classes="t") if not recall.empty
           else "<p style='margin-left:24px'>no positive control in this run</p>")
        + "<h3 style='margin-left:24px'>Per bait and metric</h3>"
        + summ.round(3).to_html(index=False, border=0, classes="t")
        + f"<h3 style='margin-left:24px'>Hits on any metric ({len(hits)})</h3>"
        + hits.to_html(index=False, border=0, classes="t") + "</body>"))
    path.write_text(html)


if __name__ == "__main__":
    sys.exit(main())
