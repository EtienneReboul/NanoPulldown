#!/usr/bin/env python3
"""
nanopulldown.report.plots.violins — Stage 2g report figure (interactive)
============================================================
Plotly twin of plot_metric_violins.py: one violin panel per metric with every
point hoverable (bait × prey names, pair, sample, chain pair, value). Written as a directory with
a self-contained index.html (plotly.js inlined) so `snakemake --report` can
embed it via report(directory(...), htmlindex="index.html") and it works
offline.

--metrics / --title / --labels make it reusable for other column sets of the
results table (the Snakefile's PLIP interaction-type section uses it with the
plip_* count columns).

One point per pair by default (--aggregate mean): the pair's mean over its
samples and chain pairs, so a pair with 5 samples weighs as much as one with
1. --aggregate none plots every sample x chain-pair row. Fragment mode keeps
its best-fragment-pair points.

When the run has controls, each panel splits into three violins -- known
partners, screen preys, decoys -- in three shades of the panel's hue
(lightest = known partners, base = preys, darkest = decoys; each triple
validated as an ordinal ramp on the dark surface). Groups follow
nanopulldown.report.calibration: a known partner counts for the bait(s) it is a
positive control for only (`<tag>_direct_of`), and a control-only partner of
another bait is left out of that bait's rows. The group is also the x tick
label and in the hover, so it never rests on color alone.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# dark-mode categorical slots (dataviz reference palette), each as
# (known partners, preys, decoys): lighter / base / darker OKLCH steps
SHADES = [("#8cbefe", "#3987e5", "#205ba2"), ("#fda080", "#d95926", "#98380f"),
          ("#73d1a7", "#199e70", "#0d6c4b"), ("#fabe6c", "#c98500", "#8e5d07"),
          ("#fd98b7", "#d55181", "#953257"), ("#5fb55a", "#008300", "#035b03"),
          ("#c6c3fe", "#9085e9", "#655da8"), ("#fdb5b1", "#e66767", "#a54646")]
GROUPS = ["known partners", "preys", "decoys"]

METRICS = ["ptm", "iptm", "pair_iptm", "mean_plddt", "ipsae", "lis", "clis",
           "ilis", "pinc", "pdockq", "pdockq2", "plip_contacts", "pydock_1vdw", "z_af_pydock_run"]


def best_fragment(df: pd.DataFrame, m: str) -> pd.DataFrame:
    """One row per bait protein x prey protein x group: the fragment pair with
    the best value of `m` (a pair's value = median over its samples / chain
    pairs), with `name` and `n_fragment_pairs`."""
    name = (df["bait"].astype(str).str.split(" ").str[0] + " × " + df["prey"].astype(str).str.split(" ").str[0]
            if {"bait", "prey"} <= set(df.columns) else df["bait_uniprot"] + " × " + df["prey_uniprot"])
    per_pair = (df.assign(name=name).groupby("pair", as_index=False)
                .agg(**{m: (m, "median")}, group=("group", "first"), name=("name", "first"),
                     bait_uniprot=("bait_uniprot", "first"), prey_uniprot=("prey_uniprot", "first")))
    per_pair = per_pair[per_pair[m].notna()]
    keys = ["bait_uniprot", "prey_uniprot", "group"]
    n = per_pair.groupby(keys).pair.transform("size")
    best = per_pair.assign(n_fragment_pairs=n).sort_values(m, ascending=False).drop_duplicates(keys)
    return best.reset_index(drop=True)


def pair_mean(df: pd.DataFrame, m: str, name: pd.Series) -> pd.DataFrame:
    """One row per pair: the mean of `m` over its samples and chain pairs, with
    `name`, `group` and `n_rows` (how many values went into the mean)."""
    per_pair = (df.assign(name=name)[df[m].notna()].groupby("pair", as_index=False)
                .agg(**{m: (m, "mean")}, group=("group", "first"), name=("name", "first"),
                     n_rows=(m, "size")))
    return per_pair.reset_index(drop=True)


def control_groups(df: pd.DataFrame, decoy_tag: str, positive_tag: str) -> pd.Series:
    """'known partners' / 'preys' / 'decoys' per row (None: a control-only
    known partner of another bait, which is neither for this bait)."""
    if "prey_sources" not in df:
        return pd.Series("preys", index=df.index)
    tags = df["prey_sources"].fillna("").astype(str).str.split(r",\s*").map(lambda t: {x for x in t if x})
    of_col = f"{positive_tag}_direct_of"
    direct_of = (df[of_col].fillna("").astype(str).str.split(";") if of_col in df
                 else pd.Series([None] * len(df), index=df.index))
    bait = df.get("bait_uniprot", pd.Series("", index=df.index)).astype(str)
    out = []
    for t, d, b in zip(tags, direct_of, bait):
        if decoy_tag in t:
            out.append("decoys")
        elif positive_tag in t and (d is None or b in d):
            out.append("known partners")
        elif positive_tag in t and t == {positive_tag}:
            out.append(None)
        else:
            out.append("preys")
    return pd.Series(out, index=df.index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--metrics", nargs="+", default=METRICS, help="results-table columns, one panel each")
    ap.add_argument("--labels", nargs="+", default=None, help="panel titles (same order as --metrics)")
    ap.add_argument("--title", default="Metric distributions (all pairs, samples, chain pairs)")
    ap.add_argument("--skip-all-zero", action="store_true", help="drop columns that are 0 everywhere")
    ap.add_argument("--decoy-tag", default="decoy", help="input-source tag of the decoys")
    ap.add_argument("--positive-tag", default="known", help="input-source tag of the positive controls")
    ap.add_argument("--fragments", choices=["auto", "best", "all"], default="auto",
                    help="best: one point per bait x prey protein (its best fragment pair); "
                         "auto: best when the table has fragment columns")
    ap.add_argument("--aggregate", choices=["mean", "none"], default="mean",
                    help="mean: one point per pair, its mean over samples and chain pairs; "
                         "none: one point per sample x chain pair (outside fragment mode)")
    a = ap.parse_args()

    df = pd.read_csv(a.table)
    df["group"] = control_groups(df, a.decoy_tag, a.positive_tag)
    df = df[df.group.notna()]
    groups = [g for g in GROUPS if (df.group == g).any()]
    frag_cols = [c for c in ("bait_fragment", "prey_fragment") if c in df]
    best = a.fragments == "best" or (a.fragments == "auto" and bool(frag_cols)
                                     and df[frag_cols].notna().any().any()
                                     and {"bait_uniprot", "prey_uniprot"} <= set(df.columns))
    label = dict(zip(a.metrics, a.labels or a.metrics))
    present = [m for m in a.metrics if m in df.columns and df[m].notna().any()
               and not (a.skip_all_zero and not df[m].fillna(0).astype(bool).any())]
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not present:
        Path(outdir / "index.html").write_text("<p>no metric values</p>")
        return 0

    ncols = min(a.ncols, len(present))
    nrows = math.ceil(len(present) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[label[m] for m in present],
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    chain = (df["chain_i"].astype(str) + "-" + df["chain_j"].astype(str)
             if "chain_i" in df else pd.Series("", index=df.index))
    # 'BPM6 × MYC2' when the results table carries display names
    name = (df["bait"].astype(str) + " × " + df["prey"].astype(str)
            if {"bait", "prey"} <= set(df.columns) else df["pair"].astype(str))
    mean = not best and a.aggregate == "mean"
    for k, m in enumerate(present):
        shades = dict(zip(GROUPS, SHADES[k % len(SHADES)]))
        per = best_fragment(df, m) if best else pair_mean(df, m, name) if mean else df
        for grp in groups:
            sub = per[per[m].notna() & (per.group == grp)]
            if sub.empty:
                continue
            if best:
                customdata = list(zip(sub["name"], sub["pair"], sub["n_fragment_pairs"]))
                hover = (f"<b>%{{customdata[0]}}</b> ({grp})<br>{label[m]}=%{{y:.4g}} (best of "
                         "%{customdata[2]} fragment pair(s))<br>best pair=%{customdata[1]}<extra></extra>")
            elif mean:
                customdata = list(zip(sub["name"], sub["pair"], sub["n_rows"]))
                hover = (f"<b>%{{customdata[0]}}</b> ({grp})<br>{label[m]}=%{{y:.4g}} (mean of "
                         "%{customdata[2]} sample × chain-pair value(s))<br>pair=%{customdata[1]}<extra></extra>")
            else:
                customdata = list(zip(name[sub.index], sub["pair"], sub["sample_index"], chain[sub.index]))
                hover = (f"<b>%{{customdata[0]}}</b> ({grp})<br>{label[m]}=%{{y:.4g}}"
                         "<br>pair=%{customdata[1]}<br>sample=%{customdata[2]}"
                         "<br>chains=%{customdata[3]}<extra></extra>")
            fig.add_trace(go.Violin(
                x=[grp] * len(sub), y=sub[m], name=grp, box_visible=True, meanline_visible=True,
                points="all", jitter=0.4, pointpos=0, marker_size=3, showlegend=False,
                line_color=shades[grp], fillcolor=shades[grp], opacity=0.85, spanmode="hard",
                customdata=customdata, hovertemplate=hover,
            ), row=k // ncols + 1, col=k % ncols + 1)
    fig.update_xaxes(showticklabels=len(groups) > 1, categoryorder="array", categoryarray=groups,
                     tickfont_size=10)
    if best:
        title = (a.title.replace("(all pairs, samples, chain pairs)", "(best fragment pair per bait × prey protein)")
                 + "<br><sup>fragment mode: one point per bait × prey protein, its best fragment pair "
                   "per metric (biased upward for proteins with more fragments)</sup>")
    elif mean:
        title = a.title.replace("(all pairs, samples, chain pairs)", "(one point per pair)") + \
            "<br><sup>each point: the pair's mean over its samples and chain pairs</sup>"
    else:
        title = a.title
    fig.update_layout(height=340 * nrows, template="plotly_dark", title=title,
                      margin=dict(t=110 if best or mean else 80))
    fig.write_html(outdir / "index.html", include_plotlyjs=True, full_html=True)
    print(f"[plot_metric_violins_interactive] {len(present)} metric(s) -> {outdir}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
