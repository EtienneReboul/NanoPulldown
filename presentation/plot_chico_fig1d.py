#!/usr/bin/env python3
"""
presentation/plot_chico_fig1d.py
================================
Redraw Fig. 1D of Chico et al. 2020 (PNAS, GFP-BPM6 co-IP volcano plot) at
high resolution from PNAS Dataset S1, Table S3 (log2 FC, adjusted p-value).

Usage (from the repo root or from presentation/):
    python presentation/plot_chico_fig1d.py
    python presentation/plot_chico_fig1d.py --dpi 600 --also-svg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

GREEN = "#00a550"   # BPM paralogs
RED = "#ff1a1a"     # candidate substrates
# AGI locus -> (label, colour, label offset in points)
HIGHLIGHT = {
    "AT3G43700": ("BPM6", GREEN, (14, 0)),
    "AT5G21010": ("BPM5", GREEN, (14, 0)),
    "AT3G03740": ("BPM4", GREEN, (22, 6)),
    "AT3G06190": ("BPM2", GREEN, (26, 4)),
    "AT5G19000": ("BPM1", GREEN, (30, -10)),
    # red labels: stacked column, label position in data coords (x, y)
    "AT4G17880": ("MYC4", RED, (0.8, 12.5)),
    "AT5G46760": ("MYC3", RED, (0.8, 10.3)),
    "AT1G32640": ("MYC2", RED, (0.8, 8.1)),
    "AT1G19180": ("JAZ1", RED, (0.8, 5.9)),
    "AT4G28910": ("NINJA", RED, (0.8, 3.7)),
}


def load(xlsx: Path) -> pd.DataFrame:
    s3 = pd.read_excel(xlsx, sheet_name="Table S3", header=6)
    s3.columns = ["accession", "logfc", "pval", "adjp", "accession2", "annotation"]
    # keep Arabidopsis loci only (drops the GFP bait construct, as in the paper)
    s3 = s3[~s3["accession"].astype(str).str.startswith("sp|")].copy()
    s3["agi"] = s3["accession"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    s3["nlogp"] = -np.log10(s3["adjp"].astype(float))
    return s3.dropna(subset=["logfc", "nlogp"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=ROOT / "data" / "pnas.1912199117.sd01.xlsx")
    ap.add_argument("--out", default=HERE / "figures" / "chico_fig1d.png")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--also-svg", action="store_true")
    a = ap.parse_args()

    df = load(Path(a.xlsx))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8, 4.9))

    ax.scatter(df["logfc"], df["nlogp"], s=9, c="#6e6e6e", alpha=0.7,
               linewidths=0, rasterized=True)
    for x in (-1, 1):
        ax.axvline(x, color="red", ls="--", lw=1.1)
    ax.axhline(-np.log10(0.05), color="red", ls="--", lw=1.1)

    for agi, (name, col, off) in HIGHLIGHT.items():
        r = df[df["agi"] == agi]
        if r.empty:
            continue
        r = r.iloc[0]
        ax.scatter(r["logfc"], r["nlogp"], s=34, c=col, edgecolors="black",
                   linewidths=0.6, zorder=4)
        data_pos = col == RED
        ax.annotate(name, (r["logfc"], r["nlogp"]), xytext=off,
                    textcoords="data" if data_pos else "offset points",
                    color=col, fontsize=13,
                    fontweight="bold" if name == "BPM6" else "normal",
                    va="center", ha="right" if data_pos else "left",
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7),
                    zorder=5)

    ax.text(0.03, 0.94, "CONTROL", transform=ax.transAxes, fontsize=14, va="top")
    ax.text(0.97, 0.94, "GFP-BPM6", transform=ax.transAxes, fontsize=14,
            va="top", ha="right", color=GREEN)
    ax.set_xlim(-12, 12)
    ax.set_ylim(0, df["nlogp"].max() * 1.05)
    ax.set_xlabel("Log2(FoldChange)")
    ax.set_ylabel("-Log10(adjusted-pvalue)")
    ax.grid(color="#e4e4e4", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = Path(a.out)
    fig.savefig(out, dpi=a.dpi, facecolor="white")
    if a.also_svg:
        fig.savefig(out.with_suffix(".svg"), facecolor="white")
    print(f"wrote {out} ({len(df)} proteins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
