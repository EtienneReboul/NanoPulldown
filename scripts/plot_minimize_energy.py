#!/usr/bin/env python3
"""
scripts/plot_minimize_energy.py — Stage 2f report figure
============================================================
One energy-vs-step line per diffusion sample, from the energy traces
(sample_<i>_energy.parquet) scripts/minimize_openmm.py writes
alongside each minimized PDB. Also writes a small failure-rate table: a
sample whose minimized PDB never appeared (its rule raised on divergence)
counts as failed, same "flag it, don't silently drop it" posture as
ab_initio_pipeline's own minimize-energy report figure.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--max-abs-energy", type=float, default=1.0e15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-table", required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    base = Path(a.results_root) / a.pair / "minimized"
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    n_ok = 0
    for i in range(a.num_samples):
        energy_pq = base / f"sample_{i}_energy.parquet"
        if not energy_pq.exists():
            continue
        df = pd.read_parquet(energy_pq)
        steps = df["step"].tolist()
        energies = df["energy_kJ_mol"].tolist()
        if any(abs(e) > a.max_abs_energy for e in energies):
            continue
        ax.plot(steps, energies, marker="o", ms=2, lw=1, label=f"sample {i}")
        n_ok += 1
    ax.set_xlabel("minimization step")
    ax.set_ylabel("potential energy (kJ/mol)")
    ax.set_title(f"{a.pair}: minimization energy ({n_ok}/{a.num_samples} sample(s))")
    if n_ok:
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.text(0.5, 0.5, "no successful minimizations", ha="center", transform=ax.transAxes)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for fmt in a.formats.split(","):
        fig.savefig(f"{out.with_suffix('')}.{fmt.strip()}")
    plt.close(fig)

    table_path = Path(a.out_table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair", "num_samples", "num_succeeded", "num_failed", "failure_rate"])
        n_failed = a.num_samples - n_ok
        writer.writerow([a.pair, a.num_samples, n_ok, n_failed,
                         round(n_failed / a.num_samples, 3) if a.num_samples else 0.0])

    print(f"[plot_minimize_energy] {a.pair}: {n_ok}/{a.num_samples} succeeded -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
