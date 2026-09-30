#!/usr/bin/env bash
# scripts/build_report.sh [repo_root]
# Rebuilds reports/report.zip from the just-completed Snakemake run.
# Invoked from Snakefile's `onsuccess:` hook, so it must never fail the
# pipeline — a broken report build is logged and swallowed.
#
# Unlike ab_initio_pipeline (one report.zip per stage), nanopulldown builds
# only ONE report — preprocessing has no report of its own, since its only
# output that matters (the annotation review) is meant to be read directly
# from data/annotation/<pair>/annotation.yaml, not packaged. Its rules just
# carry no `report()` markers, so they don't show up in the zip.
#
# The report is a .zip (not a bare .html) because `snakemake --report` only
# embeds interactive HTML items — the datavzrd table bundle — for the zip
# form. Unzip it and open report.html; the dark theme lives in
# report/custom.css, injected via --report-stylesheet.
set -uo pipefail

REPO="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SNAKEFILE="$REPO/Snakefile"
OUT="$REPO/reports/report.zip"
CSS="$REPO/report/custom.css"

mkdir -p "$REPO/reports"
echo "[build_report] -> $OUT"

args=(-s "$SNAKEFILE" --report "$OUT")
[[ -f "$CSS" ]] && args+=(--report-stylesheet "$CSS")

if command -v snakemake >/dev/null 2>&1; then
    ( cd "$REPO" && snakemake "${args[@]}" ) \
        && echo "[build_report] ok: $OUT" \
        || echo "[build_report] WARNING: report build failed (non-fatal)" >&2
else
    echo "[build_report] WARNING: snakemake not on PATH — skipping report build" >&2
fi
exit 0
