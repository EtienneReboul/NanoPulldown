#!/usr/bin/env bash
# Run by Snakemake (--use-conda) inside the new envs/sparrow.yaml env, right
# after it is created. SPARROW requires parrot as an unpinned git URL, so the
# yaml cannot pin it; pin it here instead (no deps: SPARROW already pulled them).
set -euo pipefail
python -m pip install --no-deps --force-reinstall \
  "idptools-parrot @ git+https://github.com/idptools/parrot.git@8d5c671a6796ab69ace69e5cdf98fb2469deafb2"
