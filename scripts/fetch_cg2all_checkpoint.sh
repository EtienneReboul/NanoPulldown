#!/usr/bin/env bash
# Download cg2all's CalphaBasedModel-FIX checkpoint (~49 MB, from Zenodo) into
# the minimize_cg2all conda env(s). cg2all fetches it on first use, INTO the
# env's site-packages -- so on a cluster whose compute nodes have no internet
# (CAIUS) run this once on the login node, after the rule envs exist
# (snakemake --use-conda --conda-create-envs-only), before any job needs it.
#
#   scripts/fetch_cg2all_checkpoint.sh [env_prefix ...]
#
# No argument: every .snakemake/conda/*/ env that has convert_cg2all.
set -euo pipefail
cd "$(dirname "$0")/.."

envs=("$@")
if [ ${#envs[@]} -eq 0 ]; then
  for d in .snakemake/conda/*/; do
    [ -x "$d/bin/convert_cg2all" ] && envs+=("${d%/}")
  done
fi
[ ${#envs[@]} -gt 0 ] || { echo "no env with convert_cg2all found (envs/minimize_cg2all.yaml not built?)" >&2; exit 1; }

tmp=$(mktemp -d "${TMPDIR:-/tmp}/cg2all_ckpt.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
for env in "${envs[@]}"; do
  echo "== $env"
  # 12 ALA CA atoms on an ideal alpha helix: enough for cg2all to load its model
  "$env/bin/python" - "$tmp/ca.pdb" <<'PY'
import math, sys
with open(sys.argv[1], "w") as fh:
    for i in range(12):
        a = math.radians(100 * i)
        fh.write("ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
                 % (i + 1, i + 1, 2.3 * math.cos(a), 2.3 * math.sin(a), 1.5 * i))
    fh.write("END\n")
PY
  "$env/bin/convert_cg2all" -p "$tmp/ca.pdb" -o "$tmp/all.pdb" --cg CalphaBasedModel --fix --device cpu
  ls -la "$env"/lib/python3.*/site-packages/cg2all/model/CalphaBasedModel-FIX.ckpt | head -1
done
