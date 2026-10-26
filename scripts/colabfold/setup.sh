#!/usr/bin/env bash
# One-time setup for fold.engine: colabfold on a GPU cluster (IFB, CAIUS, ...).
# Heavy downloads: submit it as a SLURM job from the checkout, never on a login
# node, AFTER the rule envs exist (scripts/<cluster>/setup.sh envs, with
# fold.engine: colabfold in the run overlay so envs/colabfold*.yaml get built).
# On a cluster whose compute nodes are offline (CAIUS), run the download steps
# where there is internet and the `databases` index step as a job.
#
#   sbatch --cpus-per-task=16 --mem=128G --time=24:00:00 \
#     --output=logs/colabfold_setup.log scripts/colabfold/setup.sh weights databases
#
#   scripts/colabfold/setup.sh [weights] [databases] [check]   # default: weights check
#
#  weights    AlphaFold2-multimer v3 parameters into $CF_DATA (-> colabfold.data_dir)
#  databases  ColabFold's setup_databases.sh (v1.6.3) into $CF_DB with GPU=1: UniRef30
#             2302 + ColabFoldDB envdb, padded for MMseqs2-GPU (-> colabfold.msa.db_dir).
#             ~1 TB of disk but few files (no PDB templates: SKIP_TEMPLATES); downloads resume.
#  check      MMseqs2 version and GPU support of the env's mmseqs, and the weights
#
# Paths: CF_DATA (default $PROJECT/colabfold_data), CF_DB (default
# $PROJECT/colabfold_db), PROJECT (default the checkout's parent).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || STEPS=(weights check)
want() { [[ " ${STEPS[*]} " == *" $1 "* ]]; }

PROJECT=${PROJECT:-$(cd .. && pwd)}
CF_DATA=${CF_DATA:-$PROJECT/colabfold_data}
CF_DB=${CF_DB:-$PROJECT/colabfold_db}
CF_VERSION=v1.6.3

# the CUDA fold env (envs/colabfold.yaml) as built by Snakemake under .snakemake/conda/
envbin=""
for d in .snakemake/conda/*/; do
  if compgen -G "$d/lib/python3*/site-packages/colabfold" >/dev/null && [ -x "$d/bin/mmseqs" ]; then
    envbin="${d%/}/bin"; break
  fi
done
[ -n "$envbin" ] || { echo "no envs/colabfold.yaml env in .snakemake/conda -- build the envs first" >&2; exit 1; }
echo "host $(hostname), steps: ${STEPS[*]}; env $envbin"

if want weights; then
  mkdir -p "$CF_DATA"
  "$envbin/python" -c '
import sys
from pathlib import Path
from colabfold.download import download_alphafold_params
download_alphafold_params("alphafold2_multimer_v3", Path(sys.argv[1]))
' "$CF_DATA"
  echo "weights in $CF_DATA/params -> set colabfold.data_dir: $CF_DATA"
fi

if want databases; then
  mkdir -p "$CF_DB"
  script=$CF_DB/setup_databases_$CF_VERSION.sh
  [ -s "$script" ] || curl -fsSL -o "$script" \
    "https://raw.githubusercontent.com/sokrypton/ColabFold/$CF_VERSION/setup_databases.sh"
  # No templates (the pipeline folds without them): SKIP_TEMPLATES stops the
  # script's rsync of the whole PDB mmCIF tree -- ~200k files, which alone
  # would blow IFB's 500k-inode project cap.
  touch "$CF_DB/SKIP_TEMPLATES"
  # GPU=1: the padded databases MMseqs2-GPU searches; the env's mmseqs first on PATH.
  # DOWNLOADS_ONLY=1 (env) stops after the downloads -- for a login node with
  # internet when compute nodes are offline (CAIUS); rerun without it in a job.
  (cd "$CF_DB" && PATH="$envbin:$PATH" GPU=1 bash "$script" "$CF_DB")
  echo "databases in $CF_DB -> set colabfold.msa.db_dir: $CF_DB"
  du -sh "$CF_DB"
fi

if want check; then
  echo "mmseqs: $("$envbin/mmseqs" version)"
  if "$envbin/mmseqs" --help | grep -q gpuserver; then   # setup_databases.sh's own test
    echo "mmseqs has GPU support (gpuserver)"
  else
    echo "WARNING: this mmseqs has no GPU support -- download https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz" >&2
    echo "         and point colabfold.msa.mmseqs at its bin/mmseqs" >&2
  fi
  ls "$CF_DATA"/params/download_complexes_multimer_v3_finished.txt 2>/dev/null \
    && echo "AF2-multimer v3 weights present" || echo "no weights in $CF_DATA yet (step: weights)"
fi
