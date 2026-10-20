#!/usr/bin/env bash
# One-time (and after an env change) setup on the IFB cluster. Heavy (conda
# solves + ~150k files), so run it as a SLURM job, NOT on the login node. From
# the checkout, with config.local.yaml copied from config.ifb.yaml.example:
#
#   sbatch --account=npf_abinitio --partition=fast --cpus-per-task=8 --mem=32G \
#     --time=03:00:00 --output=ifb_tests/logs/setup.log scripts/ifb/setup.sh
#
#   scripts/ifb/setup.sh [images] [envs] [weights]    # default: all three
#
#  images   <images_dir>/plip.sif and aiupred.sif, pulled from their docker images
#           (IFB has no docker daemon; one inode each)
#  envs     the rule conda envs, built by Snakemake from envs/*.yaml into
#           .snakemake/conda/ (--conda-create-envs-only) -- the same env files as
#           on the Mac; the fold env is envs/esmfold2_cuda.yaml (esmfold2.backend: torch)
#  weights  ESMFold2-Fast weights + ccd.pkl into $HF_HOME, so GPU jobs run offline
#
# Inodes: /shared/projects/npf_abinitio is capped at 500k files and a full
# project fails EVERY write in it. The envs are ~146k files (measured from
# Linux builds of the same env files), so `envs` refuses to start below
# MIN_FREE_INODES free. conda's package cache goes to node-local scratch, not to
# ~/.condarc's pkgs_dirs on the project (it would double the count).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || STEPS=(images envs weights)
want() { [[ " ${STEPS[*]} " == *" $1 "* ]]; }

PROJECT=/shared/projects/npf_abinitio
IMAGES=${IMAGES:-$PROJECT/nanopulldown_images}
HF_HOME=${HF_HOME:-$PROJECT/hf_cache}
MIN_FREE_INODES=${MIN_FREE_INODES:-170000}   # ~146k envs + margin
CONTROLLER=$PROJECT/conda/envs/abcfold-npf-pipeline/bin   # snakemake 9 + slurm plugin
CONDA_BIN=/shared/software/miniconda_deployment/bin       # `conda` itself (the controller has none)
export PATH=$CONTROLLER:$PATH:$CONDA_BIN

SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/nanopulldown_setup.XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
export APPTAINER_TMPDIR=$SCRATCH/tmp APPTAINER_CACHEDIR=$SCRATCH/cache CONDA_PKGS_DIRS=$SCRATCH/pkgs
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$CONDA_PKGS_DIRS" "$IMAGES" "$HF_HOME"

inodes() { getfattr --only-values -n ceph.dir.rentries "$PROJECT" 2>/dev/null; }
echo "host $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, steps: ${STEPS[*]}; scratch $SCRATCH"
echo "project inodes before: $(inodes) / 500000"

pull() {  # name, docker ref -- written under a temporary name, so a killed pull leaves no truncated .sif
  [ -e "$IMAGES/$1.sif" ] && return 0
  apptainer pull "$IMAGES/$1.sif.partial" "docker://$2"
  mv "$IMAGES/$1.sif.partial" "$IMAGES/$1.sif"
}
if want images; then
  pull plip pharmai/plip:latest
  pull aiupred ghcr.io/doszilab/aiupred:cpu
fi

if want envs; then
  free=$((500000 - $(inodes)))
  if [ "$free" -lt "$MIN_FREE_INODES" ]; then
    echo "REFUSING to build the conda envs: $free inodes free < $MIN_FREE_INODES needed" >&2
    exit 1
  fi
  snakemake --use-conda --conda-create-envs-only --cores "${SLURM_CPUS_PER_TASK:-4}"
  echo "project inodes after envs: $(inodes) / 500000"
fi

# weights, downloaded with the fold env's own huggingface_hub
if want weights; then
  py=$(ls -d .snakemake/conda/*/ | while read -r d; do
         [ -e "$d/lib/python3.12/site-packages/esm" ] && [ -e "$d/lib/python3.12/site-packages/torch" ] && echo "$d/bin/python"
       done | head -1)
  [ -n "$py" ] || { echo "no esmfold2_cuda env in .snakemake/conda -- run the envs step first" >&2; exit 1; }
  HF_HOME=$HF_HOME "$py" -c '
from huggingface_hub import hf_hub_download, snapshot_download
print(snapshot_download("biohub/ESMFold2-Fast"))
print(hf_hub_download("biohub/ESMFold2", "ccd.pkl"))   # read by ESMFold2InputBuilder()
'
fi

echo "project inodes after:  $(inodes) / 500000"
ls -lh "$IMAGES"/*.sif
