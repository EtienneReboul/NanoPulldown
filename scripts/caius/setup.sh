#!/usr/bin/env bash
# One-time (and after an env change) setup on CAIUS, the Université de
# Strasbourg mésocentre. Runs ON THE LOGIN NODE, niced: the compute nodes have
# no internet, so every download (micromamba, conda/pip packages, images,
# weights) has to happen here. From the checkout:
#
#   nice -n 19 scripts/caius/setup.sh [micromamba] [controller] [config] [images] [envs] [weights]
#                                                      # default: all six, in this order
#
#  micromamba  the static binary into ~/.local/bin, root prefix ~/micromamba
#  controller  the `nanopulldown` env from envs/controller.yaml (snakemake 9 +
#              SLURM plugin + conda, which --use-conda builds the rule envs with)
#  config      config.local.yaml from config.caius.yaml.example (if absent)
#  images      <images_dir>/plip.sif and aiupred.sif, pulled from their docker images
#  envs        the rule conda envs into .snakemake/conda/ (--conda-create-envs-only),
#              the same envs/*.yaml as on the Mac and IFB
#  weights     ESMFold2-Fast weights + ccd.pkl into $HF_HOME
#
# /tmp on hpc-login has <1 GB free, so TMPDIR and every cache go under $HOME.
set -euo pipefail
cd "$(dirname "$0")/../.."

STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || STEPS=(micromamba controller config images envs weights)
want() { [[ " ${STEPS[*]} " == *" $1 "* ]]; }

export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
MM=$HOME/.local/bin/micromamba
ENV=$MAMBA_ROOT_PREFIX/envs/nanopulldown
IMAGES=${IMAGES:-$HOME/nanopulldown_images}
HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TMPDIR=$HOME/tmp/nanopulldown_setup
export APPTAINER_TMPDIR=$TMPDIR/apptainer APPTAINER_CACHEDIR=$TMPDIR/apptainer_cache
export PIP_CACHE_DIR=$TMPDIR/pip CONDA_CHANNEL_PRIORITY=strict
mkdir -p "$HOME/.local/bin" "$TMPDIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$IMAGES" "$HF_HOME"
trap 'rm -rf "$TMPDIR"' EXIT
echo "host $(hostname), steps: ${STEPS[*]}"

if want micromamba && [ ! -x "$MM" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME/.local" bin/micromamba
fi
"$MM" --version

if want controller; then
  if [ -d "$ENV" ]; then "$MM" update -y -p "$ENV" -f envs/controller.yaml
  else "$MM" create -y -p "$ENV" -f envs/controller.yaml; fi
  # rule envs: conda's own package cache (not micromamba's) under the root prefix
  "$ENV/bin/conda" config --file "$ENV/.condarc" --add pkgs_dirs "$MAMBA_ROOT_PREFIX/conda_pkgs"
fi
export PATH=$ENV/bin:$PATH
snakemake --version

if want config && [ ! -e config.local.yaml ]; then
  sed "s|<HOME>|$HOME|g" config.caius.yaml.example > config.local.yaml
  echo "wrote config.local.yaml -- add the Mac's annotate: block (InterProScan email)"
fi

if want images; then
  { type module >/dev/null 2>&1 || source /usr/local/Modules/5.5.0/init/bash; }
  module load apptainer
  for spec in plip=pharmai/plip:latest aiupred=ghcr.io/doszilab/aiupred:cpu; do
    name=${spec%%=*}
    [ -e "$IMAGES/$name.sif" ] && continue    # temporary name: a killed pull leaves no truncated .sif
    apptainer pull "$IMAGES/$name.sif.partial" "docker://${spec#*=}"
    mv "$IMAGES/$name.sif.partial" "$IMAGES/$name.sif"
  done
  ls -lh "$IMAGES"/*.sif
fi

if want envs; then
  snakemake --use-conda --conda-create-envs-only --cores 2
fi

if want weights; then
  py=""
  for d in .snakemake/conda/*/; do
    if compgen -G "$d/lib/python3*/site-packages/esm" >/dev/null && compgen -G "$d/lib/python3*/site-packages/torch" >/dev/null; then
      py="$d/bin/python"; break
    fi
  done
  [ -n "$py" ] || { echo "no esmfold2_cuda env in .snakemake/conda -- run the envs step first" >&2; exit 1; }
  HF_HOME=$HF_HOME "$py" -c '
from huggingface_hub import hf_hub_download, snapshot_download
print(snapshot_download("biohub/ESMFold2-Fast"))
print(hf_hub_download("biohub/ESMFold2", "ccd.pkl"))   # read by ESMFold2InputBuilder()
'
fi
diskquota | tail -2
