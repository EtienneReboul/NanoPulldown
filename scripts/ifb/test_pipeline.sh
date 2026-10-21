#!/usr/bin/env bash
# End-to-end test of every pipeline stage on IFB, for one pair
# (config_ifb_test.yaml), in ONE SLURM job on a GPU node: Snakemake runs locally
# inside the job, so every rule runs exactly as written, in the same conda envs
# as on the Mac. The envs are built in the node's local scratch (--conda-prefix)
# and dropped afterwards: no conda files on the inode-capped project storage.
#
#   sbatch --account=npf_abinitio --partition=gpu --gpus=l40s:1 \
#     --exclude=gpu-node-7,gpu-node-9 --cpus-per-task=16 --mem=64G --time=06:00:00 \
#     --output=ifb_tests/logs/test_pipeline.log scripts/ifb/test_pipeline.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

PROJECT=/shared/projects/npf_abinitio
export PATH=$PROJECT/conda/envs/abcfold-npf-pipeline/bin:$PATH:/shared/software/miniconda_deployment/bin
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/nanopulldown_test.XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
export CONDA_PKGS_DIRS=$SCRATCH/pkgs APPTAINER_TMPDIR=$SCRATCH/atmp APPTAINER_CACHEDIR=$SCRATCH/acache
# strict channel priority: conda-forge first in every envs/*.yaml; much faster solves
# (an unconfigured conda took ~20 min for one env) and what conda-forge/bioconda require
export CONDA_CHANNEL_PRIORITY=strict
mkdir -p "$CONDA_PKGS_DIRS" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
export NANOPULLDOWN_RUN_CONFIG=config_ifb_test.yaml

inodes() { getfattr --only-values -n ceph.dir.rentries $PROJECT 2>/dev/null; }
echo "== $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h "$SCRATCH" | tail -1; df -i "$SCRATCH" | tail -1
echo "project inodes before: $(inodes)"

echo "== images (plip.sif, aiupred.sif)"
scripts/ifb/setup.sh images

echo "== conda envs into node-local scratch"
time snakemake --use-conda --conda-prefix "$SCRATCH/conda" --conda-create-envs-only --cores "${SLURM_CPUS_PER_TASK:-8}"
for d in "$SCRATCH"/conda/*/; do echo "  $(find "$d" | wc -l) files: $(ls "$d"/conda-meta | grep -m1 -oE '^(python|openmm|pytorch|torch|matplotlib)-[0-9]' || true) $d"; done

echo "== MINIMIZE env: OpenMM platforms on this GPU node"
# (no `cond && echo` as a loop's last command: under set -e/pipefail its false case kills the script)
mpy=""
for d in "$SCRATCH"/conda/*/; do
  if compgen -G "$d/lib/python3*/site-packages/openmm" >/dev/null; then mpy="$d/bin/python"; break; fi
done
"$mpy" -c "import openmm; print('openmm', openmm.__version__, [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())])" \
  || echo "WARNING: OpenMM platform check failed"

echo "== pipeline, one pair ($(date))"
set +e
time snakemake --use-conda --conda-prefix "$SCRATCH/conda" --cores "${SLURM_CPUS_PER_TASK:-8}" \
  --resources gpu=1 --keep-going --printshellcmds
rc=$?
set -e
echo "== snakemake exit $rc ($(date))"
echo "project inodes after: $(inodes)"
exit $rc
