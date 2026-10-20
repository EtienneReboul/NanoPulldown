#!/usr/bin/env bash
# One-time (and after an env change) setup on the IFB login node, from the
# checkout:   scripts/ifb/setup.sh
#
#  1. <images>/tar/nanopulldown-<env>.tar (scripts/ifb/build_images.sh, copied
#     from the Mac) -> <images>/nanopulldown-<env>.sif
#  2. <images>/plip.sif from docker://pharmai/plip
#  3. ESMFold2-Fast weights + ccd.pkl into $HF_HOME, so GPU jobs run offline
#     (esmfold2.hf_home, HF_HUB_OFFLINE=1)
#
# Inodes: /shared/projects/npf_abinitio is capped at 500k files and already
# near it -- a full project blocks EVERY write. So everything that unpacks
# (apptainer's layer extraction, its cache) goes to the login node's local
# /tmp; only the finished .sif files (one inode each) and the HF cache
# (~15 files) land on project storage.
set -euo pipefail
cd "$(dirname "$0")/../.."

IMAGES=${IMAGES:-/shared/projects/npf_abinitio/nanopulldown_images}
HF_HOME=${HF_HOME:-/shared/projects/npf_abinitio/hf_cache}
SCRATCH=$(mktemp -d /tmp/nanopulldown_setup.XXXXXX)
trap 'rm -rf "$SCRATCH"' EXIT
export APPTAINER_TMPDIR=$SCRATCH/tmp APPTAINER_CACHEDIR=$SCRATCH/cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$IMAGES" "$HF_HOME"

echo "project inodes before: $(getfattr --only-values -n ceph.dir.rentries /shared/projects/npf_abinitio 2>/dev/null) / 500000"

# 1. rule-env images (rebuilt only when the tarball is newer than the .sif)
for tar in "$IMAGES"/tar/nanopulldown-*.tar; do
  [ -e "$tar" ] || { echo "no tarballs in $IMAGES/tar -- run scripts/ifb/build_images.sh on the Mac first" >&2; exit 1; }
  sif="$IMAGES/$(basename "$tar" .tar).sif"
  if [ ! -e "$sif" ] || [ "$tar" -nt "$sif" ]; then
    echo "== $sif"
    apptainer build --force "$sif" "docker-archive://$tar"
  fi
done

# 2. PLIP
[ -e "$IMAGES/plip.sif" ] || apptainer pull "$IMAGES/plip.sif" docker://pharmai/plip:latest

# 3. weights, downloaded with the fold image's own huggingface_hub
apptainer exec --cleanenv --env HF_HOME="$HF_HOME" --bind /shared \
  "$IMAGES/nanopulldown-esmfold2_cuda.sif" python -c '
from huggingface_hub import hf_hub_download, snapshot_download
print(snapshot_download("biohub/ESMFold2-Fast"))
print(hf_hub_download("biohub/ESMFold2", "ccd.pkl"))   # read by ESMFold2InputBuilder()
'

echo "project inodes after:  $(getfattr --only-values -n ceph.dir.rentries /shared/projects/npf_abinitio 2>/dev/null) / 500000"
ls -lh "$IMAGES"/*.sif
