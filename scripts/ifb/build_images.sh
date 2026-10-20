#!/usr/bin/env bash
# Build the rule-env images for the IFB cluster (linux/amd64) on this Mac with
# Docker Desktop, and save each as a tarball for scripts/ifb/setup.sh to turn
# into a .sif on the cluster:
#
#   scripts/ifb/build_images.sh [env ...]      # default: all four
#   -> build/images/nanopulldown-<env>.tar
#
# One image per envs/<env>.yaml (containers/Dockerfile). The analysis image is
# built FROM the AIUPred image (see containers/Dockerfile). Builds run under
# amd64 emulation: slow (the esmfold2_cuda one pulls ~3 GB of torch/CUDA
# wheels), but only needed when an env file changes. Don't run it while a
# pipeline run is using Docker (PLIP/AIUPred share the Docker VM's memory).
set -euo pipefail
cd "$(dirname "$0")/../.."

ENVS=("$@")
[ ${#ENVS[@]} -gt 0 ] || ENVS=(analysis minimize pliparser esmfold2_cuda)
OUT=build/images
mkdir -p "$OUT"

for env in "${ENVS[@]}"; do
  [ -f "envs/$env.yaml" ] || { echo "no envs/$env.yaml" >&2; exit 1; }
  base=()
  [ "$env" = analysis ] && base=(--build-arg BASE=ghcr.io/doszilab/aiupred:cpu)
  tag="nanopulldown-$env:latest"
  echo "== $tag"
  docker buildx build --platform linux/amd64 --load -f containers/Dockerfile \
    --build-arg ENV_FILE="envs/$env.yaml" "${base[@]}" -t "$tag" .
  docker save "$tag" -o "$OUT/nanopulldown-$env.tar"
  ls -lh "$OUT/nanopulldown-$env.tar"
done

cat <<EOF

Next: copy the tarballs to the cluster and convert them there, e.g.
  rsync -avP $OUT/ core.cluster.france-bioinformatique.fr:/shared/projects/npf_abinitio/nanopulldown_images/tar/
  ssh core.cluster.france-bioinformatique.fr 'cd <checkout> && scripts/ifb/setup.sh'
EOF
