#!/usr/bin/env bash
# Launch (or resume) a run on CAIUS from the login node, inside tmux:
#
#   export TMUX_TMPDIR=~/.tmux-sock   # hpc-login1's /tmp can be full (tmux needs a socket)
#   tmux new -s <run>
#   NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml scripts/caius/run.sh [extra snakemake args]
#
# 1. annotate_domains_batch runs HERE, locally: it is the only rule calling the
#    web (InterPro / ScanProsite), and the compute nodes are offline. So does
#    colabfold_msa with fold.engine: colabfold and colabfold.msa.mode: server
#    (the ColabFold MMseqs2 web API); local MSA search goes to SLURM (GPU).
# 2. Everything else goes to SLURM via profiles/caius.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${NANOPULLDOWN_RUN_CONFIG:?set NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml}"
export NANOPULLDOWN_RUN_CONFIG
export PATH=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/envs/nanopulldown/bin:$PATH
export CONDA_CHANNEL_PRIORITY=strict
# hpc-login1's root fs (/tmp) fills up (other users): here-documents and the
# controller's temp files go under $HOME. Jobs get /tmp back from the
# profile's tmpdir resource (exported as their TMPDIR).
export TMPDIR=$HOME/tmp/controller
mkdir -p "$TMPDIR"
{ type module >/dev/null 2>&1 || source /usr/local/Modules/5.5.0/init/bash; }
module load apptainer        # PLIP / AIUPred jobs inherit the controller's environment

targets=$(PYTHONPATH=src python - <<'PY'
from nanopulldown.core.config import load_config
from nanopulldown.core.fold_table import fold_engine
cfg = load_config(".")
print(f'{cfg["dirs"]["annotation"]}/_cache/domains.done')
if fold_engine(cfg) == "colabfold" and (cfg["colabfold"].get("msa") or {}).get("mode", "server") == "server":
    print(f'{cfg["dirs"]["postproc"]}/_colabfold/msa.done')
PY
)
echo "== $(date): web-calling rules locally:" $targets
# same --rerun-triggers as the SLURM run: with the default ones, a changed pair
# list reruns this step (from its cache, same content) and its newer .done file
# then reruns every per-pair figure downstream (1,750 jobs on 2026-10-26)
triggers=()
args=("$@")
for i in "${!args[@]}"; do
  [ "${args[$i]}" = --rerun-triggers ] && triggers=(--rerun-triggers "${args[$((i + 1))]}")
done
# targets FIRST: --rerun-triggers takes several values and would swallow them
# shellcheck disable=SC2086  # one path per word
snakemake $targets --use-conda --cores 1 "${triggers[@]}"
echo "== $(date): pipeline on SLURM (profiles/caius)"
rc=0
snakemake --profile profiles/caius "$@" || rc=$?
echo "== $(date): snakemake exit $rc"
exit $rc
