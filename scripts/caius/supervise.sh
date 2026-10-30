#!/usr/bin/env bash
# Keep a CAIUS run going unattended: whenever the controller (tmux session
# $SESSION, started by scripts/caius/run.sh) is gone, dry-run the workflow and,
# if anything is left -- jobs that failed or timed out, or a controller crash --
# relaunch it. Gives up after MAX_RELAUNCH relaunches, or after two in a row
# that leave the same number of jobs to do. Writes a status summary to
# $STATUS every POLL seconds. Run it in its own tmux session on hpc-login:
#
#   export TMUX_TMPDIR=~/.tmux-sock
#   tmux new -d -s supervise "NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml scripts/caius/supervise.sh"
#
# Relaunches use --rerun-triggers mtime: a new controller re-plans the fold
# chunks, and with the default triggers the finished chunks' changed `pairs`
# param reruns them -- relinking their structures and so redoing every
# minimization / PLIP downstream (750 jobs on 2026-10-22).
# EXTRA_ARGS (e.g. --groups ... --group-components ...) is appended to every relaunch.
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${NANOPULLDOWN_RUN_CONFIG:?set NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml}"
export NANOPULLDOWN_RUN_CONFIG
export TMUX_TMPDIR=${TMUX_TMPDIR:-$HOME/.tmux-sock}
export TMPDIR=$HOME/tmp/controller            # hpc-login1's /tmp fills up (see run.sh)
mkdir -p "$TMPDIR" "$TMUX_TMPDIR"
export PATH=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/envs/nanopulldown/bin:$PATH

SESSION=${SESSION:-dne1}
POLL=${POLL:-300}
MAX_RELAUNCH=${MAX_RELAUNCH:-4}
EXTRA_ARGS=${EXTRA_ARGS:-}
STATUS=${STATUS:-$HOME/caius_logs/STATUS}
logs=$(python -c "
import os, yaml
cfg = yaml.safe_load(open('config.yaml'))
for f in ('config.local.yaml', os.environ['NANOPULLDOWN_RUN_CONFIG']):
    if os.path.exists(f):
        cfg['dirs'].update((yaml.safe_load(open(f)) or {}).get('dirs') or {})
print(cfg['dirs']['logs'], cfg['dirs']['postproc'])
")
read -r LOGS POST <<<"$logs"
LOG=$LOGS/snakemake.log

status() {
  {
    echo "updated $(date '+%a %d %b %H:%M')  (supervisor: $1)"
    echo "folded pairs:  $(ls "$POST"/_fold_cache/*/fold.parquet "$POST"/_colabfold_cache/*/fold.parquet 2>/dev/null | wc -l)"
    echo "plip reports:  $(ls -d "$POST"/*/plip/sample_*_report/*_report.txt 2>/dev/null | wc -l)"
    echo "relaunches:    $relaunches / $MAX_RELAUNCH"
    [ -f "$LOG" ] && echo "progress:      $(grep -E 'steps \(' "$LOG" | tail -1)"
    [ -f "$LOG" ] && echo "errors (this controller): $(grep -c '^Error in rule' "$LOG")"
    echo "queue:"; squeue -u "$USER" -h -o "  %P %T" | sort | uniq -c
  } > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"
}

controller_alive() {
  # "=": exact name -- a bare target prefix-matches, so dne1cf_sup would count as dne1cf
  tmux has-session -t "=$SESSION" 2>/dev/null
}

remaining() {   # jobs a dry run would still do; 0 = finished
  local out
  out=$(snakemake --profile profiles/caius -n --rerun-triggers mtime 2>&1)
  if grep -q "Nothing to be done" <<<"$out"; then echo 0; return; fi
  awk '$1 == "total" {print $2; exit}' <<<"$out"
}

relaunches=0
last_left=""
while true; do
  if controller_alive; then
    status "controller running"
    sleep "$POLL"
    continue
  fi
  status "controller exited -- checking what is left"
  sleep 120                                    # let SLURM / the filesystem settle
  # a crashed controller can leave its lock behind; none is running now
  snakemake --unlock >> "$STATUS.log" 2>&1
  left=$(remaining)
  echo "$(date '+%F %T') controller gone; dry run: ${left:-?} job(s) left" >> "$STATUS.log"
  if [ "$left" = 0 ]; then
    status "FINISHED -- nothing left to do"
    echo "$(date '+%F %T') FINISHED" >> "$STATUS.log"
    exit 0
  fi
  if [ -z "$left" ]; then
    status "STOPPED -- the dry run failed, see $STATUS.log"
    snakemake --profile profiles/caius -n --rerun-triggers mtime >> "$STATUS.log" 2>&1
    exit 1
  fi
  if [ "$relaunches" -ge "$MAX_RELAUNCH" ] || [ "$left" = "$last_left" ]; then
    status "STOPPED -- $left job(s) left after $relaunches relaunch(es) without progress"
    echo "$(date '+%F %T') STOPPED with $left left" >> "$STATUS.log"
    exit 1
  fi
  relaunches=$((relaunches + 1))
  last_left=$left
  [ -f "$LOG" ] && cp "$LOG" "$LOGS/snakemake.before_relaunch$relaunches.log"
  echo "$(date '+%F %T') relaunch $relaunches ($left job(s) left)" >> "$STATUS.log"
  tmux new -d -s "$SESSION" \
    "NANOPULLDOWN_RUN_CONFIG=$NANOPULLDOWN_RUN_CONFIG scripts/caius/run.sh --rerun-triggers mtime $EXTRA_ARGS > $LOG 2>&1"
  sleep "$POLL"
done
