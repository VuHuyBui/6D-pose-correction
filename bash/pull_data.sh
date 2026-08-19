#!/usr/bin/env bash
# =========================================================================== #
# pull_data.sh -- bring vit_hope job output back down from the Cassandra
# login node. Run this after your sbatch job has finished.
#
# Four sources, all under /data on the cluster (see slurm/run_all_objects.sh,
# which writes both of its outputs into ${OUT_ROOT}/obj_XXXXXX/):
#
#   1. predictions  ${REMOTE_RENDERED_ROOT}/obj_*/predictions_*.csv   [default]
#   2. sweeps       ${REMOTE_RENDERED_ROOT}/sweeps/*.csv              [default]
#   3. slurm logs   ${REMOTE_LOG_DIR}/slurm.*.out                     [default]
#   4. weights      ${REMOTE_RENDERED_ROOT}/obj_*/*.pt                [--weights]
#
# Weights are OFF by default. The only thing done locally with job output is
# visualize.py, which corrects poses from a predictions CSV and re-renders --
# it never loads a checkpoint. Pulling hundreds of MB of .pt per sweep buys
# nothing, so ask for them explicitly when you actually want to resume or
# fine-tune from one.
#
# (2) is the result CSV slurm/sweep_hparams.sh appends one row per (trial,
# object) to. It is a few KB, so it comes down by default -- pull it and rank
# the trials locally with `python analyze_sweep.py results/sweeps/*.csv`. The
# sweep's own checkpoints and per-trial predictions live one level deeper, in
# obj_*/sweep/, and are deliberately NOT pulled: (1) and (4) both match only
# the top level of each object dir.
#
# (1) and (4) filter hard: the rendered root also holds the input tiles, which
# are far too large to drag back.
#
# Additive: --delete is never used, so nothing already on this machine is
# removed. A source that does not exist yet logs a warning and the remaining
# sources are still pulled.
#
#   ./pull_data.sh              predictions + sweeps + logs
#   ./pull_data.sh --weights    ... and the checkpoints too
#   ./pull_data.sh --dry-run    preview, transfer nothing
# =========================================================================== #

set -euo pipefail

# ============================== EDIT ME ==================================== #
# Defaults below were read out of slurm/run_all_objects.sh (DATA_ROOT,
# RENDERED_ROOT) and vit_hope_dinov3.slurm (#SBATCH -o) -- verify them against
# your own account before the first real run.

REMOTE_HOST="cassandra-login-node"                            # ~/.ssh/config alias, or user@host
REMOTE_RENDERED_ROOT="/data/s-2657115/vit_hope_rendered"      # RENDERED_ROOT / OUT_ROOT
REMOTE_LOG_DIR="/data/s-2657115/logs"                         # #SBATCH -o directory
LOCAL_DIR="/home/vubui/daad-rise-2026/vit_hope"               # destination (this folder)
SSH_OPTS=""                                                   # e.g. "-p 22 -i ~/.ssh/id_ed25519 -J bastion"

# Toggle any of these. PULL_WEIGHTS is overridden by --weights / --no-weights,
# PULL_SWEEPS by --sweeps / --no-sweeps.
PULL_WEIGHTS=0
PULL_PREDICTIONS=1
PULL_SWEEPS=1
PULL_LOGS=1

# Which files count as "predictions" under REMOTE_RENDERED_ROOT/obj_*/.
# Add patterns here rather than widening the transfer -- everything not listed
# is left on the cluster. metrics*.csv and corrected_poses*.json are NOT here:
# only visualize.py writes those, and it cannot run on a compute node (no
# OpenGL), so they exist locally and never on the cluster.
PREDICTION_PATTERNS=(
  "predictions_*.csv"
)

# Sweep result CSVs, relative to REMOTE_RENDERED_ROOT. This mirrors RESULT_CSV
# in slurm/sweep_hparams.sh, which defaults to ${OUT_ROOT}/sweeps/ -- change it
# here too if you point OUT_ROOT somewhere other than RENDERED_ROOT.
REMOTE_SWEEPS_SUBDIR="sweeps"
# Match every CSV in that directory rather than sweep_*.csv: the default name is
# sweep_<model>_<tag>.csv, but a job submitted with RESULT_CSV=... can call it
# anything, and the directory holds nothing else.
SWEEP_PATTERNS=(
  "*.csv"
)

# Where the pulled files land locally.
LOCAL_WEIGHTS_DIR="${LOCAL_DIR%/}/weights"
LOCAL_PREDICTIONS_DIR="${LOCAL_DIR%/}/results/remote_predictions"
LOCAL_SWEEPS_DIR="${LOCAL_DIR%/}/results/sweeps"
LOCAL_LOGS_DIR="${LOCAL_DIR%/}/results/slurm_logs"
# =========================================================================== #

usage() {
  cat <<EOF
usage: ${0##*/} [--weights|--no-weights] [--sweeps|--no-sweeps]
       [--dry-run|-n] [--help|-h]

  --weights       also pull the .pt checkpoints (off by default -- nothing
                  local needs them; visualize.py works from the CSV alone)
  --no-weights    skip them (the default; here for explicitness)
  --sweeps        pull the hyper-parameter sweep result CSVs (the default)
  --no-sweeps     skip them
  --dry-run, -n   show what would transfer, move nothing
  --help,    -h   this message

Edit the "EDIT ME" block at the top of this script to set the paths and to
toggle which of weights / predictions / sweeps / logs get pulled.
EOF
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights)    PULL_WEIGHTS=1; shift ;;
    --no-weights) PULL_WEIGHTS=0; shift ;;
    --sweeps)     PULL_SWEEPS=1; shift ;;
    --no-sweeps)  PULL_SWEEPS=0; shift ;;
    -n|--dry-run) DRY_RUN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)            echo "error: unknown argument '$1'" >&2; echo >&2; usage >&2; exit 2 ;;
  esac
done

# --- guards ---------------------------------------------------------------- #
for var in REMOTE_HOST LOCAL_DIR; do
  if [[ -z "${!var}" ]]; then
    echo "error: ${var} is empty -- set it in the EDIT ME block of $0" >&2
    exit 1
  fi
done

if ! command -v rsync >/dev/null 2>&1; then
  echo "error: rsync not found on this machine" >&2
  exit 1
fi

# --- shared rsync setup ----------------------------------------------------- #
RSYNC_ARGS=(-avz --partial --progress --human-readable)
(( DRY_RUN )) && RSYNC_ARGS+=(--dry-run)

SSH_CMD="ssh"
if [[ -n "${SSH_OPTS}" ]]; then
  SSH_CMD="ssh ${SSH_OPTS}"
fi

FAILED=()
PULLED=()

ensure_dir() {
  (( DRY_RUN )) || mkdir -p "$1"
}

# do_rsync <label> <rsync args...>
# A missing remote directory makes rsync exit non-zero. Without this wrapper
# `set -e` would abort the whole script partway through -- so a run that
# produced weights but no predictions yet would never reach the logs.
do_rsync() {
  local label="$1"; shift
  echo
  echo "--- ${label} ---"
  if rsync "${RSYNC_ARGS[@]}" -e "${SSH_CMD}" "$@"; then
    PULLED+=("${label}")
  else
    local rc=$?
    echo "    WARNING: ${label} failed (rsync exit ${rc}) -- continuing" >&2
    FAILED+=("${label}")
  fi
}

echo "=== vit_hope :: pull data ==="
(( DRY_RUN )) && echo "*** DRY RUN -- nothing will be transferred ***"
echo "from: ${REMOTE_HOST}"
echo "to:   ${LOCAL_DIR%/}"
echo "============================="

# --- 1. per-object predictions ---------------------------------------------- #
# --include the obj_* dirs, --include the wanted files inside them, --exclude
# everything else. --prune-empty-dirs stops objects with no CSV from creating
# empty local directories.
if (( PULL_PREDICTIONS )); then
  if [[ -z "${REMOTE_RENDERED_ROOT}" ]]; then
    echo "error: PULL_PREDICTIONS=1 but REMOTE_RENDERED_ROOT is empty" >&2
    exit 1
  fi
  ensure_dir "${LOCAL_PREDICTIONS_DIR}"

  FILTERS=(--prune-empty-dirs --include="obj_*/")
  for pat in "${PREDICTION_PATTERNS[@]}"; do
    FILTERS+=("--include=obj_*/${pat}")
  done
  FILTERS+=(--exclude="*")

  do_rsync "predictions" \
    "${FILTERS[@]}" \
    "${REMOTE_HOST}:${REMOTE_RENDERED_ROOT%/}/" \
    "${LOCAL_PREDICTIONS_DIR%/}/"
fi

# --- 2. sweep result CSVs ---------------------------------------------------- #
# One flat directory of small CSVs, so this is a filtered copy of that directory
# alone -- not a walk of the rendered root. A sweep that has never run leaves no
# sweeps/ dir on the cluster; rsync then exits non-zero and do_rsync downgrades
# that to a warning, which is the intended behaviour here.
if (( PULL_SWEEPS )); then
  if [[ -z "${REMOTE_RENDERED_ROOT}" ]]; then
    echo "error: PULL_SWEEPS=1 but REMOTE_RENDERED_ROOT is empty" >&2
    exit 1
  fi
  ensure_dir "${LOCAL_SWEEPS_DIR}"

  SWEEP_FILTERS=()
  for pat in "${SWEEP_PATTERNS[@]}"; do
    SWEEP_FILTERS+=("--include=${pat}")
  done
  SWEEP_FILTERS+=(--exclude="*")

  do_rsync "sweeps" \
    "${SWEEP_FILTERS[@]}" \
    "${REMOTE_HOST}:${REMOTE_RENDERED_ROOT%/}/${REMOTE_SWEEPS_SUBDIR%/}/" \
    "${LOCAL_SWEEPS_DIR%/}/"
fi

# --- 3. slurm logs ---------------------------------------------------------- #
if (( PULL_LOGS )); then
  if [[ -z "${REMOTE_LOG_DIR}" ]]; then
    echo "error: PULL_LOGS=1 but REMOTE_LOG_DIR is empty" >&2
    exit 1
  fi
  ensure_dir "${LOCAL_LOGS_DIR}"
  do_rsync "slurm logs" \
    --include="slurm.*.out" --exclude="*" \
    "${REMOTE_HOST}:${REMOTE_LOG_DIR%/}/" \
    "${LOCAL_LOGS_DIR%/}/"
fi

# --- 4. trained weights (opt-in: --weights) --------------------------------- #
# Checkpoints now live beside the tiles, one per object dir. The source is left
# unquoted-glob on purpose: obj_*/*.pt is expanded by the *remote* shell, so the
# .pt files arrive flat in weights/ rather than nested under obj_XXXXXX/. Their
# filenames already carry the object id (obj_000006_dinov2_frozen.pt), so the
# directory level would add nothing. No match on the cluster makes rsync exit
# non-zero, which do_rsync reports as a warning.
if (( PULL_WEIGHTS )); then
  if [[ -z "${REMOTE_RENDERED_ROOT}" ]]; then
    echo "error: PULL_WEIGHTS=1 but REMOTE_RENDERED_ROOT is empty" >&2
    exit 1
  fi
  ensure_dir "${LOCAL_WEIGHTS_DIR}"
  do_rsync "weights" \
    "${REMOTE_HOST}:${REMOTE_RENDERED_ROOT%/}/obj_*/*.pt" \
    "${LOCAL_WEIGHTS_DIR%/}/"
fi

# --- summary ---------------------------------------------------------------- #
echo
echo "=== Summary ==="
echo "pulled (${#PULLED[@]}): ${PULLED[*]:-none}"
echo "failed (${#FAILED[@]}): ${FAILED[*]:-none}"

if (( DRY_RUN )); then
  echo
  echo "Dry run complete. Re-run without --dry-run to transfer."
fi

[[ ${#FAILED[@]} -eq 0 ]]
