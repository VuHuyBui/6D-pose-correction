#!/usr/bin/env bash
# =========================================================================== #
# push_code.sh -- send vit_hope source code up to the Cassandra login node.
#
# Code only. Results, weights, the venv and the rendered dataset are all
# excluded, so a push is small and can never clobber what a job produced on
# the cluster. Additive: --delete is never used, nothing on the remote is
# removed.
#
#   ./push_code.sh              transfer
#   ./push_code.sh --dry-run    preview, transfer nothing
#
# Then submit the job yourself on the login node:
#   ssh cassandra-login-node
#   sbatch daad-rise-2026/vit_hope/slurm/vit_hope_dinov3.slurm
#
# When the job finishes, bring the outputs back with ./pull_data.sh
# =========================================================================== #

set -euo pipefail

# ============================== EDIT ME ==================================== #
# Defaults below were read out of slurm/run_all_objects.sh (REPO_DIR) -- verify
# them against your own account before the first real run.

REMOTE_HOST="cassandra-login-node"                        # ~/.ssh/config alias, or user@host
LOCAL_DIR="/home/vubui/daad-rise-2026/vit_hope"           # source (this folder)
REMOTE_DIR="/home/s-2657115/daad-rise-2026/vit_hope"      # destination on the cluster
SSH_OPTS=""                                               # e.g. "-p 22 -i ~/.ssh/id_ed25519 -J bastion"

# Never pushed. Keep results/ and weights/ here: they are what pull_data.sh
# brings *back*, and pushing them would overwrite fresher cluster output.
EXCLUDES=(
  ".git/" ".gitignore"
  ".venv/" "venv/" "env/"                 # the cluster builds its own venv
  "__pycache__/" "*.pyc" ".ipynb_checkpoints/"
  "results/"                              # local analysis output (~400M)
  "weights/"                              # trained checkpoints come back, not up
  "*.pt" "*.pth" "*.ckpt" "*.safetensors"
  "datasets/" "data/"                     # rendered data is staged on /data
  "slurm-*.out" "*.log"
  "wandb/" ".DS_Store"
)
# =========================================================================== #

usage() {
  cat <<EOF
usage: ${0##*/} [--dry-run|-n] [--help|-h]

  --dry-run, -n   show what would transfer, move nothing
  --help,    -h   this message

Edit the "EDIT ME" block at the top of this script to set the paths.
EOF
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run) DRY_RUN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)            echo "error: unknown argument '$1'" >&2; echo >&2; usage >&2; exit 2 ;;
  esac
done

# --- guards ---------------------------------------------------------------- #
for var in REMOTE_HOST LOCAL_DIR REMOTE_DIR; do
  if [[ -z "${!var}" ]]; then
    echo "error: ${var} is empty -- set it in the EDIT ME block of $0" >&2
    exit 1
  fi
done

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "error: LOCAL_DIR '${LOCAL_DIR}' is not a directory" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "error: rsync not found on this machine" >&2
  exit 1
fi

# --- build the rsync invocation -------------------------------------------- #
# -a archive, -v verbose, -z compress on the wire, --partial so an interrupted
# transfer resumes instead of restarting from zero.
RSYNC_ARGS=(-avz --partial --progress --human-readable)

for pat in "${EXCLUDES[@]}"; do
  RSYNC_ARGS+=("--exclude=${pat}")
done

(( DRY_RUN )) && RSYNC_ARGS+=(--dry-run)

SSH_CMD="ssh"
if [[ -n "${SSH_OPTS}" ]]; then
  SSH_CMD="ssh ${SSH_OPTS}"
fi

# Trailing slashes are normalised here, not in the EDIT ME block: "SRC/" copies
# the *contents* of SRC into DEST, while "SRC" copies the directory itself and
# would produce vit_hope/vit_hope/ on the remote.
SRC="${LOCAL_DIR%/}/"
DEST="${REMOTE_HOST}:${REMOTE_DIR%/}/"

echo "=== vit_hope :: push code ==="
(( DRY_RUN )) && echo "*** DRY RUN -- nothing will be transferred ***"
echo "from: ${SRC}"
echo "to:   ${DEST}"
echo "============================="

rsync "${RSYNC_ARGS[@]}" -e "${SSH_CMD}" "${SRC}" "${DEST}"

echo
if (( DRY_RUN )); then
  echo "Dry run complete. Re-run without --dry-run to transfer."
  exit 0
fi

cat <<EOF
Push complete.

Next, submit the job on the cluster:

  ssh ${REMOTE_HOST}
  cd ${REMOTE_DIR%/}
  sbatch slurm/vit_hope_dinov3.slurm        # or vit_hope_{dinov2,swinv2,vanilla}.slurm

Watch it with:  squeue -u \$USER
When it finishes, bring the outputs back with:  ./pull_data.sh
EOF
