#!/usr/bin/env bash
# =========================================================================== #
# push_rendered.sh -- stage the jittered/rendered dataset up to the cluster's
# RENDERED_ROOT, where run_all_objects.sh expects to find it:
#
#   ${REMOTE_RENDERED_ROOT}/obj_000002/jitter_all.csv
#   ${REMOTE_RENDERED_ROOT}/obj_000002/run_0/ ... run_4/   (tiles)
#
# This is the big one: ~2 GB across ~32k mostly-PNG files. Tuned for that,
# and deliberately different from push_code.sh:
#
#   * no -z        PNGs are already compressed; -z just burns CPU on both ends
#   * progress2    one overall progress line instead of 32k filenames
#   * --partial    a dropped link resumes instead of restarting the file
#
# rsync is incremental, so an interrupted run is fixed by simply re-running --
# only what is missing goes over the wire.
#
# Additive: --delete is never used, so predictions already written into
# obj_*/ on the cluster are never removed.
#
#   ./push_rendered.sh                  push every object
#   ./push_rendered.sh --dry-run        preview, transfer nothing
#   ./push_rendered.sh --obj 2 6 14     push only these object ids
#   ./push_rendered.sh --verbose        list every file (noisy)
# =========================================================================== #

set -euo pipefail

# ============================== EDIT ME ==================================== #
# NOTE ON THE ACCOUNT NUMBER: you asked for /data/s-2657155/vit_hope_rendered,
# but your ~/.ssh/config (User s-2657115) and every slurm script
# (DATA_ROOT=/data/s-2657115) say s-26571*15*. Assuming a transposed digit and
# using the verified s-2657115 -- change it here if that is wrong.

REMOTE_HOST="cassandra-login-node"
LOCAL_RENDERED_ROOT="/home/vubui/daad-rise-2026/vit_hope_rendered"
REMOTE_RENDERED_ROOT="/data/s-2657115/vit_hope_rendered"
SSH_OPTS=""                       # e.g. "-i ~/.ssh/id_ed25519"; ProxyJump already in ssh config

COMPRESS=0                        # 1 to force -z (pointless for PNG tiles)
CREATE_REMOTE_DIR=1               # mkdir -p the destination on the far side

# Never staged.
EXCLUDES=(
  ".DS_Store" "*.tmp" "*.partial"
  "__pycache__/" "*.pyc"
  "predictions_*.csv"             # cluster output -- pulled down, not pushed up
)
# =========================================================================== #

usage() {
  cat <<EOF
usage: ${0##*/} [--dry-run|-n] [--obj ID...] [--verbose|-v] [--help|-h]

  --dry-run, -n   show what would transfer, move nothing
  --obj ID...     only these object ids (e.g. --obj 2 6 14); default is all
  --verbose, -v   list every file instead of one progress line
  --help,    -h   this message

Edit the "EDIT ME" block at the top of this script to set the paths.
EOF
}

DRY_RUN=0
VERBOSE=0
OBJ_IDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run) DRY_RUN=1; shift ;;
    -v|--verbose) VERBOSE=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    --obj)
      shift
      # consume every following bare number
      while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do OBJ_IDS+=("$1"); shift; done
      if [[ ${#OBJ_IDS[@]} -eq 0 ]]; then
        echo "error: --obj needs at least one numeric id" >&2; exit 2
      fi
      ;;
    *) echo "error: unknown argument '$1'" >&2; echo >&2; usage >&2; exit 2 ;;
  esac
done

# --- guards ---------------------------------------------------------------- #
for var in REMOTE_HOST LOCAL_RENDERED_ROOT REMOTE_RENDERED_ROOT; do
  if [[ -z "${!var}" ]]; then
    echo "error: ${var} is empty -- set it in the EDIT ME block of $0" >&2
    exit 1
  fi
done

if [[ ! -d "${LOCAL_RENDERED_ROOT}" ]]; then
  echo "error: LOCAL_RENDERED_ROOT '${LOCAL_RENDERED_ROOT}' is not a directory" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "error: rsync not found on this machine" >&2
  exit 1
fi

# Refuse to push an empty tree -- almost always a wrong path rather than a
# genuinely empty dataset.
if ! compgen -G "${LOCAL_RENDERED_ROOT%/}/obj_*" >/dev/null; then
  echo "error: no obj_* directories under ${LOCAL_RENDERED_ROOT}" >&2
  echo "       is LOCAL_RENDERED_ROOT pointing at the right place?" >&2
  exit 1
fi

# --- object selection ------------------------------------------------------- #
# With no --obj, everything under the root goes. With --obj, include only those
# stems (and all their contents) and exclude the rest.
SELECT=()
if [[ ${#OBJ_IDS[@]} -gt 0 ]]; then
  for id in "${OBJ_IDS[@]}"; do
    stem="$(printf 'obj_%06d' "${id}")"
    if [[ ! -d "${LOCAL_RENDERED_ROOT%/}/${stem}" ]]; then
      echo "error: no such local object dir: ${LOCAL_RENDERED_ROOT%/}/${stem}" >&2
      exit 1
    fi
    SELECT+=("--include=${stem}/" "--include=${stem}/**")
  done
  SELECT+=("--exclude=*")
fi

# --- build the rsync invocation --------------------------------------------- #
RSYNC_ARGS=(-a --partial --human-readable)

if (( VERBOSE )); then
  RSYNC_ARGS+=(-v --progress)
else
  # One rolling total across the whole transfer, rather than 32k filenames.
  RSYNC_ARGS+=(--info=progress2)
fi

(( COMPRESS )) && RSYNC_ARGS+=(-z)
(( DRY_RUN ))  && RSYNC_ARGS+=(--dry-run -v)

# Excludes come before the object selection so they apply within a selected
# object too.
for pat in "${EXCLUDES[@]}"; do
  RSYNC_ARGS+=("--exclude=${pat}")
done
RSYNC_ARGS+=("${SELECT[@]}")

RSYNC_ARGS+=(--stats)

SSH_CMD="ssh"
if [[ -n "${SSH_OPTS}" ]]; then
  SSH_CMD="ssh ${SSH_OPTS}"
fi

# rsync only creates the final path component. If /data/<user>/ exists but
# vit_hope_rendered/ does not, this makes the first run work anyway.
if (( CREATE_REMOTE_DIR )) && (( ! DRY_RUN )); then
  RSYNC_ARGS+=(--rsync-path="mkdir -p '${REMOTE_RENDERED_ROOT%/}' && rsync")
fi

SRC="${LOCAL_RENDERED_ROOT%/}/"
DEST="${REMOTE_HOST}:${REMOTE_RENDERED_ROOT%/}/"

# --- go ---------------------------------------------------------------------- #
if [[ ${#OBJ_IDS[@]} -gt 0 ]]; then
  SCOPE="objects: ${OBJ_IDS[*]}"
  SEL_DIRS=()
  for id in "${OBJ_IDS[@]}"; do
    SEL_DIRS+=("$(printf '%s/obj_%06d' "${LOCAL_RENDERED_ROOT%/}" "${id}")")
  done
  BYTES="$(du -shc "${SEL_DIRS[@]}" 2>/dev/null | tail -1 | cut -f1 || echo '?')"
else
  SCOPE="all objects ($(compgen -G "${LOCAL_RENDERED_ROOT%/}/obj_*" | wc -l))"
  BYTES="$(du -sh "${LOCAL_RENDERED_ROOT}" 2>/dev/null | cut -f1 || echo '?')"
fi

echo "=== vit_hope :: push rendered data ==="
(( DRY_RUN )) && echo "*** DRY RUN -- nothing will be transferred ***"
echo "from:  ${SRC}"
echo "to:    ${DEST}"
echo "scope: ${SCOPE}"
echo "local size: ${BYTES}  (rsync sends only what is missing or changed)"
echo "======================================"
echo

rsync "${RSYNC_ARGS[@]}" -e "${SSH_CMD}" "${SRC}" "${DEST}"

echo
if (( DRY_RUN )); then
  echo "Dry run complete. Re-run without --dry-run to transfer."
  exit 0
fi

cat <<EOF
Rendered data staged.

Verify from here:
  ./push_rendered.sh --dry-run        # a clean run should now send ~nothing
  ssh ${REMOTE_HOST} 'ls ${REMOTE_RENDERED_ROOT%/} | head; ls ${REMOTE_RENDERED_ROOT%/} | wc -l'

Then push the code and submit:
  ./push_code.sh
  ssh ${REMOTE_HOST}
  sbatch daad-rise-2026/vit_hope/slurm/vit_hope_dinov3.slurm
EOF
