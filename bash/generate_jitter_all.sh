#!/bin/bash
# =========================================================================== #
# Generate jittered renders for EVERY HOPE object, one object at a time.
#
# Runs LOCALLY -- the C++ renderer needs an OpenGL context, which the GPU
# compute nodes do not provide. generate_jitter.py wraps itself in xvfb-run
# when there is no DISPLAY.
#
# The renderer always writes to one fixed staging directory (Config.output_dir,
# i.e. texture-projection-opengl-cpp/results/hope), so each object would clobber
# the previous one. This script renders into that staging dir and then MOVES the
# result to ${DEST_ROOT}/obj_XXXXXX/, which is the per-object layout
# slurm/run_all_objects.sh globs for:
#
#   ${DEST_ROOT}/obj_000002/jitter_all.csv + request.json + run_*/
#   ${DEST_ROOT}/obj_000006/...
#
# Objects absent from the HOPE val split have no GT poses; prepare_poses.py
# exits on them and the loop logs a skip and moves on.
#
# Usage:
#   ./generate_jitter_all.sh                       # all objects
#   OBJ_IDS="2 6 14" ./generate_jitter_all.sh      # just these
#   FORCE=1 ./generate_jitter_all.sh               # redo finished objects
#   NO_RENDER=1 ./generate_jitter_all.sh           # CSV/JSON only, no GPU
#   DEST_ROOT=/data/me/vit_hope_rendered ./generate_jitter_all.sh
#
# Then ship it to the cluster:
#   rsync -av ${DEST_ROOT}/ USER@CLUSTER:/data/USER/vit_hope_rendered/
# =========================================================================== #

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_DIR}/src"
DEST_ROOT="${DEST_ROOT:-$(dirname "${REPO_DIR}")/vit_hope_rendered}"

JITTERS_PER_IMAGE="${JITTERS_PER_IMAGE:-5}"
DT_STD="${DT_STD:-}"              # empty = Config default (0.01 m)
DR_STD="${DR_STD:-0}"              # empty = Config default (0.05 rad)
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"                # cap instances per object (quick test)
MASK="${MASK:-0}"                 # 1 = mask projection PNGs to the instance
FORCE="${FORCE:-0}"
NO_RENDER="${NO_RENDER:-0}"
KEEP_ASSETS="${KEEP_ASSETS:-0}"   # 1 = do not delete assets PNGs between objects
OBJ_IDS="${OBJ_IDS:-}"            # empty = every obj_*.ply in the HOPE models dir

cd "${REPO_DIR}"

# cv2 / numpy / pandas live in the repo venv, not in the base interpreter.
if [[ -z "${VIRTUAL_ENV:-}" && -f "${REPO_DIR}/.venv/bin/activate" ]]; then
  source "${REPO_DIR}/.venv/bin/activate"
fi

# Ask config.py for the paths rather than hard-coding them, so the env
# overrides (VIT_HOPE_HOPE_ROOT, VIT_HOPE_RENDERER_REPO, ...) still apply.
read -r HOPE_ROOT STAGING ASSETS < <(PYTHONPATH="${SRC_DIR}" python -c '
from config import Config
c = Config()
print(c.hope_root, c.output_dir, c.assets_dir)
') || { echo "ERROR: could not import config.py -- activate the venv first" >&2; exit 1; }

MODELS_DIR="${HOPE_ROOT}/models"
[[ -d "${MODELS_DIR}" ]] || { echo "ERROR: no HOPE models dir: ${MODELS_DIR}" >&2; exit 1; }

# --- discover objects ------------------------------------------------------ #
if [[ -z "${OBJ_IDS}" ]]; then
  for ply in "${MODELS_DIR}"/obj_*.ply; do
    [[ -f "${ply}" ]] || continue
    stem="$(basename "${ply}" .ply)"                  # obj_000002
    OBJ_IDS+="$((10#${stem#obj_})) "                  # -> 2
  done
fi
read -r -a OBJ_ARRAY <<< "${OBJ_IDS}"

echo "=== vit_hope: jitter generation over all HOPE objects ==="
echo "Repo:      ${REPO_DIR}"
echo "HOPE:      ${HOPE_ROOT}"
echo "Staging:   ${STAGING}"
echo "Assets:    ${ASSETS}"
echo "Dest:      ${DEST_ROOT}"
echo "Objects:   ${#OBJ_ARRAY[@]} -> ${OBJ_ARRAY[*]:-none}"
echo "Runs/img:  ${JITTERS_PER_IMAGE}   Seed: ${SEED}   Render: $([[ ${NO_RENDER} == 1 ]] && echo no || echo yes)"
echo "========================================================"

if [[ ${#OBJ_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: no obj_*.ply under ${MODELS_DIR}" >&2
  exit 1
fi

# Refuse to rm -rf anything that does not look like the renderer's staging dir.
case "${STAGING}" in
  */results/*) ;;
  *) echo "ERROR: refusing to clean '${STAGING}' -- not a results/ path" >&2; exit 1 ;;
esac

clean_staging() {
  rm -rf "${STAGING}"/run_* "${STAGING}/jitter_all.csv" "${STAGING}/request.json"
}

PREP_FLAGS=(); [[ -n "${LIMIT}" ]] && PREP_FLAGS+=(--limit "${LIMIT}")
[[ "${MASK}" == "1" ]] && PREP_FLAGS+=(--mask)

JIT_FLAGS=(--jitters_per_image "${JITTERS_PER_IMAGE}" --seed "${SEED}")
[[ -n "${DT_STD}" ]] && JIT_FLAGS+=(--dt_std "${DT_STD}")
[[ -n "${DR_STD}" ]] && JIT_FLAGS+=(--dr_std "${DR_STD}")
[[ "${NO_RENDER}" == "1" ]] && JIT_FLAGS+=(--no_render)

DONE=(); NO_GT=(); FAILED=(); SKIPPED=()

for obj_id in "${OBJ_ARRAY[@]}"; do
  stem="$(printf 'obj_%06d' "${obj_id}")"
  dest="${DEST_ROOT}/${stem}"

  echo ""
  echo "--------------------------------------------------------------------"
  echo ">>> ${stem}  $(date -Is)"

  if [[ -f "${dest}/jitter_all.csv" && "${FORCE}" != "1" ]]; then
    echo "    ${dest}/jitter_all.csv exists -- skipping (FORCE=1 to redo)"
    SKIPPED+=("${stem}"); continue
  fi

  # Start from an empty staging dir so a previous object's run_* tiles can
  # never be moved into this object's directory.
  clean_staging
  mkdir -p "${STAGING}"

  # prepare_poses.py exits 2 when the object is simply absent from the split,
  # and 1 for anything genuinely broken. Keep those apart -- lumping them
  # together hides real errors (a missing module) in the benign "no GT" bucket.
  python "${SRC_DIR}/prepare_poses.py" --obj_id "${obj_id}" "${PREP_FLAGS[@]}"
  prep_rc=$?
  if [[ ${prep_rc} -eq 2 ]]; then
    echo "    not in the ${VIT_HOPE_SPLIT:-val} split -- skipping"
    NO_GT+=("${stem}"); continue
  elif [[ ${prep_rc} -ne 0 ]]; then
    echo "    PREPARE FAILED (exit ${prep_rc})"; FAILED+=("${stem}"); continue
  fi

  if ! python "${SRC_DIR}/generate_jitter.py" --obj_id "${obj_id}" "${JIT_FLAGS[@]}"; then
    echo "    JITTER/RENDER FAILED"; FAILED+=("${stem}"); continue
  fi

  # Move staging -> per-object destination. jitter_all.csv goes LAST so that a
  # run interrupted mid-move leaves no CSV, and the skip-existing check above
  # correctly treats the object as unfinished.
  rm -rf "${dest}"
  mkdir -p "${dest}"
  if ! mv "${STAGING}"/run_* "${STAGING}/request.json" "${dest}/" \
     || ! mv "${STAGING}/jitter_all.csv" "${dest}/"; then
    echo "    MOVE TO ${dest} FAILED"; FAILED+=("${stem}"); continue
  fi

  # The projection PNGs (~250 MB/object) are only needed while that object
  # renders. The .obj mesh is tiny and kept, so a re-run is cheap.
  [[ "${KEEP_ASSETS}" == "1" ]] || rm -f "${ASSETS}"/*.png

  echo "    -> ${dest}  ($(du -sh "${dest}" 2>/dev/null | cut -f1))"
  DONE+=("${stem}")
done

clean_staging

echo ""
echo "=== Summary ==="
echo "done    (${#DONE[@]}):    ${DONE[*]:-none}"
echo "skipped (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
echo "no GT   (${#NO_GT[@]}):   ${NO_GT[*]:-none}"
echo "failed  (${#FAILED[@]}):  ${FAILED[*]:-none}"
echo "Output: ${DEST_ROOT}/obj_*/jitter_all.csv"
echo "Total:  $(du -sh "${DEST_ROOT}" 2>/dev/null | cut -f1)"

[[ ${#FAILED[@]} -eq 0 ]]
