#!/bin/bash
# =========================================================================== #
# Visualize EVERY predictions CSV that has come back from the cluster.
#
# Runs LOCALLY -- visualize.py re-renders through the C++ renderer, which needs
# an OpenGL context the GPU compute nodes do not have. visualize.py wraps itself
# in xvfb-run when there is no DISPLAY.
#
# Two things make this more than a for-loop over CSVs:
#
#   1. The projection PNGs in assets/hope are DELETED after each object renders
#      (generate_jitter_all.sh, ~250 MB/object), so they must be regenerated
#      with prepare_poses.py before that object can be re-rendered. The .obj
#      mesh is kept, so this is cheap -- but it is per object, not per model.
#
#   2. visualize.py suffixes its outputs with _<model>_<tag> but NOT with the
#      object, and `clean/` is unsuffixed by design. Every object therefore gets
#      its own ${OUT_ROOT}/obj_XXXXXX/ directory, or run_0/ from obj_000002
#      would land on top of run_0/ from obj_000006.
#
# So the loop is object-outer, model-inner: prepare once, render clean once,
# then every model for that object reuses both.
#
#   ${OUT_ROOT}/obj_000002/clean/                       (shared, GT)
#   ${OUT_ROOT}/obj_000002/corrected_dinov3_frozen/
#   ${OUT_ROOT}/obj_000002/comparisons_dinov3_frozen/
#   ${OUT_ROOT}/obj_000002/metrics_dinov3_frozen.csv
#
# Budget the disk: comparison collages run a few hundred MB per object-model.
# NO_COMPARISONS=1 gives metrics only.
#
# Usage:
#   ./visualize_all.sh                          # everything pulled so far
#   OBJ_IDS="2 6 14" ./visualize_all.sh         # just these objects
#   FORCE=1 ./visualize_all.sh                  # redo finished ones
#   NO_COMPARISONS=1 ./visualize_all.sh         # metrics only, no collages
#   NO_RENDER=1 ./visualize_all.sh              # reuse existing tiles, no GPU
#   DRY_RUN=1 ./visualize_all.sh                # print the commands only
#   ./visualize_all.sh --ssim_threshold 10      # extra args go to visualize.py
#
# Gather the per-object metrics afterwards:
#   head -1 results/viz/obj_*/metrics_dinov3_frozen.csv | head -2
#   awk 'FNR>1' results/viz/obj_*/metrics_*.csv > results/viz/metrics_all.csv
# =========================================================================== #

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Where the predictions came down (pull_data.sh) and where the tiles live.
SRC_ROOT="${SRC_ROOT:-${REPO_DIR}/results/remote_predictions}"
RENDERED_ROOT="${RENDERED_ROOT:-$(dirname "${REPO_DIR}")/vit_hope_rendered}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/results/viz}"

OBJ_IDS="${OBJ_IDS:-}"                    # empty = every object under SRC_ROOT
FORCE="${FORCE:-0}"
MASK="${MASK:-0}"                         # must match generate_jitter_all.sh
LIMIT="${LIMIT:-}"                        # must match generate_jitter_all.sh
NO_RENDER="${NO_RENDER:-0}"
NO_COMPARISONS="${NO_COMPARISONS:-0}"
KEEP_ASSETS="${KEEP_ASSETS:-0}"           # 1 = keep the projection PNGs
DRY_RUN="${DRY_RUN:-0}"

cd "${REPO_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" && -f "${REPO_DIR}/.venv/bin/activate" ]]; then
  source "${REPO_DIR}/.venv/bin/activate"
fi

# Ask config.py for the assets dir rather than hard-coding it, so the env
# overrides (VIT_HOPE_RENDERER_REPO, ...) still apply.
ASSETS="$(python -c '
from config import Config
print(Config().assets_dir)
')" || { echo "ERROR: could not import config.py -- activate the venv first" >&2; exit 1; }

[[ -d "${SRC_ROOT}" ]] || { echo "ERROR: no predictions root: ${SRC_ROOT}" >&2; exit 1; }

# --- discover objects ------------------------------------------------------ #
if [[ -z "${OBJ_IDS}" ]]; then
  for d in "${SRC_ROOT}"/obj_*/; do
    stem="$(basename "${d}")"
    compgen -G "${d}predictions_*.csv" >/dev/null || continue
    OBJ_IDS+="$((10#${stem#obj_})) "
  done
fi
read -r -a OBJ_ARRAY <<< "${OBJ_IDS}"

echo "=== vit_hope: visualize every prediction ==="
echo "Predictions: ${SRC_ROOT}"
echo "Tiles:       ${RENDERED_ROOT}"
echo "Assets:      ${ASSETS}"
echo "Out:         ${OUT_ROOT}"
echo "Objects:     ${#OBJ_ARRAY[@]} -> ${OBJ_ARRAY[*]:-none}"
echo "Render:      $([[ ${NO_RENDER} == 1 ]] && echo no || echo yes)   Comparisons: $([[ ${NO_COMPARISONS} == 1 ]] && echo no || echo yes)"
(( DRY_RUN )) && echo "*** DRY RUN -- printing commands only ***"
echo "==========================================="

if [[ ${#OBJ_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: no obj_*/predictions_*.csv under ${SRC_ROOT}" >&2
  exit 1
fi

PREP_FLAGS=(); [[ -n "${LIMIT}" ]] && PREP_FLAGS+=(--limit "${LIMIT}")
[[ "${MASK}" == "1" ]] && PREP_FLAGS+=(--mask)

VIZ_FLAGS=()
[[ "${NO_RENDER}" == "1" ]] && VIZ_FLAGS+=(--no_render)
[[ "${NO_COMPARISONS}" == "1" ]] && VIZ_FLAGS+=(--no_comparisons)
VIZ_FLAGS+=("$@")                         # anything else passes straight through

run() {                                   # echo in dry-run, execute otherwise
  if (( DRY_RUN )); then printf '    +'; printf ' %q' "$@"; printf '\n'; return 0; fi
  "$@"
}

DONE=(); FAILED=(); SKIPPED=(); NO_GT=(); NO_TILES=()

for obj_id in "${OBJ_ARRAY[@]}"; do
  stem="$(printf 'obj_%06d' "${obj_id}")"
  src_dir="${SRC_ROOT}/${stem}"
  data_dir="${RENDERED_ROOT}/${stem}"
  out_dir="${OUT_ROOT}/${stem}"

  echo ""
  echo "--------------------------------------------------------------------"
  echo ">>> ${stem}  $(date -Is)"

  mapfile -t csvs < <(ls -1 "${src_dir}"/predictions_*.csv 2>/dev/null)
  if [[ ${#csvs[@]} -eq 0 ]]; then
    echo "    no predictions CSV -- skipping"; SKIPPED+=("${stem}"); continue
  fi

  # The original jittered tiles are the "Original" column of every collage and
  # the baseline of every metric, so without them there is nothing to compare.
  if [[ ! -f "${data_dir}/jitter_all.csv" ]]; then
    echo "    no tiles at ${data_dir} -- skipping"; NO_TILES+=("${stem}"); continue
  fi

  # Which variants still need doing? metrics_<suffix>.csv is written last, so
  # its presence means that model finished.
  todo=()
  for csv in "${csvs[@]}"; do
    suffix="$(basename "${csv}" .csv)"; suffix="${suffix#predictions_}"
    if [[ -f "${out_dir}/metrics_${suffix}.csv" && "${FORCE}" != "1" ]]; then
      echo "    ${suffix}: metrics exist -- skipping (FORCE=1 to redo)"
      continue
    fi
    todo+=("${csv}")
  done
  if [[ ${#todo[@]} -eq 0 ]]; then
    SKIPPED+=("${stem}"); continue
  fi

  # Regenerate this object's projection PNGs -- the renderer cannot texture
  # without them. Pointless when we are not rendering at all.
  if [[ "${NO_RENDER}" != "1" ]]; then
    run python prepare_poses.py --obj_id "${obj_id}" "${PREP_FLAGS[@]}"
    prep_rc=$?
    if [[ ${prep_rc} -eq 2 ]]; then
      echo "    not in the ${VIT_HOPE_SPLIT:-val} split -- skipping"
      NO_GT+=("${stem}"); continue
    elif [[ ${prep_rc} -ne 0 ]]; then
      echo "    PREPARE FAILED (exit ${prep_rc})"; FAILED+=("${stem}"); continue
    fi
  fi

  obj_failed=0
  for csv in "${todo[@]}"; do
    echo "    -> $(basename "${csv}")"
    if ! run python visualize.py \
          --predictions "${csv}" \
          --obj_id "${obj_id}" \
          --data_dir "${data_dir}" \
          --out_dir "${out_dir}" \
          "${VIZ_FLAGS[@]}"; then
      echo "    VISUALIZE FAILED: $(basename "${csv}")"
      obj_failed=1
    fi
  done

  # ~250 MB per object, and only needed while that object renders.
  [[ "${KEEP_ASSETS}" == "1" || "${NO_RENDER}" == "1" ]] || run rm -f "${ASSETS}"/*.png

  if (( obj_failed )); then FAILED+=("${stem}"); else DONE+=("${stem}"); fi
done

echo ""
echo "=== Summary ==="
echo "done     (${#DONE[@]}):     ${DONE[*]:-none}"
echo "skipped  (${#SKIPPED[@]}):  ${SKIPPED[*]:-none}"
echo "no tiles (${#NO_TILES[@]}): ${NO_TILES[*]:-none}"
echo "no GT    (${#NO_GT[@]}):    ${NO_GT[*]:-none}"
echo "failed   (${#FAILED[@]}):   ${FAILED[*]:-none}"
echo "Output: ${OUT_ROOT}/obj_*/metrics_*.csv"

[[ ${#FAILED[@]} -eq 0 ]]
