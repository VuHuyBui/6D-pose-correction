#!/bin/bash
# =========================================================================== #
# Shared body for the per-model slurm scripts: train + infer one backbone on
# EVERY object that has rendered data, sequentially inside a single job.
#
# Not submitted directly -- sourced by vit_hope_<model>.slurm, which sets MODEL
# (and optionally FREEZE / BATCH_SIZE / EPOCHS) first.
#
# Data layout expected on the node (one directory per object):
#   ${RENDERED_ROOT}/obj_000002/jitter_all.csv + run_*/ tiles
#   ${RENDERED_ROOT}/obj_000006/...
#
# Both outputs -- the checkpoint and the predictions CSV -- are written back
# into ${OUT_ROOT}/obj_XXXXXX/, i.e. onto /data. Nothing large touches /home.
#
# Objects with no jitter_all.csv are skipped, so a partial render is fine.
# An object that already has its predictions CSV is skipped too (FORCE=1 to
# redo it) -- so a job that hits the walltime can simply be resubmitted.
# A failing object logs the error and the loop moves on.
# =========================================================================== #

DATA_ROOT="${DATA_ROOT:-/data/s-2657115}"
REPO_DIR="${REPO_DIR:-/home/s-2657115/daad-rise-2026/vit_hope}"
RENDERED_ROOT="${RENDERED_ROOT:-${DATA_ROOT}/vit_hope_rendered}"
OUT_ROOT="${OUT_ROOT:-${RENDERED_ROOT}}"

MODEL="${MODEL:?set MODEL in the calling slurm script}"
FREEZE="${FREEZE:-1}"             # 1 = frozen backbone, 0 = full fine-tune
EPOCHS="${EPOCHS:-30}"

# Batch size defaults per model, unless the caller set one. This is the ONLY
# place they live -- the per-model wrappers deliberately do not set BATCH_SIZE,
# or the freeze-aware halving below could never fire for them.
#
# The multi-view models run the backbone once per view, so one sample costs ~6x
# the activations of the equivalent 18-channel stacked model. Frozen, that is
# forward-only and survivable; fine-tuning stores activations for all 6 views
# and needs a smaller batch again (an 8-sample mvswinv2 fine-tune OOMs on an
# 8GB card).
#
# The stacked defaults (vanilla/swinv2 at 16, the 518px ViTs at 8) are the ones
# RESULTS.md was produced with, for both freeze modes -- do not "tidy" them into
# the halving rule or the existing numbers stop being comparable.
BATCH_SIZE_AUTO=0
if [[ -z "${BATCH_SIZE:-}" ]]; then
  BATCH_SIZE_AUTO=1                 # the fine-tune halving below applies
  case "${MODEL}" in
    mvdinov2|mvdinov3) BATCH_SIZE=4 ;;
    mv*)               BATCH_SIZE=8 ;;
    vanilla|swinv2)    BATCH_SIZE=16 ;;
    *)                 BATCH_SIZE=8 ;;
  esac
fi
TILE_SIZE="${TILE_SIZE:-}"        # empty = the backbone's native tile size
FORCE="${FORCE:-0}"
OBJ_IDS="${OBJ_IDS:-}"            # e.g. "2 6 14"; empty = discover from disk

# --- walltime chaining (opt-in) -------------------------------------------- #
# The gpu partition caps jobs at 24h, which may not cover every object. With
# CHAIN>0 the job submits its own successor before exiting, and the skip-existing
# check makes it resume where this run stopped. Guards against a runaway loop:
#   * only chains if this run actually completed >=1 object,
#   * only chains if objects still remain,
#   * CHAIN decrements every generation and stops at 0.
# Only one job ID is ever queued per model, so the 2-job limit still holds.
CHAIN="${CHAIN:-0}"               # remaining generations; 0 = no chaining
CHAIN_SCRIPT="${CHAIN_SCRIPT:-}"  # set by the calling slurm script

PATH=/usr/local/bin:$PATH
set -uo pipefail
cd "${REPO_DIR}"
source "${REPO_DIR}/.venv/bin/activate"

# Ask models.py for the checkpoint tag rather than restating the model list in
# shell: MODEL_SPECS[...]["pretrained"] is the single source of truth that
# train.py and infer.py also use, so the filenames can never disagree. This also
# validates MODEL (an unknown key raises KeyError) before any GPU time is spent.
TAG="$(python -c 'import sys
from models import MODEL_SPECS
kind, freeze = sys.argv[1], sys.argv[2]
if kind not in MODEL_SPECS:
    sys.exit(f"unknown model {kind!r}; choose from {list(MODEL_SPECS)}")
print(("frozen" if freeze == "1" else "finetune")
      if MODEL_SPECS[kind]["pretrained"] else "scratch")' "${MODEL}" "${FREEZE}")"
# This script runs under `set -uo pipefail` but NOT `-e`, so a failed python
# would leave TAG empty and silently mislabel every output file.
if [[ -z "${TAG}" ]]; then
  echo "ERROR: could not resolve TAG for MODEL=${MODEL} (see error above)" >&2
  exit 1
fi

# Halve the auto-chosen batch for a multi-view fine-tune. Keyed on TAG, not on
# FREEZE, so a meaningless `FREEZE=0` on a from-scratch model (mvit -- tag stays
# `scratch`) does not silently halve a batch that was never going to store
# backbone activations in the first place.
if [[ "${BATCH_SIZE_AUTO}" == "1" && "${MODEL}" == mv* && "${TAG}" == "finetune" ]]; then
  BATCH_SIZE=$(( BATCH_SIZE / 2 ))
fi

# --- discover objects ------------------------------------------------------ #
if [[ -z "${OBJ_IDS}" ]]; then
  OBJ_IDS=""
  for csv in "${RENDERED_ROOT}"/obj_*/jitter_all.csv; do
    [[ -f "${csv}" ]] || continue
    stem="$(basename "$(dirname "${csv}")")"          # obj_000002
    OBJ_IDS+="$((10#${stem#obj_})) "                  # -> 2
  done
fi
read -r -a OBJ_ARRAY <<< "${OBJ_IDS}"

echo "=== vit_hope: ${MODEL} (${TAG}) over all objects ==="
echo "Job ID:    ${SLURM_JOB_ID:-local}   Node: ${SLURM_NODELIST:-local}"
echo "Repo:      ${REPO_DIR}"
echo "Data root: ${RENDERED_ROOT}"
echo "Out root:  ${OUT_ROOT}"
echo "Objects:   ${#OBJ_ARRAY[@]} -> ${OBJ_ARRAY[*]:-none}"
echo "Epochs:    ${EPOCHS}   Batch: ${BATCH_SIZE}   Tile: ${TILE_SIZE:-native}"
echo "Python:    $(which python)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
echo "==========================================="

if [[ ${#OBJ_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: no obj_*/jitter_all.csv under ${RENDERED_ROOT}" >&2
  exit 1
fi

TILE_FLAG=()
[[ -n "${TILE_SIZE}" ]] && TILE_FLAG=(--tile_size "${TILE_SIZE}")
FREEZE_FLAG=(--freeze)
[[ "${FREEZE}" == "1" ]] || FREEZE_FLAG=(--no_freeze)

DONE=(); FAILED=(); SKIPPED=()

for obj_id in "${OBJ_ARRAY[@]}"; do
  stem="$(printf 'obj_%06d' "${obj_id}")"
  data_dir="${RENDERED_ROOT}/${stem}"
  out_dir="${OUT_ROOT}/${stem}"
  # Weights land beside the tiles and the predictions CSV, on /data -- never in
  # the repo on /home.
  weights="${out_dir}/${stem}_${MODEL}_${TAG}.pt"
  preds="${out_dir}/predictions_${MODEL}_${TAG}.csv"

  echo ""
  echo "--------------------------------------------------------------------"
  echo ">>> ${stem}  [${MODEL}/${TAG}]  $(date -Is)"

  if [[ ! -f "${data_dir}/jitter_all.csv" ]]; then
    echo "    no jitter_all.csv -- skipping"; SKIPPED+=("${stem}"); continue
  fi
  if [[ -f "${preds}" && "${FORCE}" != "1" ]]; then
    echo "    ${preds} exists -- skipping (FORCE=1 to redo)"
    SKIPPED+=("${stem}"); continue
  fi

  mkdir -p "${out_dir}"
  if ! python train.py \
        --obj_id "${obj_id}" \
        --model "${MODEL}" \
        "${FREEZE_FLAG[@]}" \
        "${TILE_FLAG[@]}" \
        --data_dir "${data_dir}" \
        --weights_dir "${out_dir}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --augment; then
    echo "    TRAIN FAILED"; FAILED+=("${stem}"); continue
  fi

  if ! python infer.py \
        --weights "${weights}" \
        --data_dir "${data_dir}" \
        --out_dir "${out_dir}"; then
    echo "    INFER FAILED"; FAILED+=("${stem}"); continue
  fi

  DONE+=("${stem}")
done

echo ""
echo "=== Summary: ${MODEL}/${TAG} ==="
echo "done    (${#DONE[@]}):    ${DONE[*]:-none}"
echo "skipped (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
echo "failed  (${#FAILED[@]}):  ${FAILED[*]:-none}"
echo "Weights:     ${OUT_ROOT}/obj_*/obj_*_${MODEL}_${TAG}.pt"
echo "Predictions: ${OUT_ROOT}/obj_*/predictions_${MODEL}_${TAG}.csv"

# --- chain a successor job if there is more to do -------------------------- #
if [[ "${CHAIN}" -gt 0 ]]; then
  REMAINING=0
  for obj_id in "${OBJ_ARRAY[@]}"; do
    stem="$(printf 'obj_%06d' "${obj_id}")"
    [[ -f "${RENDERED_ROOT}/${stem}/jitter_all.csv" ]] || continue
    [[ -f "${OUT_ROOT}/${stem}/predictions_${MODEL}_${TAG}.csv" ]] || REMAINING=$((REMAINING + 1))
  done

  echo ""
  echo "--- chaining (CHAIN=${CHAIN}, ${REMAINING} object(s) left) ---"
  if [[ ${REMAINING} -eq 0 ]]; then
    echo "nothing left to do -- chain complete."
  elif [[ ${#DONE[@]} -eq 0 ]]; then
    # Nothing finished this run, so a successor would hit the same wall.
    echo "NOT chaining: this job completed 0 objects. Fix the cause, then"
    echo "resubmit by hand -- finished objects are skipped automatically."
  elif [[ -z "${CHAIN_SCRIPT}" || ! -f "${CHAIN_SCRIPT}" ]]; then
    echo "NOT chaining: CHAIN_SCRIPT=${CHAIN_SCRIPT:-unset} is not a file."
  else
    # FORCE is deliberately cleared: the successor fills gaps, never redoes
    # finished objects, so the chain is guaranteed to converge.
    echo "submitting successor: ${CHAIN_SCRIPT} (CHAIN=$((CHAIN - 1)))"
    sbatch --export=ALL,CHAIN=$((CHAIN - 1)),FORCE=0 "${CHAIN_SCRIPT}"
  fi
fi

[[ ${#FAILED[@]} -eq 0 ]]
