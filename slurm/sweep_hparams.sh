#!/bin/bash
# =========================================================================== #
# Shared body for the hyper-parameter sweep: run a random search for ONE model
# over a SMALL SUBSET of objects, sequentially inside a single job.
#
# Not submitted directly -- sourced by vit_hope_sweep.slurm, which sets MODEL
# (and optionally FREEZE / N_TRIALS / EPOCHS / SWEEP_OBJ_IDS) first.
#
# The question this answers: does tuning the architecture beat the defaults that
# RESULTS.md was produced with -- in particular the embedding dimension of the
# patch-embed linear projection -- for the stacked models and for the multi-view
# transformer models. sweep.py emits the plan; this script executes it.
#
# Trial `default` is always first and always runs train.py with NO hyper-parameter
# flags, so it reproduces the current configuration exactly. Every other trial is
# read against that row.
#
# Deliberately a SUBSET of objects, not all 28: one config over all 28 objects is
# already most of a 24h job (see run_all_objects.sh's chaining note), so a
# 25-trial search over all of them would run for weeks. Confirm the winner over
# all 28 afterwards with run_all_objects.sh.
#
# Selection metric is train.py's best VAL loss, recorded per (trial, object) in
# RESULT_CSV. --seed is left at its default for every trial so all trials share
# one train/val/test split and the losses are directly comparable. infer.py is
# skipped by default (INFER=1 to enable) -- the test split belongs to the final
# confirmation run, not to model selection.
# =========================================================================== #

DATA_ROOT="${DATA_ROOT:-/data/s-2657115}"
REPO_DIR="${REPO_DIR:-/home/s-2657115/daad-rise-2026/vit_hope}"
RENDERED_ROOT="${RENDERED_ROOT:-${DATA_ROOT}/vit_hope_rendered}"
OUT_ROOT="${OUT_ROOT:-${RENDERED_ROOT}}"

MODEL="${MODEL:?set MODEL in the calling slurm script}"
FREEZE="${FREEZE:-1}"             # 1 = frozen backbone, 0 = full fine-tune
EPOCHS="${EPOCHS:-30}"            # same as run_all_objects.sh, so the `default`
                                  # trial is comparable to RESULTS.md
N_TRIALS="${N_TRIALS:-24}"        # random draws, on top of the default trial
SWEEP_SEED="${SWEEP_SEED:-0}"

# Four objects spanning the dataset-size range (275 / 225 / 200 / 125 samples),
# so a config that only works on a large object cannot win.
SWEEP_OBJ_IDS="${SWEEP_OBJ_IDS:-21 2 11 6}"

INFER="${INFER:-0}"               # 1 = also write per-trial predictions CSVs
FORCE="${FORCE:-0}"               # 1 = redo (trial, object) pairs already recorded

# Batch size defaults per model, unless the caller set one -- the same table and
# the same fine-tune halving as run_all_objects.sh. Kept in sync deliberately:
# the `default` trial is only comparable to RESULTS.md if the batch matches.
BATCH_SIZE_AUTO=0
if [[ -z "${BATCH_SIZE:-}" ]]; then
  BATCH_SIZE_AUTO=1
  case "${MODEL}" in
    mvdinov2|mvdinov3) BATCH_SIZE=4 ;;
    mv*)               BATCH_SIZE=8 ;;
    vanilla|swinv2)    BATCH_SIZE=16 ;;
    *)                 BATCH_SIZE=8 ;;
  esac
fi
TILE_SIZE="${TILE_SIZE:-}"        # empty = the backbone's native tile size

# --- walltime chaining (opt-in) -------------------------------------------- #
# Resume granularity is the (trial, object) pair, keyed on RESULT_CSV, so a job
# killed mid-trial resumes exactly where it stopped.
CHAIN="${CHAIN:-0}"
CHAIN_SCRIPT="${CHAIN_SCRIPT:-}"

PATH=/usr/local/bin:$PATH
set -uo pipefail
cd "${REPO_DIR}"
source "${REPO_DIR}/.venv/bin/activate"

# Ask models.py for the checkpoint tag rather than restating the model list in
# shell -- MODEL_SPECS is the single source of truth train.py and infer.py use
# too. This also validates MODEL before any GPU time is spent.
TAG="$(python -c 'import sys
from models import MODEL_SPECS
kind, freeze = sys.argv[1], sys.argv[2]
if kind not in MODEL_SPECS:
    sys.exit(f"unknown model {kind!r}; choose from {list(MODEL_SPECS)}")
print(("frozen" if freeze == "1" else "finetune")
      if MODEL_SPECS[kind]["pretrained"] else "scratch")' "${MODEL}" "${FREEZE}")"
if [[ -z "${TAG}" ]]; then
  echo "ERROR: could not resolve TAG for MODEL=${MODEL} (see error above)" >&2
  exit 1
fi

if [[ "${BATCH_SIZE_AUTO}" == "1" && "${MODEL}" == mv* && "${TAG}" == "finetune" ]]; then
  BATCH_SIZE=$(( BATCH_SIZE / 2 ))
fi

RESULT_CSV="${RESULT_CSV:-${OUT_ROOT}/sweeps/sweep_${MODEL}_${TAG}.csv}"
mkdir -p "$(dirname "${RESULT_CSV}")"

# --- build the plan -------------------------------------------------------- #
FREEZE_FLAG=(--freeze)
[[ "${FREEZE}" == "1" ]] || FREEZE_FLAG=(--no_freeze)

PLAN="$(python sweep.py --model "${MODEL}" --n_trials "${N_TRIALS}" \
          --seed "${SWEEP_SEED}" "${FREEZE_FLAG[@]}")"
if [[ -z "${PLAN}" ]]; then
  echo "ERROR: sweep.py produced no trials for MODEL=${MODEL}" >&2
  exit 1
fi
N_PLANNED="$(grep -cv '^#' <<< "${PLAN}")"

read -r -a OBJ_ARRAY <<< "${SWEEP_OBJ_IDS}"

echo "=== vit_hope hyper-parameter sweep: ${MODEL} (${TAG}) ==="
echo "Job ID:    ${SLURM_JOB_ID:-local}   Node: ${SLURM_NODELIST:-local}"
echo "Repo:      ${REPO_DIR}"
echo "Data root: ${RENDERED_ROOT}"
echo "Results:   ${RESULT_CSV}"
echo "Objects:   ${#OBJ_ARRAY[@]} -> ${OBJ_ARRAY[*]:-none}"
echo "Trials:    ${N_PLANNED} (seed ${SWEEP_SEED})  => $(( N_PLANNED * ${#OBJ_ARRAY[@]} )) runs"
echo "Epochs:    ${EPOCHS}   Batch: ${BATCH_SIZE}   Tile: ${TILE_SIZE:-native}"
echo "Python:    $(which python)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
echo "--- plan ---"
python sweep.py --model "${MODEL}" --n_trials "${N_TRIALS}" \
       --seed "${SWEEP_SEED}" "${FREEZE_FLAG[@]}" --pretty
echo "==========================================="

if [[ ${#OBJ_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: SWEEP_OBJ_IDS is empty" >&2
  exit 1
fi

TILE_FLAG=()
[[ -n "${TILE_SIZE}" ]] && TILE_FLAG=(--tile_size "${TILE_SIZE}")

# Has this (trial, object) pair already been recorded? The result CSV is the
# resume record; a checkpoint file is not, since it appears at epoch 1.
already_done() {  # args: <run_tag> <obj_id>
  [[ -f "${RESULT_CSV}" ]] || return 1
  python - "${RESULT_CSV}" "$1" "$2" <<'PY'
import csv, sys
path, tag, obj = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, newline="") as fh:
    hit = any(r["run_tag"] == tag and r["obj_id"] == obj for r in csv.DictReader(fh))
sys.exit(0 if hit else 1)
PY
}

DONE=0; SKIPPED=0; FAILED=()

while IFS=$'\t' read -r run_tag flags; do
  [[ -z "${run_tag}" || "${run_tag}" == \#* ]] && continue
  # Word-split the flag string on purpose -- it is generated by sweep.py, not
  # user input, and is a flat list of --key value pairs.
  read -r -a FLAG_ARRAY <<< "${flags}"

  echo ""
  echo "===================================================================="
  echo ">>> trial ${run_tag}  [${MODEL}/${TAG}]  ${flags:-(model defaults)}"

  for obj_id in "${OBJ_ARRAY[@]}"; do
    stem="$(printf 'obj_%06d' "${obj_id}")"
    data_dir="${RENDERED_ROOT}/${stem}"
    # Sweep checkpoints go in their own subdirectory so the ~100 .pt files never
    # mix with the canonical run_all_objects.sh outputs.
    out_dir="${OUT_ROOT}/${stem}/sweep"

    echo ""
    echo "--- ${stem} / ${run_tag}  $(date -Is)"

    if [[ ! -f "${data_dir}/jitter_all.csv" ]]; then
      echo "    no jitter_all.csv -- skipping"; SKIPPED=$((SKIPPED + 1)); continue
    fi
    if [[ "${FORCE}" != "1" ]] && already_done "${run_tag}" "${obj_id}"; then
      echo "    already in ${RESULT_CSV} -- skipping (FORCE=1 to redo)"
      SKIPPED=$((SKIPPED + 1)); continue
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
          --augment \
          --run_tag "${run_tag}" \
          --result_csv "${RESULT_CSV}" \
          "${FLAG_ARRAY[@]}"; then
      echo "    TRAIN FAILED"; FAILED+=("${run_tag}/${stem}"); continue
    fi

    if [[ "${INFER}" == "1" ]]; then
      weights="${out_dir}/${stem}_${MODEL}_${TAG}_${run_tag}.pt"
      if ! python infer.py \
            --weights "${weights}" \
            --data_dir "${data_dir}" \
            --out_dir "${out_dir}" \
            --out_name "predictions_${MODEL}_${TAG}_${run_tag}.csv"; then
        echo "    INFER FAILED"; FAILED+=("${run_tag}/${stem} (infer)"); continue
      fi
    fi

    DONE=$((DONE + 1))
  done
done <<< "${PLAN}"

echo ""
echo "=== Summary: sweep ${MODEL}/${TAG} ==="
echo "runs done:    ${DONE}"
echo "runs skipped: ${SKIPPED}"
echo "runs failed  (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "Results CSV:  ${RESULT_CSV}"
echo "Rank them:    python analyze_sweep.py ${RESULT_CSV}"

# --- chain a successor job if there is more to do -------------------------- #
if [[ "${CHAIN}" -gt 0 ]]; then
  REMAINING=0
  while IFS=$'\t' read -r run_tag _; do
    [[ -z "${run_tag}" || "${run_tag}" == \#* ]] && continue
    for obj_id in "${OBJ_ARRAY[@]}"; do
      stem="$(printf 'obj_%06d' "${obj_id}")"
      [[ -f "${RENDERED_ROOT}/${stem}/jitter_all.csv" ]] || continue
      already_done "${run_tag}" "${obj_id}" || REMAINING=$((REMAINING + 1))
    done
  done <<< "${PLAN}"

  echo ""
  echo "--- chaining (CHAIN=${CHAIN}, ${REMAINING} run(s) left) ---"
  if [[ ${REMAINING} -eq 0 ]]; then
    echo "nothing left to do -- chain complete."
  elif [[ ${DONE} -eq 0 ]]; then
    # Nothing finished this run, so a successor would hit the same wall.
    echo "NOT chaining: this job completed 0 runs. Fix the cause, then resubmit"
    echo "by hand -- recorded runs are skipped automatically."
  elif [[ -z "${CHAIN_SCRIPT}" || ! -f "${CHAIN_SCRIPT}" ]]; then
    echo "NOT chaining: CHAIN_SCRIPT=${CHAIN_SCRIPT:-unset} is not a file."
  else
    # FORCE is deliberately cleared, and SWEEP_SEED carried over so the successor
    # re-derives the identical trial list -- that is what makes the chain
    # converge instead of drawing fresh configs every generation.
    echo "submitting successor: ${CHAIN_SCRIPT} (CHAIN=$((CHAIN - 1)))"
    sbatch --export=ALL,CHAIN=$((CHAIN - 1)),FORCE=0,SWEEP_SEED="${SWEEP_SEED}" \
           "${CHAIN_SCRIPT}"
  fi
fi

[[ ${#FAILED[@]} -eq 0 ]]
