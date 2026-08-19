#!/bin/bash
#SBATCH --partition=gpu # partition
#SBATCH --nodes=1  # number of nodes
#SBATCH --gres=gpu:1 # number of gpus
#SBATCH -t 0-8:00  # time (D-HH:MM)
#SBATCH -o /data/%u/logs/slurm.%N.%j.out

# =========================================================================== #
# vit_hope TRAINING + INFERENCE on a compute node.
#
# The compute node has no OpenGL/GLFW, so the rendering stages (prepare_poses,
# generate_jitter, visualize) CANNOT run here. Render locally and copy the
# renderer output (results/hope: jitter_all.csv + tiles) to the node, then this
# script trains and runs infer.py on that prepared data.
#
# Prerequisites (do once, before submitting):
#   1. Create the venv with uv, e.g. from ${REPO_DIR}:
#        uv venv && uv pip install -r requirements.txt
#        (install torch/torchvision matched to the node's CUDA first)
#   2. Copy the locally-rendered data to the node so that
#        ${RENDERED_DATA_DIR}/jitter_all.csv and its run_*/ tile PNGs exist.
#   3. Make sure /data/%u/logs exists (SBATCH -o needs it at submit time):
#        mkdir -p /data/$USER/logs
#
# Submit:   sbatch vit_hope/slurm/vit_hope.slurm
# Override: sbatch --export=ALL,OBJ_ID=1,EPOCHS=50,RENDERED_DATA_DIR=/data/me/hope \
#                  vit_hope/slurm/vit_hope.slurm
# =========================================================================== #

# --- Configurable paths (edit before submitting) ---------------------------- #
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/s-2657115}"

# vit_hope package (this script lives in vit_hope/slurm/).
REPO_DIR="/home/s-2657115/daad-rise-2026/vit_hope"

# Directory copied from the local renderer's results/hope folder. Training reads
# jitter_all.csv and run_*/ tiles directly; the renderer is not needed here.
RENDERED_DATA_DIR="${RENDERED_DATA_DIR:-${DATA_ROOT}/vit_hope_rendered}"

# --- Run parameters (override via --export) --------------------------------- #
OBJ_ID="${OBJ_ID:-2}"
TILE_SIZE="${TILE_SIZE:-518}"     # must be a multiple of 14 (DINOv2 patch size)
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-30}"

# Which models to train (1 = run, 0 = skip).
TRAIN_DINOV2_FROZEN="${TRAIN_DINOV2_FROZEN:-1}"
TRAIN_DINOV2_FINETUNE="${TRAIN_DINOV2_FINETUNE:-0}"
TRAIN_VANILLA="${TRAIN_VANILLA:-0}"

PATH=/usr/local/bin:$PATH
set -euo pipefail
cd "${REPO_DIR}"

# Project venv created with: uv venv && uv pip install -r requirements.txt
source "${REPO_DIR}/.venv/bin/activate"

OBJ_STEM="$(printf 'obj_%06d' "${OBJ_ID}")"
JITTER_CSV="${RENDERED_DATA_DIR}/jitter_all.csv"
WEIGHTS_DIR="${REPO_DIR}/weights"

echo "=== vit_hope training + inference ==="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node(s):       ${SLURM_NODELIST:-local}"
echo "Repo:          ${REPO_DIR}"
echo "Prepared data: ${RENDERED_DATA_DIR}"
echo "Object:        ${OBJ_STEM} (obj_id=${OBJ_ID})"
echo "Tile size:     ${TILE_SIZE}   Batch: ${BATCH_SIZE}   Epochs: ${EPOCHS}"
if [[ "${TRAIN_DINOV2_FROZEN}" == "1" ]]; then
  echo "Experiment:    vit_hope_dinov2_frozen"
elif [[ "${TRAIN_DINOV2_FINETUNE}" == "1" ]]; then
  echo "Experiment:    vit_hope_dinov2_finetune"
else
  echo "Experiment:    vit_hope_vanilla"
fi
echo "Python:        $(which python)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')"
echo "========================="

# --- sanity check: prepared data must be present --------------------------- #
if [[ ! -f "${JITTER_CSV}" ]]; then
  echo "ERROR: rendered data not found: ${JITTER_CSV}" >&2
  echo "       Render locally, copy results/hope to the node, and set" >&2
  echo "       RENDERED_DATA_DIR to the copied directory." >&2
  exit 1
fi

# --- train ------------------------------------------------------------------ #
train_one() {  # args: <extra flags...>
  echo ">>> train.py $*"
  python train.py \
    --obj_id "${OBJ_ID}" \
    --data_dir "${RENDERED_DATA_DIR}" \
    --tile_size "${TILE_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --augment \
    "$@"
}

# --- infer ------------------------------------------------------------------ #
# Writes predictions_<model>_<tag>.csv next to the prepared data.
infer_one() {  # args: <weights basename without .pt>
  local stem="$1"
  local weights="${WEIGHTS_DIR}/${stem}.pt"
  echo ">>> infer.py --weights ${weights}"
  python infer.py \
    --weights "${weights}" \
    --data_dir "${RENDERED_DATA_DIR}" \
    --out_dir "${RENDERED_DATA_DIR}"
}

[[ "${TRAIN_DINOV2_FROZEN}"   == "1" ]] && {
  train_one --model dinov2 --freeze
  infer_one "${OBJ_STEM}_dinov2_frozen"
}
[[ "${TRAIN_DINOV2_FINETUNE}" == "1" ]] && {
  train_one --model dinov2 --no_freeze --lr 3e-5
  infer_one "${OBJ_STEM}_dinov2_finetune"
}
[[ "${TRAIN_VANILLA}"         == "1" ]] && {
  train_one --model vanilla
  infer_one "${OBJ_STEM}_vanilla_scratch"
}

echo "=== Done ==="
echo "Weights:     ${WEIGHTS_DIR}"
echo "Predictions: ${RENDERED_DATA_DIR}/predictions_*.csv"
