#!/bin/bash
#SBATCH --partition=gpu # partition
#SBATCH --nodes=1  # number of nodes
#SBATCH --gres=gpu:1 # number of gpus
#SBATCH -t 0-1:00  # time (D-HH:MM)
#SBATCH -o /data/%u/logs/slurm.%N.%j.out

# =========================================================================== #
# vit_hope INFERENCE ONLY on a compute node.
#
# Loads a trained checkpoint and writes predicted 6D jitter for the test split.
# No OpenGL/renderer required — only the prepared tiles + jitter_all.csv.
#
# Prerequisites:
#   1. Venv at ${REPO_DIR}/.venv (same as training).
#   2. ${RENDERED_DATA_DIR}/jitter_all.csv and run_*/ tiles present.
#   3. Checkpoint under weights/ (or pass WEIGHTS explicitly).
#   4. mkdir -p /data/$USER/logs
#
# Submit:   sbatch vit_hope/slurm/vit_hope_infer.slurm
# Override: sbatch --export=ALL,OBJ_ID=2,WEIGHTS=/path/to.pt,\
#                  RENDERED_DATA_DIR=/data/me/vit_hope_rendered \
#                  vit_hope/slurm/vit_hope_infer.slurm
# =========================================================================== #

DATA_ROOT="${DATA_ROOT:-/data/s-2657115}"
REPO_DIR="/home/s-2657115/daad-rise-2026/vit_hope"
RENDERED_DATA_DIR="${RENDERED_DATA_DIR:-${DATA_ROOT}/vit_hope_rendered}"

OBJ_ID="${OBJ_ID:-2}"
OUT_DIR="${OUT_DIR:-${RENDERED_DATA_DIR}}"

PATH=/usr/local/bin:$PATH
set -euo pipefail
cd "${REPO_DIR}"
source "${REPO_DIR}/.venv/bin/activate"

SRC_DIR="${REPO_DIR}/src"
OBJ_STEM="$(printf 'obj_%06d' "${OBJ_ID}")"
WEIGHTS="${WEIGHTS:-${REPO_DIR}/weights/${OBJ_STEM}_dinov2_frozen.pt}"
JITTER_CSV="${RENDERED_DATA_DIR}/jitter_all.csv"

echo "=== vit_hope inference ==="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node(s):       ${SLURM_NODELIST:-local}"
echo "Repo:          ${REPO_DIR}"
echo "Prepared data: ${RENDERED_DATA_DIR}"
echo "Weights:       ${WEIGHTS}"
echo "Out dir:       ${OUT_DIR}"
echo "Object:        ${OBJ_STEM} (obj_id=${OBJ_ID})"
echo "Python:        $(which python)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')"
echo "========================="

if [[ ! -f "${JITTER_CSV}" ]]; then
  echo "ERROR: rendered data not found: ${JITTER_CSV}" >&2
  exit 1
fi
if [[ ! -f "${WEIGHTS}" ]]; then
  echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2
  exit 1
fi

python "${SRC_DIR}/infer.py" \
  --weights "${WEIGHTS}" \
  --data_dir "${RENDERED_DATA_DIR}" \
  --out_dir "${OUT_DIR}"

echo "=== Done ==="
echo "Predictions under: ${OUT_DIR}"
