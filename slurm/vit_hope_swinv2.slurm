#!/bin/bash
#SBATCH --job-name=vh_swinv2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00
#SBATCH -o /data/%u/logs/slurm.%N.%j.out

# Swin V2-B window16 @ 256, frozen backbone -- every object with rendered data.
# Smaller tiles than the ViTs, so the batch can be larger.
#
#   sbatch vit_hope/slurm/vit_hope_swinv2.slurm
#   sbatch --export=ALL,FREEZE=0,EPOCHS=50 vit_hope/slurm/vit_hope_swinv2.slurm

export REPO_DIR="${REPO_DIR:-/home/s-2657115/daad-rise-2026/vit_hope}"
export MODEL=swinv2
export FREEZE="${FREEZE:-1}"
# BATCH_SIZE is intentionally unset: run_all_objects.sh owns the per-model
# defaults and halves them for a fine-tune. Override per submission with
# --export=ALL,BATCH_SIZE=N if you need to.

export CHAIN_SCRIPT="${REPO_DIR}/slurm/vit_hope_swinv2.slurm"
export CHAIN="${CHAIN:-0}"   # e.g. CHAIN=8 to keep resubmitting until done

source "${REPO_DIR}/slurm/run_all_objects.sh"
