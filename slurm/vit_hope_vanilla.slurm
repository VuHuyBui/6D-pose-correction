#!/bin/bash
#SBATCH --job-name=vh_vanilla
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00
#SBATCH -o /data/%u/logs/slurm.%N.%j.out

# From-scratch ViT baseline @ 518 -- every object with rendered data.
# No pretrained backbone, so FREEZE is ignored and the tag is always "scratch".
#
#   sbatch vit_hope/slurm/vit_hope_vanilla.slurm
#   sbatch --export=ALL,EPOCHS=60 vit_hope/slurm/vit_hope_vanilla.slurm

export REPO_DIR="${REPO_DIR:-/home/s-2657115/daad-rise-2026/vit_hope}"
export MODEL=vanilla
# BATCH_SIZE is intentionally unset: run_all_objects.sh owns the per-model
# defaults and halves them for a fine-tune. Override per submission with
# --export=ALL,BATCH_SIZE=N if you need to.

export CHAIN_SCRIPT="${REPO_DIR}/slurm/vit_hope_vanilla.slurm"
export CHAIN="${CHAIN:-0}"   # e.g. CHAIN=8 to keep resubmitting until done

source "${REPO_DIR}/slurm/run_all_objects.sh"
