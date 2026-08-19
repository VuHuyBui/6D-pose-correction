#!/bin/bash
#SBATCH --job-name=vh_dinov2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00
#SBATCH -o /data/%u/logs/slurm.%N.%j.out

# DINOv2 ViT-B/14 @ 518, frozen backbone -- every object with rendered data.
# ViT-B, not ViT-L: 18 channels at 518x518 makes ViT-L OOM-prone on one GPU.
# VIT_HOPE_DINOV2_BACKBONE overrides the timm name if you want ViT-L back.
#
#   sbatch vit_hope/slurm/vit_hope_dinov2.slurm
#   sbatch --export=ALL,FREEZE=0,EPOCHS=50 vit_hope/slurm/vit_hope_dinov2.slurm
#   sbatch --export=ALL,OBJ_IDS="2 6 14" vit_hope/slurm/vit_hope_dinov2.slurm
#   sbatch --export=ALL,CHAIN=8 vit_hope/slurm/vit_hope_dinov2.slurm  # auto-resume

export REPO_DIR="${REPO_DIR:-/home/s-2657115/daad-rise-2026/vit_hope}"
export MODEL=dinov2
export FREEZE="${FREEZE:-1}"
# BATCH_SIZE is intentionally unset: run_all_objects.sh owns the per-model
# defaults and halves them for a fine-tune. Override per submission with
# --export=ALL,BATCH_SIZE=N if you need to.

export CHAIN_SCRIPT="${REPO_DIR}/slurm/vit_hope_dinov2.slurm"
export CHAIN="${CHAIN:-0}"   # e.g. CHAIN=8 to keep resubmitting until done

source "${REPO_DIR}/slurm/run_all_objects.sh"
