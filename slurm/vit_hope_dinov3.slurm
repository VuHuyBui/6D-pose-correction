#!/bin/bash
#SBATCH --job-name=vh_dinov3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00
#SBATCH -o /data/%u/logs/slurm.%N.%j.out

# DINOv3 ViT-B/16 @ 512, frozen backbone -- every object with rendered data.
# ViT-B, not ViT-L, for the same memory reason as dinov2; override the timm
# name with VIT_HOPE_DINOV3_BACKBONE.
#
# The DINOv3 weights are gated on the Hugging Face hub: accept the licence once
# and export HF_TOKEN (or pre-download to HF_HOME) before submitting, otherwise
# every object fails on the first timm.create_model call.
#
#   sbatch vit_hope/slurm/vit_hope_dinov3.slurm
#   sbatch --export=ALL,FREEZE=0,EPOCHS=50 vit_hope/slurm/vit_hope_dinov3.slurm

export REPO_DIR="${REPO_DIR:-/home/s-2657115/daad-rise-2026/vit_hope}"
export MODEL=dinov3
export FREEZE="${FREEZE:-1}"
# BATCH_SIZE is intentionally unset: run_all_objects.sh owns the per-model
# defaults and halves them for a fine-tune. Override per submission with
# --export=ALL,BATCH_SIZE=N if you need to.

export CHAIN_SCRIPT="${REPO_DIR}/slurm/vit_hope_dinov3.slurm"
export CHAIN="${CHAIN:-0}"   # e.g. CHAIN=8 to keep resubmitting until done

source "${REPO_DIR}/slurm/run_all_objects.sh"
