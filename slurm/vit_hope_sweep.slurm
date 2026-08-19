#!/bin/bash
#SBATCH --job-name=vh_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00
#SBATCH -o /data/%u/logs/slurm.%N.%j.out

# Hyper-parameter random search for ONE model over a small subset of objects.
# Generic over every MODEL_SPECS key, exactly like vit_hope_model.slurm -- pass
# the model via --export rather than copying this file. Which knobs are actually
# searched comes from models.ARCH_KEYS:
#
#   vanilla            embed_dim (the 18-channel patch-embed projection), depth,
#                      num_heads, mlp_ratio, lr, dropout
#   mvit               the above for the per-view patch embed + local trunk, plus
#                      fuser_embed_dim / fuser_depth / fuser_heads /
#                      fuser_mlp_ratio, lr, dropout
#   mvdinov2/3,        the fuser four + lr/dropout (the backbone width is fixed
#   mvswinv2           by the pretrained weights)
#   mvswinlo           the fuser four + stage_dim + lr/dropout
#   dinov2/3, swinv2   lr/dropout only -- fixed reference points
#
# Trial `default` always runs first with no flags at all, reproducing the current
# configuration; it is the row everything else is measured against.
#
# The from-scratch pair first -- cheapest, and the only models where embed_dim is
# the projection width:
#   sbatch -J vh_sw_mvit    --export=ALL,MODEL=mvit    vit_hope/slurm/vit_hope_sweep.slurm
#   sbatch -J vh_sw_vanilla --export=ALL,MODEL=vanilla vit_hope/slurm/vit_hope_sweep.slurm
#
# Then the pretrained ones as GPU time frees up (CHAIN to survive the 24h cap):
#   sbatch -J vh_sw_mvdino2 --export=ALL,MODEL=mvdinov2,N_TRIALS=12,CHAIN=4 vit_hope/slurm/vit_hope_sweep.slurm
#   sbatch -J vh_sw_mvswlo  --export=ALL,MODEL=mvswinlo,N_TRIALS=16,CHAIN=4 vit_hope/slurm/vit_hope_sweep.slurm
#   sbatch -J vh_sw_dino2   --export=ALL,MODEL=dinov2,N_TRIALS=8           vit_hope/slurm/vit_hope_sweep.slurm
#
# Fine-tune instead of frozen (sweep.py narrows the lr range automatically):
#   sbatch --export=ALL,MODEL=mvdinov2,FREEZE=0,N_TRIALS=8,CHAIN=4 vit_hope/slurm/vit_hope_sweep.slurm
#
# Other knobs: EPOCHS, N_TRIALS, SWEEP_SEED, SWEEP_OBJ_IDS="21 2 11 6",
# BATCH_SIZE, TILE_SIZE, RESULT_CSV, INFER=1, FORCE=1. Smoke-test a submission
# with N_TRIALS=1,EPOCHS=2,SWEEP_OBJ_IDS=6 before committing a day of GPU.
#
# dinov3 / mvdinov3 need HF_TOKEN exported before submitting (gated weights).
#
# The job name can't interpolate ${MODEL} -- sbatch parses #SBATCH before the
# shell runs -- so pass -J per submission if you want a readable queue.

export REPO_DIR="${REPO_DIR:-/home/s-2657115/daad-rise-2026/vit_hope}"
export MODEL="${MODEL:?set MODEL, e.g. sbatch --export=ALL,MODEL=mvit $0}"
export FREEZE="${FREEZE:-1}"

export CHAIN_SCRIPT="${REPO_DIR}/slurm/vit_hope_sweep.slurm"
export CHAIN="${CHAIN:-0}"   # e.g. CHAIN=4 to keep resubmitting until done

source "${REPO_DIR}/slurm/sweep_hparams.sh"
