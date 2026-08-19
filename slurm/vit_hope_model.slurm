#!/bin/bash
#SBATCH --job-name=vh_model
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00
#SBATCH -o /data/%u/logs/slurm.%N.%j.out

# Generic per-model launcher: train + infer ONE model over every object with
# rendered data. Pass the model key via --export instead of copying this file.
#
#   sbatch --export=ALL,MODEL=mvdinov2               vit_hope/slurm/vit_hope_model.slurm
#   sbatch --export=ALL,MODEL=mvdinov2,FREEZE=0      vit_hope/slurm/vit_hope_model.slurm
#   sbatch --export=ALL,MODEL=mvit,CHAIN=8           vit_hope/slurm/vit_hope_model.slurm
#   sbatch --export=ALL,MODEL=mvswinlo,OBJ_IDS="2 6" vit_hope/slurm/vit_hope_model.slurm
#
# The job name can't interpolate ${MODEL} -- sbatch parses #SBATCH directives
# before the shell runs -- so override it per submission if you want it readable:
#   sbatch -J vh_mvdinov2 --export=ALL,MODEL=mvdinov2 vit_hope/slurm/vit_hope_model.slurm
#
# Valid MODEL keys are whatever MODEL_SPECS lists in models.py; an unknown one
# fails in run_all_objects.sh before any GPU time is spent. BATCH_SIZE defaults
# per model there too -- set it explicitly to override.
#
# The four vit_hope_<model>.slurm wrappers still work; they predate this and set
# MODEL themselves.

export REPO_DIR="${REPO_DIR:-/home/s-2657115/daad-rise-2026/vit_hope}"
export MODEL="${MODEL:?set MODEL, e.g. sbatch --export=ALL,MODEL=mvdinov2 $0}"
export FREEZE="${FREEZE:-1}"

export CHAIN_SCRIPT="${REPO_DIR}/slurm/vit_hope_model.slurm"
export CHAIN="${CHAIN:-0}"   # e.g. CHAIN=8 to keep resubmitting until done

source "${REPO_DIR}/slurm/run_all_objects.sh"
