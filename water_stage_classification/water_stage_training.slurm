#!/bin/bash
#SBATCH --job-name=water_stage
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --error=logs/pxlct-%A.err
#SBATCH --output=logs/pxlct-%A.out

## Load modules (adjust if needed)
source /opt/apps/software/lang/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate streamflow-env

## Run your SimCLR training script
python train.py