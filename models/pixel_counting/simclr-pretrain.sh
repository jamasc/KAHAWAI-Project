#!/bin/bash
#SBATCH --job-name=simclr-pretrain
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --error=logs/simclr-%A.err
#SBATCH --output=logs/simclr-%A.out

## Load modules (adjust if needed)
module load anaconda3
source activate streamflow-env

## Run your SimCLR training script
python code/streamflow_pretrain_3.py \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    
