#!/bin/bash

#SBATCH -t 06:00:00
#SBATCH --nodes 1
#SBATCH -C gpu
#SBATCH --gpus-per-task=4
#SBATCH -A m3443_g
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out

# Submit from Tracking-ML-Exa.TrkX/Pipelines/LRT
# to ensure path is correct.

source activate $HOME/.conda/envs/HSF

srun traintrack configs/pipelines/train_filter.yaml
