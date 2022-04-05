#!/bin/bash

#SBATCH -t 6:00:00
#SBATCH -C gpu
#SBATCH -c 10
#SBATCH -G 1
#SBATCH -A m3443_g
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out

# Submit from Tracking-ML-Exa.TrkX/Pipelines/LRT
# to ensure path is correct.

source activate $HOME/.conda/envs/HSF

srun -n 1 traintrack /configs/pipelines/fulltrain.yaml
