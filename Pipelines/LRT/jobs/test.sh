#!/bin/bash

#SBATCH -t 00:10:00
#SBATCH --nodes 1
#SBATCH -C gpu
#SBATCH --gpus-per-task=4
#SBATCH -A m3443_g
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out

source activate $HOME/.conda/envs/HSF

srun traintrack ./configs/pipelines/fulltrain_test.yaml

