#!/bin/bash

#SBATCH -t 0:01:00
#SBATCH -C gpu
#SBATCH -c 10
#SBATCH -G 1
#SBATCH -A m3443_g
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out

source activate $HOME/.conda/envs/HSF

srun -n 1 traintrack ./configs/pipelines/fulltrain_test.yaml

