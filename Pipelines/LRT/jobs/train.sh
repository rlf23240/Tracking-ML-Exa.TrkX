#!/bin/bash

#SBATCH -t 6:00:00
#SBATCH -C gpu
#SBATCH -c 10
#SBATCH -G 1
#SBATCH -A m3443_g

conda activate HFS

srun -n 1 traintrack ../configs/pipelines/fulltrain.yaml
