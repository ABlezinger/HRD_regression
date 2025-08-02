#!/bin/bash

#SBATCH --job-name=TCGA_download
#SBATCH --output=./logs/%j-%x-log.txt
#SBATCH --error=./logs/%j-%x-error.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/alexander.blezinger/HRD_regression
#SBATCH --time=90:00:00


srun -u python3 tcga_download.py
