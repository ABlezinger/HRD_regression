#!/bin/bash

#SBATCH --output=./logs/launch-log.txt
#SBATCH --error=./logs/launch-error.txt
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --partition=ampere
#SBATCH --chdir=/home/alexander.blezinger/HRD_regression

EXTRACTION_MODELS=("UNI" "UNI_2" "RetCCL" "GPFM" "CONCH")
# EXTRACTION_MODELS=("UNI_2")
# EXTRACTION_MODELS=("UNI" "RetCCL" "GPFM" "UNI_2")

for MODEL in "${EXTRACTION_MODELS[@]}"
do
    sbatch --job-name="${MODEL}_crossval" multi_crossval.sh --extraction_model "$MODEL"
done