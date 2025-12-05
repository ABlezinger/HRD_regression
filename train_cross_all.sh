#!/bin/bash

#SBATCH --output=./logs/launch-log.txt
#SBATCH --error=./logs/launch-error.txt
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --partition=ampere
#SBATCH --chdir=/home/alexander.blezinger/HRD_regression

# EXTRACTION_MODELS=("Virchow_2")
# EXTRACTION_MODELS=("UNI" "UNI_2" "RetCCL" "GPFM" "CONCH" "Virchow_2")
BAGSIZES=(50 100 200 400 600 800 1000 1200 1400 1600 2000 3000 4000)
EXTRACTION_MODELS=("UNI_2")
# EXTRACTION_MODELS=("UNI" "RetCCL" "GPFM" "UNI_2")

for MODEL in "${EXTRACTION_MODELS[@]}"
do
    for BAGSIZE in "${BAGSIZES[@]}"
    do
        sbatch --job-name="${MODEL}_rand_${BAGSIZE}_crossval" multi_crossval.sh --extraction_model "$MODEL" --bag_size "$BAGSIZE"
    done
done    