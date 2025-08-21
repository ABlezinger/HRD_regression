#!/bin/bash

#SBATCH --job-name=CONCH_extractFeatures
#SBATCH --output=./logs/%j-%x-log.txt
#SBATCH --error=./logs/%j-%x-error.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=a100:1
#SBATCH --chdir=/home/alexander.blezinger/HRD_regression

#parameters
extraction_model="CONCH"  # Options: "GPFM", "RetCCL", "CONCH", "UNI", "UNI_2"
output_dir="/data/datasets/images/CPTAC/PDA/features"
jpg_dir="/data/datasets/images/CPTAC/PDA/PNG"
wsi_dir="/data/datasets/images/CPTAC/PDA"

final_output_path="$output_dir/$extraction_model"

module load conda


if [[ $extraction_model = "CONCH" ]]; then
    conda activate cenv_2
else
    conda activate cenv
fi

srun -u python3 custom_WSI_pipeline/process_WSI.py \
    --wsi-dir $wsi_dir  \
    --model $extraction_model  \
    --output-dir $final_output_path \
    --jpg-dir $jpg_dir \
# python3 custom_WSI_pipeline/process_WSI.py --wsi-dir "test_WSI_test" --model "UNI" --output-dir "test_output/UNI" --jpg-dir "test_JPGs" 
# python3 custom_WSI_pipeline/process_WSI.py --wsi-dir "/data/datasets/images/CPTAC/PDA" --model "RetCCL" --output-dir "/data/datasets/images/CPTAC/PDA/features/RetCCL" --jpg-dir "/data/datasets/images/CPTAC/PDA/PNG" 