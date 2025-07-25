#!/bin/bash

#SBATCH --job-name=extractFeatues
#SBATCH --output=./logs/%j-%x-log.txt
#SBATCH --error=./logs/%j-%x-error.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=a100:1
#SBATCH --chdir=/home/alexander.blezinger/HRD_regression

#parameters
extraction_model="UNI"
output_dir="test_output"
jpg_dir="test_JPGs"
wsi_dir="test_WSI_test"

final_output_path="${output_path}/${extraction_model}"

module load conda
conda activate cenv

srun -u python3 custom_WSI_pipeline/process_WSI.py \
    --wsi-dir $wsi_dir  \
    --model $extraction_model  \
    --output-dir $final_output_path \
    --jpg-dir $jpg_dir \
# python3 custom_WSI_pipeline/process_WSI.py --wsi-dir "test_WSI_test" --model "UNI" --output-dir "test_output/UNI" --jpg-dir "test_JPGs" 