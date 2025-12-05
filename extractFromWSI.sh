#!/bin/bash

#SBATCH --job-name=rev_extract_Virchow2-TBRCA
#SBATCH --output=./logs/extraction/%j-%x-log.txt
#SBATCH --error=./logs/extraction/%j-%x-error.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=1
#SBATCH --chdir=/home/alexander.blezinger/HRD_regression
#SBATCH --partition=ampere
#SBATCH --time=90:00:00


#parameters
extraction_model="Virchow_2"  # Options: "GPFM", "RetCCL", "CONCH", "UNI", "UNI_2", "Virchow_2"
output_dir="/data/datasets/images/TCGA/BRCA/features"
jpg_dir="/data/datasets/images/TCGA/BRCA/PNG"
wsi_dir="/data/datasets/images/CPTAC/BRCA"

final_output_path="$output_dir/$extraction_model"

module load conda


if [[ $extraction_model = "CONCH" ]]; then
    conda activate cenv_2
else
    conda activate cenv
fi

srun --cpu-bind=none -u python3 custom_WSI_pipeline/process_WSI.py \
    --wsi-dir $wsi_dir  \
    --model $extraction_model  \
    --output-dir $final_output_path \
    --jpg-dir $jpg_dir \
# python3 custom_WSI_pipeline/process_WSI.py --wsi-dir "test_WSI_test" --model "UNI" --output-dir "test_output/UNI" --jpg-dir "test_JPGs" 
# python3 custom_WSI_pipeline/process_WSI.py --wsi-dir "/data/datasets/images/CPTAC/PDA" --model "RetCCL" --output-dir "/data/datasets/images/CPTAC/PDA/features/RetCCL" --jpg-dir "/data/datasets/images/CPTAC/PDA/PNG" 