#!/bin/bash

#SBATCH --job-name=RetCCL_TrainMIL
#SBATCH --output=./logs/%x/%j-log.txt
#SBATCH --error=./logs/%x/%j-error.txt
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --partition=ampere
#SBATCH --chdir=/home/alexander.blezinger/HRD_regression


module load conda 
conda activate hrd_pred

datafile="datafiles/TCGA_CPTAC_data.xlsx"
MIL_type="marugoto"
extraction_model="RetCCL"  # Options: "GPFM", "RetCCL", "CONCH", "UNI", "UNI_2"
cohort="TCGA_UCEC"  # Options: "CPTAC_PDA", "CPTAC_BRCA" (TCGA), "TCGA_UCEC", "TCGA_LUAD", 
target_label="HRD_sum"
epochs=25
prediciton_level="slide" # patient or slide
bag_size=300
sample_amount=1



srun -u python3 hrd_prediction/train_crossvalidation.py \
    --MIL_type $MIL_type\
    --extraction_model $extraction_model \
    --cohort $cohort \
    --target_label $target_label \
    --epochs $epochs \
    --prediction_level $prediciton_level \
# python3 hrd_prediction/train_crossvalidation.py --MIL_type "marugoto" --extraction_model "CONCH" --cohort "CPTAC_PDA" --target_label "HRD_sum" --prediction_level "slide"

srun -u python3 hrd_prediction/train_crossvalidation.py \
    --MIL_type $MIL_type\
    --extraction_model $extraction_model \
    --cohort $cohort \
    --target_label $target_label \
    --epochs $epochs \
    --prediction_level $prediciton_level \
    --sample_bag_size $bag_size \
    --sample_amount $sample_amount
# python3 hrd_prediction/train_crossvalidation.py --MIL_type "marugoto" --extraction_model "CONCH" --cohort "CPTAC_PDA" --target_label "HRD_sum" --prediction_level "slide" --sample_bag_size 300 --sample_amount 1


    # --patient_data_file $datafile \

# srun -u python3 hrd_prediction/train_crossvalidation.py \
#     --MIL_type "marugoto" \
#     --extraction_model "UNI"\
#     --cohort "CPTAC_PDA" \
#     --target_label "HRD_sum"