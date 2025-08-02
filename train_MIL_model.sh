#!/bin/bash

#SBATCH --job-name=TrainMILModel
#SBATCH --output=./logs/%x/%j-log.txt
#SBATCH --error=./logs/%x/%j-error.txt
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1 
#SBATCH --chdir=/home/alexander.blezinger/HRD_regression


module load conda 
conda activate hrd_pred

datafile="datafiles/TCGA_CPTAC_data.xlsx"
MIL_type="marugoto"
extraction_model="RetCCL"
cohort="TCGA_LUAD"
target_label="HRD_sum"
epochs=25



srun -u python3 hrd_prediction/train_prediction_model.py \
    --MIL_type $MIL_type\
    --extraction_model $extraction_model \
    --cohort $cohort \
    --target_label $target_label \
    --patient_data_file $datafile \
    --epochs $epochs \
# python3 hrd_prediction/train_prediction_model.py --MIL_type "test" --extraction_model "resnet50" --cohort "TCGA_LUAD" --target_label "HRD_sum" --patient_data_file "datafiles/TCGA_CPTAC_data.xlsx" --epochs 25 