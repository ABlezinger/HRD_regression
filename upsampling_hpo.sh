#!/bin/bash

#SBATCH --job-name=HPO_UPSAMPLE
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
MIL_model="marugoto"  #["marugoto", "random_attn_topk", "random_4_quantile"]
extraction_model="CONCH"  # Options: "GPFM", "RetCCL", "CONCH", "UNI", "UNI_2"
cohort="LUAD"  # Options: "UCEC", "LUAD", 
target_label="HRD_sum"
epochs=25
prediciton_level="patient" # patient or slide
bag_size=800
sample_amount=1


## STANDARD marugoto

# srun --cpu-bind=none -u python3 hrd_prediction/train_prediction_model.py \
#     --MIL_model $MIL_model\
#     --extraction_model $extraction_model \
#     --cohort $cohort \
#     --target_label $target_label \
#     --epochs $epochs \
#     --prediction_level $prediciton_level \
# python3 hrd_prediction/train_prediction_model.py --MIL_model "marugoto" --extraction_model "CONCH" --cohort "UCEC" --target_label "HRD_sum" --prediction_level "patient"

## CLUSTER_WEIGHTED SAMPLING MARUGOTO and SURE

# srun --cpu-bind=none -u python3 hrd_prediction/train_prediction_model.py \
#     --MIL_model $MIL_model\
#     --extraction_model $extraction_model \
#     --cohort $cohort \
#     --target_label $target_label \
#     --epochs $epochs \
#     --prediction_level $prediciton_level \
#     --sample_bag_size $bag_size \
#     --sample_amount $sample_amount
# python3 hrd_prediction/train_prediction_model.py --MIL_model "marugoto" --extraction_model "UNI" --cohort "UCEC" --target_label "HRD_sum" --prediction_level "patient" --sample_bag_size 600 --sample_amount 1

## CLUSTER BASED UPSAMPLING 
srun --cpu-bind=none -u python3 hrd_prediction/upsampling_hpo.py \
    --MIL_model $MIL_model\
    --extraction_model $extraction_model \
    --cohort $cohort \
    --target_label $target_label \
    --epochs $epochs \
    --prediction_level $prediciton_level \
    --sample_bag_size $bag_size \
    --sample_amount $sample_amount\
    --use_cluster_based_upsampling \
#python3 hrd_prediction/upsampling_hpo.py --MIL_model "marugoto" --extraction_model "CONCH" --cohort "LUAD" --target_label "HRD_sum" --prediction_level "patient" --sample_bag_size 800 --sample_amount 1 --use_cluster_based_upsampling
