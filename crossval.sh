#!/bin/bash

#SBATCH --job-name=RetCCL_2000
#SBATCH --output=./logs/%x/%j-log.txt
#SBATCH --error=./logs/%x/%j-error.txt
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --partition=ampere
#SBATCH --chdir=/home/alexander.blezinger/HRD_regression


module load conda 
conda activate hrd_new

datafile="datafiles/TCGA_CPTAC_data.xlsx"
MIL_model="marugoto"  #["marugoto", "random_attn_topk", "random_4_quantile"]
extraction_model="RetCCL"  # Options: "GPFM", "RetCCL", "CONCH", "UNI", "UNI_2", "Virchow_2"
cohort="TCGA_BRCA"  # Options: "CPTAC_PDA", "CPTAC_BRCA" (TCGA), "TCGA_UCEC", "TCGA_LUAD", 
target_label="HRD_sum"
epochs=25
prediciton_level="patient" # patient or slide
bag_size=2000
sample_amount=1
clustersized_upsampling=false
sampling_strategy="clustered_random" # Options: "clustered_random", "random", "cluster_size"
## STANDARD marugoto

# srun -u python3 hrd_prediction/train_crossvalidation.py \
#     --MIL_model $MIL_model\
#     --extraction_model $extraction_model \
#     --cohort $cohort \
#     --target_label $target_label \
#     --epochs $epochs \
#     --prediction_level $prediciton_level \
# python3 hrd_prediction/train_crossvalidation.py --MIL_type "marugoto" --extraction_model "CONCH" --cohort "CPTAC_PDA" --target_label "HRD_sum" --prediction_level "slide"

## CLUSTER_WEIGHTED SAMPLING MARUGOTO and SURE

# srun -u python3 hrd_prediction/train_crossvalidation.py \
#     --MIL_model $MIL_model\
#     --extraction_model $extraction_model \
#     --cohort $cohort \
#     --target_label $target_label \
#     --epochs $epochs \
#     --prediction_level $prediciton_level \
#     --sample_bag_size $bag_size \
#     --sample_amount $sample_amount
# # python3 hrd_prediction/train_crossvalidation.py --MIL_model "marugoto" --extraction_model "CONCH" --cohort "CPTAC_PDA" --target_label "HRD_sum" --prediction_level "patient" --sample_bag_size 600 --sample_amount 1

# CLUSTER BASED UPSAMPLING 
# srun -u python3 hrd_prediction/train_crossvalidation.py \
#     --MIL_model $MIL_model\
#     --extraction_model $extraction_model \
#     --cohort $cohort \
#     --target_label $target_label \
#     --epochs $epochs \
#     --prediction_level $prediciton_level \
#     --sample_bag_size $bag_size \
#     --sample_amount $sample_amount\
#     --use_cluster_based_upsampling \
#     --upsampling_bins 10
#python3 hrd_prediction/train_crossvalidation.py --MIL_model "marugoto" --extraction_model "CONCH" --cohort "CPTAC_PDA" --target_label "HRD_sum" --prediction_level "patient" --sample_bag_size 600 --sample_amount 1 --use_cluster_based_upsampling --upsampling_bins 10

## Clustered Random Sampling
srun -u python3 hrd_prediction/train_crossvalidation.py \
    --MIL_model $MIL_model \
    --extraction_model $extraction_model \
    --cohort $cohort \
    --target_label $target_label \
    --epochs $epochs \
    --prediction_level $prediciton_level \
    --sample_bag_size $bag_size \
    --sample_amount $sample_amount\
    --sampling_strategy $sampling_strategy