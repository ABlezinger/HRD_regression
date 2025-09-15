#!/bin/bash

#SBATCH --output=./logs/%x/%j-log.txt
#SBATCH --error=./logs/%x/%j-error.txt
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --partition=ampere
#SBATCH --chdir=/home/alexander.blezinger/HRD_regression


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --extraction_model) extraction_model="$2"; shift ;;
    esac
    shift
done

datafile="datafiles/TCGA_CPTAC_data.xlsx"
MIL_type="random_4_quantile" #["marugoto", "random_attn_topk", "random_4_quantile"]
# extraction_model="RetCCL"  # Options: "GPFM", "RetCCL", "CONCH", "UNI", "UNI_2"
cohort="TCGA_LUAD"  # Options: "CPTAC_PDA", "CPTAC_BRCA" (TCGA), "TCGA_UCEC", "TCGA_LUAD", 
target_label="HRD_sum"
epochs=25
prediciton_level="patient" # patient or slide
bag_size=1200
sample_amount=1
clustersized_upsampling=true

echo $extraction_model

module load conda 
conda activate hrd_pred

# srun -u python3 hrd_prediction/train_crossvalidation.py \
#     --MIL_model $MIL_type\
#     --extraction_model $extraction_model \
#     --cohort $cohort \
#     --target_label $target_label \
#     --epochs $epochs \
#     --prediction_level $prediciton_level \
# python3 hrd_prediction/train_crossvalidation.py --MIL_type "marugoto" --extraction_model "CONCH" --cohort "CPTAC_PDA" --target_label "HRD_sum" --prediction_level "slide"

srun -u python3 hrd_prediction/train_crossvalidation.py \
    --MIL_model $MIL_type\
    --extraction_model $extraction_model \
    --cohort $cohort \
    --target_label $target_label \
    --epochs $epochs \
    --prediction_level $prediciton_level \
    --sample_bag_size $bag_size \
    --sample_amount $sample_amount
# python3 hrd_prediction/train_crossvalidation.py --MIL_model "marugoto" --extraction_model "CONCH" --cohort "TCGA_UCEC" --target_label "HRD_sum" --prediction_level "patient" --sample_bag_size 600 --sample_amount 1


    # --patient_data_file $datafile \

# srun -u python3 hrd_prediction/train_crossvalidation.py \
#     --MIL_model "marugoto" \
#     --extraction_model "UNI"\
#     --cohort "CPTAC_PDA" \
#     --target_label "HRD_sum"