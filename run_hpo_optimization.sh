#!/bin/bash
#SBATCH --job-name=Virchow_2_HPO_LUAD
#SBATCH --output=./logs/HPO/%x/%j-log.txt
#SBATCH --error=./logs/HPO/%x/%j-error.txt
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --partition=ampere
#SBATCH --chdir=/home/alexander.blezinger/HRD_regression

# Load modules or activate your environment if needed
# module load python/3.10
# source ~/myenv/bin/activate

datafile="datafiles/TCGA_CPTAC_data.xlsx"
MIL_type="marugoto" #["marugoto", "random_attn_topk", "random_4_quantile"]
extraction_model="Virchow_2"  # Options: "GPFM", "RetCCL", "CONCH", "UNI", "UNI_2", "Virchow_2"
cohort="LUAD"  # Options: "LUAD", "UCEC"
target_label="HRD_sum"
epochs=25
prediction_level="patient" # patient or slide
# bag_size=${bag_size:-800}
sample_amount=1
sampling_strategy="clustered_random" # Options: "clustered_random", "random", "cluster_size"

echo "Extraction model: $extraction_model"
# echo "Bag size: $bag_size"

module load conda 
conda activate hrd_new

# python3 hrd_prediction/hpo_optimization.py --extraction_model "UNI" --cohort "LUAD" --epochs 1 --prediction_level "patient" --sampling_strategy "clustered_random" --test_mode
srun --cpu-bind=none -u python3 hrd_prediction/hpo_optimization.py \
    --extraction_model $extraction_model \
    --cohort $cohort \
    --target_label $target_label \
    --epochs $epochs \
    --prediction_level $prediction_level \
    --sampling_strategy $sampling_strategy \
    --test_mode