import yaml
import os 

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from mil_train import train_CAMIL_model_crossval

def main(args):
    np.random.seed(42)
    patient_data = pd.read_excel(args.patient_data_file)
    patient_data = patient_data[patient_data["process_error"] != True]
    
    patient_data.reset_index(drop=True, inplace=True)
    
    dataset, cohort = args.cohort.split("_")
    
    feature_path = f"{args.dataset_path}/{dataset}/{cohort}/features/{args.extraction_model}"
    
    train_CAMIL_model_crossval(
        MIL_model=args.MIL_model,
        extraction_model=args.extraction_model,
        patient_data=patient_data,
        feature_path=feature_path,
        prediction_level=args.prediction_level,
        cohort=args.cohort,
        target_label=args.target_label,
        epochs=args.epochs,
        n_splits=args.n_splits,
        sample_bag_size=args.sample_bag_size,
        sample_amount=args.sample_amount,
        sampling_strategy= args.sampling_strategy,
        use_cluster_based_upsampling=args.use_cluster_based_upsampling,
        upsampling_bins=args.upsampling_bins,
        )
        
    print("DONE!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CAMIL model for HRD prediction.")
    
    parser.add_argument("--MIL_model", type=str, default="test",choices=["marugoto", "random_attn_topk", "random_4_quantile"])
    parser.add_argument("--extraction_model", type=str, required=True, 
                        choices=["UNI", "UNI_2", "RetCCL", "GPFM", "CONCH", "Virchow_2"], help="Name of the feature extraction model.")
    parser.add_argument("--cohort", type=str, required=True, 
                        choices=["TCGA_UCEC", "TCGA_LUAD", "CPTAC_PDA", "TCGA_BRCA", "CPTAC_UCEC"], help="Cohort to filter the data.") #TODO add more cohorts
    parser.add_argument("--target_label", type=str, default="HRD_sum", 
                        choices=["HRD_sum", "HRD_Binary"], help="Target label for regression. HRD_sum for regression, HRD_Binary for classification.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--patient_data_file", type=Path, default="datafiles/TCGA_CPTAC_data_backup.xlsx", help="Path to a XLSX file containing the patient Data with labels.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for cross-validation.")
    parser.add_argument("--dataset_path", type=Path, default="/data/datasets/images", help="Path to the directory containing the datasets.")
    parser.add_argument("--prediction_level", type=str, choices=["slide", "patient"], help="Wether to predict HRD for each slide or per patient, based on all slides of the patient. ")
    parser.add_argument("--sample_bag_size", type=int, default=None, help="Number of instances to sample per bag during training. If None all patches of the slides will be used.  With the transormer a bagsize is necessary")
    parser.add_argument("--sample_amount", type=int, default=1, help="Amount of times a cluster-weighted sample is drawn from each slide/patient for training.")
    parser.add_argument("--sample_randomly", action="store_true", help="Use random sampling instead of cluster-based sampling for bag creation during training.")
    parser.add_argument("--use_cluster_based_upsampling", action="store_true", help="Usage of cluster-based Upsampling for rare HRD values. Multiple samples of the bag size are drawn from the same patient during training.")
    parser.add_argument("--upsampling_bins", type=int, default= 10, help="Amount of bins to use for the Upsampling.")
    parser.add_argument("--sampling_strategy", type=str, default="cluster_size", choices=["cluster_size", "random", "clustered_random"], help="Sampling strategy to use for bag creation during training.")
    
    
    # config = yaml.safe_load(open("hrd_prediction/train_config.yaml", "r"))
    

    args = parser.parse_args()
    
    print("----------------------------------------------------------")
    print(f"Starting model Crossvalidation Training with the following parameters:\n")
    print(f"MIL Model:\t\t\t\t\t{args.MIL_model}")
    print(f"Cohort: \t\t\t\t\t{args.cohort}")
    print(f"Extraction Model: \t\t\t{args.extraction_model}")
    print(f"Target Label: \t\t\t\t{args.target_label}")
    print(f"Prediction Level: \t\t\t{args.prediction_level}")
    print(f"Training Epochs: \t\t\t{args.epochs}")
    print(f"Crossvalidation Folds: \t{args.n_splits}")
    print(f"sample bag size:            {args.sample_bag_size}")
    print(f"sample amount: \t\t\t\t{args.sample_amount}")
    print(f"cluster based upsampling:   {args.use_cluster_based_upsampling}")
    print(f"upsampling bins:            {args.upsampling_bins}")
    print(f"sampling strategy:          {args.sampling_strategy}")
    print("----------------------------------------------------------")
    
    main(args)