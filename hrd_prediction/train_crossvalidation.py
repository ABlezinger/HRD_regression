from train_mil_marugoto import train_CAMIL_model_crossval
import yaml
import os 

import argparse
from pathlib import Path
import pandas as pd

def main(args):
    patient_data = pd.read_excel(args.patient_data_file)
    patient_data = patient_data[patient_data["process_error"] != True]
    
    patient_data.reset_index(drop=True, inplace=True)
    
    dataset, cohort = args.cohort.split("_")
    
    feature_path = f"{args.dataset_path}/{dataset}/{cohort}/features/{args.extraction_model}"
    
    if args.MIL_type == "marugoto":
        train_CAMIL_model_crossval(
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
            )
        
    print("DONE!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CAMIL model for HRD prediction.")
    
    parser.add_argument("--MIL_type", type=str, default="test",choices=["marugoto", "transformer", "test"])
    parser.add_argument("--extraction_model", type=str, required=True, 
                        choices=["UNI", "UNI_2", "RetCCL", "GPFM", "CONCH"], help="Name of the feature extraction model.")
    parser.add_argument("--cohort", type=str, required=True, 
                        choices=["TCGA_UCEC", "TCGA_LUAD", "CPTAC_PDA"], help="Cohort to filter the data.") #TODO add more cohorts
    parser.add_argument("--target_label", type=str, default="HRD_sum", 
                        choices=["HRD_sum", "HRD_Binary"], help="Target label for regression. HRD_sum for regression, HRD_Binary for classification.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--patient_data_file", type=Path, default="datafiles/TCGA_CPTAC_data.xlsx", help="Path to a XLSX file containing the patient Data with labels.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for cross-validation.")
    parser.add_argument("--dataset_path", type=Path, default="/data/datasets/images", help="Path to the directory containing the datasets.")
    parser.add_argument("--prediction_level", type=str, choices=["slide", "patient"], help="Wether to predict HRD for each slide or per patient, based on all slides of the patient. ")
    parser.add_argument("--sample_bag_size", type=int, default=None, help="Number of instances to sample per bag during training. If None all patches of the slides will be used.  With the transormer a bagsize is necessary")
    parser.add_argument("--sample_amount", type=int, default=1, help="Amount of times a cluster-weighted sample is drawn from each slide/patient for training.")
    
    
    # config = yaml.safe_load(open("hrd_prediction/train_config.yaml", "r"))
    

    args = parser.parse_args()
    
    print("----------------------------------------------------------")
    print(f"Starting model Crossvalidation Training with the following parameters:\n")
    print(f"MIL Type:\t\t\t\t{args.MIL_type}")
    print(f"Cohort: \t\t\t\{args.cohort}")
    print(f"Extraction Model: \t\t\t{args.extraction_model}")
    print(f"Target Label: \t\t\t\t{args.target_label}")
    print(f"Prediction Level: \t\t\t{args.prediction_level}")
    print(f"Training Epochs: \t\t\t{args.epochs}")
    print(f"Crossvalidation Folds : \t{args.n_splits}")
    print(f"sample bag size: \t\t\t{args.sample_bag_size}")
    print(f"sample amount: \t\t\t\t{args.sample_amount}")
    print("----------------------------------------------------------")
    
    main(args)