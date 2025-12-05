import argparse
import logging
from pathlib import Path
import pandas as pd

import optuna

from mil_train import train_CAMIL_model

def train_cycle(trial):
    patient_data = pd.read_excel(args.patient_data_file)
    patient_data = patient_data[~patient_data["process_error"]]
    
    patient_data.reset_index(drop=True, inplace=True)
    
    train_feature_path = f"{args.dataset_path}/TCGA/{args.cohort}/features/{args.extraction_model}"
    test_feature_path = f"{args.dataset_path}/CPTAC/{args.cohort}/features/{args.extraction_model}"

    upsampling_bins = trial.suggest_int("upsampling_bins", 5, 35)
    alpha = trial.suggest_float("alpha", 0.05, 0.7, step=0.05)
    beta = trial.suggest_float("beta", 0.1, 0.8, step= 0.05)
    return train_CAMIL_model(
        MIL_model=args.MIL_model,
        extraction_model=args.extraction_model,
        patient_data=patient_data,
        train_feature_path=train_feature_path,
        test_feature_path=test_feature_path,
        prediction_level=args.prediction_level,
        cohort=args.cohort,
        target_label=args.target_label,
        epochs=args.epochs,
        sample_bag_size=args.sample_bag_size,
        sample_amount=args.sample_amount,
        use_cluster_based_upsampling=args.use_cluster_based_upsampling,
        upsampling_bins=upsampling_bins,
        alpha=alpha,
        beta=beta,
        no_save = True
        )
    


def main():
    

    # Get Optuna's logger
    optuna_logger = optuna.logging.get_logger("optuna")

    # Create a file handler
    file_handler = logging.FileHandler("optuna_trials.log")
    file_handler.setLevel(logging.INFO)

    # (Optional) create a formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the handler to the Optuna logger
    optuna_logger.addHandler(file_handler)    
    
    
    study = optuna.create_study(direction="maximize", storage="sqlite:///upsampling_hpo.db", study_name="upsampling_hpo", load_if_exists=True)
    study.optimize(train_cycle, timeout=86400) # 15h
    
    
    print("DONE!")


parser = argparse.ArgumentParser(description="Train a CAMIL model for HRD prediction.")


parser.add_argument("--MIL_model", type=str, default="test",choices=["marugoto", "random_attn_topk", "random_4_quantile"])
parser.add_argument("--extraction_model", type=str, required=True, 
                    choices=["UNI", "UNI_2", "RetCCL", "GPFM", "CONCH"], help="Name of the feature extraction model.")
parser.add_argument("--cohort", type=str, required=True, 
                    choices=["UCEC", "LUAD"], help="Cohort to filter the data.") #TODO add more cohorts
parser.add_argument("--target_label", type=str, default="HRD_sum", 
                    choices=["HRD_sum", "HRD_Binary"], help="Target label for regression. HRD_sum for regression, HRD_Binary for classification.")
parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
parser.add_argument("--patient_data_file", type=Path, default="datafiles/TCGA_CPTAC_data.xlsx", help="Path to a XLSX file containing the patient Data with labels.")
parser.add_argument("--dataset_path", type=Path, default="/data/datasets/images", help="Path to the directory containing the datasets.")
parser.add_argument("--prediction_level", type=str, choices=["slide", "patient"], help="Wether to predict HRD for each slide or per patient, based on all slides of the patient. ")
parser.add_argument("--sample_bag_size", type=int, default=None, help="Number of instances to sample per bag during training. If None all patches of the slides will be used.  With the transormer a bagsize is necessary")
parser.add_argument("--sample_amount", type=int, default=1, help="Amount of times a cluster-weighted sample is drawn from each slide/patient for training.")
parser.add_argument("--use_cluster_based_upsampling", action="store_true", help="Usage of cluster-based Upsampling for rare HRD values. Multiple samples of the bag size are drawn from the same patient during training.")


# config = yaml.safe_load(open("hrd_prediction/train_config.yaml", "r"))


args = parser.parse_args()

print("----------------------------------------------------------")
print(f"Starting model Crossvalidation Training with the following parameters:\n")
print(f"MIL Model:                  {args.MIL_model}")
print(f"Cohort:                     {args.cohort}")
print(f"Extraction Model:           {args.extraction_model} --ALL")
print(f"Target Label:               {args.target_label}")
print(f"Prediction Level:           {args.prediction_level}")
print(f"Training Epochs:            {args.epochs}")
print(f"sample bag size:            {args.sample_bag_size}")
print(f"sample amount:              {args.sample_amount}")
print(f"cluster based upsampling:   {args.use_cluster_based_upsampling}")
print("----------------------------------------------------------")

main()
