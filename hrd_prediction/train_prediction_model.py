
import argparse
from pathlib import Path
import pandas as pd
from mil_train import train_CAMIL_model
from config_utils import get_MIL_config, get_training_config, get_setup_config

def train_and_test_model(args, config: None):
    # Load configs
    MIL_model_config = get_MIL_config(args)
    training_config = get_training_config(args)
    setup_config = get_setup_config(args)
    
    print("EPOCHS: ", training_config["epochs"])

    if config is not None:
        MIL_model_config.update(config)
        training_config.update(config)
        setup_config.update(config)

    patient_data = pd.read_excel(args.patient_data_file)
    patient_data = patient_data[~patient_data["process_error"]]
    patient_data.reset_index(drop=True, inplace=True)

    if setup_config["cohort"] in ["UCEC-LUAD", "LUAD-UCEC"]:
        train_cohort, test_cohort = setup_config["cohort"].split("-")
        train_feature_path = f"{args.dataset_path}/TCGA/{train_cohort}/features/{setup_config['extraction_model']}"
        test_feature_path = f"{args.dataset_path}/CPTAC/{test_cohort}/features/{setup_config['extraction_model']}"
    else:
        train_feature_path = f"{args.dataset_path}/TCGA/{setup_config['cohort']}/features/{setup_config['extraction_model']}"
        test_feature_path = f"{args.dataset_path}/CPTAC/{setup_config['cohort']}/features/{setup_config['extraction_model']}"

    metrics = train_CAMIL_model(
        setup_config=setup_config,
        training_config=training_config,
        MIL_model_config=MIL_model_config,
        patient_data=patient_data,
        train_feature_path=train_feature_path,
        test_feature_path=test_feature_path,
    )
    
    if config is not None:
        return metrics
    print("DONE!")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CAMIL model for HRD prediction.")
   
    parser.add_argument("--MIL_model", type=str, default="marugoto", choices=["marugoto", "random_attn_topk", "random_4_quantile"])
    parser.add_argument("--extraction_model", type=str, required=True, choices=["UNI", "UNI_2", "RetCCL", "GPFM", "CONCH", "Virchow_2"], help="Name of the feature extraction model.")
    parser.add_argument("--cohort", type=str, required=True, choices=["UCEC", "LUAD", "UCEC-LUAD", "LUAD-UCEC"], help="Cohort to filter the data.")
    parser.add_argument("--target_label", type=str, default="HRD_sum", choices=["HRD_sum", "HRD_Binary"], help="Target label for regression. HRD_sum for regression, HRD_Binary for classification.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--patient_data_file", type=Path, default="datafiles/TCGA_CPTAC_data.xlsx", help="Path to a XLSX file containing the patient Data with labels.")
    parser.add_argument("--dataset_path", type=Path, default="/data/datasets/images", help="Path to the directory containing the datasets.")
    parser.add_argument("--prediction_level", type=str, choices=["slide", "patient"], help="Whether to predict HRD for each slide or per patient, based on all slides of the patient.")
    parser.add_argument("--sample_bag_size", type=int, default=None, help="Number of instances to sample per bag during training. If None all patches of the slides will be used.")
    parser.add_argument("--sample_amount", type=int, default=1, help="Amount of times a cluster-weighted sample is drawn from each slide/patient for training.")
    parser.add_argument("--use_cluster_based_upsampling", action="store_true", help="Usage of cluster-based Upsampling for rare HRD values. Multiple samples of the bag size are drawn from the same patient during training.")
    parser.add_argument("--upsampling_bins", type=int, default=10, help="Amount of bins to use for the Upsampling.")
    # Add more arguments as needed for config_utils

    args = parser.parse_args()

    print("----------------------------------------------------------")
    print(f"Starting model Training with the following parameters:\n")
    print(f"MIL Model:                  {args.MIL_model}")
    print(f"Cohort:                     {args.cohort}")
    print(f"Extraction Model:           {args.extraction_model}")
    print(f"Target Label:               {args.target_label}")
    print(f"Prediction Level:           {args.prediction_level}")
    print(f"Training Epochs:            {args.epochs}")
    print(f"sample bag size:            {args.sample_bag_size}")
    print(f"sample amount:              {args.sample_amount}")
    print(f"cluster based upsampling:   {args.use_cluster_based_upsampling}")
    print("----------------------------------------------------------")

    train_and_test_model(args, config=None)