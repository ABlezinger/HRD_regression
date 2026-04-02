import argparse
from pathlib import Path

import optuna 
from train_crossvalidation import start_crossvalidation
from train_prediction_model import train_and_test_model
from functools import partial

def cycle(trial: optuna.trial.Trial, args):
    
    # Define your hyperparameters and their search space here   
    config = build_config(trial)
    
    # Train your model with the given hyperparameters and return the evaluation metric
    metrics= train_and_test_model(args, config)
    
    return metrics[args.metric]


def main():
    print("Starting HPO optimization...")
    
    global args
    args = parse_args()
    print("Arguments parsed. Starting Optuna study...")
    
    # setup Optuna study with SQLite storage
    name = f"NAS_hpo_{args.extraction_model}_{args.cohort}"
    db_directory = Path("HPO_dbs")
    db_directory.mkdir(exist_ok=True)  
    storage=f"sqlite:///{db_directory}/{name}.db"
    print(f"Creating Optuna study in {db_directory}/{name}.db...")
    
    study = optuna.create_study(direction="maximize", study_name=name, load_if_exists=True, storage=storage, )
    print(f"Optuna study created and saved in {db_directory}/{name}.db. Starting optimization...")
    
    study.optimize(partial(cycle, args=args), n_trials=200, timeout=86400, )    
    
    print("Best hyperparameters: ", study.best_params)
    print("Best metric: ", study.best_value)
    

def build_config(trial):
    # Build your argument configuration based on the trial's suggested hyperparameters
    config = {
            "MIL_model": trial.suggest_categorical("MIL_model", ["marugoto", "random_attn_topk"]),
            "bag_size": trial.suggest_int("bag_size", 600, 1600, step=200)
    }
    
    if config["MIL_model"] == "marugoto":
        config["encoding_dim"] = trial.suggest_categorical("encoding_dim", [128, 256, 512])
        config["attention_layers"] = trial.suggest_int("attention_layers", 1, 3)
        config["head_depth"] = trial.suggest_int("head_depth", 1, 3)
        config["dropout"] = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)

    elif config["MIL_model"] == "random_attn_topk":
        config["transformer_layers"] = trial.suggest_int("transformer_layers", 2, 12, step=2)
        config["heads"] = trial.suggest_int("heads", 2, 12, step=2)
        config["dim_head"] = trial.suggest_categorical("dim_head", [32, 64, 128, 256])
        config["dim"] = config["dim_head"]*config["heads"]
        # config["mlp_dim"] = trial.suggest_categorical("mlp_dim", [128, 256, 512])
        config["mlp_dim"] = config["dim"]
        config["dropout"] = trial.suggest_float("dropout", 0.0, 0.5, step=0.1) 
        
        
        # Add more parameters as needed
    
    return config
    
def parse_args():
    parser = argparse.ArgumentParser(description="Train a CAMIL model for HRD prediction.")
    
    parser.add_argument("--extraction_model", type=str, required=True, 
                        choices=["UNI", "UNI_2", "RetCCL", "GPFM", "CONCH", "Virchow_2"], help="Name of the feature extraction model.")
    parser.add_argument("--cohort", type=str, required=True, 
                        choices=["LUAD", "UCEC"], help="Cohort to filter the data.") #TODO add more cohorts
    parser.add_argument("--target_label", type=str, default="HRD_sum", 
                        choices=["HRD_sum", "HRD_Binary"], help="Target label for regression. HRD_sum for regression, HRD_Binary for classification.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--patient_data_file", type=Path, default="datafiles/TCGA_CPTAC_data.xlsx", help="Path to a XLSX file containing the patient Data with labels.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for cross-validation.")
    parser.add_argument("--dataset_path", type=Path, default="/data/datasets/images", help="Path to the directory containing the datasets.")
    parser.add_argument("--prediction_level", type=str, choices=["slide", "patient"], help="Wether to predict HRD for each slide or per patient, based on all slides of the patient. ")
    # parser.add_argument("--sample_bag_size", type=int, default=None, help="Number of instances to sample per bag during training. If None all patches of the slides will be used.  With the transformer a bagsize is necessary")
    parser.add_argument("--sample_amount", type=int, default=1, help="Amount of times a cluster-weighted sample is drawn from each slide/patient for training.")
    parser.add_argument("--sample_randomly", action="store_true", help="Use random sampling instead of cluster-based sampling for bag creation during training.")
    parser.add_argument("--use_cluster_based_upsampling", action="store_true", help="Usage of cluster-based Upsampling for rare HRD values. Multiple samples of the bag size are drawn from the same patient during training.")
    parser.add_argument("--upsampling_bins", type=int, default= 10, help="Amount of bins to use for the Upsampling.")
    parser.add_argument("--sampling_strategy", type=str, default="cluster_size", choices=["cluster_size", "random", "clustered_random"], help="Sampling strategy to use for bag creation during training.")
    parser.add_argument("--test_mode", action="store_true", help="If set to True, the code will run in test mode with reduced dataset size and training epochs for quick testing.")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["balanced_accuracy", "rmse", "accuracy"], help="Metric to optimize during HPO.")
    
    arguments = parser.parse_args()
    return arguments

if __name__ == "__main__":
    main()