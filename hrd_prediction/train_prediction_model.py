from train_mil_marugoto import train_CAMIL_model

import argparse
from pathlib import Path
import pandas as pd

def main(args):
    patient_data = pd.read_excel(args.patient_data_file)
    print(patient_data.head())
    
    if args.MIL_type == "marugoto":
        train_CAMIL_model(
            extraction_model=args.extraction_model,
            patient_data=patient_data,
            cohort=args.cohort,
            target_label=args.target_label,
            epochs=args.epochs
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CAMIL model for HRD prediction.")
    
    parser.add_argument("--MIL_type", type=str, default="test",choices=["marugoto", "transformer", "test"])
    parser.add_argument("--extraction_model", type=str, required=True, 
                        choices=["UNI", "UNI_2", "RetCCL", "GPFM", "CONCH"], help="Name of the feature extraction model.")
    parser.add_argument("--cohort", type=str, required=True, 
                        choices=["TCGA-UCEC", "TCGA-LUAD"], help="Cohort to filter the data.") #TODO add more cohorts
    parser.add_argument("--target_label", type=str, default="HRD_sum", 
                        choices=["HRD_sum", "HRD_Binary"], help="Target label for regression. HRD_sum for regression, HRD_Binary for classification.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--patient_data_file", type=Path, required=True, help="Path to a XLSX file containing the patient Data with labels.")

    args = parser.parse_args()
    
    main(args)