import os
import pandas as pd
import numpy as np
from MLstatkit import Delong_test

BASELINE_MODEL = "RetCCL"

# result_dir = "hrd_prediction/results/marugoto_full_train/no_sampling/patient/LUAD"
result_dir = "hrd_prediction/results/random_attn_topk_full_train/bagsize_1200_nSamples_1/patient/LUAD"

baseline_results = pd.read_csv(f"{result_dir}/{BASELINE_MODEL}/patient-preds.csv")
print(result_dir)
for extraction_model in os.listdir(result_dir):
    if extraction_model == BASELINE_MODEL:
        continue
    else:
        try:
            preds = pd.read_csv(f"{result_dir}/{extraction_model}/patient-preds.csv")
            
            ground_truth = np.where(baseline_results["HRD_sum"] > 42, 1, 0)
            baseline_preds = baseline_results["pred"]
            test_preds = preds["pred"]
            
            z, p = Delong_test(ground_truth, baseline_preds, test_preds)
            print(extraction_model)
            print(z)
            print(p)
            
            # raise ValueError
        except FileNotFoundError:
            print(f"{extraction_model} does not have patient-preds")
            
