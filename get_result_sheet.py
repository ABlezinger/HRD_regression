import os 
from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error, accuracy_score, roc_auc_score, f1_score, mean_absolute_error
from scipy import stats
import pandas as pd
import warnings
warnings.filterwarnings('ignore') 





def get_stat_dict(pred_df: pd.DataFrame) -> dict:
    rmse = root_mean_squared_error(pred_df["HRD_sum"], pred_df["pred"])
    mae = mean_absolute_error(pred_df["HRD_sum"], pred_df["pred"])
    pcc = stats.pearsonr(pred_df["HRD_sum"], pred_df["pred"])[0]
    pred_binary, true_binary = pd.Series(pred_df["pred"]> 42),  pd.Series(pred_df["HRD_sum"]>42) 
    acc = accuracy_score(true_binary, pred_binary)
    b_acc = balanced_accuracy_score(true_binary, pred_binary)
    auroc = roc_auc_score(true_binary, pred_df["pred"])
    f1 = f1_score(true_binary, pred_binary)
    micro_f1 = f1_score(true_binary, pred_binary, average="micro")
    return {
        "RMSE": rmse,
        "MAE": mae,
        "Pearson Correlation Coefficient": pcc, 
        "Accuracy": acc,
        "Balanced Accuracy": b_acc,
        "AUROC": auroc,
        "F1-Score": f1,
        "Micro-averaged F1-Score": micro_f1
    }
    



rows = []
for MIL_model in os.listdir("hrd_prediction/results"):
    if os.path.isdir(f"hrd_prediction/results/{MIL_model}"):
        for sampling in os.listdir(f"hrd_prediction/results/{MIL_model}"):
            for prediction_level in os.listdir(f"hrd_prediction/results/{MIL_model}/{sampling}"):
                for cohort in os.listdir(f"hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}"):
                    for extraction_model in os.listdir(f"hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}"):
                        if os.path.isdir(f"hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}/{extraction_model}"):
                            for fold in os.listdir(f"hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}/{extraction_model}"):
                                if os.path.isdir(f"hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}/{extraction_model}/{fold}"):
                                    try:
                                        pred_df = pd.read_csv(f"hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}/{extraction_model}/{fold}/patient-preds.csv")
                                        row = {
                                            "MIL_type": MIL_model,
                                            "Cluster-based Sampling": sampling,
                                            "Prediction Level": prediction_level,
                                            "Cohort": cohort,
                                            "Extraction Model": extraction_model,
                                            "Fold": fold,
                                        }
                                        
                                        stat_dict = get_stat_dict(pred_df)
                                        row.update(stat_dict)
                                        rows.append(row)
                                    except FileNotFoundError:
                                        print(f"ERROR: hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}/{extraction_model}/{fold}/patient-preds.csv does not exist")
stat_df = pd.DataFrame(rows)
                                    
stat_df.to_excel("hrd_prediction/results/result_stats.xlsx")
                        
        