import os 
from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error, accuracy_score, roc_auc_score, f1_score, mean_absolute_error, recall_score
from scipy import stats
import pandas as pd
import warnings
from matplotlib import pyplot as plt
import numpy as np
warnings.filterwarnings('ignore') 





def get_stat_dict(pred_df: pd.DataFrame, cohort) -> dict:
    rmse = root_mean_squared_error(pred_df["HRD_sum"], pred_df["pred"])
    mae = mean_absolute_error(pred_df["HRD_sum"], pred_df["pred"])
    pcc = stats.pearsonr(pred_df["HRD_sum"], pred_df["pred"])[0]
    pred_binary, true_binary = pd.Series(pred_df["pred"]>= 42),  pd.Series(pred_df["HRD_sum"]>=42) 
    acc = accuracy_score(true_binary, pred_binary)
    b_acc = balanced_accuracy_score(true_binary, pred_binary)
    auroc = roc_auc_score(true_binary, pred_df["pred"])
    f1 = f1_score(true_binary, pred_binary)
    micro_f1 = f1_score(true_binary, pred_binary, average="micro")
    recall = recall_score(true_binary, pred_binary)
    spearmanR = stats.spearmanr(pred_df["HRD_sum"], pred_df["pred"])[0]
    
    mHRD_auroc = None
    tHRD_auroc = None
    
    if cohort in ["TCGA_BRCA", "TCGA_UCEC", "TCGA_LUAD"]:
    
        thresholds_mHRD = {"TCGA_BRCA": 27, "TCGA_UCEC": 5, "TCGA_LUAD": 35}
        thresholds_tHRD = {"TCGA_BRCA": (17, 37), "TCGA_UCEC": (3, 9), "TCGA_LUAD": (26, 43)}
        
        mHRD_True_binary =  pd.Series(pred_df["pred"] > thresholds_mHRD[cohort])
        
        mHRD_auroc = roc_auc_score(true_binary, mHRD_True_binary)
        
        tHRD_data = pred_df[(pred_df["HRD_sum"] <= thresholds_tHRD[cohort][0]) | (pred_df["HRD_sum"] >= thresholds_tHRD[cohort][1])]
        
        tHRD_binary = pd.Series(tHRD_data["HRD_sum"] >= thresholds_tHRD[cohort][1])
        tHRD_pred = pd.Series(tHRD_data["pred"])
        
        tHRD_auroc = roc_auc_score(tHRD_binary, tHRD_pred)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "Pearson Correlation Coefficient": pcc, 
        "Accuracy": acc,
        "Balanced Accuracy": b_acc,
        "AUROC": auroc,
        "F1-Score": f1,
        "Micro-averaged F1-Score": micro_f1,
        "Recall": recall,
        "Spearman R": spearmanR,
        "mHRD_AUROC": mHRD_auroc,
        "tHRD_AUROC": tHRD_auroc,
    }
    
def plot_binned_error(pred_df: pd.DataFrame, path: str, n_bins: int = 10):
    true = pred_df["HRD_sum"]
    pred = pred_df["pred"]
    edges = np.histogram_bin_edges(true, bins=n_bins) 
    rmses = []
    for i in range(len(edges)):
        if i == len(edges)-2:
            indexes = np.where((true >= edges[i]) & (true <= edges[i+1]))[0]
        else:
            indexes = np.where((true >= edges[i]) & (true < edges[i+1]))[0]
        
        if len(indexes) > 0:
            rmses.append(root_mean_squared_error(true[indexes], pred[indexes]))
        else:
            rmses.append(0)
        
    
        if i == len(edges)-2:
            break
        
    plt.clf()
    plt.stairs(rmses, edges)
    plt.ylabel("RMSE")
    plt.xlabel("HRD score bins")
    plt.title("Binned RMSE")
    plt.savefig(f"{path}/binned_rmse.png")



rows = []
for MIL_model in os.listdir("hrd_prediction/results"):
    if os.path.isdir(f"hrd_prediction/results/{MIL_model}"):
        for sampling in os.listdir(f"hrd_prediction/results/{MIL_model}"):
            for prediction_level in os.listdir(f"hrd_prediction/results/{MIL_model}/{sampling}"):
                for cohort in os.listdir(f"hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}"):
                    for extraction_model in os.listdir(f"hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}"):
                        if MIL_model.endswith("full_train"):
                            try:
                                path = f"hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}/{extraction_model}"
                                pred_df = pd.read_csv(f"{path}/patient-preds.csv")
                                row = {
                                        "MIL_type": MIL_model,
                                        "Cluster-based Sampling": sampling,
                                        "Prediction Level": prediction_level,
                                        "Cohort": cohort,
                                        "Extraction Model": extraction_model,
                                        "Fold": None,
                                    }
                                stat_dict = get_stat_dict(pred_df, cohort)
                                
                                if not os.path.exists(f"{path}/binned_rmse.png"):
                                    plot_binned_error(pred_df, path)
                                row.update(stat_dict)
                                rows.append(row)
                                print("processed", path)
                            except FileNotFoundError:
                                print(f"ERROR: hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}/{extraction_model}/patient-preds.csv does not exist")

                        else:
                            if os.path.isdir(f"hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}/{extraction_model}"):
                                for fold in os.listdir(f"hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}/{extraction_model}"):
                                    if os.path.isdir(f"hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}/{extraction_model}/{fold}"):
                                        try:
                                            path = f"hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}/{extraction_model}/{fold}"
                                            pred_df = pd.read_csv(f"{path}/patient-preds.csv")
                                            row = {
                                                "MIL_type": MIL_model,
                                                "Cluster-based Sampling": sampling,
                                                "Prediction Level": prediction_level,
                                                "Cohort": cohort,
                                                "Extraction Model": extraction_model,
                                                "Fold": fold,
                                            }
                                            
                                            stat_dict = get_stat_dict(pred_df, cohort)
                                            
                                            if not os.path.exists(f"{path}/binned_rmse.png"):
                                                plot_binned_error(pred_df, path)
                                            row.update(stat_dict)
                                            rows.append(row)
                                            print("processed", path)
                                            
                                        except FileNotFoundError:
                                            print(f"ERROR: hrd_prediction/results/{MIL_model}/{sampling}/{prediction_level}/{cohort}/{extraction_model}/{fold}/patient-preds.csv does not exist")

stat_df = pd.DataFrame(rows)


                                    
stat_df.to_excel("hrd_prediction/results/result_stats.xlsx")
                        
        