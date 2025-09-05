import argparse
import glob
import os 
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


import pandas as pd

def main(args):
    auroc_values = plot_roc_curves(args)

    plot_auroc_values(args, auroc_values)

def plot_auroc_values(args, auroc_values):
    plot_path = Path(args.save_path)/'auroc_plots'/args.prediction_level/args.cohort
    plot_path.mkdir(exist_ok=True, parents=True)
    n_models = len(auroc_values)
    
    
    plt.clf()
    plt.boxplot(auroc_values.values(), label=auroc_values.keys())
    plt.xticks(range(1, len(auroc_values) + 1), list(auroc_values.keys()))
    plt.title(f"AUROC for different feature extraction models on {args.cohort}")
    plt.ylabel('AUROC')
    plt.ylim(0.0, 1.0)
    plt.xlabel('Extraction model')
    plt.savefig(plot_path/f'{args.target_label}.jpg')
    





def plot_roc_curves(args):
    
    plot_path = Path(args.save_path)/'roc_plots'/args.prediction_level/args.cohort
    plot_path.mkdir(exist_ok=True, parents=True)
    model_dirs = [d for d in glob.glob(f"{args.result_path}/CAMIL_crossval/{args.prediction_level}/{args.cohort}/*") if os.path.isdir(d)]  
    print(f"{args.result_path}/{args.prediction_level}/{args.cohort}/*") 
    n_models = len(model_dirs)
    
    print(n_models)
    fig, axes = plt.subplots(n_models, 5, figsize=(5*5, 3*n_models))
    
    auroc_values = dict()

    
    for m, model in enumerate(model_dirs):
        model_name = model.split('/')[-1]
        auroc_list = []
        for f, fold in enumerate(glob.glob(model + '/fold*')):
            pred_df = pd.read_csv(Path(fold)/'patient-preds.csv')
            y_true = pd.Series(pred_df[args.target_label]>42)
            y_pred = pred_df['pred']
            
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            
            axes[m, f].plot(fpr, tpr)
            axes[m, f].fill_between(fpr, tpr, alpha=0.2, color='blue')
            axes[m, f].plot([0, 1], [0, 1], linestyle='--', color='gray')
            
            auroc = roc_auc_score(y_true, y_pred)
            axes[m, f].set_title(f"{model_name}; Fold {f}; AUROC: {auroc:0.2f}")
            auroc_list.append(auroc)
        auroc_values[model_name] = auroc_list
           
    fig.suptitle(f"ROC Curves for {args.cohort} ({args.target_label})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave more space at the top
    fig.savefig(plot_path/f'{args.target_label}.jpg')
    
    plt.close()
    return auroc_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize clustering results')
    parser.add_argument('--result_path', type=str, default='hrd_prediction/results', help='Path to the results directory containing model outputs')
    parser.add_argument('--cohort', type=str, default='TCGA_LUAD', help='Name of the patient cohort to plot')
    parser.add_argument('--target_label', type=str, default='HRD_sum', help='Target label to plot ROC for')
    parser.add_argument('--save_path', type=str, default='visualization', help='Path to save the plot')
    parser.add_argument('--prediction_level', default="slide", type=str, choices=['slide', 'patient'], help='Wether to predict HRD for each slide or per patient, based on all slides of the patient.')
    parser.add_argument('--MIL_type', type=str, default='marugoto', choices=['marugoto', 'transformer'], help='Type of MIL model used')
    

    args = parser.parse_args()
    main(args)