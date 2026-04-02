from datetime import datetime
import json
from pathlib import Path
from pyexpat import features
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split, StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error, accuracy_score, roc_auc_score, f1_score, mean_absolute_error

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from fastai.vision.learner import load_learner
import torch
from sklearn.preprocessing import MinMaxScaler
import sklearn
import seaborn as sns
import scipy
import os
from collections import defaultdict
import gc
#from marugoto.data import FunctionTransformer

from _mil import train_marugoto, train_marugoto_crossval, deploy
# from .data import get_cohort_df, get_target_enc
from marugoto_helpers.data import get_patient_df


def train_CAMIL_model(
    setup_config: dict,
    training_config: dict,
    MIL_model_config: dict,
    patient_data: pd.DataFrame,
    train_feature_path: Path,
    test_feature_path: Path,
):
    
    # Unpack configs
    MIL_model = MIL_model_config["MIL_model"]
    extraction_model = setup_config["extraction_model"]
    prediction_level = setup_config["prediction_level"]
    cohort = setup_config["cohort"]
    target_label = setup_config.get("target_label", "HRD_sum")
    cat_labels = setup_config.get("cat_labels", [])
    cont_labels = setup_config.get("cont_labels", [])
    epochs = training_config.get("epochs", 25)
    bagsize = setup_config.get("bag_size", None)
    sample_amount = setup_config.get("sample_amount", 1)
    use_cluster_based_upsampling = setup_config.get("use_cluster_based_upsampling", False)
    upsampling_bins = setup_config.get("upsampling_bins", 10)
    seed = setup_config.get("seed", 42)
    alpha = training_config.get("alpha", 0.65)
    beta = training_config.get("beta", 0.25)

    ## DATA LOADING AND PREPROCESSING 
    if bagsize is None:
        output_path = Path(f"hrd_prediction/results/real_{MIL_model}_full_train/no_sampling_seed_{seed}/{prediction_level}/{cohort}/{extraction_model}")
    elif use_cluster_based_upsampling:
        output_path = Path(f"hrd_prediction/results/real_{MIL_model}_full_train/upsampling_{upsampling_bins}_bins_bagsize_{bagsize}_nSamples_{sample_amount}_seed_{seed}/{prediction_level}/{cohort}/{extraction_model}")
    else:
        output_path = Path(f"hrd_prediction/results/real_{MIL_model}_full_train/bagsize_{bagsize}_nSamples_{sample_amount}_seed_{seed}/{prediction_level}/{cohort}/{extraction_model}")
    
    if not setup_config["test_mode"]:
        os.makedirs(output_path, exist_ok=True)
    
    # set random seeds
    set_seed(seed)    
    
    data_df = patient_data.dropna(subset=[target_label])
        
    if cohort in ["UCEC-LUAD", "LUAD-UCEC"]:
        train_cohort, test_cohort = cohort.split("-")
        train_df = data_df[data_df['cohort'] == f"TCGA_{train_cohort}"].reset_index(drop=True)
        test_df = data_df[data_df['cohort'] ==  f"CPTAC_{test_cohort}"].reset_index(drop=True)
    elif cohort is not None:
        train_df = data_df[data_df['cohort'] == f"TCGA_{cohort}"].reset_index(drop=True)
        test_df = data_df[data_df['cohort'] ==  f"CPTAC_{cohort}"].reset_index(drop=True)
    
    # get Dataframe with patients, hrd_values and list feature files
    train_df = get_patient_df(patient_df=train_df, data_path=train_feature_path)
    test_df = get_patient_df(patient_df=test_df, data_path=test_feature_path)
    
    
    if target_label == "HRD_sum":
        train_df[target_label] = train_df[target_label].astype('float32')
        test_df[target_label] = test_df[target_label].astype('float32')

    
    # Min-Max normalize the target label
    scaler = MinMaxScaler()
    train_targets = train_df[[target_label]].to_numpy()
    train_df[target_label] = scaler.fit_transform(train_targets).ravel()

    # split into train and validation sets
    train_patients, valid_patients = train_test_split(train_df.patient_id, random_state=seed) #, stratify=df[target_label]
    # train_df = train_df[train_df.patient_id.isin(train_patients)]
    # valid_df = train_df[train_df.patient_id.isin(valid_patients)]

    # TODO: additional features neben slides, sollte leer sien 
    add_features = []
    # if cat_labels: add_features.append((_make_cat_enc(train_df, cat_labels), df[cat_labels].values))
    # if cont_labels: add_features.append((_make_cont_enc(train_df, cont_labels), df[cont_labels].values))
    
    
    if (preds_csv := output_path/'patient-preds.csv').exists() and not setup_config["test_mode"]:
        print(f'{preds_csv} already exists!  Skipping...')
        return
    elif (output_path/'export.pkl').exists() and not setup_config["test_mode"]:
        learn = load_learner(output_path/'export.pkl')
    else:
        print("#### training_model...")
        learn = train_marugoto(
            setup_config=setup_config,
            training_config=training_config,
            MIL_model_config=MIL_model_config,
            bags=train_df.feature_files.values,
            targets=(train_df[target_label].values).reshape(-1,1),
            valid_idxs=train_df.patient_id.isin(valid_patients).values,
            path=output_path,
        )
        
        learn.target_label = setup_config.get("target_label", "HRD_sum")
        learn.cat_labels = setup_config.get("cat_labels", [])
        learn.cont_labels = setup_config.get("cont_labels", [])
        if not setup_config["test_mode"]:
            print("#### saving model...")
            learn.export()
    
    print("Evaluation on Test Dataset...")

    test_targets = test_df[[target_label]].to_numpy()
    test_df[target_label] = scaler.transform(test_targets).ravel()

    test_patient_preds = deploy(
        test_df=test_df,
        learn=learn,
        setup_config=setup_config,
    )
    
    test_patient_preds[target_label] = scaler.inverse_transform(test_patient_preds[target_label].values.reshape(-1,1)).ravel()
    test_patient_preds['pred'] = scaler.inverse_transform(test_patient_preds['pred'].values.reshape(-1,1)).ravel()
    
    #cleanup 
    del learn
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    
    if not setup_config["test_mode"]:
        
        plot_pcc(test_patient_preds, target_label, output_path, extraction_model)
    
        print("saving predictions")
        test_patient_preds.to_csv(preds_csv, index=False)
    else: 
        
        bal_accuracy = balanced_accuracy_score(pd.Series(test_patient_preds[target_label]>42), pd.Series(test_patient_preds["pred"]> 42))
        rmse = root_mean_squared_error(test_patient_preds[target_label], test_patient_preds["pred"])
        acc = accuracy_score(test_patient_preds[target_label]>42, test_patient_preds["pred"]> 42)
        metrics = {
            "balanced_accuracy": float(bal_accuracy),
            "rmse": float(rmse),
            "accuracy": float(acc)
        }
        
        print(f"Metrics: {metrics}")
        return metrics


def train_CAMIL_model_crossval(
    setup_config: dict,
    training_config: dict,
    MIL_model_config: dict,
    # MIL_model: str,
    # extraction_model: str,
    patient_data: pd.DataFrame,
    feature_path: Path,
    # prediction_level: str,
    # cohort: str|None = None,
    # binary_label: str|None = None,
    # target_label: str = "HRD_sum",
    # cat_labels: Sequence[str] = [],
    # cont_labels: Sequence[str] = [],
    # epochs: int = 25,
    # n_splits: int = 5,
    # bagsize: int = None,
    # sample_amount: int = 1,
    # sampling_strategy: str  = "cluster_size",
    # use_cluster_based_upsampling:bool = False,
    # upsampling_bins:int = 10,
    # alpha: float = 0.65,
    # beta: float = 0.25,
    ):
    
    # prefix for different k-Means level
    prefix = ""
    
    print(patient_data)
    
    if setup_config["bag_size"] is None:
        output_path = Path(f"hrd_prediction/results/{prefix}real_{MIL_model_config['MIL_model']}_crossval/no_sampling/{setup_config['prediction_level']}/{setup_config['cohort']}/{setup_config['extraction_model']}")
        fold_save_path = Path(f"hrd_prediction/results/{prefix}real_{MIL_model_config['MIL_model']}_crossval/no_sampling/{setup_config['prediction_level']}/{setup_config['cohort']}")
    
    elif setup_config["use_cluster_based_upsampling"]:
        output_path = Path(f"hrd_prediction/results/{prefix}real_{MIL_model_config['MIL_model']}_crossval/upsampling_{setup_config['upsampling_bins']}_bins_bagsize_{setup_config['bag_size']}_nSamples_{setup_config['sample_amount']}/{setup_config['prediction_level']}/{setup_config['cohort']}/{setup_config['extraction_model']}")
        fold_save_path = Path(f"hrd_prediction/results/{prefix}real_{MIL_model_config['MIL_model']}_crossval/upsampling_{setup_config['upsampling_bins']}_bins_bagsize_{setup_config['bag_size']}_nSamples_{setup_config['sample_amount']}/{setup_config['prediction_level']}/{setup_config['cohort']}") 
    
    elif setup_config["sampling_strategy"] == "random":
        output_path = Path(f"hrd_prediction/results/{prefix}real_{MIL_model_config['MIL_model']}_crossval/random_sampling_bagsize_{setup_config['bag_size']}_nSamples_{setup_config['sample_amount']}/{setup_config['prediction_level']}/{setup_config['cohort']}/{setup_config['extraction_model']}")
        fold_save_path = Path(f"hrd_prediction/results/{prefix}real_{MIL_model_config['MIL_model']}_crossval/random_sampling_bagsize_{setup_config['bag_size']}_nSamples_{setup_config['sample_amount']}/{setup_config['prediction_level']}/{setup_config['cohort']}")
    
    elif setup_config["sampling_strategy"] == "clustered_random":
        output_path = Path(f"hrd_prediction/results/{prefix}real_{MIL_model_config['MIL_model']}_crossval/clustered_random_sampling_bagsize_{setup_config['bag_size']}_nSamples_{setup_config['sample_amount']}/{setup_config['prediction_level']}/{setup_config['cohort']}/{setup_config['extraction_model']}")
        fold_save_path = Path(f"hrd_prediction/results/{prefix}real_{MIL_model_config['MIL_model']}_crossval/clustered_random_sampling_bagsize_{setup_config['bag_size']}_nSamples_{setup_config['sample_amount']}/{setup_config['prediction_level']}/{setup_config['cohort']}")
    
    elif setup_config["sampling_strategy"] == "cluster_size":
        output_path = Path(f"hrd_prediction/results/{prefix}real_{MIL_model_config['MIL_model']}_crossval/bagsize_{setup_config['bag_size']}_nSamples_{setup_config['sample_amount']}/{setup_config['prediction_level']}/{setup_config['cohort']}/{setup_config['extraction_model']}")
        fold_save_path = Path(f"hrd_prediction/results/{prefix}real_{MIL_model_config['MIL_model']}_crossval/bagsize_{setup_config['bag_size']}_nSamples_{setup_config['sample_amount']}/{setup_config['prediction_level']}/{setup_config['cohort']}")
        
    else:
        raise NotImplementedError(f"There is a mistake somewhere, check your the sampling setup.")
    
    os.makedirs(output_path, exist_ok=True)
    print(fold_save_path)
    info = {
        'description': f'{setup_config["extraction_model"]} MIL cross-validation',
        'target_label': str(setup_config['target_label']),
        'output_path': str(output_path.absolute()),
        'n_splits': setup_config['n_splits'],
        'datetime': datetime.now().astimezone().isoformat(),
    }

    ## DATA LOADING AND PREPROCESSING
    data_df = patient_data.dropna(subset=[setup_config['target_label']])
    
    print(data_df['cohort'].value_counts())
    print(f"SETUP COHORT: {setup_config['cohort']}")
    print(data_df[data_df['cohort'] == "CPTAC_PDA"])
    if setup_config['cohort'] is not None:
        data_df = data_df[data_df['cohort'] == setup_config['cohort']].reset_index(drop=True)
    print(data_df)
    
    # get Dataframe with patients, hrd_values and list feature files --> each entry corresponds to one patient
    data_df = get_patient_df(data_df, data_path=feature_path) #categories
    
    
    if setup_config['target_label'] == "HRD_sum":
        data_df[setup_config['target_label']] = data_df[setup_config['target_label']].astype('float32')

    # print(data_df)
    # create folds 
    if (fold_path := fold_save_path/'folds.pt').exists():
        folds = torch.load(fold_path)
    else:
        #added shuffling with seed 1337
        skf = KFold(n_splits=setup_config['n_splits'], shuffle=True, random_state=1337)
        patient_df = data_df.groupby('patient_id').first().reset_index()
        
        # print(patient_df)
        
        folds = tuple(skf.split(patient_df.patient_id, patient_df[setup_config['target_label']])) # patient_df['SITE_CODE'])) with stratified potentially
        torch.save(folds, fold_path)
    
    info['folds'] = [
        {
            part: list(data_df.patient_id[folds[fold_i][i]])
            for i, part in enumerate(['train', 'test'])
        }
        for fold_i in range(info['n_splits']) ]
    
    info["setup_config"] = setup_config
    info["training_config"] = training_config
    info["MIL_model_config"] = MIL_model_config
    
    with open(output_path/'info.json', 'w') as f:
        json.dump(info, f)
        
    
    metrics = []
    
    ## FOLDWISE TRAINING 
    for fold, (train_idxs, test_idxs) in enumerate(folds):
        fold_path = output_path/f'fold-{fold}'
        
        #minmax normalisation for train set, save distrib for test
        fold_train_df = pd.DataFrame(data_df.iloc[train_idxs])
        scaler = MinMaxScaler().fit(fold_train_df[[setup_config['target_label']]].to_numpy())
        fold_train_df[setup_config['target_label']] = scaler.transform(fold_train_df[[setup_config['target_label']]].to_numpy()).ravel()

        # Train the model on train split
        if (preds_csv := fold_path/'patient-preds.csv').exists():
            print(f'{preds_csv} already exists!  Skipping...')
            continue
        elif (fold_path/'export.pkl').exists():
            learn = load_learner(fold_path/'export.pkl')
        else:         
            learn = train_marugoto_crossval(
                setup_config=setup_config,
                training_config=training_config,
                MIL_model_config=MIL_model_config,
                # MIL_model=MIL_model,
                fold_path=fold_path, 
                fold_df=fold_train_df,
                # target_label=target_label, #, target_enc=target_enc,
                # cat_labels=cat_labels, 
                # cont_labels=cont_labels,
                # binary_label=binary_label,
                # n_epochs=epochs,
                # prediction_level=prediction_level, 
                # sample_bag_size=bagsize,
                # sample_amount=sample_amount,
                # sampling_strategy=sampling_strategy,
                # use_cluster_based_upsampling=use_cluster_based_upsampling,
                # upsampling_bins=upsampling_bins,
                ) #added weights #fold_weights_train=fold_weights_train
            learn.export()
            
            
        # Test model on test split 
        #minmax normalisation for test set with train distrib (same scaler object)
        fold_test_df = pd.DataFrame(data_df.iloc[test_idxs])
        fold_test_df.drop(columns='feature_files').to_csv(fold_path/'test.csv', index=False)
        fold_test_df[setup_config["target_label"]] = scaler.transform(fold_test_df[[setup_config["target_label"]]].to_numpy()).ravel()
        
        patient_preds_df = deploy(
            test_df=fold_test_df, 
            learn=learn, #send weights to be all ones, i.e. nothing changes weights=np.ones(test_idxs.shape)
            setup_config=setup_config,
            # target_label=setup_config["target_label"], 
            # cat_labels=cat_labels, 
            # cont_labels=cont_labels,
            # prediction_level=prediction_level,
            # sampling_strategy=sampling_strategy,
            # sample_bag_size=bagsize
        )

        #rescale ground truth and patient predictions to original range
        patient_preds_df[setup_config["target_label"]] = scaler.inverse_transform(patient_preds_df[setup_config["target_label"]].values.reshape(-1,1)).ravel()
        patient_preds_df['pred'] = scaler.inverse_transform(patient_preds_df['pred'].values.reshape(-1,1)).ravel()

        
        if not setup_config["test_mode"]:

            #obtain pearson's R and create plot per fold
            plot_pearsr_df = patient_preds_df[[setup_config["target_label"], "pred"]]
            pears = scipy.stats.pearsonr(plot_pearsr_df[setup_config["target_label"]], plot_pearsr_df['pred'])[0]
            pval = scipy.stats.pearsonr(plot_pearsr_df[setup_config["target_label"]], plot_pearsr_df['pred'])[1]
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(plot_pearsr_df[setup_config["target_label"]], plot_pearsr_df['pred'])
            ax = sns.lmplot(x=setup_config["target_label"], y='pred', data=plot_pearsr_df, )
            ax.set(title=f"{os.path.basename(output_path)}\nR^2: {np.round(r_value**2, 2)} | Pearson's R: {np.round(pears,2)} | p-value: {np.round(pval, 7)}")
            #ax.set(ylim=(0,1), xlim=(0,1)) #set a x/y-limit to get the same plots for a specific project
            max_value = max(patient_preds_df[setup_config["target_label"]].max(), patient_preds_df['pred'].max()) 
            for a in ax.axes.flat:
                a.set_ylim(0, max_value + 4)
                a.set_xlim(0, max_value + 4)
            ax.savefig(fold_path/"correlation_plot.png")

        if not setup_config["test_mode"]:
            patient_preds_df.to_csv(preds_csv, index=False)
        
        
        #cleanup 
        del learn
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        
        
        
        
        
def plot_pcc(patient_preds_df, target_label, save_path, extraction_model):

        #obtain pearson's R and create plot per fold
        plot_pearsr_df = patient_preds_df[[target_label, "pred"]]
        pears = scipy.stats.pearsonr(plot_pearsr_df[target_label], plot_pearsr_df['pred'])[0]
        pval = scipy.stats.pearsonr(plot_pearsr_df[target_label], plot_pearsr_df['pred'])[1]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(plot_pearsr_df[target_label], plot_pearsr_df['pred'])
        ax = sns.lmplot(x=target_label, y='pred', data=plot_pearsr_df, )
        ax.set(title=f"{extraction_model}\nR^2: {np.round(r_value**2, 2)} | Pearson's R: {np.round(pears,2)} | p-value: {np.round(pval, 7)}")
        #ax.set(ylim=(0,1), xlim=(0,1)) #set a x/y-limit to get the same plots for a specific project
        max_value = max(patient_preds_df[target_label].max(), patient_preds_df['pred'].max()) 
        for a in ax.axes.flat:
            a.set_ylim(0, max_value + 4)
            a.set_xlim(0, max_value + 4)
        ax.savefig(save_path/"correlation_plot.png")

def set_seed(seed:int = 42):
    sklearn.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)