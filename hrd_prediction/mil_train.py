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
#from marugoto.data import FunctionTransformer

from _mil import train_marugoto, train_marugoto_crossval, deploy
# from .data import get_cohort_df, get_target_enc
from marugoto_helpers.data import get_patient_df


def train_CAMIL_model(
    MIL_model: str,
    extraction_model: str,
    patient_data: pd.DataFrame,
    train_feature_path: Path,
    test_feature_path: Path,
    prediction_level: str,
    cohort: str|None = None,
    target_label: str = "HRD_sum",
    cat_labels: Sequence[str] = [],
    cont_labels: Sequence[str] = [],
    epochs: int = 25,
    sample_bag_size: int = None,
    sample_amount: int = 1,
    use_cluster_based_upsampling:bool = False,
    upsampling_bins:int = 10,
    seed:int = 42,
    alpha: float = 0.65,
    beta: float = 0.25,
    no_save: bool = False
):
    
    ## DATA LOADING AND PREPROCESSING 
    if sample_bag_size is None:
        output_path = Path(f"hrd_prediction/results/{MIL_model}_full_train/no_sampling_seed_{seed}/{prediction_level}/{cohort}/{extraction_model}")
    elif use_cluster_based_upsampling:
        output_path = Path(f"hrd_prediction/results/{MIL_model}_full_train/upsampling_{upsampling_bins}_bins_bagsize_{sample_bag_size}_nSamples_{sample_amount}_seed_{seed}/{prediction_level}/{cohort}/{extraction_model}")
    else:
        output_path = Path(f"hrd_prediction/results/{MIL_model}_full_train/bagsize_{sample_bag_size}_nSamples_{sample_amount}_seed_{seed}/{prediction_level}/{cohort}/{extraction_model}")
    
    if not no_save: 
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
    train_df[target_label] = scaler.fit_transform(train_df[[target_label]])

    # split into train and validation sets
    train_patients, valid_patients = train_test_split(train_df.patient_id, random_state=seed) #, stratify=df[target_label]
    # train_df = train_df[train_df.patient_id.isin(train_patients)]
    # valid_df = train_df[train_df.patient_id.isin(valid_patients)]

    # TODO: additional features neben slides, sollte leer sien 
    add_features = []
    # if cat_labels: add_features.append((_make_cat_enc(train_df, cat_labels), df[cat_labels].values))
    # if cont_labels: add_features.append((_make_cont_enc(train_df, cont_labels), df[cont_labels].values))
    
    
    if (preds_csv := output_path/'patient-preds.csv').exists():
            print(f'{preds_csv} already exists!  Skipping...')
            return

    elif (output_path/'export.pkl').exists():
        learn = load_learner(output_path/'export.pkl')
    else:       
    # TRAINING THE MODEL 
        print("#### training_model...")
        
        
        learn = train_marugoto(
            MIL_model=MIL_model,
            bags=train_df.feature_files.values,
            targets=(train_df[target_label].values).reshape(-1,1),
            # add_features=add_features,
            valid_idxs=train_df.patient_id.isin(valid_patients).values, # boolean Series of validation patients
            prediction_level=prediction_level,
            n_epoch=epochs,
            path=output_path,
            sample_bag_size=sample_bag_size,
            sample_amount=sample_amount,
            use_cluster_based_upsampling=use_cluster_based_upsampling,
            upsampling_bins=upsampling_bins,
            alpha=alpha,
            beta=beta,
            no_save=no_save,
        )
    
        learn.target_label = target_label
        learn.cat_labels, learn.cont_labels = cat_labels, cont_labels

        #save the mode
        print("#### saving model...")
        
        if not no_save: 
            learn.export()
    
    print("Evaluation on Test Dataset...")

    test_df[target_label] = scaler.transform(test_df[target_label].values.reshape(-1,1))


    test_patient_preds = deploy(
            test_df=test_df, 
            learn=learn, #send weights to be all ones, i.e. nothing changes weights=np.ones(test_idxs.shape)
            target_label=target_label, 
            cat_labels=cat_labels, 
            cont_labels=cont_labels,
            prediction_level=prediction_level,
            sample_bag_size=sample_bag_size
        )
    
    test_patient_preds[target_label] = scaler.inverse_transform(test_patient_preds[target_label].values.reshape(-1,1))
    test_patient_preds['pred'] = scaler.inverse_transform(test_patient_preds['pred'].values.reshape(-1,1))
    
    if not no_save:
        
        plot_pcc(test_patient_preds, target_label, output_path, extraction_model)
    
        print("saving predictions")
        test_patient_preds.to_csv(preds_csv, index=False)
    else: 
        
        bal_accuracy = balanced_accuracy_score(pd.Series(test_patient_preds[target_label]>42), pd.Series(test_patient_preds["pred"]> 42))
        
        return float(bal_accuracy)



def train_CAMIL_model_crossval(
    MIL_model: str,
    extraction_model: str,
    patient_data: pd.DataFrame,
    feature_path: Path,
    prediction_level: str,
    cohort: str|None = None,
    binary_label: str|None = None,
    target_label: str = "HRD_sum",
    cat_labels: Sequence[str] = [],
    cont_labels: Sequence[str] = [],
    epochs: int = 25,
    n_splits: int = 5,
    sample_bag_size: int = None,
    sample_amount: int = 1,
    sample_randomly:bool = False,
    use_cluster_based_upsampling:bool = False,
    upsampling_bins:int = 10,
    alpha: float = 0.65,
    beta: float = 0.25,
    ):
    
    if sample_bag_size is None:
        output_path = Path(f"hrd_prediction/results/{MIL_model}_crossval/no_sampling/{prediction_level}/{cohort}/{extraction_model}")
        fold_save_path = Path(f"hrd_prediction/results/{MIL_model}_crossval/no_sampling/{prediction_level}/{cohort}")
    elif use_cluster_based_upsampling:
        output_path = Path(f"hrd_prediction/results/{MIL_model}_crossval/upsampling_{upsampling_bins}_bins_bagsize_{sample_bag_size}_nSamples_{sample_amount}/{prediction_level}/{cohort}/{extraction_model}")
        fold_save_path = Path(f"hrd_prediction/results/{MIL_model}_crossval/upsampling_{upsampling_bins}_bins_bagsize_{sample_bag_size}_nSamples_{sample_amount}/{prediction_level}/{cohort}") 
    
    elif sample_randomly:
        output_path = Path(f"hrd_prediction/results/{MIL_model}_crossval/random_sampling_bagsize_{sample_bag_size}_nSamples_{sample_amount}/{prediction_level}/{cohort}/{extraction_model}")
        fold_save_path = Path(f"hrd_prediction/results/{MIL_model}_crossval/random_sampling_bagsize_{sample_bag_size}_nSamples_{sample_amount}/{prediction_level}/{cohort}")
    else:
        output_path = Path(f"hrd_prediction/results/{MIL_model}_crossval/bagsize_{sample_bag_size}_nSamples_{sample_amount}/{prediction_level}/{cohort}/{extraction_model}")
        fold_save_path = Path(f"hrd_prediction/results/{MIL_model}_crossval/bagsize_{sample_bag_size}_nSamples_{sample_amount}/{prediction_level}/{cohort}")
    os.makedirs(output_path, exist_ok=True)
    info = {
        'description': f'{extraction_model} MIL cross-validation',
        'target_label': str(target_label),
        'output_path': str(output_path.absolute()),
        'n_splits': n_splits,
        'datetime': datetime.now().astimezone().isoformat(),
    }

    ## DATA LOADING AND PREPROCESSING
    data_df = patient_data.dropna(subset=[target_label])
    
    if cohort is not None:
        data_df = data_df[data_df['cohort'] == cohort].reset_index(drop=True)
    
    # get Dataframe with patients, hrd_values and list feature files --> each entry corresponds to one patient
    data_df = get_patient_df(data_df, data_path=feature_path) #categories
    
    if target_label == "HRD_sum":
        data_df[target_label] = data_df[target_label].astype('float32')

    # create folds 
    if (fold_path := fold_save_path/'folds.pt').exists():
        folds = torch.load(fold_path)
    else:
        #added shuffling with seed 1337
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=1337)
        patient_df = data_df.groupby('patient_id').first().reset_index()
        folds = tuple(skf.split(patient_df.patient_id, patient_df[target_label])) # patient_df['SITE_CODE'])) with stratified potentially
        torch.save(folds, fold_path)
    
    info['folds'] = [
        {
            part: list(data_df.patient_id[folds[fold_i][i]])
            for i, part in enumerate(['train', 'test'])
        }
        for fold_i in range(info['n_splits']) ]
    with open(output_path/'info.json', 'w') as f:
        json.dump(info, f)
        
        
    ## FOLDWISE TRAINING 
    for fold, (train_idxs, test_idxs) in enumerate(folds):
        fold_path = output_path/f'fold-{fold}'
        
        #minmax normalisation for train set, save distrib for test
        fold_train_df = pd.DataFrame(data_df.iloc[train_idxs])
        scaler=MinMaxScaler().fit(fold_train_df[target_label].values.reshape(-1,1))
        fold_train_df[target_label] = scaler.transform(fold_train_df[target_label].values.reshape(-1,1))

        # Train the model on train split
        if (preds_csv := fold_path/'patient-preds.csv').exists():
            print(f'{preds_csv} already exists!  Skipping...')
            continue
        elif (fold_path/'export.pkl').exists():
            learn = load_learner(fold_path/'export.pkl')
        else:         
            learn = train_marugoto_crossval(
                MIL_model=MIL_model,
                fold_path=fold_path, 
                fold_df=fold_train_df,
                target_label=target_label, #, target_enc=target_enc,
                cat_labels=cat_labels, 
                cont_labels=cont_labels,
                binary_label=binary_label,
                n_epochs=epochs,
                prediction_level=prediction_level, 
                sample_bag_size=sample_bag_size,
                sample_amount=sample_amount,
                sample_randomly=sample_randomly,
                use_cluster_based_upsampling=use_cluster_based_upsampling,
                upsampling_bins=upsampling_bins,
                ) #added weights #fold_weights_train=fold_weights_train
            learn.export()
            
            
        # Test model on test split 
        #minmax normalisation for test set with train distrib (same scaler object)
        fold_test_df = pd.DataFrame(data_df.iloc[test_idxs])
        fold_test_df.drop(columns='feature_files').to_csv(fold_path/'test.csv', index=False)
        fold_test_df[target_label] = scaler.transform(fold_test_df[target_label].values.reshape(-1,1))
        
        patient_preds_df = deploy(
            test_df=fold_test_df, 
            learn=learn, #send weights to be all ones, i.e. nothing changes weights=np.ones(test_idxs.shape)
            target_label=target_label, 
            cat_labels=cat_labels, 
            cont_labels=cont_labels,
            prediction_level=prediction_level,
        )

        #rescale ground truth and patient predictions to original range
        patient_preds_df[target_label] = scaler.inverse_transform(patient_preds_df[target_label].values.reshape(-1,1))
        patient_preds_df['pred'] = scaler.inverse_transform(patient_preds_df['pred'].values.reshape(-1,1))

        #obtain pearson's R and create plot per fold
        plot_pearsr_df = patient_preds_df[[target_label, "pred"]]
        pears = scipy.stats.pearsonr(plot_pearsr_df[target_label], plot_pearsr_df['pred'])[0]
        pval = scipy.stats.pearsonr(plot_pearsr_df[target_label], plot_pearsr_df['pred'])[1]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(plot_pearsr_df[target_label], plot_pearsr_df['pred'])
        ax = sns.lmplot(x=target_label, y='pred', data=plot_pearsr_df, )
        ax.set(title=f"{os.path.basename(output_path)}\nR^2: {np.round(r_value**2, 2)} | Pearson's R: {np.round(pears,2)} | p-value: {np.round(pval, 7)}")
        #ax.set(ylim=(0,1), xlim=(0,1)) #set a x/y-limit to get the same plots for a specific project
        max_value = max(patient_preds_df[target_label].max(), patient_preds_df['pred'].max()) 
        for a in ax.axes.flat:
            a.set_ylim(0, max_value + 4)
            a.set_xlim(0, max_value + 4)
        ax.savefig(fold_path/"correlation_plot.png")

        patient_preds_df.to_csv(preds_csv, index=False)
        
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