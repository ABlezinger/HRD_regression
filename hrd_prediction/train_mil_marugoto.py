from datetime import datetime
import json
from pathlib import Path
from pyexpat import features
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from fastai.vision.learner import load_learner
import torch
from sklearn.preprocessing import MinMaxScaler
# import seaborn as sns
import scipy
import os
from collections import defaultdict
#from marugoto.data import FunctionTransformer

from marugoto_helpers._mil import train, deploy
# from .data import get_cohort_df, get_target_enc
from marugoto_helpers.data import get_patient_df


def train_CAMIL_model(
    extraction_model: str,
    patient_data: pd.DataFrame,
    cohort: str|None = None,
    target_label: str = "HRD_sum",
    cat_labels: Sequence[str] = [],
    cont_labels: Sequence[str] = [],
    epochs: int = 25,
):
    output_path = Path(f"hrd_prediction/models/CAMIL/{extraction_model}")
    os.makedirs(output_path, exist_ok=True)
    
    model_path = output_path / 'best_model.pth'
    
    
    data_df = patient_data.dropna(subset=[target_label])
    
    if cohort is not None:
        data_df = data_df[data_df['cohort'] == cohort]
    
    # get 
    data_df = get_patient_df(data_df)
    
    if target_label == "HRD_sum":
        data_df[target_label] = data_df[target_label].astype('float32')
    
    # Min-Max normalize the target label
    scaler = MinMaxScaler()
    data_df[target_label] = scaler.fit_transform(data_df[[target_label]])

    # plit into train and validation sets
    train_patients, valid_patients = train_test_split(data_df.patient_id) #, stratify=df[target_label]
    train_df = data_df[data_df.patient_id.isin(train_patients)]
    valid_df = data_df[data_df.patient_id.isin(valid_patients)]

    # TODO: additional features neben slides, sollte leer sien 
    add_features = []
    # if cat_labels: add_features.append((_make_cat_enc(train_df, cat_labels), df[cat_labels].values))
    # if cont_labels: add_features.append((_make_cont_enc(train_df, cont_labels), df[cont_labels].values))
    
    #train model
    print("#### training_model...")
    learn = train(
        bags=data_df.feature_files.values,
        targets=data_df[target_label].values,
        add_features=add_features,
        valid_idxs=data_df.patient_id.isin(valid_patients), # boolean Series of validation patients
        path=output_path,
        n_epoch=epochs,
        
    )
    
    learn.target_label = target_label
    learn.cat_labels, learn.cont_labels = cat_labels, cont_labels

    #save the mode
    print("#### saving model...")
    learn.export()

    pass


def _make_cat_enc(df, cats) -> FunctionTransformer:
    # create a scaled one-hot encoder for the categorical values
    #
    # due to weirdeties in sklearn's OneHotEncoder.fit we fill NAs with other values
    # randomly sampled with the same probability as their distribution in the
    # dataset.  This is necessary for correctly determining StandardScaler's weigth
    fitting_cats = []
    for cat in cats:
        weights = df[cat].value_counts(normalize=True)
        non_na_samples = df[cat].fillna(pd.Series(np.random.choice(weights.index, len(df), p=weights)))
        fitting_cats.append(non_na_samples)
    cat_samples = np.stack(fitting_cats, axis=1)
    cat_enc = make_pipeline(
        FunctionTransformer(), #OneHotEncoder(sparse=False, handle_unknown='ignore'),
        StandardScaler(),
    ).fit(cat_samples)
    return cat_enc


def _make_cont_enc(df, conts) -> FunctionTransformer:
    cont_enc = make_pipeline(
        StandardScaler(),
        SimpleImputer(fill_value=0)
    ).fit(df[conts].values)
    return cont_enc

