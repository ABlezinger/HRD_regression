from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, TypeVar
from pathlib import Path
import os

import h5py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from fastai.vision.all import (
    Learner, DataLoader, DataLoaders, RocAuc,
    SaveModelCallback, CSVLogger, GradientAccumulation)
#from fastai.callback import tensorboard

#from fastai.callback.tensorboard import TensorBoardCallback
import pandas as pd
import numpy as np


#Deep imbalanced regression
from marugoto_helpers.loss import WeightedMSELoss, WeightedL1Loss, WeightedHuberLoss
from fastai.optimizer import OptimWrapper
from fastai.optimizer import SGD

from collections import Counter
from scipy.ndimage import convolve1d
from marugoto_helpers.utils import get_lds_kernel_window
#######################################


# from marugoto.data import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from marugoto_helpers.loss import mean_squared_error
#from sklearn.metrics.pairwise import manhattan_distances

# from sklearn.neural_network import MLPRegressor #just to test a simpler model
# from fastai.vision.all import * #same as avive

from marugoto_helpers.data import make_whole_slide_dataset
from suRe_helpers.data import get_clustered_samples, make_clustered_dataset, arrange_bags_for_upsamling
from suRe_helpers.sure_model import get_suRe_emodel
# from suRe_helpers.data import make_image_level_dataset
from marugoto_helpers.model import MILModel


__all__ = ['train_marugoto', 'deploy']


T = TypeVar('T')

#CHANGED
def train_marugoto(
    *,
    MIL_model: str,
    bags: Sequence[Iterable[Path]],
    targets: np.ndarray,
    add_features: Iterable[Tuple[FunctionTransformer, Sequence[Any]]] = [],
    valid_idxs: np.ndarray,
    prediction_level: str,
    n_epoch: int = 25, #32
    patience: int = 8,
    path: Optional[Path] = None,
    sample_bag_size: Optional[int] = None,
    sample_amount: int = 1,
    cluster_based_upsampling = False
) -> Learner:
    """Train a MLP on image features.

    Args:
        bags:  H5s containing bags of tiles.
        targets:  An (encoder, targets) pair.
        add_features:  An (encoder, targets) pair for each additional input.
        valid_idxs:  Indices of the datasets to use for validation.
    """

    def get_bin_idx(x, bins):
    #TODO: find optimal binning strategy
        '''
        x is a continuous variable (normalised) between 0-1
        to get the bins, x is rounded to its nearest decimal,
        and then multiplied by ten. Totalling in 11 bins which
        will be weighed accordingly to its frequency within
        '''
        label = None
        for i, bin in enumerate(bins):
            if x <= bin:
                label = i
                break


        return label


    def weighting_continuous_values(labels) -> torch.FloatTensor:
        """implements Label Distribution Smoothing (LDS) for continuous values 

        Args:
            labels (nd.array): labels as continuous values, e.g. HRD scores

        Returns:
            torch.FloatTensor: _description_
        """
        
        edges = np.histogram_bin_edges(labels, bins='auto')
        bin_index_per_label = np.array([get_bin_idx(label, edges) for label in labels])
        # calculate empirical (original) label distribution: [Nb,]
        # "Nb" is the number of bins
        Nb = max(bin_index_per_label) + 1

        #i.e., np.histogram(bin_index_per_label)
        unique, counts = np.unique(bin_index_per_label, return_counts=True)
        num_samples_of_bins = dict(zip(unique, counts))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
        # emp_label_dist[i] = number of samples in bin i

        # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
        # calculate effective label distribution: [Nb,]
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
        # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
        eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
        weights = torch.FloatTensor([np.float32(1 / x) for x in eff_num_per_label])

        return weights
    
    
    ### arrange Data according the configuration 

    # for slide level predicitons each bag contains only features from one file; targets and indexes have to be appended multiple times
    if prediction_level == "slide":
        new_bags = []
        targs = []
        idx = []
        
        for i, bag_paths in enumerate(bags):
            for bag_path in bag_paths:
                new_bags.append([bag_path])
                targs.append(targets[i])
                idx.append(valid_idxs[i])
        bags = np.array(new_bags, dtype=object)
        targs = np.array(targs)
        valid_idxs = np.array(idx)

    # for patient predictions each bag contains features from mutliple slides
    elif(prediction_level == "patient"):    
        targs = targets

    if cluster_based_upsampling:
        arrange_bags_for_upsamling(bags, targs, valid_idxs)
        
        
    #create samples with cluster-size weighted sampling        
    if sample_bag_size is not None:
        print("Creating clustered samples with fixed bag size: ", sample_bag_size)
        bags, targs, valid_idxs, sampled_indexes = get_clustered_samples(
            bags=bags, 
            targs=targs, 
            bag_size=sample_bag_size, 
            num_samples=sample_amount, 
            prediction_level=prediction_level, 
            valid_idxs=valid_idxs)
    
    
    # enable LDS-Smoothing  for continuous values, since high HRD_scores are rare
    weights = weighting_continuous_values(targs).reshape(-1,1)
    


    
    if sample_bag_size is not None:
        #create dataset with cluster-size weighted sampled bags
        train_ds = make_clustered_dataset(
            bags=bags[~valid_idxs],
            targets= (targs[~valid_idxs], weights[~valid_idxs]),
            sampled_idxs=sampled_indexes[~valid_idxs],
            bag_size=sample_bag_size)
        
        valid_ds = make_clustered_dataset(
            bags=bags[valid_idxs],
            targets= (targs[valid_idxs], weights[valid_idxs]),
            sampled_idxs=sampled_indexes[valid_idxs],
            bag_size=sample_bag_size)

    else:
        #create datasets with the whole image 
        train_ds = make_whole_slide_dataset(
            bags=bags[~valid_idxs],
            targets= (targs[~valid_idxs], weights[~valid_idxs]),
            add_features=[
                (enc, vals[~valid_idxs])
                for enc, vals in add_features],
            bag_size=None) #512 # NONE = use all features of the bag


        #CHANGED
        valid_ds = make_whole_slide_dataset(
            bags=bags[valid_idxs],
            targets=(targs[valid_idxs], weights[valid_idxs]),
            add_features=[
                (enc, vals[valid_idxs])
                for enc, vals in add_features],
            bag_size=None) #None


    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')
    
    # build dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=1, shuffle=False, num_workers=1) #batch_size=64, shuffle=True drop_last=True
    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False, num_workers=1) #batch_size=1, shuffle=False , drop_last=True

    #Graziani et al: batch_size_bag = 1, shuffle=True for both
    batch = train_dl.one_batch()
    
    # print(f"SHAPE INPUT: {batch[0][0].shape}")
    # print(f"BATCH {batch[0]}")


    #added extra [0] because of the new tuple structure
    if MIL_model == "marugoto":
        model = MILModel(batch[0][0].shape[-1], 1) #batch[-1].shape[-1]
    elif MIL_model == "random_attn_topk":
        model = get_suRe_emodel("random_attn_topk", batch[0][0].shape[-1], 1)
    elif MIL_model == "random_4_quantile":
        model = get_suRe_emodel("random_4_quantile", batch[0][0].shape[-1], 1)
        

    # print(model)
    # MILModel(
    # (encoder): Sequential(
    #     (0): Linear(in_features=2048, out_features=256, bias=True)
    #     (1): ReLU()
    # )
    # (attention): Sequential(
    #     (0): Linear(in_features=256, out_features=128, bias=True)
    #     (1): Tanh()
    #     (2): Linear(in_features=128, out_features=1, bias=True)
    # )
    # (head): Sequential(
    #     (0): Flatten(start_dim=1, end_dim=-1)
    #     (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (2): Dropout(p=0.5, inplace=False)
    #     (3): Linear(in_features=256, out_features=1, bias=True)
    # )
    # )
    
    #for imbalanced regression
    loss_func = WeightedMSELoss()

    dls = DataLoaders(train_dl, valid_dl)
    
    #SGD instead of Adam standard, from Graziani et al.
    #def opt_func(params, **kwargs): return OptimWrapper(SGD(params, lr=.0001, mom=.9, wd=0.01))

    #mean squared error metric is 'handmade' from .loss file
    learn = Learner(
        dls,
        model,
        loss_func=loss_func,
        lr=.0001, 
        wd=0.01,
        metrics=[mean_squared_error],
        path=path,
        )


    cbs = [
        SaveModelCallback(fname=f'best_valid'),
        #EarlyStoppingCallback(monitor='roc_auc_score',
        #                      min_delta=0.01, patience=patience),
        #GradientAccumulation(n_acc=64),
        #TensorBoardCallback(),
        CSVLogger()]

    # with learn.no_bar():
    #     learn.fit_one_cycle(n_epoch=n_epoch, lr_max=1e-4, cbs=cbs, )
    learn.fit_one_cycle(n_epoch=n_epoch, lr_max=1e-4, cbs=cbs, )
    return learn

def train_marugoto_crossval(
    *, 
    MIL_model: str,
    fold_path,
    fold_df, 
    target_label,
    cat_labels,
    cont_labels,
    binary_label, #target_enc,fold_weights_train 
    prediction_level,
    n_epochs=25,
    sample_bag_size=None,
    sample_amount=1,
    ) -> Learner:

    """Helper function for training the folds."""
    assert fold_df.patient_id.nunique() == len(fold_df)
    fold_path.mkdir(exist_ok=True, parents=True)


    if binary_label is not None:
        train_patients, valid_patients = train_test_split(
            fold_df.patient_id, stratify=fold_df[binary_label], random_state=1337)
    else:
        train_patients, valid_patients = train_test_split(
            fold_df.patient_id, random_state=1337)
        
    train_df = fold_df[fold_df.patient_id.isin(train_patients)]
    valid_df = fold_df[fold_df.patient_id.isin(valid_patients)]
    train_df.drop(columns='feature_files').to_csv(fold_path/'train.csv', index=False)
    valid_df.drop(columns='feature_files').to_csv(fold_path/'valid.csv', index=False)


    learn = train_marugoto(
        MIL_model=MIL_model,
        bags=fold_df.feature_files.values,
        targets=(fold_df[target_label].values).reshape(-1,1),
        # add_features=add_features,
        valid_idxs=fold_df.patient_id.isin(valid_patients).values,
        path=fold_path,
        prediction_level=prediction_level,
        n_epoch=n_epochs,
        sample_bag_size=sample_bag_size,
        sample_amount=sample_amount,
    )
    
    learn.target_label = target_label
    learn.cat_labels, learn.cont_labels = cat_labels, cont_labels

    return learn

def deploy(
    test_df: pd.DataFrame, 
    learn: Learner,
    *,
    prediction_level: str,
    target_label: Optional[str] = None,
    cat_labels: Optional[Sequence[str]] = None, 
    cont_labels: Optional[Sequence[str]] = None,
    sample_bag_size: Optional[int] = None,
) -> pd.DataFrame:
    assert test_df.patient_id.nunique() == len(test_df), 'duplicate patients!'


    if target_label is None: target_label = learn.target_label
    if cat_labels is None: cat_labels = learn.cat_labels
    if cont_labels is None: cont_labels = learn.cont_labels

    #CHANGED
    add_features = []
    if cat_labels:
        cat_enc = learn.dls.dataset._datasets[-2]._datasets[0].encode
        add_features.append((cat_enc, test_df[cat_labels].values))
    if cont_labels:
        cont_enc = learn.dls.dataset._datasets[-2]._datasets[1].encode
        add_features.append((cont_enc, test_df[cont_labels].values))
        
    if prediction_level == "slide":
        patient_ids = []
        bags = test_df.feature_files.values
        targets = test_df[target_label].values
        new_bags = []
        targs = []
        for i, bag_paths in enumerate(bags):
            for bag_path in bag_paths:
                new_bags.append([bag_path])
                targs.append(targets[i])
                patient_ids.append(test_df.patient_id.values[i])
        bags = np.array(new_bags)
        targs = np.array(targs)
        print(f"Number of test slides: {len(bags)}")
    elif prediction_level == "patient":
        patient_ids = test_df.patient_id.values
        targs = test_df[target_label].values
        bags = test_df.feature_files.values
        

    if sample_bag_size is not None:
        # only create one sample per bag for inference
        targs, bags, _,  sampled_indexes = get_clustered_samples(
            bags=bags, 
            targs=targs, 
            bag_size=sample_bag_size, 
            num_samples=1, 
            prediction_level=prediction_level)
        
        test_ds = make_clustered_dataset(
            bags=bags,
            targets=(targs, np.ones(targs.shape).reshape(-1,1)), #(target_enc, )
            sampled_idxs=sampled_indexes,
            bag_size=sample_bag_size)
    else:
    
    
        #CHANGED
        test_ds = make_whole_slide_dataset(
            bags=bags,
            #weights=weights.reshape(-1,1),
            targets=(targs, np.ones(targs.shape).reshape(-1,1)), #(target_enc, )
            add_features=add_features,
            bag_size=None)

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0) #shuffle=True #drop_last=True


    patient_preds, patient_targs = learn.get_preds(dl=test_dl)
    patient_targs = patient_targs[0]

    # make into DF w/ ground truth
    #CHANGED
    patient_preds_df = pd.DataFrame.from_dict({
        'patient_id': patient_ids,
        target_label: targs})

    patient_preds_df['loss'] = F.mse_loss(
        patient_preds.clone().detach().squeeze(), patient_targs.clone().detach().squeeze(),
        reduction='none')

    patient_preds_df['pred'] = patient_preds

    patient_preds_df = patient_preds_df[[
        'patient_id',
        target_label,
        'pred',
        'loss']]

    return patient_preds_df
