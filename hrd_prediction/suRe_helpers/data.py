from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Tuple, Sequence, Iterable, Optional, Any, Callable
from sklearn.cluster import KMeans
import h5py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

@dataclass
class ClusteredBagDataset(Dataset):
    """A dataset of bags of instances."""
    bags: Sequence[Iterable[Path]]
    """The `.h5` files containing the bags.

    Each bag consists of the features taken from one or multiple h5 files.
    Each of the h5 files needs to have a dataset called `feats` of shape N x
    F, where N is the number of instances and F the number of features per
    instance.
    """
    sample_idxs: Sequence[np.ndarray]
    """inidices of the the features to be used for each bag. 
    Can be used to select and order the features according to their clusters.
    """
    bag_size: Optional[int] = None
    """The number of instances in each bag.

    For bags containing more instances, a random sample of `bag_size`
    instances will be drawn.  Smaller bags are padded with zeros.  If
    `bag_size` is None, all the samples will be used.
    """

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # collect all the features
        feats = []
        image_clusters = []
        for bag_file in self.bags[index]:
            with h5py.File(bag_file, 'r') as f:
                feats.append(torch.from_numpy(f['feats'][:]))

        feats = torch.concat(feats).float()
        
        #select subset and order the features according to the clusters        
        feats = feats[self.sample_idxs[index].astype(int)]

        return feats, len(feats)
    
class MapDataset(Dataset):
    def __init__(
            self,
            func: Callable,
            *datasets: Sequence[Any],
            strict: bool = True
    ) -> None:
        """A dataset mapping over a function over other datasets.

        Args:
            func:  Function to apply to the underlying datasets.  Has to accept
                `len(dataset)` arguments.
            datasets:  The datasets to map over.
            strict:  Enforce the datasets to have the same length.  If
                false, then all datasets will be truncated to the shortest
                dataset's length.
        """
        #breakpoint()
        #ds[0] contains bags, ds[1] contains (targets,weights)
        if strict:
            #assert all(len(ds) == len(datasets[[0]]) for ds in datasets)
            self._len = len(datasets[0])
        elif datasets:
            self._len = min(len(ds) for ds in datasets)
        else:
            self._len = 0

        self._datasets = datasets
        self.func = func

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Any:
        #breakpoint()

        items = [(ds[index] if len(ds)!=2 else (ds[0][index], ds[1][index])) for ds in self._datasets]
        return items #self.func(*[ds[index] for ds in self._datasets])

    def new_empty(self):
        #FIXME hack to appease fastai's export
        return self



def get_clustered_samples(
    bags: np.ndarray, 
    targs: np.ndarray, 
    bag_size: int,
    num_samples: int,
    prediction_level: str,
    valid_idxs: Optional[np.ndarray] = None
    ) -> Tuple[list, np.ndarray, np.ndarray, list]:
    """Create a given amount of samples from the feature files, based on the tile feature clusters

    Args:
        bags (np.ndarray): Array containing the feature file paths
        targs (np.ndarray): Array of target values for each bag
        bag_size (int): max amount of tile features per sample 
        num_samples (int): amount of samples per bag
        prediction_level (str): Wether to predict HRD for each slide or per patient, based on all slides of the patient.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - new_bags: Array of sampled bag file paths
            - new_targs: Array of target values for each sampled bag
            - valid_idxs: validation indicies for each bag
            - sample_idxs: Array of indices used to select features for each sample
    """
    new_bags = []
    new_targs = []
    sample_idxs = []
    # clus = []
    new_valid_idxs = []
    # Iterate through bags (patients or images, depends on prediction_level)
    for bag, target, i in zip(bags, targs, range(len(bags))):
        bag_features = []
        bag_clusters = []
        perform_clustering = False
        # for slide level prediction, bag contains only one file 
        for file in bag:
            # collect all features of the bag and cluster labels
            h5_file = h5py.File(file, 'r')
            feats = torch.Tensor(np.array(h5_file['feats']))
            bag_features.append(feats)
            
            #collect clusters for each feature_vector 
            if prediction_level == "slide":
                bag_clusters.extend(np.array(h5_file["cluster_labels"]))
                # perform_clustering = True
            elif prediction_level == "patient" and "patient_cluster_labels" in h5_file.keys():
                bag_clusters.extend(np.array(h5_file["patient_cluster_labels"]))
                # perform_clustering = True
            elif prediction_level == "patient" and "patient_cluster_labels" not in h5_file.keys():
                # perform initial clustering in the next step if it has not been done yet
                perform_clustering = True
            else:
                raise ValueError("prediction_level must be either 'slide' or 'patient'")
            
            h5_file.close()
        bag_features = torch.concat(bag_features)
        
        # Cluster features on patient level and save the cluster labels in each h5 file.
        if perform_clustering:
            print("creating initial patient clusters")
            bag_clusters = kmeans_clustering(bag_features.numpy(), n_clusters=50)
            n = 0
            for i, file in enumerate(bag):
                h5_file = h5py.File(file, 'r+')
                file_length = h5_file["feats"].shape[0]
                if "patient_cluster_labels" in h5_file:
                    del h5_file["patient_cluster_labels"]
                if "patient_cluster_labels" not in h5_file.keys():
                    h5_file.create_dataset("patient_cluster_labels", data=bag_clusters[n : n+file_length])
                n += file_length
                h5_file.close()
        
        bag_clusters = np.array(bag_clusters)
        
        # sample clustersize weighted features
        unique_ids, counts = np.unique(bag_clusters, return_counts=True)
        
        # if less features than bag, just sort the indexes according to clusters
        if len(bag_features) <= bag_size:
            new_bags.append(bag)
            new_targs.append(target)
            if valid_idxs is not None:
                new_valid_idxs.append(valid_idxs[i])
            sample_idxs.append(np.argsort(bag_clusters))
            continue
                
        #create samples
        for _ in range(num_samples):    
            sampled_indices = []
            cluster_count = np.zeros_like(unique_ids)
            for cluster_id, count in zip(unique_ids, counts):
                if bag_size == 50:
                    # if bagsize = k from kMeans (50), sample one feature per cluster 
                    n_cluster_instances = 1
                else:
                    n_cluster_instances = round((count/len(bag_clusters)) * bag_size)
                    n_cluster_instances = max(1, n_cluster_instances)  # at least one instance per cluster
                indices = np.where(bag_clusters == cluster_id)[0]
                
                
                chosen = np.random.choice(indices, n_cluster_instances, replace=False)
                cluster_count[cluster_id] = n_cluster_instances

                sampled_indices.extend(chosen)
            sampled_indices = np.array(sampled_indices, dtype=int)
            
            # if len(sampled_indices) != bag_size:
            while len(sampled_indices) != bag_size:
                clus = bag_clusters[sampled_indices]
                             
                
                # remove features from the biggest cluster from the sample if the sample is too big due to rounding
                if len(sampled_indices) > bag_size:
                    biggest_cluster = unique_ids[np.argmax(cluster_count)]   
                    
                    # excess = len(sampled_indices) - bag_size
                    excess = 1
                    drop_idxs = np.random.choice(np.where(clus == biggest_cluster)[0], excess, replace=False)
                 
                    sampled_indices = np.delete(sampled_indices, drop_idxs)
                    clus = bag_clusters[sampled_indices]
                    cluster_count[biggest_cluster] -= excess
                
                # add features from the biggest cluster to the sample if the sample is too small due to rounding
                if len(sampled_indices) < bag_size:
                    biggest_cluster = unique_ids[np.argmax(counts)]
                    needed = bag_size - len(sampled_indices)
                    
                    index_pool = np.where(bag_clusters == biggest_cluster)[0]
                    add_idxs = np.random.choice(index_pool, needed, replace=False)
                    sampled_indices = np.concatenate([sampled_indices, add_idxs])
            
            sampled_indices = sampled_indices[np.argsort(bag_clusters[sampled_indices])]
        
                    
            new_bags.append(bag)
            new_targs.append(target)
            sample_idxs.append(np.array(sampled_indices))
            if valid_idxs is not None:
                new_valid_idxs.append(valid_idxs[i])
                
                
    new_bags = np.array(new_bags, dtype=object)
    new_targs = np.array(new_targs)
    new_valid_idxs = np.array(new_valid_idxs)
    sample_idxs = np.array(sample_idxs, dtype=object)
    # print("LENGE: " , len(sample_idxs[0]))
        
    return new_bags, new_targs, new_valid_idxs, sample_idxs  #, np.array(clus)


def get_random_samples(
    bags: np.ndarray, 
    targs: np.ndarray, 
    bag_size: int,
    num_samples: int,
    prediction_level: str,
    valid_idxs: Optional[np.ndarray] = None
    ) -> Tuple[list, np.ndarray, np.ndarray, list]:
    """
    Create a given amount of samples from the feature files by randomly sampling
    tile features (no clustering).
    """

    new_bags = []
    new_targs = []
    sample_idxs = []
    new_valid_idxs = []

    for bag, target, i in zip(bags, targs, range(len(bags))):
        bag_features = []

        # Load all slide-level features into one array
        for file in bag:
            with h5py.File(file, 'r') as h5_file:
                feats = torch.Tensor(np.array(h5_file['feats']))
                bag_features.append(feats)

        bag_features = torch.concat(bag_features)
        num_features = len(bag_features)

        # If fewer features than bag_size → use all features
        if num_features <= bag_size:
            new_bags.append(bag)
            new_targs.append(target)
            sample_idxs.append(np.arange(num_features))
            if valid_idxs is not None:
                new_valid_idxs.append(valid_idxs[i])
            continue

        # Create samples
        for _ in range(num_samples):

            # Randomly sample features without replacement
            sampled_indices = np.random.choice(
                np.arange(num_features),
                size=bag_size,
                replace=False
            )

            new_bags.append(bag)
            new_targs.append(target)
            sample_idxs.append(sampled_indices)

            if valid_idxs is not None:
                new_valid_idxs.append(valid_idxs[i])

    new_bags = np.array(new_bags, dtype=object)
    new_targs = np.array(new_targs)
    new_valid_idxs = np.array(new_valid_idxs)
    sample_idxs = np.array(sample_idxs, dtype=object)

    return new_bags, new_targs, new_valid_idxs, sample_idxs

def kmeans_clustering(features, n_clusters=50) -> np.ndarray:
    """
    Perform KMeans clustering on the provided features.

    Parameters:
    path (str): The path to the data file.
    n_clusters (int): The number of clusters to form.
    """
            
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(features)
    labels = kmeans.labels_
    
    return labels


def make_clustered_dataset(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[Sequence[Any], Sequence[Any]],
    bag_size: Optional[int] = None,
    sampled_idxs: Sequence[np.ndarray],
    ) -> MapDataset:
    
    ds = MapDataset(
        zip_bag_targ,
        ClusteredBagDataset(bags, bag_size=bag_size, sample_idxs=sampled_idxs),
        targets
    )


    return ds

def zip_bag_targ(bag, targets):

    features, lengths = bag

    return (
        features,
        lengths,
        targets
    )
    

def arrange_bags_for_upsamling(
    bags: np.ndarray, 
    targs:np.ndarray, 
    valid_idxs:np.ndarray, 
    n_bins: int, 
    alpha: int = 0.1, 
    beta: int = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function arranges the bags, targets and validation_indexes to allow cluster based upsampling .

    Args:
        bags (np.ndarray): _description_
        targs (np.ndarray): _description_
        valid_idxs (np.ndarray): _description_
        n_bins (int): Number of bins to use for the upsampling
        alpha (int): Weighting factor to compute the budget cap
        beta (int): Weighting factor to coompute the budget for each bin

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Bags, Targets and Valid Indexes arranged accordingly for umpsampling. 
    """
    print(f"Amount of bags before upsampling: {len(bags)}")
    new_bags = []
    new_targs = []
    new_valid_idxs = []
    hist, edges = np.histogram(targs, bins=n_bins, density=False)
    
    # print(f"Bin distribution: {hist}")
    max_val = max(hist)
    budgets = np.full(hist.shape, max_val, dtype=int) - hist
    budgets = np.where(budgets == max_val, 0, budgets)
    
    # Cap the budget to alpha of max_val
    budget_cap = int(max_val * alpha)
    budgets = np.minimum(budgets * beta, budget_cap).astype(int)
    budgets 
    
    
    for bin, budget in enumerate(budgets):
        if budget == max_val:
            continue 
        if bin < len(budgets) -1:
            bag_pool = np.where((edges[bin] <= targs) & (targs < edges[bin+1]))[0]
        else:
            bag_pool = np.where((edges[bin] <= targs) & (targs <= edges[bin+1]))[0]
            
        upsampled_indices = np.random.choice(bag_pool, budget, replace=True)
         
        new_bags.append(bags[upsampled_indices])
        new_targs.extend(targs[upsampled_indices])
        new_valid_idxs.extend(valid_idxs[upsampled_indices])

    new_bags = np.concatenate(new_bags)
    print(f"Amount of upsampled bags: {len(new_bags)}")
    

    
    
    return new_bags, np.vstack(new_targs), np.array(new_valid_idxs)



if __name__ == "__main__":
    
    # bags = np.array([['/data/datasets/images/CPTAC/PDA/features/RetCCL/C3L-00017-21.h5',
    #                  '/data/datasets/images/CPTAC/PDA/features/RetCCL/C3L-00017-22.h5',
    #                  '/data/datasets/images/CPTAC/PDA/features/RetCCL/C3L-00017-23.h5',
    #                  '/data/datasets/images/CPTAC/PDA/features/RetCCL/C3L-00017-24.h5',
    #                  '/data/datasets/images/CPTAC/PDA/features/RetCCL/C3L-00017-25.h5']])
    
    bags = np.array([['1'], ['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9'],], dtype=object)
    valid_idxs = np.array([False,True,False,False,False,False,False,False,False])
    targs = np.array([42, 42, 42, 1 ,2, 3,4,4,10])
    prediction_level = "patient"
    bag_size = 600
    num_samples = 3
    
    # new_bags, new_targs, sample_idxs, c = get_clustered_samples(bags, targs, bag_size, num_samples, prediction_level)
    # print(new_bags)
    # print(new_targs)
    # print(sample_idxs)
    # print(c)
    bags, targs, valid_idxs = arrange_bags_for_upsamling(bags, targs, valid_idxs=valid_idxs, n_bins=10)       
    print(type(bags))
    print(targs)
    print(valid_idxs)
    
    pass    
    
    