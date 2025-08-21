import h5py
from sklearn.cluster import KMeans
import torch
import numpy as np

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
    

if __name__ == "__main__":
    import argparse
    import h5py
    
    # parser = argparse.ArgumentParser(description="KMeans Clustering Script")
    # parser.add_argument("path", type=str, help="Path to the data file")
    # parser.add_argument("--n_clusters", type=int, default=50, help="Number of clusters to form")
    
    # args = parser.parse_args()
    
    # h5_file = h5py.File(args.path, 'r')
    h5_file = h5py.File("/home/alexander.blezinger/HRD_regression/test_output/UNI/TCGA-G4-6298-01Z-00-DX1.83055d52-71f7-46ec-be53-11d86b19b4cf.h5", 'r')
    features = torch.Tensor(np.array(h5_file['feats']))
    coords = h5_file['coords']
    
    kmeans_clustering(features=features, n_clusters=50)