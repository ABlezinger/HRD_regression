import os
import h5py
import argparse
from pathlib import Path
from PIL import Image
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None  # Disable PIL's image size limit to avoid errors with large images


def loadDataSet(h5_path):
    h5_file = h5py.File(h5_path)
    if 'feats' in h5_file.keys():   
        features = torch.Tensor(np.array(h5_file['feats']))
        coords = torch.Tensor(np.array(h5_file['coords']))
        cluster_lables = torch.Tensor(np.array(h5_file['cluster_labels'], dtype=int)).int()
        num_features = features.shape[0]
        feat_indices = np.arange(num_features, dtype=int)
        return feat_indices, coords, features, cluster_lables
    else:
        raise ValueError("Required dataset 'feats' not found in the file")

def get_colors(num_colors):
    cmap = plt.get_cmap('tab20',num_colors)
    colors = [cmap(i) for i in range(num_colors)]
    return colors
    
def showClass(ax, coordinates, labels, color_list):
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    for i, coord in enumerate(coordinates):
        x, y = int(coord[0].item()), int(coord[1].item())  
        x_min, x_max = min(x, x_min), max(x, x_max)
        y_min, y_max = min(y, y_min), max(y, y_max)
        ax.add_patch(plt.Rectangle((x, y), 256, 256, linewidth=1, edgecolor=color_list[labels[i]], facecolor=color_list[labels[i]]))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_min - 5000, x_max + 5000)
    ax.set_ylim(y_min - 5000, y_max + 5000)
    # ax.invert_yaxis()
    ax.axis('off')

def main(args):
    plot_path = Path(args.save_path)
    plot_path.mkdir(exist_ok=True, parents=True)

    png_image = Image.open(args.dataset_path + f'/PNG/{args.sample_id}/slide.jpg')
    colors = get_colors(50)
    model_dirs = glob.glob(args.dataset_path + "/features/*")
    print(f"Models: {model_dirs}")
    n_models = len(model_dirs)
    
    # Get image size for axis limits
    img_width, img_height = png_image.size
    xlim = (0, img_width)
    ylim = (0, img_height)

    fig, axes = plt.subplots(2, round((n_models + 1)/2), figsize=(3 * (n_models + 1), 10))
    
    axes = axes.flatten()
    # Plot original image
    axes[0].imshow(png_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for idx, model_dir in enumerate(model_dirs):
        model = Path(model_dir).name
        ids, coords, feats, cluster_lables = loadDataSet(os.path.join(args.dataset_path, 'features', model, f'{args.sample_id}.h5'))
        showClass(axes[idx + 1], coords, cluster_lables, colors)
        axes[idx + 1].set_title(model)
        axes[idx + 1].set_ylim(ylim)
        axes[idx + 1].set_xlim(xlim)
        axes[idx + 1].invert_yaxis()

    used = n_models + 1     # original image + number of models
    for ax in axes[used:]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    path = os.path.join(plot_path, f"{args.sample_id}_clusters.jpg")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize clustering results')
    parser.add_argument('--sample_id', type=str, default="C3N-01011-22", help='Name of the sample to visualize')
    parser.add_argument('--save_path', type=str, default='./visualization/figures', help='Path to save the visualization results')
    parser.add_argument('--dataset_path', type=str, default='/data/datasets/images/CPTAC/PDA', help='Dataset the case belongs to')

    args = parser.parse_args()
    main(args)


    
    
    