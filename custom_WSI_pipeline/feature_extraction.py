from pathlib import Path
import hashlib
import re
import json
from typing import Optional
from pathlib import Path

import PIL
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import h5py

from models.RetCCL.RetCCL_model import get_RetCCL_model
from models.GPFM.gpfm_model import get_gpfm
from models.UNI.uni_model import get_uni_model
from models.UNI_2.uni_2_model import get_uni2_model



class SlideTileDataset(Dataset):
    def __init__(self, patches: np.array, transform=None, *, repetitions: int = 1) -> None:
        self.tiles = patches
        #assert self.tiles, f'no tiles found in {slide_dir}'
        self.tiles *= repetitions
        self.transform = transform

    # patchify returns a NumPy array with shape (n_rows, n_cols, 1, H, W, N), if image is N-channels.
    # H W N is Height Width N-channels of the extracted patch
    # n_rows is the number of patches for each column and n_cols is the number of patches for each row
    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        image = PIL.Image.fromarray(self.tiles[i])
        if self.transform:
            image = self.transform(image)

        return image


def _get_coords(filename) -> Optional[np.ndarray]:
    if matches := re.match(r'.*\((\d+),(\d+)\)\.jpg', str(filename)):
        coords = tuple(map(int, matches.groups()))
        assert len(coords) == 2, 'Error extracting coordinates'
        return np.array(coords)
    else:
        return None

def get_mask_from_thumb(thumb, threshold: int) -> np.ndarray:
    thumb = thumb.convert('L')
    return np.array(thumb) < threshold

#TODO: replace slide_tile_paths with the actual tiles which are in memory
def extract_features_(
        *,
        model, model_name, norm_wsi_img: np.ndarray, coords: list, wsi_name: str, outdir: Path, 
) -> None:
    """Extracts features from slide tiles.

    Args:
        slide_tile_paths:  A list of paths containing the slide tiles, one
            per slide.
        outdir:  Path to save the features to.
        augmented_repetitions:  How many additional iterations over the
            dataset with augmentation should be performed.  0 means that
            only one, non-augmentation iteration will be done.
    """

    # ImageNet Normalization
    normal_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    # patchpatch = Path(f'{outdir}/patches')
    # patchpatch.mkdir(exist_ok=True,parents=True)

    extractor_string = f'marugoto-extract_{model_name}'
    with open(outdir/'info.json', 'w') as f:
        json.dump({
            'extractor': extractor_string,
                  }, f)

    
    ds = SlideTileDataset(norm_wsi_img, normal_transform)
    
    #clean up memory
    del norm_wsi_img

    dl = torch.utils.data.DataLoader(
        ds, batch_size=64, shuffle=False, num_workers=12, drop_last=False)

    model = model.eval()

    feats = []
    for batch in tqdm(dl, leave=False):
        feats.append(
                model(batch.type_as(next(model.parameters()))).half().cpu().detach())

    with h5py.File(f'{outdir}/{wsi_name}.h5', 'w') as f:
        f['coords'] = coords
        f['feats'] = torch.concat(feats).cpu().numpy()
        f.attrs['extractor'] = extractor_string
        
    print("saved features to", f'{outdir}/{wsi_name}.h5')

def extract_and_save_features(norm_wsi_img: np.ndarray, wsi_name: str, coords: list, extraction_model: str, outdir: Path, **kwargs):
    """Extracts features from slide tiles.
    Args:
        checkpoint_path:  Path to the model checkpoint file.  Can be downloaded
            from <https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL>.
    """
    
    # create model 
    
    if extraction_model == "UNI":
        model = get_uni_model()
    elif extraction_model == "UNI_2":
        model = get_uni2_model()
    elif extraction_model == "RetCCL":
        model = get_RetCCL_model()
    elif extraction_model == "GPFM":
        model = get_gpfm()
    else:
        raise ValueError(f"Unknown model: {extraction_model}")
    
    return extract_features_(norm_wsi_img=norm_wsi_img, wsi_name=wsi_name, coords=coords, model=model, outdir=outdir, model_name=extraction_model, **kwargs)
