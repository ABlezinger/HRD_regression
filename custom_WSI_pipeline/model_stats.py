import json
import PIL
from PIL import Image
import numpy as np
from pathlib import Path
import torch
import torchprofile
import time

from tqdm import tqdm

from models.GPFM.gpfm_model import get_gpfm_model
from models.UNI.uni_model import get_uni_model
from models.UNI_2.uni_2_model import get_uni2_model
from models.RetCCL.RetCCL_model import get_RetCCL_model
from models.CONCH.conch_model import get_conch_model

Image.MAX_IMAGE_PIXELS = None  # Disable PIL's image size limit to avoid errors with large images


def load_one_image_tile() -> np.ndarray:
    slide_jpg_dir = Path('test_JPGs/TCGA-G4-6298-01Z-00-DX1.83055d52-71f7-46ec-be53-11d86b19b4cf/')
    slide_png = 'test_JPGs/TCGA-G4-6298-01Z-00-DX1.83055d52-71f7-46ec-be53-11d86b19b4cf/norm_slide.png'
    print("WSI has been normalized and converted into jpg, now extracting features...")
    # If the normalised slide png already exists, load it
    img_norm_wsi_jpg = PIL.Image.open(slide_png)
    image_array = np.array(img_norm_wsi_jpg)
    canny_patch_list = []
    coords_list=[]
    patch_saved=0
    coords_path = slide_jpg_dir / "coords.npy"
    if coords_path.exists():
        coords_list = np.load(coords_path, allow_pickle=True)
        canny_patch_list = [image_array[y:y+224, x:x+224, :] for (x, y) in coords_list]
        patch_saved = len(canny_patch_list)
    else:
        print("no coords")
    tile = canny_patch_list[0]

    return tile
    
    
model_list = {
    "RetCCL":get_RetCCL_model, 
    "GPFM": get_gpfm_model,
    "UNI": get_uni_model,
    "UNI-2": get_uni2_model,
    "CONCH": get_conch_model,}

image = load_one_image_tile()
print("loaded image tile")

stats = dict()

for name, get_model in tqdm(model_list.items(), total=len(model_list), desc=f"Processingmodels "):
    model = get_model()
    start = time.time()
    input = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
    print(input.shape)
    process_time = time.time() - start

    out = model(input)
    
    macs = torchprofile.profile_macs(model, input)
    params = sum(p.numel() for p in model.parameters())
    macs
    
    stats[name] = {
        "macs": macs,
        "params": params,
        "Inference time per image": process_time,
        "embed_dim": out.shape
    }
    
# Save the stats to a file
stats_path = Path("custom_WSI_pipeline/model_stats.json")
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=4)
    