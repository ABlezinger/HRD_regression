import os
import torch
from torchvision import transforms
import timm
import json
from huggingface_hub import login, hf_hub_download

def get_uni_model() -> torch.nn.Module:
    
    token = json.load(open('huggingface_config.json'))['token']
    login(token=token)  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    
    local_dir = "custom_WSI_pipeline/models/UNI"
    os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=False)
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True), strict=True)
    
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model

if __name__ == "__main__":
    model = get_uni_model()
    print("UNI model loaded successfully.")
    # You can now use the model for inference or further processing.