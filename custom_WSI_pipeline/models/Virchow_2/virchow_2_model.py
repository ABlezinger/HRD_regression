import os
import torch
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
import json
from huggingface_hub import login, hf_hub_download

def get_virchow_2_model() -> torch.nn.Module:
    
    token = json.load(open('huggingface_config.json'))['token']
    login(token=token)  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    
    local_dir = "custom_WSI_pipeline/models/Virchow_2"

    hf_hub_download("paige-ai/Virchow2", filename="pytorch_model.bin", local_dir=local_dir, force_download=False)
    
    model = timm.create_model("hf-hub:paige-ai/Virchow2", mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, pretrained = False)
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True), strict=True)
    model = model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
            
    return model

if __name__ == "__main__":
    model = get_virchow_2_model()
    print(model)
    print("Virchow_2 model loaded successfully.")
    # You can now use the model for inference or further processing.