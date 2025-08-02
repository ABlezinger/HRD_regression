import os
import torch
import json
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download


def get_uni2_model() -> torch.nn.Module:
    
    token = json.load(open('huggingface_config.json'))['token']
    login(token=token)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

    local_dir = "custom_WSI_pipeline/models/UNI_2"
    os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=False)
    timm_kwargs = {
                'model_name': 'vit_giant_patch14_224',
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )
    model.load_state_dict(torch.load(
        os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True), strict=True)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model

if __name__ == "__main__":
    model = get_uni2_model()
    print("UNI-2 model loaded successfully.")
    # You can now use the model for inference or further processing.