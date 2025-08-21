import os
import torch
import torch.nn as nn
from huggingface_hub import login, hf_hub_download
from conch.open_clip_custom import create_model_from_pretrained


class ConchEncodingModel(nn.Module):
    def __init__(self, model):
        super(ConchEncodingModel, self).__init__()
        self.model = model
    
    def forward(self, x):
        x = self.model.encode_image(x)
        return x


def get_conch_model() -> nn.Module:
    local_dir = "custom_WSI_pipeline/models/CONCH"
    os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    hf_hub_download("MahmoodLab/CONCH", filename="pytorch_model.bin", local_dir=local_dir)
    c_model = create_model_from_pretrained('conch_ViT-B-16', local_dir + "/pytorch_model.bin", return_transform=False) 
    model = ConchEncodingModel(c_model)
    model.cuda()
    return model 

if __name__ == "__main__":
    model = get_conch_model()
    print(model)
    print("CONCH model loaded successfully.")
