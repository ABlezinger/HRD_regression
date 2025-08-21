import hashlib
import torch
import torch.nn as nn

from .ResNet import resnet50

def get_RetCCL_model() -> nn.Module:
    sha256 = hashlib.sha256()
    with open("custom_WSI_pipeline/models/RetCCL/RetCCL.pth", 'rb') as f:
        while True:
            data = f.read(1 << 16)
            if not data:
                break
            sha256.update(data)

    assert sha256.hexdigest() == '931956f31d3f1a3f6047f3172b9e59ee3460d29f7c0c2bb219cbc8e9207795ff'

    model = resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
    #put the model on the CPU for HPC
    pretext_model = torch.load("custom_WSI_pipeline/models/RetCCL/RetCCL.pth", 
                               map_location=torch.device('cpu'), weights_only=True)
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
        
    return model