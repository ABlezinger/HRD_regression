import torch
from .models.aggregators.transformer_gpt_random_attn_topk import Transformer_gpt_random_attn_topk
from .models.aggregators.transformer_gpt_random_4_quantile import Transformer_gpt_random_4_quantile
import argparse


def get_suRe_emodel(model_type, input_dim, output_dim, model_config=None):
    
    
    # config = yaml.safe_load(open("hrd_prediction/train_config.yaml", "r"))
    
    # config = {"depth": 6,
    #           "heads": 6,
    #           "dim_head": 64,
    #           "dim": 384,
    #           "mlp_dim": 384,
    #           "dropout": 0
    # }
    
    # if True:
    if model_type == "random_attn_topk":
        model = Transformer_gpt_random_attn_topk(
            num_classes=output_dim,
            input_dim=input_dim,
            depth=model_config.get("transformer_layers", 6) if model_config else 6,
            heads=model_config.get("heads", 6) if model_config else 6,
            dim_head=model_config.get("dim_head", 64) if model_config else 64,
            dim=model_config.get("dim", 384) if model_config else 384,
            mlp_dim=model_config.get("mlp_dim", 384) if model_config else 384,
            dropout=model_config.get("dropout", 0) if model_config else 0
        )
    elif model_type == "random_4_quantile":
        model = Transformer_gpt_random_4_quantile(
            num_classes=output_dim,
            input_dim=input_dim,
            depth=model_config.get("transformer_layers", 6) if model_config else 6,
            heads=model_config.get("heads", 6) if model_config else 6,
            dim_head=model_config.get("dim_head", 64) if model_config else 64,
            dim=model_config.get("dim", 384) if model_config else 384,
            mlp_dim=model_config.get("mlp_dim", 384) if model_config else 384,
            dropout=model_config.get("dropout", 0) if model_config else 0
        )
        

    # print(model(torch.rand(1, 200, 2048)))
   
    return model      
    # print(model)
    # print("Success")

    
    
if __name__ == "__main__":    
    # parser = argparse.ArgumentParser()
    
    # parser.add_argument("--sure_type", type=str, default="random_4_quantile", choices=["random_attn_tok", "random_4_quantile"])
    # args = parser.parse_args()
    # get_suRe_emodel(args, 2048)
    pass
    