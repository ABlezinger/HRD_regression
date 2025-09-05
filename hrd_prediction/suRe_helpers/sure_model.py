import torch
from models.aggregators.transformer_gpt_random_attn_topk import Transformer_gpt_random_attn_topk
from models.aggregators.transformer_gpt_random_4_quantile import Transformer_gpt_random_4_quantile
import argparse


def get_model(args, input_dim):
    
    
    # config = yaml.safe_load(open("hrd_prediction/train_config.yaml", "r"))
    
    # config = {"depth": 6,
    #           "heads": 6,
    #           "dim_head": 64,
    #           "dim": 384,
    #           "mlp_dim": 384,
    #           "dropout": 0
    # }
    
    # if True:
    if args.sure_type == "random_attn_tok":
        model = Transformer_gpt_random_attn_topk(
            num_classes=1,
            input_dim=input_dim,
            depth=6,
            heads=6,
            dim_head=64,
            dim=384,
            mlp_dim=384,
            dropout=0
        )
    elif args.sure_type == "random_4_quantile":
        model = Transformer_gpt_random_4_quantile(
            num_classes=1,
            input_dim=input_dim,
            depth=6,
            heads=6,
            dim_head=64,
            dim=384,
            mlp_dim=384,
            dropout=0
        )
        

    print(model(torch.rand(1, 200, 2048)))
   
    return model      
    # print(model)
    # print("Success")

    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sure_type", type=str, default="random_4_quantile", choices=["random_attn_tok", "random_4_quantile"])
    args = parser.parse_args()
    get_model(args, 2048)
    