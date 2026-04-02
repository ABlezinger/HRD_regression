def get_MIL_config(args):
    """Get model configuration based on provided arguments.

    Args:
        args: Parsed command-line arguments containing model configuration options.

    Returns:
        A dictionary containing model configuration parameters.
    """
    model_config = {
        'MIL_model': args.MIL_model if hasattr(args, "MIL_model") else "marugoto",
    }
    
    if model_config["MIL_model"] == "marugoto":
        model_config["encoding_dim"] = args.encoding_dim if hasattr(args, "encoding_dim") else 256
        model_config["attention_layers"] = args.attention_layers if hasattr(args, "attention_layers") else 1
        model_config["head_depth"] = args.head_depth if hasattr(args, "head_depth") else 1
        model_config["dropout"] = args.dropout if hasattr(args, "dropout") else 0.0
    
    
    elif model_config["MIL_model"] == "random_attn_topk":
        model_config["transformer_layers"] = args.transformer_layers if hasattr(args, "transformer_layers") else 6
        model_config["heads"] = args.heads if hasattr(args, "heads") else 6
        model_config["dim_head"] = args.dim_head if hasattr(args, "dim_head") else 64
        model_config["dim"] = args.dim if hasattr(args, "dim") else 384
        model_config["mlp_dim"] = args.mlp_dim if hasattr(args, "mlp_dim") else 384
        model_config["dropout"] = args.dropout if hasattr(args, "dropout") else 0.0
    
    
    return model_config

def get_training_config(args):
    """Get training configuration based on provided arguments.

    Args:
        args: Parsed command-line arguments containing training configuration options.

    Returns:
        A dictionary containing training configuration parameters.
    """
    training_config = {
        'epochs': args.epochs if hasattr(args, "epochs") else 20,
        'batch_size': args.batch_size if hasattr(args, "batch_size") else 1,
        'learning_rate': args.learning_rate if hasattr(args, "learning_rate") else .0001,
    }
    return training_config

def get_setup_config(args):
    """Get setup configuration based on provided arguments.

    Args:
        args: Parsed command-line arguments containing setup configuration options.

    Returns:
        A dictionary containing setup configuration parameters.
    """
    setup_config = {
        'extraction_model': args.extraction_model,
        'prediction_level': args.prediction_level,
        'cohort': args.cohort,
        'target_label': args.target_label,
        'binary_label': None,
        # 'dataset_path': args.dataset_path,
        # 'output_path': args.output_path,
        # 'log_interval': args.log_interval,
        # 'save_model': args.save_model,
        'bag_size': args.sample_bag_size if hasattr(args, "sample_bag_size") else 1000,
        'n_splits': args.n_splits if hasattr(args, "n_splits") else 5,
        'sampling_strategy': args.sampling_strategy if hasattr(args, "sampling_strategy") else "clustered_random",
        'use_cluster_based_upsampling': args.use_cluster_based_upsampling if hasattr(args, "use_cluster_based_upsampling") else False,
        'upsampling_bins': args.upsampling_bins if hasattr(args, "upsampling_bins") else None,
        'sample_amount': args.sample_amount if hasattr(args, "sample_amount") else None,
        'upsampling_alpha': args.alpha if hasattr(args, "alpha") else None,
        'upsampling_beta': args.beta if hasattr(args, "beta") else None, 
        'upsampling_bins': args.upsampling_bins if hasattr(args, "upsampling_bins") else None,
        'test_mode': args.test_mode if hasattr(args, "test_mode") else False,
    }
    return setup_config
