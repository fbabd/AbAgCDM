from .tokenizer_esm2 import ESM2Tokenizer 
from .encoder_esm2 import ESM2Encoder 
from .dataset import create_dataloaders, create_single_dataloader 
from .model import create_model
from .training import train, test, load_checkpoint 

import torch
import argparse
import json 
import os

def get_AbAgCDM( model_directory: str = 'AbAgCDM',
              checkpoint_folder: str = 'checkpoints',
              config_filepath: str = None, 
              verbose: bool = False):
    # LOAD ARGUMENTS from config file
    if config_filepath is None:
        config_filepath = os.path.join(model_directory, checkpoint_folder, 'config_trained_model.json')
    with open(config_filepath, "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)  
       
    if args.device == 'cuda' and not torch.cuda.is_available():
        if verbose: print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device) 
    
    # CREATE MODEL & LOAD CHECKPOINT WEIGHTS 
    model = create_model(
        encoder_name=args.encoder_name,
        freeze_encoder=args.freeze_encoder,
        contrastive_dim=args.contrastive_dim,
        dropout=args.dropout,
        temperature=args.temperature,
        lambda_contrastive=args.lambda_contrastive,
        pooling_method=args.pooling_method,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        pos_weight=args.pos_weight
    )
    model.to(device) 
    
    # COUNT PARAMETERS 
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        print(f"\nModel created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen: {args.freeze_encoder}")
    
    # LOAD MODEL WEIGHTS from checkpoint 
    model_weights_path = os.path.join(model_directory, checkpoint_folder, 'best_model.pt') 
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint) 
    
    if verbose: print(f"\nModel weights loaded from checkpoint.")
    
    return {
        "model": model ,
        "weight_path": model_weights_path, 
        "config_path": config_filepath,
        "device": device
    }