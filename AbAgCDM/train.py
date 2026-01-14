"""
Main script for training AbAgCDM model.

Usage:
    python train.py --config config.json 
"""

import argparse
import json
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime

try:
    from .dataset import create_dataloaders
    from .model import create_model
    from .training import train, test 
except ImportError:
    from dataset import create_dataloaders
    from model import create_model
    from training import train, test 

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train multi-task AbAgCDM binding model'
    )
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON configuration file')
    
    # Parse only --config
    args = parser.parse_args()
    
    # Load all config and return as Namespace
    with open(args.config, "r") as f:
        config = json.load(f)
    
    return argparse.Namespace(**config) 


# ============================================================================
# SEED SETTING
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


# ============================================================================
# SAVE CONFIGURATION
# ============================================================================

def save_config(args, save_dir: Path, train: bool = True):
    save_dir.mkdir(parents=True, exist_ok=True)
    
    config_dict = dict(vars(args))
    if not train:
        config_dict["test_only"] = True

    config_path = save_dir / "config_trained_model.json"

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    print(f"Configuration saved to {config_path}") 


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main training/testing pipeline."""
    
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # Create save directory  
    save_dir = Path(args.save_dir) 
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(args, save_dir)
    
    print(f"\n{'='*80}")
    print("Multi-Task AbAgCDM Binding Model")
    print(f"{'='*80}")
    print(f"Save directory: {save_dir}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_pq_file=args.train_pq_file,
        val_pq_file=args.val_pq_file,
        test_pq_file=args.test_pq_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_negatives=args.max_negatives,
        num_workers=args.num_workers,
        shuffle_train=True,
        load_frac_data=args.load_frac_data,
        random_negatives=args.random_negatives  
    )
    
    print(f"Data loaded:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    print("\nCreating model...")
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen: {args.freeze_encoder}")
    
    # ========================================================================
    # RESUME FROM CHECKPOINT (if specified)
    # ========================================================================
    # last_epoch = 5 
    # print(f"Loading model weight after Epoch {last_epoch}") 
    # checkpoint = torch.load("./checkpoints_e05/best_model.pt", map_location=device)
    # model.load_state_dict(checkpoint) 
    # print(f"Resumed from checkpoint {checkpoint}")
    
    # ========================================================================
    # TEST ONLY MODE
    # ========================================================================
    if args.test_only:
        print("\n" + "="*80)
        print("Test Only Mode")
        print("="*80)
        
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for test-only mode")
        
        test_results = test(
            model=model,
            test_loader=test_loader,
            device=device,
            checkpoint_path=args.checkpoint
        )
        
        # Save test results
        results_path = save_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to {results_path}")
        
        return
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    # Convert scheduler type
    scheduler_type = args.scheduler_type if args.scheduler_type != 'none' else None
    
    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        save_dir=save_dir,
        gradient_clip=args.gradient_clip,
        scheduler_type=scheduler_type,
        log_interval=args.log_interval,
        early_stopping_patience=args.early_stopping_patience, 
        save_best_only=args.save_best_only
    )
    
    # ========================================================================
    # TESTING
    # ========================================================================
    print("\n" + "="*80)
    print("Running Final Test Evaluation")
    print("="*80)
    
    # Load best model for testing
    best_checkpoint_path = save_dir / 'best_model.pt'
    
    test_results = test(
        model=model,
        test_loader=test_loader,
        device=device,
        checkpoint_path=str(best_checkpoint_path)
    )
    
    # Save test results
    results_path = save_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"Test results saved to {results_path}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Save directory: {save_dir}")
    print(f"Best model: {best_checkpoint_path}")
    print(f"Training history: {save_dir / 'training_history.json'}")
    print(f"Test results: {results_path}")
    print("="*80 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
    
    
    