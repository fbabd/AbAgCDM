"""
Training script for multi-task AbAgCDM model.

Functions:
- train_epoch: Train for one epoch
- validate_epoch: Validate on validation set
- train: Main training loop with early stopping and  
- test: Test model performance on test data 
- load_checkpoint: Utility function to load saved weights 
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Optional, Tuple, List
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time
import os
from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, average_precision_score,
        confusion_matrix
    )


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_binding_metrics(
    binding_probs: np.ndarray,
    binding_labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute metrics with focus on positive class."""
    
    preds = (binding_probs >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(binding_labels, preds).ravel()
    
    # Specificity (important for imbalanced data)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Balanced accuracy
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    
    metrics = {
        'accuracy': accuracy_score(binding_labels, preds),
        'balanced_accuracy': balanced_acc,
        'precision': precision_score(binding_labels, preds, zero_division=0),
        'recall': recall_score(binding_labels, preds, zero_division=0),
        'specificity': specificity,
        'f1': f1_score(binding_labels, preds, zero_division=0),
        'auroc': roc_auc_score(binding_labels, binding_probs),
        'auprc': average_precision_score(binding_labels, binding_probs),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }
    
    return metrics


# ============================================================================
# TRAINING EPOCH
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_clip: Optional[float] = 1.0,
    log_interval: int = 10
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: AbAgCDM 
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        gradient_clip: Max gradient norm (None to disable)
        log_interval: Steps between logging
    
    Returns:
        Dictionary of average losses
    """
    model.train()
    
    total_loss = 0.0
    total_binding_loss = 0.0
    total_contrastive_loss = 0.0
    num_batches = 0
    
    all_probs = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]") 
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_binding_loss += outputs['loss_binding']
        total_contrastive_loss += outputs['loss_contrastive']
        num_batches += 1
        
        # Collect predictions for metrics
        all_probs.extend(outputs['binding_probs'].detach().cpu().numpy())
        all_labels.extend(batch['binding_label'].cpu().numpy())
        
        # Update progress bar
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'bind': f'{total_binding_loss/num_batches:.4f}',
                'contr': f'{total_contrastive_loss/num_batches:.4f}',
            })
    
    # Compute average losses
    avg_losses = {
        'loss': total_loss / num_batches,
        'loss_binding': total_binding_loss / num_batches,
        'loss_contrastive': total_contrastive_loss / num_batches,
    }
    
    # Compute metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    metrics = compute_binding_metrics(all_probs, all_labels)
    
    # Combine losses and metrics
    results = {**avg_losses, **metrics}
    
    return results


# ============================================================================
# VALIDATION EPOCH
# ============================================================================

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: AbAgCDM 
        val_loader: Validation data loader
        device: Device to validate on
        epoch: Current epoch number
    
    Returns:
        Dictionary of average losses and metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_binding_loss = 0.0
    total_contrastive_loss = 0.0
    num_batches = 0
    
    all_probs = []
    all_labels = []
    
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch)
        
        # Accumulate losses
        total_loss += outputs['loss'].item()
        total_binding_loss += outputs['loss_binding']
        total_contrastive_loss += outputs['loss_contrastive']
        num_batches += 1
        
        # Collect predictions
        all_probs.extend(outputs['binding_probs'].cpu().numpy())
        all_labels.extend(batch['binding_label'].cpu().numpy())
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Compute average losses
    avg_losses = {
        'loss': total_loss / num_batches,
        'loss_binding': total_binding_loss / num_batches,
        'loss_contrastive': total_contrastive_loss / num_batches,
    }
    
    # Compute metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    metrics = compute_binding_metrics(all_probs, all_labels)
    
    # Combine losses and metrics
    results = {**avg_losses, **metrics}
    
    return results


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    save_dir: str = './checkpoints',
    gradient_clip: Optional[float] = 1.0,
    scheduler_type: str = 'plateau',  # 'plateau', 'cosine', or None
    log_interval: int = 10,
    early_stopping_patience: int = 10,
    save_best_only: bool = True
) -> Dict[str, List[float]]:
    """
    Main training loop with validation and checkpointing.
    
    Args:
        model: AbAgCDM 
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        device: Device to train on
        save_dir: Directory to save checkpoints
        gradient_clip: Gradient clipping threshold
        scheduler_type: Learning rate scheduler type
        log_interval: Steps between logging
        save_best_only: Only save checkpoint if validation improves
    
    Returns:
        Dictionary of training history
    """
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Setup learning rate scheduler
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
    else:
        scheduler = None
    
    # Training history
    history = {
        'train_loss': [],
        'train_binding_loss': [],
        'train_contrastive_loss': [],
        'train_accuracy': [],
        'train_f1': [],
        'train_auroc': [],
        'val_loss': [],
        'val_binding_loss': [],
        'val_contrastive_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auroc': [],
        'learning_rate': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_val_auroc = 0.0
    patience_counter = 0
    best_epoch = 0
    
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Total epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Gradient clip: {gradient_clip}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*80}\n")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_results = train_epoch(
            model, train_loader, optimizer, device, epoch,
            gradient_clip, log_interval
        )
        
        # Validate
        val_results = validate_epoch(
            model, val_loader, device, epoch
        )
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_results['loss'])
            else:
                scheduler.step()
        
        # Record history
        history['train_loss'].append(train_results['loss'])
        history['train_binding_loss'].append(train_results['loss_binding'])
        history['train_contrastive_loss'].append(train_results['loss_contrastive'])
        history['train_accuracy'].append(train_results['accuracy'])
        history['train_f1'].append(train_results['f1'])
        history['train_auroc'].append(train_results['auroc'])
        
        history['val_loss'].append(val_results['loss'])
        history['val_binding_loss'].append(val_results['loss_binding'])
        history['val_contrastive_loss'].append(val_results['loss_contrastive'])
        history['val_accuracy'].append(val_results['accuracy'])
        history['val_f1'].append(val_results['f1'])
        history['val_auroc'].append(val_results['auroc'])
        history['learning_rate'].append(current_lr)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch}/{num_epochs} Summary ({epoch_time:.2f}s):")
        print(f"  Train - Loss: {train_results['loss']:.4f} | "
              f"Acc: {train_results['accuracy']:.4f} | "
              f"F1: {train_results['f1']:.4f} | "
              f"AUROC: {train_results['auroc']:.4f}")
        print(f"  Val   - Loss: {val_results['loss']:.4f} | "
              f"Acc: {val_results['accuracy']:.4f} | "
              f"F1: {val_results['f1']:.4f} | "
              f"AUROC: {val_results['auroc']:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Check for improvement
        improved = val_results['loss'] < best_val_loss
        
        if improved:
            best_val_loss = val_results['loss']
            best_val_auroc = val_results['auroc']
            best_epoch = epoch
            patience_counter = 0
            
            # Get the actual model (unwrap DDP if needed)
            model_to_save = model.module if hasattr(model, 'module') else model
            # model_to_save = model_to_save.cpu() 
            
            # Save to temporary file first
            temp_path = save_dir / 'best_model.pt.tmp'
            checkpoint_path = save_dir / 'best_model.pt'
            
            # Get size info
            state_dict = model_to_save.state_dict()
            # print(f"State dict size: {len(state_dict)} parameters")
            print(f"Estimated model size: {sum(p.numel() * p.element_size() for p in state_dict.values()) / (1024**3):.2f} GB")

            # # Check filesystem
            # save_path = temp_path
            # print(f"Save directory: {save_dir}")
            # print(f"Directory writable: {os.access(save_dir, os.W_OK)}")
            # print(f"Filesystem type: {os.statvfs(save_dir).f_frsize}")
            
            # print(train_results)
            # print(val_results)
    
            try:
                torch.save(state_dict, temp_path, _use_new_zipfile_serialization=False)
                # Atomic rename after successful write
                os.replace(temp_path, checkpoint_path)
                print(f"  ✓ New best model saved to {checkpoint_path} (Val Loss: {best_val_loss:.4f})")
                
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")
                if temp_path.exists():
                    temp_path.unlink()
                raise
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
        
        # Save latest model
        if not save_best_only:      
            # Get the actual model (unwrap DDP if needed)
            model_to_save = model.module if hasattr(model, 'module') else model
            # Save to temporary file first
            temp_path = save_dir / f'last_epoch.pt'
            checkpoint_path = save_dir / f'last_epoch.pt'            

            try:
                torch.save(model_to_save.state_dict(), temp_path,  
                           _use_new_zipfile_serialization=False)
                # Atomic rename after successful write
                os.replace(temp_path, checkpoint_path)
                print(f"  ✓ Latest model saved to {checkpoint_path} (Val Loss: {best_val_loss:.4f})")
                
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")
                if temp_path.exists():
                    temp_path.unlink()
                raise
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n{'='*80}")
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")
            print(f"{'='*80}")
            break
        
        print()
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation AUROC: {best_val_auroc:.4f}")
    print(f"{'='*80}\n")
    
    # Save training history
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    return history


# ============================================================================
# TESTING
# ============================================================================

@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Test the model on test set.
    
    Args:
        model: AbAgCDM model 
        test_loader: Test data loader
        device: Device to test on
        checkpoint_path: Path to checkpoint to load (optional)
    
    Returns:
        Dictionary of test metrics
    """
    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    all_probs = []
    all_labels = []
    
    print("\nRunning test evaluation...")
    progress_bar = tqdm(test_loader, desc="Testing")
    
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch)
        
        # Accumulate
        total_loss += outputs['loss'].item()
        num_batches += 1
        
        all_probs.extend(outputs['binding_probs'].cpu().numpy())
        all_labels.extend(batch['binding_label'].cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / num_batches
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    metrics = compute_binding_metrics(all_probs, all_labels)
    
    results = {'loss': avg_loss, **metrics}
    
    # Print results
    print(f"\n{'='*80}")
    print("Test Results:")
    print(f"{'='*80}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"AUROC: {results['auroc']:.4f}")
    print(f"AUPRC: {results['auprc']:.4f}")
    print(f"{'='*80}\n")
    
    return results


# ============================================================================
# UTILITY: LOAD CHECKPOINT
# ============================================================================

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Load model checkpoint.
    Args:
        model: AbAgCDM 
        checkpoint_path: Path to checkpoint file
        device: Device to load onto
    
    Returns:
        Tuple of (model, optimizer, checkpoint_dict)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint) 
    model = model.to(device) 
    
    print(f"Checkpoint loaded")
    
    return model 


