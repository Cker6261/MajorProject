# =============================================================================
# TRAINING MODULE
# Training and evaluation loops for the classifier
# =============================================================================
"""
Training Module for Lung Cancer Classification.

This module provides:
    - Training loop with progress tracking
    - Validation loop for monitoring
    - Test evaluation with detailed metrics
    - Checkpoint saving/loading

WHY SEPARATE TRAINING MODULE?
    - Keeps training logic clean and reusable
    - Easy to modify training strategy
    - Clear separation from model definition
"""

import os
import time
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.utils.metrics import calculate_metrics, print_classification_report
from src.utils.helpers import save_checkpoint


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Validate the model on validation set.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Tuple of (average_loss, accuracy, all_labels, all_predictions)
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    checkpoint_dir: str = "checkpoints",
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Full training loop with validation.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs to train
        scheduler: Optional learning rate scheduler
        checkpoint_dir: Directory to save checkpoints
        class_names: List of class names for reporting
    
    Returns:
        Dictionary containing training history
    """
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("=" * 60 + "\n")
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation phase
        val_loss, val_acc, val_labels, val_preds = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc, best_path
            )
            print(f"  ★ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("=" * 60 + "\n")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    save_checkpoint(model, optimizer, num_epochs, val_loss, val_acc, final_path)
    
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str]
) -> Dict:
    """
    Evaluate the model on test set with detailed metrics.
    
    Args:
        model: The trained model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        class_names: List of class names
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    
    # Get predictions
    test_loss, test_acc, y_true, y_pred = validate(
        model, test_loader, criterion, device
    )
    
    # Calculate detailed metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print results
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"\nDetailed Metrics:")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall: {metrics['recall']*100:.2f}%")
    print(f"  F1 Score: {metrics['f1_score']*100:.2f}%")
    
    # Print classification report
    print_classification_report(y_true, y_pred, class_names)
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'y_true': y_true,
        'y_pred': y_pred,
        **metrics
    }


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot training history curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy curve
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")
    
    plt.show()
    
    return fig
