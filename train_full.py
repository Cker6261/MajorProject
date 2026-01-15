# =============================================================================
# FULL TRAINING SCRIPT - Complete Model Training Pipeline
# =============================================================================
"""
Complete Training Script for Lung Cancer Classification.

SAFE VERSION - Uses only D: drive, handles memory properly.

Usage:
    python train_full.py                    # Train with default settings
    python train_full.py --epochs 50        # Train for 50 epochs
"""

import os
import sys

# =============================================================================
# CRITICAL: Force ALL caches to D: drive BEFORE any imports
# =============================================================================
os.environ['TORCH_HOME'] = r'D:\Major Project\.cache\torch'
os.environ['HF_HOME'] = r'D:\Major Project\.cache\huggingface'
os.environ['XDG_CACHE_HOME'] = r'D:\Major Project\.cache'
os.environ['TRANSFORMERS_CACHE'] = r'D:\Major Project\.cache\transformers'
os.environ['PIP_CACHE_DIR'] = r'D:\Major Project\.cache\pip'
os.environ['TMPDIR'] = r'D:\Major Project\.cache\tmp'
os.environ['TEMP'] = r'D:\Major Project\.cache\tmp'
os.environ['TMP'] = r'D:\Major Project\.cache\tmp'

# Create cache directories
for cache_dir in [r'D:\Major Project\.cache\torch', 
                  r'D:\Major Project\.cache\huggingface',
                  r'D:\Major Project\.cache\tmp']:
    os.makedirs(cache_dir, exist_ok=True)

# Add project root to path
sys.path.insert(0, r'D:\Major Project')

import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to save memory
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
)
from tqdm import tqdm

from src.utils.config import Config
from src.utils.helpers import set_seed, get_device, save_checkpoint
from src.data.dataloader import create_dataloaders
from src.models.model_factory import create_model, get_optimizer, get_loss_function, get_scheduler


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0
        self.early_stop = False
    
    def __call__(self, val_acc):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False, ncols=80)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False, ncols=80):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return running_loss / len(val_loader), 100. * correct / total, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_training_curves(history, save_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', lw=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', lw=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', lw=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val', lw=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Training curves saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    short_names = ['Adeno', 'Benign', 'Large Cell', 'Normal', 'Squamous']
    
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names, annot_kws={'size': 14})
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title('Confusion Matrix - Lung Cancer Classification', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Confusion matrix saved to {save_path}")
    return cm


def plot_roc_curves(y_true, y_probs, class_names, save_path):
    """Plot ROC curves."""
    n_classes = len(class_names)
    short_names = ['Adeno', 'Benign', 'Large Cell', 'Normal', 'Squamous']
    
    fig = plt.figure(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{short_names[i]} (AUC={roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ ROC curves saved to {save_path}")


def compute_metrics(y_true, y_pred, y_probs):
    """Compute all evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0) * 100,
    }
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
    except:
        metrics['auc_roc'] = 0.0
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train Lung Cancer Classifier')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("     LUNGXAI - COMPLETE MODEL TRAINING")
    print("=" * 60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  All cache stored on: D: drive")
    print("=" * 60)
    
    # Setup
    config = Config()
    set_seed(config.random_seed)
    device = get_device()
    
    print(f"\n[CONFIG]")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Device: {device}")
    
    # Load data with num_workers=0 for Windows stability
    print(f"\n[1/5] Loading Dataset...")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        dataset_dir=config.dataset_dir,
        class_names=config.class_names,
        image_size=config.image_size,
        batch_size=args.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        num_workers=0,  # 0 workers for Windows stability
        random_seed=config.random_seed
    )
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\n[2/5] Creating Model...")
    model = create_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
        freeze_backbone=config.freeze_base,
        device=device
    )
    
    # Training components
    criterion = get_loss_function()
    optimizer = get_optimizer(model, "adamw", args.lr, config.weight_decay)
    scheduler = get_scheduler(optimizer, "step", step_size=config.lr_scheduler_step, gamma=config.lr_scheduler_gamma)
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    start_time = time.time()
    
    # Training loop
    print(f"\n[3/5] Training Model...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | LR: {current_lr:.2e} | {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_path)
            print(f"  ★ New best! Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if early_stopping(val_acc):
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break
        
        # Clear CUDA cache periodically
        if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    final_path = os.path.join(config.checkpoint_dir, "final_model.pth")
    save_checkpoint(model, optimizer, epoch, val_loss, val_acc, final_path)
    
    # Plot training curves
    plot_training_curves(history, os.path.join(config.results_dir, "training_curves.png"))
    
    # Save training history
    with open(os.path.join(config.results_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation
    print(f"\n[4/5] Evaluating on Test Set...")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, y_true, y_pred, y_probs = validate(model, test_loader, criterion, device)
    
    metrics = compute_metrics(y_true, y_pred, y_probs)
    metrics['test_loss'] = test_loss
    metrics['test_acc'] = test_acc
    metrics['best_val_acc'] = best_val_acc
    metrics['total_epochs'] = epoch + 1
    metrics['training_time_minutes'] = total_time / 60
    
    print(f"\n" + "=" * 60)
    print("                 FINAL RESULTS")
    print("=" * 60)
    print(f"  Test Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision:      {metrics['precision_weighted']:.2f}%")
    print(f"  Recall:         {metrics['recall_weighted']:.2f}%")
    print(f"  F1-Score:       {metrics['f1_weighted']:.2f}%")
    print(f"  AUC-ROC:        {metrics['auc_roc']:.4f}")
    print("=" * 60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=config.class_names, zero_division=0))
    
    # Save visualizations
    print(f"\n[5/5] Saving Results...")
    plot_confusion_matrix(y_true, y_pred, config.class_names, os.path.join(config.results_dir, "confusion_matrix.png"))
    plot_roc_curves(y_true, y_probs, config.class_names, os.path.join(config.results_dir, "roc_curves.png"))
    
    # Save metrics
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    metrics['class_names'] = config.class_names
    with open(os.path.join(config.results_dir, "final_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("                 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Best Model: {best_path}")
    print(f"  Results: {config.results_dir}")
    print("=" * 60)
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return metrics


if __name__ == "__main__":
    main()
