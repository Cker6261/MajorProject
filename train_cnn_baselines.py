# =============================================================================
# TRAIN CNN BASELINE MODELS (VGG-16, DenseNet-121, EfficientNet-B0)
# Script to train additional CNN baseline models from scratch
# =============================================================================
"""
CNN Baseline Training Script for VGG-16, DenseNet-121, EfficientNet-B0.

This script trains models WITHOUT pretrained ImageNet weights (from scratch)
to get actual baseline results for comparison in the research paper.

Models trained:
    1. VGG-16 (baseline)
    2. DenseNet-121 (baseline)
    3. EfficientNet-B0 (baseline)
"""

import os
import sys
import json
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataloader import create_dataloaders
from src.models.model_factory import create_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import numpy as np

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for baseline training."""
    
    def __init__(self):
        self.base_dir = r"D:\Major Project"
        self.dataset_dir = os.path.join(self.base_dir, "archive (1)", "Lung Cancer Dataset")
        self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints", "baseline")
        self.results_dir = os.path.join(self.base_dir, "results", "baseline")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.class_names = [
            "adenocarcinoma",
            "Benign cases",
            "large cell carcinoma",
            "Normal cases",
            "squamous cell carcinoma"
        ]
        self.num_classes = 5
        
        # Training config
        self.batch_size = 32
        self.num_epochs = 30
        self.learning_rate = 1e-3
        self.weight_decay = 0
        self.image_size = (224, 224)
        
        # Data split
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # Baseline = no pretrained weights
        self.pretrained = False
        self.dropout_rate = 0.5
        
        # Models to train
        self.models_to_train = ["vgg16", "densenet121", "efficientnet_b0"]
        
        self.model_display_names = {
            "vgg16": "VGG-16 (Baseline)",
            "densenet121": "DenseNet-121 (Baseline)",
            "efficientnet_b0": "EfficientNet-B0 (Baseline)"
        }
        
        self.random_seed = 42
        self.num_workers = 4


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[OK] Using CPU")
    return device


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict:
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def train_baseline_model(
    model_name: str,
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device
) -> Dict:
    """Train a single baseline model."""
    
    print(f"\n{'='*60}")
    print(f"TRAINING: {config.model_display_names[model_name]}")
    print(f"{'='*60}")
    
    # Create model (baseline = pretrained=False)
    model = create_model(
        model_name=model_name,
        num_classes=config.num_classes,
        pretrained=False,  # BASELINE!
        dropout_rate=config.dropout_rate,
        device=device
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Record history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            checkpoint_path = os.path.join(
                config.checkpoint_dir, 
                f"best_model_{model_name}_baseline.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"[OK] Saved best model (Val Acc: {val_acc:.2f}%)")
    
    training_time = time.time() - start_time
    
    # Load best model for evaluation
    checkpoint_path = os.path.join(config.checkpoint_dir, f"best_model_{model_name}_baseline.pth")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print(f"\n{'='*40}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*40}")
    
    test_results = evaluate_model(model, test_loader, device, config.class_names)
    
    print(f"\nTest Results for {config.model_display_names[model_name]}:")
    print(f"  Accuracy:  {test_results['accuracy']*100:.2f}%")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall:    {test_results['recall']:.4f}")
    print(f"  F1-Score:  {test_results['f1_score']:.4f}")
    
    # Save final model
    final_path = os.path.join(config.checkpoint_dir, f"final_model_{model_name}_baseline.pth")
    torch.save(model.state_dict(), final_path)
    
    # Compile results
    results = {
        'model_name': model_name,
        'display_name': config.model_display_names[model_name],
        'pretrained': False,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'training_time_seconds': training_time,
        'test_results': test_results,
        'training_history': training_history
    }
    
    # Clear memory
    del model
    clear_gpu_memory()
    
    return results


def main():
    """Main training function."""
    print("\n" + "=" * 70)
    print("TRAINING CNN BASELINE MODELS (VGG-16, DenseNet-121, EfficientNet-B0)")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize
    config = Config()
    device = get_device()
    
    # Set random seeds
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    
    # Create data loaders
    print("\n" + "-" * 50)
    print("LOADING DATASET")
    print("-" * 50)
    
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        dataset_dir=config.dataset_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )
    
    print(f"[OK] Train batches: {len(train_loader)}")
    print(f"[OK] Val batches: {len(val_loader)}")
    print(f"[OK] Test batches: {len(test_loader)}")
    
    # Train each model
    all_results = {}
    
    for model_name in config.models_to_train:
        try:
            results = train_baseline_model(
                model_name=model_name,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device
            )
            all_results[model_name] = results
            
            # Save individual results
            result_path = os.path.join(config.results_dir, f"{model_name}_baseline_results.json")
            with open(result_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = results.copy()
                serializable_results['test_results'] = {
                    k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in results['test_results'].items()
                }
                json.dump(serializable_results, f, indent=2)
            print(f"[OK] Saved results to {result_path}")
            
        except Exception as e:
            print(f"\n[ERROR] Training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Update all_baseline_results.json with new models
    all_results_path = os.path.join(config.results_dir, "all_baseline_results.json")
    
    # Load existing results
    if os.path.exists(all_results_path):
        with open(all_results_path, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = {}
    
    # Merge new results
    for model_name, results in all_results.items():
        existing_results[model_name] = {
            'accuracy': results['test_results']['accuracy'],
            'precision': results['test_results']['precision'],
            'recall': results['test_results']['recall'],
            'f1_score': results['test_results']['f1_score'],
            'training_time_seconds': results['training_time_seconds']
        }
    
    # Save updated results
    with open(all_results_path, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    
    for model_name, results in all_results.items():
        test = results['test_results']
        print(f"{config.model_display_names[model_name]:<25} "
              f"{test['accuracy']*100:.2f}%{'':<6} "
              f"{test['precision']:.4f}{'':<6} "
              f"{test['recall']:.4f}{'':<6} "
              f"{test['f1_score']:.4f}")
    
    print("\n" + "=" * 70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {config.results_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
