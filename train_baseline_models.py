# =============================================================================
# TRAIN BASELINE MODELS (WITHOUT PRETRAINED WEIGHTS)
# Script to train models from scratch for comparison with fine-tuned models
# =============================================================================
"""
Baseline Model Training Script for Lung Cancer Classification.

This script trains models WITHOUT pretrained ImageNet weights (from scratch)
to compare with fine-tuned models that use pretrained weights.

Key Differences from Fine-tuned Training:
    - pretrained=False: Random weight initialization
    - No transfer learning benefits
    - Shows the value of pretrained weights

Models trained:
    1. ResNet-50 (baseline)
    2. MobileNetV2 (baseline)
    3. ViT-B/16 (baseline)
    4. Swin-T (baseline)
"""

import os
import sys
import json
import gc
import torch
import torch.nn as nn
import warnings
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataloader import create_dataloaders
from src.models.model_factory import create_model, get_model_info
from src.training import train_model, evaluate_model

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION FOR BASELINE TRAINING
# =============================================================================

class BaselineConfig:
    """Configuration for baseline (non-pretrained) model training."""
    
    def __init__(self):
        # Base paths - use D: drive
        self.base_dir = r"D:\Major Project"
        self.dataset_dir = os.path.join(self.base_dir, "archive (1)", "Lung Cancer Dataset")
        self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints", "baseline")
        self.results_dir = os.path.join(self.base_dir, "results", "baseline")
        self.cache_dir = os.path.join(self.base_dir, "cache")
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Class configuration
        self.class_names = [
            "adenocarcinoma",
            "Benign cases",
            "large cell carcinoma",
            "Normal cases",
            "squamous cell carcinoma"
        ]
        self.num_classes = 5
        
        # Training configuration - SIMPLE, NO HYPERPARAMETER TUNING
        self.batch_size = 32
        self.num_epochs = 30  # Reasonable epochs for baseline
        self.learning_rate = 1e-3  # Simple learning rate
        self.weight_decay = 0  # No weight decay for baseline
        self.image_size = (224, 224)
        
        # Data split
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # Model settings - BASELINE (NO PRETRAINED WEIGHTS)
        self.pretrained = False  # KEY DIFFERENCE!
        self.dropout_rate = 0.5
        self.freeze_backbone = False
        
        # Models to train (including new models DeiT and MobileViT)
        self.models_to_train = [
            "resnet50",
            "mobilenetv2",
            "vit_b_16",
            "swin_t",
            "deit_small",
            "mobilevit_s"
        ]
        
        # Model display names
        self.model_display_names = {
            "resnet50": "ResNet-50 (Baseline)",
            "mobilenetv2": "MobileNetV2 (Baseline)",
            "vit_b_16": "ViT-B/16 (Baseline)",
            "swin_t": "Swin-T (Baseline)",
            "deit_small": "DeiT-Small (Baseline)",
            "mobilevit_s": "MobileViT-S (Baseline)"
        }
        
        # Random seed
        self.random_seed = 42
        self.num_workers = 4
    
    def get_checkpoint_path(self, model_name: str, checkpoint_type: str = "best") -> str:
        """Get checkpoint path for baseline model."""
        filename = f"{checkpoint_type}_model_{model_name}_baseline.pth"
        return os.path.join(self.checkpoint_dir, filename)
    
    def is_model_trained(self, model_name: str) -> bool:
        """Check if baseline model already exists."""
        return os.path.exists(self.get_checkpoint_path(model_name, "best"))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("✓ GPU memory cleared")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    return device


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'accuracy': val_acc,
    }, path)
    print(f"✓ Checkpoint saved: {path}")


def save_results(results: Dict, path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"✓ Results saved: {path}")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_baseline_model(
    config: BaselineConfig,
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    force_retrain: bool = False
) -> Dict:
    """
    Train a single baseline model (without pretrained weights).
    
    Args:
        config: Baseline configuration
        model_name: Name of the model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to train on
        force_retrain: Force retraining even if cached
    
    Returns:
        Dictionary with training and evaluation results
    """
    print("\n" + "=" * 70)
    print(f"BASELINE MODEL: {model_name.upper()} (No Pretrained Weights)")
    print("=" * 70)
    
    results = {
        'model_name': model_name,
        'display_name': config.model_display_names.get(model_name, model_name),
        'pretrained': False,
        'type': 'baseline'
    }
    
    # Check if already trained
    if not force_retrain and config.is_model_trained(model_name):
        print(f"✓ Baseline model already trained: {model_name}")
        print("  Loading cached results...")
        
        # Load and evaluate cached model
        checkpoint_path = config.get_checkpoint_path(model_name, "best")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model and load weights
        model = create_model(
            model_name=model_name,
            num_classes=config.num_classes,
            pretrained=False,
            dropout_rate=config.dropout_rate,
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        criterion = nn.CrossEntropyLoss()
        eval_results = evaluate_model(
            model, test_loader, criterion, device, config.class_names
        )
        
        results['test_acc'] = eval_results['test_acc']
        results['test_loss'] = eval_results['test_loss']
        results['precision'] = eval_results['precision']
        results['recall'] = eval_results['recall']
        results['f1_score'] = eval_results['f1_score']
        results['val_acc'] = checkpoint.get('accuracy', 0)
        results['cached'] = True
        
        del model
        clear_gpu_memory()
        return results
    
    # Create model WITHOUT pretrained weights
    print(f"\nCreating baseline model (pretrained=False)...")
    model = create_model(
        model_name=model_name,
        num_classes=config.num_classes,
        pretrained=False,  # NO PRETRAINED WEIGHTS
        dropout_rate=config.dropout_rate,
        freeze_backbone=False,
        device=device
    )
    
    # Simple optimizer - no fancy hyperparameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    )
    
    # Simple loss function
    criterion = nn.CrossEntropyLoss()
    
    # No learning rate scheduler for baseline
    scheduler = None
    
    print(f"\nTraining Configuration:")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Batch Size: {config.batch_size}")
    print(f"  - Learning Rate: {config.learning_rate}")
    print(f"  - Pretrained: {config.pretrained}")
    print(f"  - Optimizer: Adam (simple)")
    
    # Training loop
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    try:
        for epoch in range(config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Record history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{config.num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    model, optimizer, epoch, avg_val_loss, val_acc,
                    config.get_checkpoint_path(model_name, "best")
                )
                print(f"  ★ New best! Val Acc: {val_acc:.2f}%")
        
        # Save final model
        save_checkpoint(
            model, optimizer, config.num_epochs - 1, avg_val_loss, val_acc,
            config.get_checkpoint_path(model_name, "final")
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        eval_results = evaluate_model(
            model, test_loader, criterion, device, config.class_names
        )
        
        # Compile results
        results['test_acc'] = eval_results['test_acc']
        results['test_loss'] = eval_results['test_loss']
        results['precision'] = eval_results['precision']
        results['recall'] = eval_results['recall']
        results['f1_score'] = eval_results['f1_score']
        results['val_acc'] = best_val_acc
        results['history'] = history
        results['cached'] = False
        results['epochs_trained'] = config.num_epochs
        
        print(f"\n{'='*60}")
        print(f"BASELINE {model_name.upper()} COMPLETE")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        print(f"Test Acc: {eval_results['test_acc']:.2f}%")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        results['error'] = str(e)
    
    finally:
        del model
        clear_gpu_memory()
    
    return results


def train_all_baseline_models(force_retrain: bool = False) -> Dict:
    """
    Train all baseline models and save results.
    
    Args:
        force_retrain: Force retraining even if cached
    
    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 80)
    print("BASELINE MODEL TRAINING (WITHOUT PRETRAINED WEIGHTS)")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Initialize configuration
    config = BaselineConfig()
    
    # Set random seeds
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    
    # Get device
    device = get_device()
    
    # Create data loaders
    print("\n" + "-" * 60)
    print("LOADING DATASET")
    print("-" * 60)
    
    # Load data using config parameters
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        dataset_dir=config.dataset_dir,
        class_names=config.class_names,
        image_size=config.image_size,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        num_workers=config.num_workers,
        random_seed=config.random_seed
    )
    
    print(f"✓ Training samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    # Train all models
    all_results = {
        'training_type': 'baseline',
        'pretrained': False,
        'config': {
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'optimizer': 'Adam'
        },
        'models': {}
    }
    
    for model_name in config.models_to_train:
        print(f"\n{'='*80}")
        print(f"Training {model_name.upper()} (Baseline)")
        print(f"{'='*80}")
        
        try:
            results = train_baseline_model(
                config=config,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                force_retrain=force_retrain
            )
            all_results['models'][model_name] = results
            
            # Save individual model results
            model_results_path = os.path.join(
                config.results_dir, f"{model_name}_baseline_results.json"
            )
            save_results(results, model_results_path)
            
        except Exception as e:
            print(f"✗ Error training {model_name}: {e}")
            all_results['models'][model_name] = {'error': str(e)}
        
        # Clear memory between models
        clear_gpu_memory()
    
    # Save all results
    all_results_path = os.path.join(config.results_dir, "all_baseline_results.json")
    save_results(all_results, all_results_path)
    
    print("\n" + "=" * 80)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nResults Summary:")
    print("-" * 60)
    
    for model_name, results in all_results['models'].items():
        if 'error' not in results:
            print(f"{model_name:20s} | Test Acc: {results.get('test_acc', 0):.2f}% | "
                  f"F1: {results.get('f1_score', 0)*100:.2f}%")
        else:
            print(f"{model_name:20s} | ERROR: {results['error'][:50]}")
    
    print("-" * 60)
    print(f"\nResults saved to: {config.results_dir}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    
    return all_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train baseline models without pretrained weights")
    parser.add_argument('--force', action='store_true', help='Force retraining even if cached')
    args = parser.parse_args()
    
    results = train_all_baseline_models(force_retrain=args.force)
    
    print("\n✓ Done!")
