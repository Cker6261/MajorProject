# =============================================================================
# TRAIN ALL MODELS
# Script to train and compare multiple models with caching support
# =============================================================================
"""
Multi-Model Training Script for Lung Cancer Classification.

This script:
    1. Trains multiple models (ResNet-50, MobileNetV2, ViT, Swin)
    2. Uses caching to skip already trained models
    3. Saves all checkpoints with model names
    4. Generates comparison metrics
    
IMPORTANT:
    - All data stored on D: drive to avoid C: drive issues
    - Uses caching to avoid retraining
    - Runs models one at a time to prevent memory issues
"""

import os
import sys
import json
import gc
import torch
import warnings
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.data.dataloader import create_dataloaders
from src.models.model_factory import create_model, get_model_info
from src.training import train_model, evaluate_model
from src.models.model_factory import get_optimizer, get_loss_function, get_scheduler

warnings.filterwarnings('ignore')


def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("✓ GPU memory cleared")


def check_model_cached(config: Config, model_name: str) -> bool:
    """Check if a model has already been trained."""
    checkpoint_path = config.get_model_checkpoint_path(model_name, "best")
    if os.path.exists(checkpoint_path):
        print(f"✓ Found cached model: {checkpoint_path}")
        return True
    return False


def load_cached_model(config: Config, model_name: str, device: torch.device):
    """Load a cached model from checkpoint."""
    checkpoint_path = config.get_model_checkpoint_path(model_name, "best")
    
    # Create the model
    model = create_model(
        model_name=model_name,
        num_classes=config.num_classes,
        pretrained=False,  # We'll load weights from checkpoint
        dropout_rate=config.dropout_rate,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded cached model from {checkpoint_path}")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Val Accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%")
    
    return model, checkpoint


def train_single_model(
    config: Config,
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    force_retrain: bool = False
) -> dict:
    """
    Train a single model with caching support.
    
    Args:
        config: Configuration object
        model_name: Name of the model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to train on
        force_retrain: Force retraining even if cached
    
    Returns:
        Dictionary with training results
    """
    print("\n" + "=" * 70)
    print(f"MODEL: {model_name.upper()}")
    print("=" * 70)
    
    results = {
        'model_name': model_name,
        'display_name': config.model_display_names.get(model_name, model_name),
        'cached': False,
        'trained': False
    }
    
    # Check if model is cached
    if config.use_cache and config.skip_if_cached and not force_retrain:
        if check_model_cached(config, model_name):
            results['cached'] = True
            
            # Load cached model for evaluation
            model, checkpoint = load_cached_model(config, model_name, device)
            
            # Evaluate on test set
            criterion = get_loss_function()
            eval_results = evaluate_model(
                model, test_loader, criterion, device, config.class_names
            )
            
            results['test_acc'] = eval_results['test_acc']
            results['test_loss'] = eval_results['test_loss']
            results['precision'] = eval_results['precision']
            results['recall'] = eval_results['recall']
            results['f1_score'] = eval_results['f1_score']
            results['val_acc'] = checkpoint.get('accuracy', 0)
            
            # Clear memory
            del model
            clear_gpu_memory()
            
            return results
    
    # Train the model
    print(f"\nTraining {model_name}...")
    
    try:
        # Create model
        model = create_model(
            model_name=model_name,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout_rate=config.dropout_rate,
            freeze_backbone=config.freeze_base,
            device=device
        )
        
        # Create optimizer and scheduler
        optimizer = get_optimizer(
            model,
            optimizer_name="adamw",
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        criterion = get_loss_function()
        
        scheduler = get_scheduler(
            optimizer,
            scheduler_name="step",
            step_size=config.lr_scheduler_step,
            gamma=config.lr_scheduler_gamma
        )
        
        # Train
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=config.num_epochs,
            scheduler=scheduler,
            checkpoint_dir=config.checkpoint_dir,
            class_names=config.class_names,
            model_name=model_name
        )
        
        # Evaluate on test set
        eval_results = evaluate_model(
            model, test_loader, criterion, device, config.class_names
        )
        
        # Store results
        results['trained'] = True
        results['test_acc'] = eval_results['test_acc']
        results['test_loss'] = eval_results['test_loss']
        results['precision'] = eval_results['precision']
        results['recall'] = eval_results['recall']
        results['f1_score'] = eval_results['f1_score']
        results['val_acc'] = history.get('best_val_acc', 0)
        results['training_time'] = history.get('total_time', 0)
        results['history'] = {
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_loss': history['val_loss'],
            'val_acc': history['val_acc']
        }
        
        # Save training history
        history_path = os.path.join(
            config.get_model_results_path(model_name),
            'training_history.json'
        )
        with open(history_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"✓ Training history saved to {history_path}")
        
        # Clear memory
        del model, optimizer, scheduler
        clear_gpu_memory()
        
    except Exception as e:
        print(f"✗ Error training {model_name}: {str(e)}")
        results['error'] = str(e)
        clear_gpu_memory()
    
    return results


def train_all_models(force_retrain: bool = False):
    """
    Train all supported models.
    
    Args:
        force_retrain: Force retraining even if cached
    """
    print("\n" + "=" * 70)
    print("LUNG CANCER CLASSIFICATION - MULTI-MODEL TRAINING")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize configuration
    config = Config()
    
    # Model-specific batch sizes (smaller for transformers due to GPU memory)
    model_batch_sizes = {
        'resnet50': 32,
        'mobilenetv2': 32,
        'vit_b_16': 8,    # ViT needs small batch size on 6GB GPU
        'swin_t': 16,     # Swin can use slightly larger
    }
    
    # Ensure all paths are on D: drive
    assert config.base_dir.upper().startswith("D:"), "Base directory must be on D: drive!"
    
    print(f"\nConfiguration:")
    print(f"  Base Directory: {config.base_dir}")
    print(f"  Dataset: {config.dataset_dir}")
    print(f"  Checkpoints: {config.checkpoint_dir}")
    print(f"  Results: {config.results_dir}")
    print(f"  Cache: {config.cache_dir}")
    print(f"  Use Caching: {config.use_cache}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Default Batch Size: {config.batch_size}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Train each model
    all_results = {}
    models_to_train = config.available_models
    
    print(f"\nModels to train: {models_to_train}")
    
    for i, model_name in enumerate(models_to_train, 1):
        print(f"\n[{i}/{len(models_to_train)}] Processing {model_name}...")
        
        # Get model info
        info = get_model_info(model_name)
        print(f"  Name: {info['name']}")
        print(f"  Parameters: {info['params']}")
        print(f"  Description: {info['description']}")
        
        # Get model-specific batch size
        batch_size = model_batch_sizes.get(model_name, config.batch_size)
        print(f"  Batch Size: {batch_size}")
        
        # Create model-specific dataloaders with appropriate batch size
        print(f"  Creating dataloaders with batch_size={batch_size}...")
        train_loader, val_loader, test_loader, _ = create_dataloaders(
            dataset_dir=config.dataset_dir,
            class_names=config.class_names,
            batch_size=batch_size,
            num_workers=0,  # Use 0 workers for stability
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            random_seed=config.random_seed
        )
        
        # Train or load from cache
        results = train_single_model(
            config=config,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            force_retrain=force_retrain
        )
        
        all_results[model_name] = results
        
        # Clear memory between models
        clear_gpu_memory()
    
    # Generate comparison report
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    
    comparison_data = []
    for model_name, results in all_results.items():
        if 'error' not in results:
            comparison_data.append({
                'Model': results['display_name'],
                'Test Acc (%)': f"{results.get('test_acc', 0):.2f}",
                'Precision (%)': f"{results.get('precision', 0)*100:.2f}",
                'Recall (%)': f"{results.get('recall', 0)*100:.2f}",
                'F1 Score (%)': f"{results.get('f1_score', 0)*100:.2f}",
                'Cached': '✓' if results.get('cached', False) else '✗'
            })
    
    # Print comparison table
    if comparison_data:
        headers = comparison_data[0].keys()
        print(f"\n{'Model':<30} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Cached':<8}")
        print("-" * 86)
        for row in comparison_data:
            print(f"{row['Model']:<30} {row['Test Acc (%)']:<12} {row['Precision (%)']:<12} {row['Recall (%)']:<12} {row['F1 Score (%)']:<12} {row['Cached']:<8}")
    
    # Save comparison results
    comparison_path = os.path.join(config.results_dir, 'model_comparison.json')
    with open(comparison_path, 'w') as f:
        # Convert results for JSON serialization
        serializable_results = {}
        for model_name, results in all_results.items():
            serializable_results[model_name] = {
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in results.items()
                if k != 'history'  # Skip history to reduce file size
            }
        json.dump(serializable_results, f, indent=2)
    print(f"\n✓ Comparison results saved to {comparison_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train all models for lung cancer classification")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if cached models exist"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to train (e.g., resnet50 mobilenetv2)"
    )
    
    args = parser.parse_args()
    
    # If specific models are requested, update config
    if args.models:
        config = Config()
        config.available_models = args.models
    
    train_all_models(force_retrain=args.force_retrain)
