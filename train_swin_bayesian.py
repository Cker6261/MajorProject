# =============================================================================
# BAYESIAN OPTIMIZATION FOR SWIN TRANSFORMER HYPERPARAMETER TUNING
# =============================================================================
"""
Bayesian Optimization for Swin Transformer Hyperparameter Tuning.

This script uses Optuna for Bayesian optimization to find the best hyperparameters
for the Swin Transformer model on lung cancer CT classification.

Key Hyperparameters to Optimize:
    - Learning rate
    - Weight decay
    - Dropout rate
    - Batch size
    - Learning rate scheduler parameters
    - Optimizer settings (beta1, beta2)

WHY BAYESIAN OPTIMIZATION?
    - More efficient than grid search or random search
    - Uses probabilistic model to guide search
    - Balances exploration and exploitation
    - Finds better hyperparameters with fewer trials
"""

import os
import sys
import json
import gc
import time
import warnings
from datetime import datetime
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    print("✓ Optuna imported successfully")
except ImportError:
    print("Installing optuna...")
    os.system("pip install optuna")
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

from src.utils.config import Config
from src.data.dataloader import create_dataloaders
from src.models.model_factory import SwinTransformerClassifier
from src.utils.metrics import calculate_metrics
from src.utils.helpers import save_checkpoint

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Optimization settings
N_TRIALS = 30  # Number of Bayesian optimization trials
EPOCHS_PER_TRIAL = 15  # Epochs for each trial (reduced for speed)
FINAL_EPOCHS = 50  # Epochs for final training with best params

# Output directories
BASE_DIR = r"D:\Major Project"
RESULTS_DIR = os.path.join(BASE_DIR, "results", "bayesian_swin_t")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "bayesian_swin_t")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def create_model_with_params(
    num_classes: int,
    dropout_rate: float,
    device: torch.device
) -> nn.Module:
    """Create Swin Transformer model with specified parameters."""
    model = SwinTransformerClassifier(
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=dropout_rate,
        freeze_backbone=False
    )
    return model.to(device)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
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
    
    return running_loss / len(train_loader), 100. * correct / total


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
    
    return running_loss / len(val_loader), 100. * correct / total


def objective(trial: optuna.Trial, train_loader: DataLoader, val_loader: DataLoader, 
              device: torch.device, num_classes: int) -> float:
    """
    Optuna objective function for Bayesian optimization.
    
    Args:
        trial: Optuna trial object
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_classes: Number of output classes
    
    Returns:
        Validation accuracy (to maximize)
    """
    # Sample hyperparameters using Bayesian optimization
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7),
        'beta1': trial.suggest_float('beta1', 0.85, 0.99),
        'beta2': trial.suggest_float('beta2', 0.9, 0.9999),
        'lr_scheduler_gamma': trial.suggest_float('lr_scheduler_gamma', 0.1, 0.9),
        'lr_scheduler_step': trial.suggest_int('lr_scheduler_step', 3, 10),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 0, 5),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
    }
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number + 1}/{N_TRIALS}")
    print(f"{'='*60}")
    print("Hyperparameters:")
    for k, v in hyperparams.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    # Clear GPU memory
    clear_gpu_memory()
    
    try:
        # Create model with sampled dropout
        model = create_model_with_params(
            num_classes=num_classes,
            dropout_rate=hyperparams['dropout_rate'],
            device=device
        )
        
        # Create optimizer with sampled hyperparameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hyperparams['learning_rate'],
            weight_decay=hyperparams['weight_decay'],
            betas=(hyperparams['beta1'], hyperparams['beta2'])
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=hyperparams['lr_scheduler_step'],
            gamma=hyperparams['lr_scheduler_gamma']
        )
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=hyperparams['label_smoothing'])
        
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(EPOCHS_PER_TRIAL):
            # Warmup learning rate
            if epoch < hyperparams['warmup_epochs']:
                warmup_factor = (epoch + 1) / hyperparams['warmup_epochs']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = hyperparams['learning_rate'] * warmup_factor
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            if epoch >= hyperparams['warmup_epochs']:
                scheduler.step()
            
            print(f"  Epoch {epoch+1}/{EPOCHS_PER_TRIAL}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Report intermediate value for pruning
            trial.report(val_acc, epoch)
            
            # Handle pruning
            if trial.should_prune():
                print(f"  Trial pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()
        
        print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
        
    except Exception as e:
        print(f"  Trial failed with error: {e}")
        clear_gpu_memory()
        raise
    finally:
        # Clean up
        del model, optimizer, criterion
        clear_gpu_memory()
    
    return best_val_acc


def run_bayesian_optimization():
    """Run Bayesian optimization to find best hyperparameters."""
    print("\n" + "=" * 80)
    print("BAYESIAN OPTIMIZATION FOR SWIN TRANSFORMER")
    print("=" * 80)
    print(f"Number of trials: {N_TRIALS}")
    print(f"Epochs per trial: {EPOCHS_PER_TRIAL}")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 80 + "\n")
    
    # Setup
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create data loaders with default batch size (will vary in final training)
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, full_dataset = create_dataloaders(
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
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create Optuna study with TPE sampler (Bayesian optimization)
    sampler = TPESampler(seed=42, n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    study = optuna.create_study(
        direction='maximize',  # Maximize validation accuracy
        sampler=sampler,
        pruner=pruner,
        study_name='swin_t_bayesian_optimization'
    )
    
    # Run optimization
    start_time = time.time()
    
    study.optimize(
        lambda trial: objective(
            trial, train_loader, val_loader, device, config.num_classes
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    
    optimization_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Time: {optimization_time / 60:.1f} minutes")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.2f}%")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Save study results
    results = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'optimization_time_minutes': optimization_time / 60,
        'n_trials': N_TRIALS,
        'epochs_per_trial': EPOCHS_PER_TRIAL,
        'all_trials': [
            {
                'number': trial.number,
                'value': trial.value if trial.value is not None else None,
                'params': trial.params,
                'state': str(trial.state)
            }
            for trial in study.trials
        ]
    }
    
    results_path = os.path.join(RESULTS_DIR, 'bayesian_optimization_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return study.best_params, train_loader, val_loader, test_loader, config, device


def train_with_best_params(
    best_params: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: Config,
    device: torch.device
) -> Dict:
    """Train the final model with best hyperparameters."""
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("=" * 80)
    print(f"Epochs: {FINAL_EPOCHS}")
    print("\nBest Hyperparameters:")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 80 + "\n")
    
    clear_gpu_memory()
    
    # Create model with best dropout
    model = create_model_with_params(
        num_classes=config.num_classes,
        dropout_rate=best_params['dropout_rate'],
        device=device
    )
    model.print_model_summary()
    
    # Create optimizer with best hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        betas=(best_params['beta1'], best_params['beta2'])
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=best_params['lr_scheduler_step'],
        gamma=best_params['lr_scheduler_gamma']
    )
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=best_params['label_smoothing'])
    
    # Training history
    history = {
        'model_name': 'swin_t_bayesian',
        'best_params': best_params,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(FINAL_EPOCHS):
        epoch_start = time.time()
        
        # Warmup learning rate
        if epoch < best_params['warmup_epochs']:
            warmup_factor = (epoch + 1) / best_params['warmup_epochs']
            for param_group in optimizer.param_groups:
                param_group['lr'] = best_params['learning_rate'] * warmup_factor
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        if epoch >= best_params['warmup_epochs']:
            scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{FINAL_EPOCHS} ({epoch_time:.1f}s) [LR: {current_lr:.2e}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(CHECKPOINT_DIR, "best_model_swin_t_bayesian.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_path)
            print(f"  ★ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    total_time = time.time() - start_time
    
    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "final_model_swin_t_bayesian.pth")
    save_checkpoint(model, optimizer, FINAL_EPOCHS, val_loss, val_acc, final_path)
    
    print("\n" + "=" * 80)
    print("FINAL TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total Time: {total_time / 60:.1f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_path}")
    print(f"Final model saved to: {final_path}")
    print("=" * 80 + "\n")
    
    history['best_val_acc'] = best_val_acc
    history['total_time'] = total_time
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100. * correct / total
    test_loss = test_loss / len(test_loader)
    
    # Calculate detailed metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    test_metrics = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'confusion_matrix': conf_matrix.tolist()
    }
    
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("=" * 80)
    
    history['test_metrics'] = test_metrics
    
    # Save complete history
    history_path = os.path.join(RESULTS_DIR, 'training_history_bayesian.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    
    return history


def compare_with_original():
    """Compare Bayesian-optimized model with original random-tuned model."""
    print("\n" + "=" * 80)
    print("COMPARING BAYESIAN VS ORIGINAL SWIN TRANSFORMER")
    print("=" * 80)
    
    # Load original model results
    original_results_path = os.path.join(BASE_DIR, "results", "model_comparison.json")
    bayesian_results_path = os.path.join(RESULTS_DIR, "training_history_bayesian.json")
    
    comparison = {}
    
    if os.path.exists(original_results_path):
        with open(original_results_path, 'r') as f:
            original_results = json.load(f)
        
        if 'swin_t' in original_results:
            original_swin = original_results['swin_t']
            comparison['original'] = {
                'model': 'Swin-T (Random Tuned)',
                'val_accuracy': original_swin.get('val_accuracy', 'N/A'),
                'test_accuracy': original_swin.get('test_accuracy', 'N/A'),
                'f1_score': original_swin.get('f1_score', 'N/A')
            }
            print("\nOriginal Swin-T (Random Tuned):")
            print(f"  Validation Accuracy: {comparison['original']['val_accuracy']}")
            print(f"  Test Accuracy: {comparison['original']['test_accuracy']}")
    
    if os.path.exists(bayesian_results_path):
        with open(bayesian_results_path, 'r') as f:
            bayesian_results = json.load(f)
        
        comparison['bayesian'] = {
            'model': 'Swin-T (Bayesian Optimized)',
            'val_accuracy': bayesian_results.get('best_val_acc', 'N/A'),
            'test_accuracy': bayesian_results.get('test_metrics', {}).get('test_accuracy', 'N/A'),
            'f1_score': bayesian_results.get('test_metrics', {}).get('f1_score', 'N/A'),
            'best_params': bayesian_results.get('best_params', {})
        }
        print("\nBayesian Optimized Swin-T:")
        print(f"  Validation Accuracy: {comparison['bayesian']['val_accuracy']:.2f}%")
        print(f"  Test Accuracy: {comparison['bayesian']['test_accuracy']:.2f}%")
        print(f"  F1 Score: {comparison['bayesian']['f1_score']:.2f}%")
    
    # Calculate improvement
    if 'original' in comparison and 'bayesian' in comparison:
        if isinstance(comparison['original']['val_accuracy'], (int, float)):
            orig_val = comparison['original']['val_accuracy']
            bayes_val = comparison['bayesian']['val_accuracy']
            improvement = bayes_val - orig_val
            print(f"\n★ Improvement: {improvement:+.2f}% validation accuracy")
    
    # Save comparison
    comparison_path = os.path.join(RESULTS_DIR, 'bayesian_vs_original_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nComparison saved to: {comparison_path}")
    
    print("=" * 80)
    
    return comparison


def main():
    """Main function to run Bayesian optimization and training."""
    print("\n" + "=" * 80)
    print("SWIN TRANSFORMER BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    total_start = time.time()
    
    # Step 1: Run Bayesian optimization
    best_params, train_loader, val_loader, test_loader, config, device = run_bayesian_optimization()
    
    # Step 2: Train final model with best parameters
    history = train_with_best_params(
        best_params, train_loader, val_loader, test_loader, config, device
    )
    
    # Step 3: Compare with original model
    comparison = compare_with_original()
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 80)
    print("ALL COMPLETE")
    print("=" * 80)
    print(f"Total Time: {total_time / 60:.1f} minutes")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved in: {RESULTS_DIR}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
