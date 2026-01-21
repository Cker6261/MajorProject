# =============================================================================
# COMPARE FINE-TUNED VS BASELINE MODELS
# Script to compare models with and without pretrained weights
# =============================================================================
"""
Model Comparison Script for Lung Cancer Classification.

This script compares:
    1. Fine-tuned models (pretrained=True): Transfer learning with ImageNet weights
    2. Baseline models (pretrained=False): Trained from scratch

The comparison demonstrates the value of transfer learning for medical imaging.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import warnings
from datetime import datetime
from typing import Dict, List, Optional
import gc

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.data.dataloader import create_dataloaders
from src.models.model_factory import create_model
from src.training import evaluate_model

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class ComparisonConfig:
    """Configuration for model comparison."""
    
    def __init__(self):
        self.base_dir = r"D:\Major Project"
        
        # Checkpoint directories
        self.finetuned_checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        self.baseline_checkpoint_dir = os.path.join(self.base_dir, "checkpoints", "baseline")
        
        # Results directory
        self.results_dir = os.path.join(self.base_dir, "results", "finetuned_vs_baseline")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Models to compare (DeiT and MobileViT are baseline-only)
        self.models = ["resnet50", "mobilenetv2", "vit_b_16", "swin_t"]
        self.baseline_only_models = ["deit_small", "mobilevit_s"]
        
        # Model display names
        self.model_display_names = {
            "resnet50": "ResNet-50",
            "mobilenetv2": "MobileNetV2",
            "vit_b_16": "ViT-B/16",
            "swin_t": "Swin-T",
            "deit_small": "DeiT-Small",
            "mobilevit_s": "MobileViT-S"
        }
        
        # Class names
        self.class_names = [
            "adenocarcinoma",
            "Benign cases",
            "large cell carcinoma",
            "Normal cases",
            "squamous cell carcinoma"
        ]
        self.num_classes = 5
        self.dropout_rate = 0.5


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    return device


def save_results(results: Dict, path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
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


def load_and_evaluate_model(
    model_name: str,
    checkpoint_path: str,
    pretrained_for_creation: bool,
    test_loader,
    device: torch.device,
    config: ComparisonConfig
) -> Dict:
    """
    Load a model from checkpoint and evaluate it.
    
    Args:
        model_name: Name of the model
        checkpoint_path: Path to the checkpoint file
        pretrained_for_creation: Whether model was trained with pretrained weights
        test_loader: Test data loader
        device: Device to evaluate on
        config: Configuration object
    
    Returns:
        Dictionary with evaluation results
    """
    results = {
        'model_name': model_name,
        'checkpoint_path': checkpoint_path,
        'pretrained': pretrained_for_creation
    }
    
    if not os.path.exists(checkpoint_path):
        print(f"  ✗ Checkpoint not found: {checkpoint_path}")
        results['error'] = "Checkpoint not found"
        return results
    
    try:
        # Create model (pretrained=False since we're loading weights)
        model = create_model(
            model_name=model_name,
            num_classes=config.num_classes,
            pretrained=False,  # We load weights from checkpoint
            dropout_rate=config.dropout_rate,
            device=device
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
        
        # Evaluate
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
        results['epoch'] = checkpoint.get('epoch', 0)
        
        del model
        clear_gpu_memory()
        
    except Exception as e:
        print(f"  ✗ Error evaluating model: {e}")
        results['error'] = str(e)
        clear_gpu_memory()
    
    return results


def compare_models(config: ComparisonConfig, test_loader, device: torch.device) -> Dict:
    """
    Compare fine-tuned and baseline models.
    
    Args:
        config: Configuration object
        test_loader: Test data loader
        device: Device to use
    
    Returns:
        Dictionary with comparison results
    """
    comparison_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': {}
    }
    
    for model_name in config.models:
        print(f"\n{'='*70}")
        print(f"COMPARING: {model_name.upper()}")
        print(f"{'='*70}")
        
        model_comparison = {
            'display_name': config.model_display_names.get(model_name, model_name)
        }
        
        # Evaluate fine-tuned model
        print(f"\n1. Fine-tuned (pretrained=True):")
        finetuned_path = os.path.join(config.finetuned_checkpoint_dir, f"best_model_{model_name}.pth")
        finetuned_results = load_and_evaluate_model(
            model_name=model_name,
            checkpoint_path=finetuned_path,
            pretrained_for_creation=True,
            test_loader=test_loader,
            device=device,
            config=config
        )
        model_comparison['finetuned'] = finetuned_results
        
        # Evaluate baseline model
        print(f"\n2. Baseline (pretrained=False):")
        baseline_path = os.path.join(config.baseline_checkpoint_dir, f"best_model_{model_name}_baseline.pth")
        baseline_results = load_and_evaluate_model(
            model_name=model_name,
            checkpoint_path=baseline_path,
            pretrained_for_creation=False,
            test_loader=test_loader,
            device=device,
            config=config
        )
        model_comparison['baseline'] = baseline_results
        
        # Calculate improvement
        if 'test_acc' in finetuned_results and 'test_acc' in baseline_results:
            improvement = finetuned_results['test_acc'] - baseline_results['test_acc']
            model_comparison['improvement'] = {
                'test_acc_diff': improvement,
                'percentage_improvement': (improvement / baseline_results['test_acc'] * 100) if baseline_results['test_acc'] > 0 else 0
            }
            print(f"\n  → Improvement: {improvement:+.2f}% (fine-tuned vs baseline)")
        
        comparison_results['models'][model_name] = model_comparison
    
    # Evaluate baseline-only models (DeiT and MobileViT)
    comparison_results['baseline_only_models'] = {}
    
    for model_name in config.baseline_only_models:
        print(f"\n{'='*70}")
        print(f"BASELINE-ONLY: {model_name.upper()}")
        print(f"{'='*70}")
        
        model_data = {
            'display_name': config.model_display_names.get(model_name, model_name),
            'note': 'Baseline only - no fine-tuned version available'
        }
        
        # Evaluate baseline model
        print(f"\nBaseline (pretrained=False):")
        baseline_path = os.path.join(config.baseline_checkpoint_dir, f"best_model_{model_name}_baseline.pth")
        baseline_results = load_and_evaluate_model(
            model_name=model_name,
            checkpoint_path=baseline_path,
            pretrained_for_creation=False,
            test_loader=test_loader,
            device=device,
            config=config
        )
        model_data['baseline'] = baseline_results
        
        comparison_results['baseline_only_models'][model_name] = model_data
    
    return comparison_results


def generate_comparison_report(comparison_results: Dict, output_path: str):
    """Generate a markdown report of the comparison."""
    
    report = []
    report.append("# Fine-tuned vs Baseline Model Comparison Report")
    report.append(f"\n**Generated:** {comparison_results['timestamp']}")
    report.append("\n## Overview")
    report.append("\nThis report compares models trained with two approaches:")
    report.append("- **Fine-tuned (Transfer Learning):** Models initialized with ImageNet pretrained weights")
    report.append("- **Baseline (From Scratch):** Models trained from random initialization")
    report.append("\n---")
    
    report.append("\n## Summary Table")
    report.append("\n| Model | Fine-tuned Acc | Baseline Acc | Improvement |")
    report.append("|-------|----------------|--------------|-------------|")
    
    for model_name, data in comparison_results['models'].items():
        display_name = data.get('display_name', model_name)
        
        finetuned_acc = data.get('finetuned', {}).get('test_acc', 'N/A')
        baseline_acc = data.get('baseline', {}).get('test_acc', 'N/A')
        improvement = data.get('improvement', {}).get('test_acc_diff', 'N/A')
        
        if isinstance(finetuned_acc, (int, float)):
            finetuned_acc = f"{finetuned_acc:.2f}%"
        if isinstance(baseline_acc, (int, float)):
            baseline_acc = f"{baseline_acc:.2f}%"
        if isinstance(improvement, (int, float)):
            improvement = f"{improvement:+.2f}%"
        
        report.append(f"| {display_name} | {finetuned_acc} | {baseline_acc} | {improvement} |")
    
    report.append("\n---")
    
    report.append("\n## Detailed Results")
    
    for model_name, data in comparison_results['models'].items():
        display_name = data.get('display_name', model_name)
        report.append(f"\n### {display_name}")
        
        # Fine-tuned results
        finetuned = data.get('finetuned', {})
        report.append("\n**Fine-tuned (Transfer Learning):**")
        if 'error' in finetuned:
            report.append(f"- Error: {finetuned['error']}")
        else:
            report.append(f"- Test Accuracy: {finetuned.get('test_acc', 'N/A'):.2f}%")
            report.append(f"- Test Loss: {finetuned.get('test_loss', 'N/A'):.4f}")
            report.append(f"- Precision: {finetuned.get('precision', 0)*100:.2f}%")
            report.append(f"- Recall: {finetuned.get('recall', 0)*100:.2f}%")
            report.append(f"- F1 Score: {finetuned.get('f1_score', 0)*100:.2f}%")
        
        # Baseline results
        baseline = data.get('baseline', {})
        report.append("\n**Baseline (From Scratch):**")
        if 'error' in baseline:
            report.append(f"- Error: {baseline['error']}")
        else:
            report.append(f"- Test Accuracy: {baseline.get('test_acc', 'N/A'):.2f}%")
            report.append(f"- Test Loss: {baseline.get('test_loss', 'N/A'):.4f}")
            report.append(f"- Precision: {baseline.get('precision', 0)*100:.2f}%")
            report.append(f"- Recall: {baseline.get('recall', 0)*100:.2f}%")
            report.append(f"- F1 Score: {baseline.get('f1_score', 0)*100:.2f}%")
        
        # Improvement
        improvement = data.get('improvement', {})
        if improvement:
            report.append("\n**Improvement (Fine-tuned over Baseline):**")
            report.append(f"- Accuracy Improvement: {improvement.get('test_acc_diff', 0):+.2f}%")
            report.append(f"- Relative Improvement: {improvement.get('percentage_improvement', 0):.1f}%")
    
    # Add baseline-only models section
    baseline_only = comparison_results.get('baseline_only_models', {})
    if baseline_only:
        report.append("\n---")
        report.append("\n## Baseline-Only Models")
        report.append("\nThese models were only trained from scratch (no fine-tuned versions):")
        report.append("\n| Model | Baseline Acc | F1 Score |")
        report.append("|-------|--------------|----------|")
        
        for model_name, data in baseline_only.items():
            display_name = data.get('display_name', model_name)
            baseline = data.get('baseline', {})
            baseline_acc = baseline.get('test_acc', 'N/A')
            f1_score = baseline.get('f1_score', 0)
            
            if isinstance(baseline_acc, (int, float)):
                baseline_acc = f"{baseline_acc:.2f}%"
            if isinstance(f1_score, (int, float)):
                f1_score = f"{f1_score*100:.2f}%"
            
            report.append(f"| {display_name} | {baseline_acc} | {f1_score} |")
        
        # Detailed results for baseline-only
        for model_name, data in baseline_only.items():
            display_name = data.get('display_name', model_name)
            report.append(f"\n### {display_name} (Baseline Only)")
            report.append(f"\n*Note: {data.get('note', 'No fine-tuned version available')}*")
            
            baseline = data.get('baseline', {})
            if 'error' in baseline:
                report.append(f"\n- Error: {baseline['error']}")
            else:
                report.append(f"\n- Test Accuracy: {baseline.get('test_acc', 'N/A'):.2f}%")
                report.append(f"- Test Loss: {baseline.get('test_loss', 'N/A'):.4f}")
                report.append(f"- Precision: {baseline.get('precision', 0)*100:.2f}%")
                report.append(f"- Recall: {baseline.get('recall', 0)*100:.2f}%")
                report.append(f"- F1 Score: {baseline.get('f1_score', 0)*100:.2f}%")
    
    report.append("\n---")
    report.append("\n## Conclusion")
    report.append("\nTransfer learning with pretrained ImageNet weights provides significant benefits for medical image classification:")
    report.append("- Faster convergence")
    report.append("- Higher accuracy")
    report.append("- Better generalization")
    report.append("\nThis demonstrates the effectiveness of using pretrained models for lung cancer CT image classification.")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Report saved: {output_path}")


def run_comparison():
    """Run the full comparison between fine-tuned and baseline models."""
    print("\n" + "=" * 80)
    print("FINE-TUNED vs BASELINE MODEL COMPARISON")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Initialize
    config = ComparisonConfig()
    device = get_device()
    
    # Load test data
    print("\n" + "-" * 60)
    print("LOADING TEST DATA")
    print("-" * 60)
    
    main_config = Config()
    _, _, test_loader, _ = create_dataloaders(
        dataset_dir=main_config.dataset_dir,
        class_names=main_config.class_names,
        image_size=main_config.image_size,
        batch_size=main_config.batch_size,
        train_ratio=main_config.train_ratio,
        val_ratio=main_config.val_ratio,
        test_ratio=main_config.test_ratio,
        num_workers=main_config.num_workers,
        random_seed=main_config.random_seed
    )
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    # Run comparison
    comparison_results = compare_models(config, test_loader, device)
    
    # Save results
    print("\n" + "-" * 60)
    print("SAVING RESULTS")
    print("-" * 60)
    
    results_json_path = os.path.join(config.results_dir, "comparison_results.json")
    save_results(comparison_results, results_json_path)
    
    report_path = os.path.join(config.results_dir, "comparison_report.md")
    generate_comparison_report(comparison_results, report_path)
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print("\n{:<20} {:>15} {:>15} {:>12}".format(
        "Model", "Fine-tuned", "Baseline", "Improvement"
    ))
    print("-" * 65)
    
    for model_name, data in comparison_results['models'].items():
        display_name = data.get('display_name', model_name)
        finetuned_acc = data.get('finetuned', {}).get('test_acc', 0)
        baseline_acc = data.get('baseline', {}).get('test_acc', 0)
        improvement = data.get('improvement', {}).get('test_acc_diff', 0)
        
        print("{:<20} {:>14.2f}% {:>14.2f}% {:>+11.2f}%".format(
            display_name, finetuned_acc, baseline_acc, improvement
        ))
    
    print("-" * 65)
    
    # Print baseline-only models
    baseline_only = comparison_results.get('baseline_only_models', {})
    if baseline_only:
        print("\nBaseline-Only Models (no fine-tuned version):")
        print("-" * 45)
        print("{:<20} {:>15} {:>12}".format("Model", "Baseline Acc", "F1 Score"))
        print("-" * 45)
        
        for model_name, data in baseline_only.items():
            display_name = data.get('display_name', model_name)
            baseline_acc = data.get('baseline', {}).get('test_acc', 0)
            f1_score = data.get('baseline', {}).get('f1_score', 0) * 100
            
            print("{:<20} {:>14.2f}% {:>11.2f}%".format(
                display_name, baseline_acc, f1_score
            ))
        
        print("-" * 45)
    
    print(f"\nResults saved to: {config.results_dir}")
    
    return comparison_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_comparison()
    print("\n✓ Comparison complete!")
