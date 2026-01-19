# =============================================================================
# MODEL COMPARISON SCRIPT
# Compare all trained models and generate visualizations
# =============================================================================
"""
Model Comparison Script for Lung Cancer Classification.

This script:
    1. Loads all trained models from cache
    2. Evaluates them on the test set
    3. Generates comparison charts and metrics
    4. Creates a comprehensive comparison report
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.data.dataloader import create_dataloaders
from src.models.model_factory import create_model, get_model_info
from src.training import validate
from src.models.model_factory import get_loss_function
from src.utils.metrics import calculate_metrics
from sklearn.metrics import confusion_matrix, classification_report


def load_model_from_cache(config: Config, model_name: str, device: torch.device):
    """Load a trained model from cache."""
    checkpoint_path = config.get_model_checkpoint_path(model_name, "best")
    
    if not os.path.exists(checkpoint_path):
        print(f"✗ Model not found: {checkpoint_path}")
        return None, None
    
    # Create model
    model = create_model(
        model_name=model_name,
        num_classes=config.num_classes,
        pretrained=False,
        dropout_rate=config.dropout_rate,
        device=device
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded {model_name} from cache")
    return model, checkpoint


def evaluate_all_models(config: Config, test_loader, device: torch.device) -> dict:
    """Evaluate all cached models on the test set."""
    results = {}
    criterion = get_loss_function()
    
    for model_name in config.available_models:
        print(f"\nEvaluating {model_name}...")
        
        model, checkpoint = load_model_from_cache(config, model_name, device)
        
        if model is None:
            continue
        
        # Evaluate
        test_loss, test_acc, y_true, y_pred = validate(
            model, test_loader, criterion, device
        )
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results[model_name] = {
            'display_name': config.model_display_names.get(model_name, model_name),
            'test_acc': test_acc,
            'test_loss': test_loss,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'confusion_matrix': cm.tolist(),
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist()
        }
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results


def plot_comparison_charts(results: dict, config: Config, save_dir: str):
    """Generate comparison visualizations."""
    
    # Prepare data
    models = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for model_name, data in results.items():
        models.append(data['display_name'])
        accuracies.append(data['test_acc'])
        precisions.append(data['precision'] * 100)
        recalls.append(data['recall'] * 100)
        f1_scores.append(data['f1_score'] * 100)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Bar chart comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    # Accuracy comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, accuracies, color=colors[:len(models)])
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Precision comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, precisions, color=colors[:len(models)])
    ax2.set_ylabel('Precision (%)', fontsize=12)
    ax2.set_title('Precision Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 100])
    for bar, prec in zip(bars2, precisions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{prec:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Recall comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, recalls, color=colors[:len(models)])
    ax3.set_ylabel('Recall (%)', fontsize=12)
    ax3.set_title('Recall Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 100])
    for bar, rec in zip(bars3, recalls):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rec:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # F1 Score comparison
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, f1_scores, color=colors[:len(models)])
    ax4.set_ylabel('F1 Score (%)', fontsize=12)
    ax4.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 100])
    for bar, f1 in zip(bars4, f1_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{f1:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    chart_path = os.path.join(save_dir, 'model_comparison_charts.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison charts saved to {chart_path}")
    plt.close()
    
    # 2. Radar chart for overall comparison
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    
    for i, (model_name, data) in enumerate(results.items()):
        values = [
            data['test_acc'],
            data['precision'] * 100,
            data['recall'] * 100,
            data['f1_score'] * 100
        ]
        values += values[:1]  # Close the polygon
        
        ax.plot(angles, values, 'o-', linewidth=2, label=data['display_name'], color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    radar_path = os.path.join(save_dir, 'model_comparison_radar.png')
    plt.savefig(radar_path, dpi=150, bbox_inches='tight')
    print(f"✓ Radar chart saved to {radar_path}")
    plt.close()
    
    # 3. Confusion matrices for each model
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, data) in enumerate(results.items()):
        cm = np.array(data['confusion_matrix'])
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=config.class_names,
            yticklabels=config.class_names,
            ax=axes[i]
        )
        axes[i].set_title(f"{data['display_name']}\nAcc: {data['test_acc']:.1f}%", fontsize=12)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(axes[i].yaxis.get_majorticklabels(), rotation=0)
    
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrices_comparison.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrices saved to {cm_path}")
    plt.close()


def generate_comparison_report(results: dict, config: Config, save_dir: str):
    """Generate a detailed comparison report."""
    
    report_path = os.path.join(save_dir, 'model_comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Lung Cancer Classification - Model Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Model | Test Accuracy | Precision | Recall | F1 Score |\n")
        f.write("|-------|--------------|-----------|--------|----------|\n")
        
        best_model = None
        best_acc = 0
        
        for model_name, data in results.items():
            f.write(f"| {data['display_name']} | {data['test_acc']:.2f}% | ")
            f.write(f"{data['precision']*100:.2f}% | {data['recall']*100:.2f}% | ")
            f.write(f"{data['f1_score']*100:.2f}% |\n")
            
            if data['test_acc'] > best_acc:
                best_acc = data['test_acc']
                best_model = data['display_name']
        
        f.write(f"\n## Best Model: **{best_model}** ({best_acc:.2f}% accuracy)\n\n")
        
        f.write("## Model Details\n\n")
        for model_name, data in results.items():
            info = get_model_info(model_name)
            f.write(f"### {data['display_name']}\n\n")
            f.write(f"- **Parameters:** {info['params']}\n")
            f.write(f"- **Description:** {info['description']}\n")
            f.write(f"- **Test Accuracy:** {data['test_acc']:.2f}%\n")
            f.write(f"- **Test Loss:** {data['test_loss']:.4f}\n")
            f.write(f"- **Precision:** {data['precision']*100:.2f}%\n")
            f.write(f"- **Recall:** {data['recall']*100:.2f}%\n")
            f.write(f"- **F1 Score:** {data['f1_score']*100:.2f}%\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("Based on the comparison results:\n\n")
        f.write(f"1. **Best Overall Performance:** {best_model}\n")
        f.write("2. **For Deployment:** Consider MobileNetV2 for resource-constrained environments\n")
        f.write("3. **For Research:** Vision Transformer and Swin Transformer provide novel approaches\n")
        f.write("4. **For Explainability:** ResNet-50 works best with Grad-CAM visualizations\n")
    
    print(f"✓ Comparison report saved to {report_path}")


def compare_models():
    """Main function to compare all models."""
    print("\n" + "=" * 70)
    print("LUNG CANCER CLASSIFICATION - MODEL COMPARISON")
    print("=" * 70)
    
    # Initialize
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nDevice: {device}")
    print(f"Models to compare: {config.available_models}")
    
    # Load test data
    print("\nLoading test dataset...")
    _, _, test_loader, _ = create_dataloaders(
        dataset_dir=config.dataset_dir,
        class_names=config.class_names,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed
    )
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    # Evaluate all models
    results = evaluate_all_models(config, test_loader, device)
    
    if not results:
        print("✗ No trained models found. Run train_all_models.py first.")
        return
    
    # Create comparison directory
    comparison_dir = os.path.join(config.results_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating comparison charts...")
    plot_comparison_charts(results, config, comparison_dir)
    
    # Generate report
    print("\nGenerating comparison report...")
    generate_comparison_report(results, config, comparison_dir)
    
    # Save raw results
    results_path = os.path.join(comparison_dir, 'comparison_results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for model_name, data in results.items():
        serializable_results[model_name] = {
            k: (v if not isinstance(v, np.ndarray) else v.tolist())
            for k, v in data.items()
        }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"✓ Raw results saved to {results_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<35} {'Accuracy':<12} {'F1 Score':<12}")
    print("-" * 59)
    
    for model_name, data in sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True):
        print(f"{data['display_name']:<35} {data['test_acc']:.2f}%{'':<6} {data['f1_score']*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    compare_models()
