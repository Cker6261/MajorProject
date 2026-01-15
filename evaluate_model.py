# =============================================================================
# EVALUATE MODEL - Get actual accuracy and metrics for the prototype
# =============================================================================
"""
Script to evaluate the trained model and display all metrics.

Run: python evaluate_model.py
"""

import os
import sys
import json

# Force cache to D: drive
os.environ['TORCH_HOME'] = r'd:\Major Project\.cache\torch'

sys.path.insert(0, r'd:\Major Project')

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import Config
from src.data.dataloader import create_dataloaders
from src.models.model_factory import create_model
from src.utils.helpers import get_device, load_checkpoint


def evaluate():
    """Evaluate the trained model and display all metrics."""
    
    print("=" * 70)
    print("LUNGXAI PROTOTYPE - MODEL EVALUATION")
    print("=" * 70)
    
    # Setup
    device = get_device()
    config = Config()
    
    # Check for model
    checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No trained model found at {checkpoint_path}")
        print("Please train the model first: python main.py --mode train")
        return
    
    print(f"\n[1/4] Loading model from: {checkpoint_path}")
    
    # Create model
    model = create_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=False
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   Model loaded successfully!")
    print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Checkpoint val accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%" if isinstance(checkpoint.get('val_acc'), (int, float)) else "")
    
    # Load data
    print(f"\n[2/4] Loading test dataset...")
    
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        dataset_dir=config.dataset_dir,
        class_names=config.class_names,
        image_size=config.image_size,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        num_workers=0  # Use 0 for evaluation
    )
    
    print(f"   Test samples: {len(test_loader.dataset)}")
    print(f"   Classes: {config.class_names}")
    
    # Evaluate
    print(f"\n[3/4] Running evaluation on test set...")
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Calculate metrics
    print(f"\n[4/4] Calculating metrics...")
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    
    # Calculate AUC-ROC (one-vs-rest)
    try:
        auc_roc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
    except:
        auc_roc = 0.0
    
    # Print results
    print("\n" + "=" * 70)
    print("                    PROTOTYPE METRICS SUMMARY")
    print("=" * 70)
    
    print(f"""
    ┌────────────────────────────────────────────────────────────────┐
    │                    OVERALL PERFORMANCE                         │
    ├────────────────────────────────────────────────────────────────┤
    │  Accuracy:           {accuracy:>6.2f}%                                │
    │  Precision:          {precision_weighted:>6.2f}%  (weighted)                     │
    │  Recall:             {recall_weighted:>6.2f}%  (weighted)                     │
    │  F1-Score:           {f1_weighted:>6.2f}%  (weighted)                     │
    │  AUC-ROC:            {auc_roc:>6.4f}  (weighted, one-vs-rest)        │
    ├────────────────────────────────────────────────────────────────┤
    │  Precision (Macro):  {precision_macro:>6.2f}%                                │
    │  Recall (Macro):     {recall_macro:>6.2f}%                                │
    │  F1-Score (Macro):   {f1_macro:>6.2f}%                                │
    └────────────────────────────────────────────────────────────────┘
    """)
    
    # Per-class metrics
    print("\n" + "-" * 70)
    print("                      PER-CLASS METRICS")
    print("-" * 70)
    
    report = classification_report(y_true, y_pred, target_names=config.class_names, zero_division=0)
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "-" * 70)
    print("                      CONFUSION MATRIX")
    print("-" * 70)
    print("\nRows = Actual, Columns = Predicted\n")
    
    # Print header
    header = "            "
    for i, name in enumerate(config.class_names):
        short_name = name[:8] + ".." if len(name) > 10 else name[:10]
        header += f"{short_name:>12}"
    print(header)
    print("            " + "-" * (12 * len(config.class_names)))
    
    for i, row in enumerate(cm):
        short_name = config.class_names[i][:8] + ".." if len(config.class_names[i]) > 10 else config.class_names[i][:10]
        row_str = f"{short_name:>10} |"
        for val in row:
            row_str += f"{val:>12}"
        print(row_str)
    
    # Save metrics to file
    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'auc_roc': auc_roc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm.tolist(),
        'class_names': config.class_names
    }
    
    metrics_path = os.path.join(config.results_dir, "prototype_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(10, 8))
    
    # Shorter names for display
    short_names = []
    for name in config.class_names:
        if 'adenocarcinoma' in name.lower():
            short_names.append('Adeno')
        elif 'squamous' in name.lower():
            short_names.append('Squamous')
        elif 'large' in name.lower():
            short_names.append('Large Cell')
        elif 'benign' in name.lower():
            short_names.append('Benign')
        elif 'normal' in name.lower():
            short_names.append('Normal')
        else:
            short_names.append(name[:10])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix - LungXAI Prototype', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    cm_path = os.path.join(config.results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Confusion matrix saved to: {cm_path}")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return metrics


if __name__ == "__main__":
    evaluate()
