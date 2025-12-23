# =============================================================================
# METRICS MODULE
# Evaluation metrics for classification
# =============================================================================
"""
Metrics utilities for evaluating the lung cancer classification model.

WHY THESE METRICS?
    - Accuracy: Overall correctness (but can be misleading for imbalanced data)
    - Precision: How many predicted positives are actually positive
    - Recall: How many actual positives were correctly identified
    - F1 Score: Harmonic mean of precision and recall
    
For medical imaging, RECALL is particularly important because:
    - Missing a cancer case (false negative) is more dangerous
    - We want to minimize false negatives
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging strategy ('weighted', 'macro', 'micro')
    
    Returns:
        Dictionary containing accuracy, precision, recall, and F1 score
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> str:
    """
    Print a detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        The classification report as a string
    """
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        zero_division=0
    )
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    return report


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred)
