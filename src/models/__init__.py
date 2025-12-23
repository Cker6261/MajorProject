# =============================================================================
# MODELS MODULE
# Neural network architectures for lung cancer classification
# =============================================================================
"""
Models Module for Lung Cancer Classification.

This module contains:
    - LungCancerClassifier: Main classification model (ResNet-50 based)
    - Model factory functions
    - Weight initialization utilities

WHY SEPARATE MODELS MODULE?
    - Allows easy swapping between different architectures
    - Keeps model code clean and focused
    - Easy to compare multiple models (future enhancement)
"""

from .classifier import LungCancerClassifier
from .model_factory import create_model

__all__ = [
    'LungCancerClassifier',
    'create_model'
]
