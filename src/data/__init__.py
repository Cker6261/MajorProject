# =============================================================================
# DATA MODULE
# Handles dataset loading, preprocessing, and augmentation
# =============================================================================
"""
Data Module for Lung Cancer CT Image Classification.

This module contains:
    - LungCancerDataset: Custom PyTorch Dataset class
    - Data transforms and augmentation pipelines
    - Train/Val/Test split utilities

WHY SEPARATE DATA MODULE?
    - Keeps data handling logic isolated from model logic
    - Makes it easy to swap datasets or modify preprocessing
    - Follows PyTorch best practices
"""

from .dataset import LungCancerDataset
from .transforms import get_train_transforms, get_val_transforms
from .dataloader import create_dataloaders

__all__ = [
    'LungCancerDataset',
    'get_train_transforms',
    'get_val_transforms',
    'create_dataloaders'
]
