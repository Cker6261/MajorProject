# =============================================================================
# UTILS MODULE
# Helper functions, configuration, and metrics
# =============================================================================
"""
Utilities Module for Lung Cancer Classification Project.

This module contains:
    - config: Centralized configuration management
    - metrics: Evaluation metrics (accuracy, precision, recall, F1)
    - helpers: General utility functions
    - visualization: Plotting utilities

WHY SEPARATE UTILS MODULE?
    - Keeps common functionality in one place
    - Reduces code duplication
    - Makes main logic files cleaner
"""

from .config import Config
from .metrics import calculate_metrics, print_classification_report
from .helpers import set_seed, get_device, save_checkpoint, load_checkpoint

__all__ = [
    'Config',
    'calculate_metrics',
    'print_classification_report',
    'set_seed',
    'get_device',
    'save_checkpoint',
    'load_checkpoint'
]
