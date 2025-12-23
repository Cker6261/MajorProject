# =============================================================================
# XAI MODULE (EXPLAINABLE AI)
# Grad-CAM implementation for model interpretability
# =============================================================================
"""
Explainable AI Module for Lung Cancer Classification.

This module contains:
    - GradCAM: Gradient-weighted Class Activation Mapping
    - Visualization utilities for heatmaps
    - Attention region extraction

WHY SEPARATE XAI MODULE?
    - XAI is a core component of this project
    - Keeps explainability logic separate from model training
    - Easy to add other XAI methods (LIME, SHAP) in future

WHY GRAD-CAM?
    - Works directly with CNN architectures
    - No modification to model architecture required
    - Produces intuitive visual explanations
    - Well-documented in literature (Selvaraju et al., 2017)
"""

from .gradcam import GradCAM
from .visualize import create_heatmap_overlay, visualize_gradcam

__all__ = [
    'GradCAM',
    'create_heatmap_overlay',
    'visualize_gradcam'
]
