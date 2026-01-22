# =============================================================================
# XAI MODULE (EXPLAINABLE AI)
# Grad-CAM and Transformer interpretability for model explainability
# =============================================================================
"""
Explainable AI Module for Lung Cancer Classification.

This module contains:
    - GradCAM: Gradient-weighted Class Activation Mapping (for CNNs)
    - HighQualityTransformerXAI: Best XAI for transformers (occlusion-based)
    - Visualization utilities for heatmaps

WHICH METHOD TO USE?
    For CNNs (ResNet, MobileNet):
        → GradCAM - fast and produces excellent focused heatmaps
    
    For Transformers (ViT, Swin, DeiT):
        → HighQualityTransformerXAI / best_transformer_xai() - BEST quality
        → OcclusionSensitivity - alternative, slower
        → RISE - good quality, faster

Example usage:
    # For CNN
    from src.xai import GradCAM
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor, predicted_class)
    
    # For Transformer
    from src.xai.best_transformer_xai import best_transformer_xai
    heatmap = best_transformer_xai(model, input_tensor, predicted_class)
"""

from .gradcam import GradCAM
from .visualize import create_heatmap_overlay, visualize_gradcam

# Best Transformer XAI (RECOMMENDED - produces GradCAM-quality heatmaps)
try:
    from .best_transformer_xai import (
        HighQualityTransformerXAI,
        best_transformer_xai,
        GradientWeightedAttention,
    )
    BEST_TRANSFORMER_XAI_AVAILABLE = True
except ImportError:
    BEST_TRANSFORMER_XAI_AVAILABLE = False

# Localized XAI (alternative methods)
try:
    from .localized_xai import (
        OcclusionSensitivity,
        RISE,
        LocalizedXAI,
    )
    LOCALIZED_XAI_AVAILABLE = True
except ImportError:
    LOCALIZED_XAI_AVAILABLE = False

# Gradient-based Transformer XAI (faster but less focused)
try:
    from .transformer_interpretability import (
        GradientAttribution,
        SmoothGradient,
        InputXGradient,
        IntegratedGradientsV2,
        TransformerXAI,
    )
    TRANSFORMER_XAI_AVAILABLE = True
except ImportError:
    TRANSFORMER_XAI_AVAILABLE = False

__all__ = [
    'GradCAM',
    'create_heatmap_overlay',
    'visualize_gradcam',
    # Best Transformer XAI (RECOMMENDED)
    'HighQualityTransformerXAI',
    'best_transformer_xai',
    # Localized alternatives
    'LocalizedXAI',
    # Gradient-based
    'SmoothGradient',
    'IntegratedGradientsV2',
]
