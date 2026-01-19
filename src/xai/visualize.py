# =============================================================================
# XAI VISUALIZATION
# Utilities for visualizing Grad-CAM heatmaps and explanations
# =============================================================================
"""
Visualization utilities for Explainable AI.

This module provides:
    - Heatmap overlay on original images
    - Side-by-side visualization (original, heatmap, overlay)
    - Batch visualization
    - Saving visualizations to files

COLOR MAPS:
    We use the 'jet' colormap which goes from:
    - Blue (low activation) → Green → Yellow → Red (high activation)
    
    This is standard in medical imaging XAI literature.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import textwrap
from typing import Optional, Tuple, List
from pathlib import Path

import torch


def _wrap_text(text: str, width: int = 80) -> str:
    """Wrap text to specified width."""
    lines = text.split('\n')
    wrapped_lines = []
    for line in lines:
        if len(line) > width:
            wrapped_lines.extend(textwrap.wrap(line, width=width))
        else:
            wrapped_lines.append(line)
    return '\n'.join(wrapped_lines)


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Create an overlay of Grad-CAM heatmap on the original image.
    
    Args:
        image: Original image [H, W, 3] in range [0, 255] or [0, 1]
        heatmap: Grad-CAM heatmap [H, W] in range [0, 1]
        alpha: Transparency of the heatmap (0 = only image, 1 = only heatmap)
        colormap: OpenCV colormap to use
    
    Returns:
        Overlay image [H, W, 3] in range [0, 255]
    """
    # Ensure image is in [0, 255] range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Ensure image is 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize heatmap to match image size if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def visualize_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    predicted_class: str,
    confidence: float,
    true_class: Optional[str] = None,
    alpha: float = 0.4,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
    explanation: Optional[object] = None
) -> plt.Figure:
    """
    Create a comprehensive Grad-CAM visualization.
    
    Shows three panels:
        1. Original image
        2. Grad-CAM heatmap
        3. Overlay (heatmap on image)
    
    Optionally includes explanation text panel.
    
    Args:
        image: Original image [H, W, 3]
        heatmap: Grad-CAM heatmap [H, W]
        predicted_class: Name of predicted class
        confidence: Prediction confidence (0-1)
        true_class: Optional ground truth class name
        alpha: Heatmap transparency for overlay
        figsize: Figure size
        save_path: Optional path to save the figure
        explanation: Optional Explanation object with medical context
    
    Returns:
        matplotlib Figure object
    """
    # Determine layout based on whether we have an explanation
    if explanation is not None:
        # Create figure with 2 rows: images on top, explanation on bottom
        # Increased height for text panel to prevent citation cutoff
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.4, wspace=0.2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax_text = fig.add_subplot(gs[1, :])
    else:
        # Original layout: just 3 images
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        ax1, ax2, ax3 = axes
        ax_text = None
    
    # Ensure image is displayable
    if image.max() <= 1.0:
        display_image = image
    else:
        display_image = image / 255.0
    
    # Panel 1: Original image
    ax1.imshow(display_image)
    ax1.set_title('Original CT Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Heatmap
    im = ax2.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    ax2.set_title('Grad-CAM Attention Map', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Panel 3: Overlay
    overlay = create_heatmap_overlay(
        (display_image * 255).astype(np.uint8) if display_image.max() <= 1.0 else display_image,
        heatmap,
        alpha=alpha
    )
    ax3.imshow(overlay)
    ax3.set_title('Overlay (Model Focus)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Add prediction info as main title
    class_display = predicted_class.replace('_', ' ').title()
    title = f"Prediction: {class_display} ({confidence*100:.1f}%)"
    if true_class is not None:
        true_display = true_class.replace('_', ' ').title()
        title += f"  |  Ground Truth: {true_display}"
        if predicted_class.lower() == true_class.lower():
            title += " ✓"
        else:
            title += " ✗"
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Panel 4: Explanation text (if provided)
    if ax_text is not None and explanation is not None:
        ax_text.axis('off')
        
        # Build explanation text
        text_parts = []
        
        # Visual evidence
        text_parts.append("=" * 80)
        text_parts.append("VISUAL EVIDENCE (What the model focused on):")
        text_parts.append("=" * 80)
        visual_evidence = getattr(explanation, 'visual_evidence', '')
        if visual_evidence:
            # Wrap text for display
            wrapped = _wrap_text(visual_evidence, 100)
            text_parts.append(wrapped)
        
        text_parts.append("")
        
        # Medical context
        text_parts.append("=" * 80)
        text_parts.append("MEDICAL CONTEXT (Retrieved knowledge):")
        text_parts.append("=" * 80)
        medical_context = getattr(explanation, 'medical_context', '')
        if medical_context:
            # Truncate if too long
            if len(medical_context) > 600:
                medical_context = medical_context[:600] + "..."
            wrapped = _wrap_text(medical_context, 100)
            text_parts.append(wrapped)
        
        text_parts.append("")
        
        # Sources
        sources = getattr(explanation, 'sources', [])
        if sources:
            text_parts.append("=" * 80)
            text_parts.append("SOURCES & CITATIONS:")
            text_parts.append("=" * 80)
            for i, source in enumerate(sources[:4], 1):  # Limit to 4 sources
                # Truncate long sources
                if len(source) > 80:
                    source = source[:77] + "..."
                text_parts.append(f"  [{i}] {source}")
        
        full_text = '\n'.join(text_parts)
        
        ax_text.text(
            0.02, 0.98, full_text,
            transform=ax_text.transAxes,
            fontsize=9,
            fontfamily='monospace',
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.9)
        )
    
    # Adjust layout with more bottom margin for text
    if explanation is not None:
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.3)
        print(f"✓ Visualization saved to {save_path}")
    
    return fig


def visualize_batch_gradcam(
    images: List[np.ndarray],
    heatmaps: List[np.ndarray],
    predictions: List[str],
    confidences: List[float],
    true_classes: Optional[List[str]] = None,
    cols: int = 4,
    figsize_per_image: Tuple[float, float] = (4, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize Grad-CAM for multiple images in a grid.
    
    Args:
        images: List of original images
        heatmaps: List of Grad-CAM heatmaps
        predictions: List of predicted class names
        confidences: List of confidence scores
        true_classes: Optional list of ground truth class names
        cols: Number of columns in grid
        figsize_per_image: Size per image in the grid
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    figsize = (figsize_per_image[0] * cols, figsize_per_image[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx in range(n_images):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        
        # Create overlay
        image = images[idx]
        heatmap = heatmaps[idx]
        
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        overlay = create_heatmap_overlay(image_uint8, heatmap, alpha=0.4)
        
        ax.imshow(overlay)
        
        # Title with prediction
        title = f"{predictions[idx]}\n{confidences[idx]*100:.1f}%"
        if true_classes is not None:
            correct = predictions[idx] == true_classes[idx]
            title += f"\n{'✓' if correct else '✗'}"
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Batch visualization saved to {save_path}")
    
    return fig


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a numpy image for visualization.
    
    Handles denormalization if the tensor was normalized with ImageNet stats.
    
    Args:
        tensor: Image tensor [C, H, W] or [B, C, H, W]
    
    Returns:
        Numpy array [H, W, 3] in range [0, 1]
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # ImageNet normalization values
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize
    tensor = tensor.cpu().clone()
    tensor = tensor * std + mean
    
    # Clip to valid range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and rearrange dimensions
    image = tensor.permute(1, 2, 0).numpy()
    
    return image


def save_gradcam_visualization(
    image: np.ndarray,
    heatmap: np.ndarray,
    output_dir: str,
    filename: str,
    predicted_class: str,
    confidence: float
) -> None:
    """
    Save Grad-CAM visualization components to files.
    
    Saves:
        - Original image
        - Heatmap
        - Overlay
    
    Args:
        image: Original image
        heatmap: Grad-CAM heatmap
        output_dir: Output directory
        filename: Base filename (without extension)
        predicted_class: Predicted class name
        confidence: Confidence score
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original
    plt.figure(figsize=(6, 6))
    plt.imshow(image if image.max() <= 1 else image / 255)
    plt.axis('off')
    plt.savefig(output_dir / f"{filename}_original.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(output_dir / f"{filename}_heatmap.png", bbox_inches='tight')
    plt.close()
    
    # Save overlay
    overlay = create_heatmap_overlay(
        (image * 255).astype(np.uint8) if image.max() <= 1 else image,
        heatmap
    )
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(f"{predicted_class} ({confidence*100:.1f}%)")
    plt.axis('off')
    plt.savefig(output_dir / f"{filename}_overlay.png", bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved Grad-CAM visualization to {output_dir}")
