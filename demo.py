# =============================================================================
# VISUAL DEMO - Explainable Lung Cancer Classification
# Run this to see visual predictions with Grad-CAM heatmaps
# =============================================================================
"""
Visual Demo Script for Explainable Lung Cancer Classification.

This script demonstrates the full pipeline with:
    - Model prediction
    - Grad-CAM heatmap visualization
    - RAG-based medical explanation
    - Side-by-side visual comparison

Usage:
    python demo.py                    # Run demo on random test image
    python demo.py path/to/image.png  # Run demo on specific image
"""

import os
import sys
import random
import argparse
from pathlib import Path

# Force cache to D: drive
os.environ['TORCH_HOME'] = r'd:\Major Project\.cache\torch'

sys.path.insert(0, r'd:\Major Project')

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.config import Config
from src.pipeline import ExplainablePipeline


def get_device():
    """Get the best available device (GPU if available)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠ GPU not available, using CPU")
    return device


def get_random_test_image(config: Config) -> str:
    """Get a random image from the dataset."""
    dataset_path = Path(config.dataset_dir)
    all_images = []
    
    for class_name in config.class_names:
        class_dir = dataset_path / class_name
        if class_dir.exists():
            images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            all_images.extend(images)
    
    return str(random.choice(all_images))


def visualize_prediction(result, save_path: str = None):
    """
    Create a comprehensive visualization of the prediction.
    
    Shows:
        - Original image
        - Grad-CAM heatmap
        - Overlay
        - Explanation text
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 0.8], hspace=0.3, wspace=0.25)
    
    # =========================================================================
    # Row 1: Images
    # =========================================================================
    
    # Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(result.original_image)
    ax1.set_title('Original CT Scan', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Grad-CAM Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(result.heatmap, cmap='jet')
    ax2.set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(result.overlay)
    ax3.set_title('Attention Overlay', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # =========================================================================
    # Row 2: Prediction and Explanation
    # =========================================================================
    
    # Prediction Bar Chart
    ax4 = fig.add_subplot(gs[1, 0])
    classes = list(result.all_probabilities.keys())
    probs = list(result.all_probabilities.values())
    
    # Clean class names for display
    display_names = [c.replace('_', ' ').replace('cell ', '\ncell ') for c in classes]
    
    colors = ['#e74c3c' if c == result.predicted_class else '#3498db' for c in classes]
    bars = ax4.barh(display_names, [p * 100 for p in probs], color=colors)
    ax4.set_xlabel('Confidence (%)', fontsize=11)
    ax4.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, 100)
    
    # Add percentage labels
    for bar, prob in zip(bars, probs):
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', fontsize=9)
    
    # Explanation Text
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')
    
    # Format explanation
    pred_text = f"PREDICTION: {result.predicted_class.replace('_', ' ').title()}"
    conf_text = f"CONFIDENCE: {result.confidence*100:.1f}%"
    
    # Get visual and medical explanation
    explanation = result.explanation
    visual_text = explanation.visual_evidence[:300] + "..." if len(explanation.visual_evidence) > 300 else explanation.visual_evidence
    medical_text = explanation.medical_context[:400] + "..." if len(explanation.medical_context) > 400 else explanation.medical_context
    
    full_text = f"""
{pred_text}
{conf_text}

VISUAL EVIDENCE:
{visual_text}

MEDICAL CONTEXT:
{medical_text}
"""
    
    ax5.text(0.02, 0.98, full_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    ax5.set_title('AI Explanation', fontsize=14, fontweight='bold')
    
    # Main title
    fig.suptitle('Explainable Lung Cancer Classification', fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Visualization saved to: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visual Demo - Explainable Lung Cancer Classification')
    parser.add_argument('image_path', nargs='?', help='Path to CT scan image (optional)')
    parser.add_argument('--save', type=str, help='Save visualization to file')
    parser.add_argument('--no-display', action='store_true', help='Do not display the plot')
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXPLAINABLE LUNG CANCER CLASSIFICATION - VISUAL DEMO")
    print("=" * 60)
    
    # Get device
    device = get_device()
    
    # Load config
    config = Config()
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ No trained model found at {checkpoint_path}")
        print("  Please train the model first: python main.py train")
        return
    
    pipeline = ExplainablePipeline(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device
    )
    
    # Get image path
    if args.image_path:
        image_path = args.image_path
    else:
        print("\nSelecting random test image...")
        image_path = get_random_test_image(config)
    
    print(f"Image: {image_path}")
    
    # Run prediction
    print("\nRunning prediction...")
    result = pipeline.predict(image_path)
    
    print(f"\n{'=' * 60}")
    print(f"PREDICTION: {result.predicted_class.replace('_', ' ').title()}")
    print(f"CONFIDENCE: {result.confidence*100:.1f}%")
    print("=" * 60)
    
    # Create visualization
    print("\nGenerating visualization...")
    save_path = args.save or os.path.join(config.results_dir, "demo_output.png")
    fig = visualize_prediction(result, save_path=save_path)
    
    if not args.no_display:
        plt.show()
    
    print("\n✓ Demo completed!")


if __name__ == "__main__":
    main()
