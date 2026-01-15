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


def wrap_text(text, max_width=80):
    """Wrap text to specified width without cutting words."""
    import textwrap
    return '\n'.join(textwrap.wrap(text, width=max_width))


def visualize_prediction(result, save_path: str = None):
    """
    Create a comprehensive visualization of the prediction.
    
    Shows:
        - Original image
        - Grad-CAM heatmap
        - Overlay
        - Explanation text (properly formatted, no truncation)
    """
    # Large figure for proper display
    fig = plt.figure(figsize=(22, 16))
    
    # Create grid: 2 rows for images/charts, then explanation below
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.2], hspace=0.3, wspace=0.3)
    
    # =========================================================================
    # Row 1: Images (3 images spread across 4 columns)
    # =========================================================================
    
    # Original Image
    ax1 = fig.add_subplot(gs[0, 0:1])
    ax1.imshow(result.original_image)
    ax1.set_title('Original CT Scan', fontsize=13, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # Grad-CAM Heatmap
    ax2 = fig.add_subplot(gs[0, 1:2])
    im = ax2.imshow(result.heatmap, cmap='jet')
    ax2.set_title('Grad-CAM Heatmap', fontsize=13, fontweight='bold', pad=10)
    ax2.axis('off')
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    
    # Overlay
    ax3 = fig.add_subplot(gs[0, 2:3])
    ax3.imshow(result.overlay)
    ax3.set_title('Attention Overlay', fontsize=13, fontweight='bold', pad=10)
    ax3.axis('off')
    
    # Prediction Result Box
    ax_pred = fig.add_subplot(gs[0, 3:4])
    ax_pred.axis('off')
    
    pred_class_display = result.predicted_class.replace('_', ' ').title()
    
    # Create a clean prediction display
    pred_box_text = f"PREDICTION RESULT\n\n"
    pred_box_text += f"Class:\n{pred_class_display}\n\n"
    pred_box_text += f"Confidence:\n{result.confidence*100:.1f}%"
    
    ax_pred.text(0.5, 0.5, pred_box_text, transform=ax_pred.transAxes, fontsize=14,
                 verticalalignment='center', horizontalalignment='center',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#e8f5e9', edgecolor='#4caf50', linewidth=3))
    
    # =========================================================================
    # Row 2: Bar Chart and Explanation (side by side)
    # =========================================================================
    
    # Prediction Bar Chart (left side)
    ax4 = fig.add_subplot(gs[1, 0:2])
    classes = list(result.all_probabilities.keys())
    probs = list(result.all_probabilities.values())
    
    # Full class names (no truncation)
    display_names = []
    for c in classes:
        name = c.replace('_', ' ').title()
        display_names.append(name)
    
    colors = ['#4caf50' if c == result.predicted_class else '#2196f3' for c in classes]
    y_pos = np.arange(len(display_names))
    
    bars = ax4.barh(y_pos, [p * 100 for p in probs], color=colors, edgecolor='white', linewidth=1, height=0.6)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(display_names, fontsize=11)
    ax4.set_xlabel('Confidence (%)', fontsize=12)
    ax4.set_title('Class Probabilities', fontsize=14, fontweight='bold', pad=15)
    ax4.set_xlim(0, 110)
    ax4.invert_yaxis()  # Highest probability on top
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probs):
        ax4.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    # Add grid for readability
    ax4.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax4.set_axisbelow(True)
    
    # =========================================================================
    # Explanation Text (right side - full text, no truncation)
    # =========================================================================
    
    ax5 = fig.add_subplot(gs[1, 2:4])
    ax5.axis('off')
    
    # Get explanation components
    explanation = result.explanation
    
    # Wrap text properly (no truncation, full content)
    visual_text = wrap_text(explanation.visual_evidence, max_width=55)
    medical_text = wrap_text(explanation.medical_context, max_width=55)
    
    # Get sources (full, no truncation)
    sources = explanation.sources if hasattr(explanation, 'sources') and explanation.sources else ["Medical Knowledge Base"]
    sources_text = wrap_text("; ".join(sources), max_width=55)
    
    # Build explanation sections
    explanation_content = ""
    explanation_content += "VISUAL EVIDENCE\n"
    explanation_content += "(What the model focused on)\n"
    explanation_content += "-" * 45 + "\n"
    explanation_content += visual_text + "\n\n"
    
    explanation_content += "MEDICAL CONTEXT\n"
    explanation_content += "(Why this matters clinically)\n"
    explanation_content += "-" * 45 + "\n"
    explanation_content += medical_text + "\n\n"
    
    explanation_content += "SOURCES\n"
    explanation_content += "-" * 45 + "\n"
    explanation_content += sources_text
    
    # Display explanation with proper formatting
    ax5.text(0.02, 0.98, explanation_content, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             fontfamily='sans-serif', linespacing=1.3,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fafafa', edgecolor='#bdbdbd', linewidth=1.5))
    ax5.set_title('AI-Generated Explanation', fontsize=14, fontweight='bold', pad=15)
    
    # Main title
    fig.suptitle('LungXAI: Explainable Lung Cancer Classification', fontsize=22, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05)
    
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
    
    # Create visualization and save to output folder
    print("\nGenerating visualization...")
    
    # Create output folder in project directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = args.save or os.path.join(output_dir, f"demo_output_{timestamp}.png")
    
    fig = visualize_prediction(result, save_path=save_path)
    
    print(f"\n{'=' * 60}")
    print(f"OUTPUT SAVED TO:")
    print(f"{save_path}")
    print(f"{'=' * 60}")
    
    # Auto-open the saved image
    import subprocess
    subprocess.Popen(['start', '', save_path], shell=True)
    print("\n✓ Opening output image...")
    
    # Close the matplotlib figure (don't show the small window)
    plt.close(fig)
    
    print("\n✓ Demo completed!")


if __name__ == "__main__":
    main()
