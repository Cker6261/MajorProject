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
    python demo.py                              # Run demo with MobileNetV2 (default)
    python demo.py path/to/image.png            # Run demo on specific image
    python demo.py --model resnet50             # Use ResNet-50 model
    python demo.py --model swin_t               # Use Swin Transformer model
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
    # Create figure - fill the page properly
    fig = plt.figure(figsize=(22, 12))
    
    # Main title at top
    fig.suptitle('LungXAI: Explainable Lung Cancer Classification', fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid: Row 1 for images (3 cols) + chart (wider), Row 2 for explanation
    gs = fig.add_gridspec(2, 5, height_ratios=[0.45, 0.55], hspace=0.12, wspace=0.08,
                          left=0.01, right=0.99, top=0.93, bottom=0.02,
                          width_ratios=[1, 1, 1, 0.25, 1.5])
    
    # =========================================================================
    # Row 1: Images, Prediction, and Probabilities
    # =========================================================================
    
    # Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(result.original_image)
    ax1.set_title('Original CT Scan', fontsize=11, fontweight='bold', pad=5)
    ax1.axis('off')
    
    # Grad-CAM Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(result.heatmap, cmap='jet')
    ax2.set_title('Grad-CAM Heatmap', fontsize=11, fontweight='bold', pad=5)
    ax2.axis('off')
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    
    # Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(result.overlay)
    ax3.set_title('Attention Overlay', fontsize=11, fontweight='bold', pad=5)
    ax3.axis('off')
    
    # Empty spacer column (gs[0, 3]) - creates gap between overlay and chart
    
    # Combined: Prediction + Probabilities (wider column)
    ax4 = fig.add_subplot(gs[0, 4])
    pred_class_display = result.predicted_class.replace('_', ' ').title()
    conf_color = '#2e7d32' if result.confidence > 0.8 else '#f57c00' if result.confidence > 0.5 else '#c62828'
    
    classes = list(result.all_probabilities.keys())
    probs = list(result.all_probabilities.values())
    # Full class names - no truncation
    display_names = [c.replace('_', ' ').title() for c in classes]
    colors = [conf_color if c == result.predicted_class else '#bbdefb' for c in classes]
    y_pos = np.arange(len(display_names))
    
    bars = ax4.barh(y_pos, [p * 100 for p in probs], color=colors, edgecolor='white', height=0.55)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(display_names, fontsize=9)
    ax4.set_xlabel('Confidence (%)', fontsize=9)
    ax4.set_title(f'Prediction: {pred_class_display} ({result.confidence*100:.1f}%)', 
                  fontsize=11, fontweight='bold', pad=6, color=conf_color)
    ax4.set_xlim(0, 110)
    ax4.invert_yaxis()
    for bar, prob in zip(bars, probs):
        if prob > 0.01:
            ax4.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.0f}%', va='center', fontsize=8, fontweight='bold')
    ax4.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax4.set_axisbelow(True)
    ax4.tick_params(axis='y', pad=5)
    
    # =========================================================================
    # Row 2: AI-Generated Explanation (3 boxes: Visual | Medical | References)
    # =========================================================================
    
    # Create a separate gridspec for bottom row with better proportions
    gs_bottom = fig.add_gridspec(1, 3, left=0.02, right=0.98, top=0.47, bottom=0.02, 
                                  wspace=0.02, width_ratios=[0.85, 1.3, 0.85])
    
    # Get explanation components
    explanation = result.explanation
    references = explanation.sources if hasattr(explanation, 'sources') and explanation.sources else []
    
    # ===== VISUAL EVIDENCE =====
    ax_visual = fig.add_subplot(gs_bottom[0, 0])
    ax_visual.set_facecolor('#e3f2fd')
    ax_visual.set_xlim(0, 1)
    ax_visual.set_ylim(0, 1)
    
    # Header
    ax_visual.fill_between([0, 1], [1, 1], [0.88, 0.88], color='#1976d2', alpha=0.9)
    ax_visual.text(0.5, 0.94, '🔍 VISUAL EVIDENCE', fontsize=14, fontweight='bold', 
                   color='white', ha='center', va='center')
    
    # Content - left aligned with larger font
    visual_text = wrap_text(explanation.visual_evidence, max_width=35)
    ax_visual.text(0.05, 0.82, visual_text, fontsize=13, va='top', ha='left',
                   wrap=True, linespacing=1.5)
    
    ax_visual.set_xticks([])
    ax_visual.set_yticks([])
    for spine in ax_visual.spines.values():
        spine.set_color('#1976d2')
        spine.set_linewidth(2)
    
    # ===== MEDICAL CONTEXT =====
    ax_medical = fig.add_subplot(gs_bottom[0, 1])
    ax_medical.set_facecolor('#fff8e1')
    ax_medical.set_xlim(0, 1)
    ax_medical.set_ylim(0, 1)
    
    # Header
    ax_medical.fill_between([0, 1], [1, 1], [0.88, 0.88], color='#f57c00', alpha=0.9)
    ax_medical.text(0.5, 0.94, '📋 MEDICAL CONTEXT', fontsize=14, fontweight='bold',
                    color='white', ha='center', va='center')
    
    # Parse medical context - separate KB and PubMed clearly
    medical_raw = explanation.medical_context
    
    # Extract KB section
    kb_text = ""
    pubmed_text = ""
    
    if '【 CLINICAL KNOWLEDGE BASE 】' in medical_raw:
        parts = medical_raw.split('【 RECENT RESEARCH (PubMed) 】')
        kb_part = parts[0].replace('【 CLINICAL KNOWLEDGE BASE 】', '').strip()
        kb_part = kb_part.replace('─' * 40, '').strip()
        kb_text = kb_part
        if len(parts) > 1:
            pubmed_text = parts[1].replace('─' * 40, '').strip()
    else:
        kb_text = medical_raw
    
    # Knowledge Base section
    ax_medical.text(0.02, 0.84, '📚 KNOWLEDGE BASE:', fontsize=13, fontweight='bold', 
                    va='top', ha='left', color='#e65100')
    kb_wrapped = wrap_text(kb_text, max_width=65)
    ax_medical.text(0.02, 0.75, kb_wrapped, fontsize=12, va='top', ha='left', linespacing=1.4)
    
    # Separator line
    ax_medical.axhline(y=0.42, xmin=0.02, xmax=0.98, color='#ffb74d', linewidth=3, linestyle='-')
    
    # PubMed section
    ax_medical.text(0.02, 0.38, '📖 PUBMED RESEARCH:', fontsize=13, fontweight='bold',
                    va='top', ha='left', color='#e65100')
    if pubmed_text:
        pubmed_wrapped = wrap_text(pubmed_text, max_width=65)
        ax_medical.text(0.02, 0.29, pubmed_wrapped, fontsize=12, va='top', ha='left', linespacing=1.4)
    else:
        ax_medical.text(0.02, 0.29, 'No recent PubMed articles found.', fontsize=12, 
                        va='top', ha='left', style='italic', color='#666666')
    
    ax_medical.set_xticks([])
    ax_medical.set_yticks([])
    for spine in ax_medical.spines.values():
        spine.set_color('#f57c00')
        spine.set_linewidth(2)
    
    # ===== REFERENCES =====
    ax_refs = fig.add_subplot(gs_bottom[0, 2])
    ax_refs.set_facecolor('#f3e5f5')
    ax_refs.set_xlim(0, 1)
    ax_refs.set_ylim(0, 1)
    
    # Header
    ax_refs.fill_between([0, 1], [1, 1], [0.88, 0.88], color='#7b1fa2', alpha=0.9)
    ax_refs.text(0.5, 0.94, '📚 REFERENCES', fontsize=14, fontweight='bold',
                 color='white', ha='center', va='center')
    
    ax_refs.text(0.5, 0.84, '(Latest Year First)', fontsize=10, ha='center', va='top', 
                 style='italic', color='#444444')
    
    # Reference list - extract year, sort by year, and display properly
    import re
    y_ref_pos = 0.76
    
    if references:
        # Parse references to extract year for each
        refs_with_years = []
        for ref in references:
            # Remove existing [X] prefix if present
            ref_clean = re.sub(r'^\[\d+\]\s*', '', ref)
            # Extract year (look for 4-digit year)
            year_match = re.search(r'(20\d{2}|19\d{2})', ref_clean)
            year = year_match.group(1) if year_match else '2000'
            refs_with_years.append((ref_clean, year))
        
        # Sort by year descending (latest first)
        refs_with_years.sort(key=lambda x: x[1], reverse=True)
        
        # Display up to 4 references
        for i, (ref_text, year) in enumerate(refs_with_years[:4]):
            ref_display = wrap_text(f"[{i+1}] {ref_text}", max_width=34)
            ax_refs.text(0.03, y_ref_pos, ref_display, fontsize=11, va='top', ha='left', linespacing=1.3)
            y_ref_pos -= 0.17
    else:
        ax_refs.text(0.03, y_ref_pos, "[1] Medical Knowledge Base", fontsize=11, va='top', ha='left')
    
    ax_refs.set_xticks([])
    ax_refs.set_yticks([])
    for spine in ax_refs.spines.values():
        spine.set_color('#7b1fa2')
        spine.set_linewidth(2)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Visualization saved to: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visual Demo - Explainable Lung Cancer Classification')
    parser.add_argument('image_path', nargs='?', help='Path to CT scan image (optional)')
    parser.add_argument('--save', type=str, help='Save visualization to file')
    parser.add_argument('--no-display', action='store_true', help='Do not display the plot')
    parser.add_argument('--model', type=str, default='mobilenetv2', 
                        choices=['resnet50', 'mobilenetv2', 'vit_b_16', 'swin_t'],
                        help='Model to use for demo (default: mobilenetv2)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXPLAINABLE LUNG CANCER CLASSIFICATION - VISUAL DEMO")
    print("=" * 60)
    
    # Get device
    device = get_device()
    
    # Load config
    config = Config()
    config.model_name = args.model
    
    # Initialize pipeline using model-specific checkpoint
    print(f"\nInitializing pipeline with {config.model_display_names.get(args.model, args.model)}...")
    
    # Check if model is trained
    if not config.is_model_trained(args.model):
        print(f"⚠ Model '{args.model}' not found in cache.")
        print(f"  Checkpoint path: {config.get_model_checkpoint_path(args.model, 'best')}")
        print(f"\n  To train this model, run:")
        print(f"    python train_all_models.py --models {args.model}")
        return
    
    pipeline = ExplainablePipeline(
        model_name=args.model,
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
