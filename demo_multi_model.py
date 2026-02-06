# =============================================================================
# MULTI-MODEL DEMO SCRIPT
# Demonstrates the complete pipeline with multiple model support
# =============================================================================
"""
Multi-Model Demo Script for Explainable Lung Cancer Classification.

This script demonstrates:
    1. Loading trained models with caching support
    2. Running inference on sample images
    3. Comparing different model architectures
    4. Generating Grad-CAM visualizations for each model
    
USAGE:
    python demo_multi_model.py                          # Demo with default (mobilenetv2)
    python demo_multi_model.py --model resnet50         # Use ResNet-50
    python demo_multi_model.py --model vit_b_16         # Use Vision Transformer
    python demo_multi_model.py --compare                # Compare all models

All data is stored on D: drive to prevent C: drive issues.
"""

import os
import sys
import argparse
import gc
import random
import subprocess
import textwrap
from pathlib import Path
from datetime import datetime

# Force cache to D: drive
os.environ['TORCH_HOME'] = r'D:\Major Project\.cache\torch'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.pipeline import ExplainablePipeline
from src.utils.config import Config


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


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def wrap_text(text, max_width=80):
    """Wrap text to specified width without cutting words."""
    return '\n'.join(textwrap.wrap(text, width=max_width))


def visualize_prediction(result, model_name: str, save_path: str = None):
    """
    Create a comprehensive visualization of the prediction.
    
    Shows:
        - Original image
        - Grad-CAM heatmap
        - Overlay
        - Explanation text (properly formatted, no truncation)
    """
    # Large figure for proper display
    fig = plt.figure(figsize=(24, 14))
    
    # Create grid: Row 1 for images, Row 2 for bar chart (small) and explanation (large)
    gs = fig.add_gridspec(2, 6, height_ratios=[0.8, 1.2], hspace=0.35, wspace=0.25)
    
    # =========================================================================
    # Row 1: Images (4 panels)
    # =========================================================================
    
    # Original Image
    ax1 = fig.add_subplot(gs[0, 0:1])
    ax1.imshow(result.original_image)
    ax1.set_title('Original CT Scan', fontsize=12, fontweight='bold', pad=8)
    ax1.axis('off')
    
    # Grad-CAM Heatmap
    ax2 = fig.add_subplot(gs[0, 1:2])
    im = ax2.imshow(result.heatmap, cmap='jet')
    ax2.set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold', pad=8)
    ax2.axis('off')
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    
    # Overlay
    ax3 = fig.add_subplot(gs[0, 2:3])
    ax3.imshow(result.overlay)
    ax3.set_title('Attention Overlay', fontsize=12, fontweight='bold', pad=8)
    ax3.axis('off')
    
    # Prediction Result Box (compact)
    ax_pred = fig.add_subplot(gs[0, 3:4])
    ax_pred.axis('off')
    
    pred_class_display = result.predicted_class.replace('_', ' ').title()
    conf_color = '#4caf50' if result.confidence > 0.8 else '#ff9800' if result.confidence > 0.5 else '#f44336'
    
    pred_box_text = f"PREDICTION\n{model_name}\n\n{pred_class_display}\n\n{result.confidence*100:.1f}%"
    
    ax_pred.text(0.5, 0.5, pred_box_text, transform=ax_pred.transAxes, fontsize=12,
                 verticalalignment='center', horizontalalignment='center',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#e8f5e9', edgecolor=conf_color, linewidth=3))
    
    # Class Probabilities (compact bar chart in top right)
    ax4 = fig.add_subplot(gs[0, 4:6])
    classes = list(result.all_probabilities.keys())
    probs = list(result.all_probabilities.values())
    
    display_names = [c.replace('_', ' ').title()[:18] for c in classes]
    colors = ['#4caf50' if c == result.predicted_class else '#90caf9' for c in classes]
    y_pos = np.arange(len(display_names))
    
    bars = ax4.barh(y_pos, [p * 100 for p in probs], color=colors, edgecolor='white', height=0.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(display_names, fontsize=9)
    ax4.set_xlabel('Confidence (%)', fontsize=9)
    ax4.set_title('Class Probabilities', fontsize=11, fontweight='bold', pad=8)
    ax4.set_xlim(0, 105)
    ax4.invert_yaxis()
    
    for bar, prob in zip(bars, probs):
        if prob > 0.01:
            ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.1f}%', va='center', fontsize=8, fontweight='bold')
    ax4.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax4.set_axisbelow(True)
    
    # =========================================================================
    # Row 2: AI-Generated Explanation (FULL WIDTH)
    # =========================================================================
    
    ax5 = fig.add_subplot(gs[1, :])
    ax5.axis('off')
    
    # Get explanation components
    explanation = result.explanation
    references = explanation.sources if hasattr(explanation, 'sources') and explanation.sources else []
    
    # Build three-column layout text
    # Column 1: Visual Evidence
    visual_header = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    visual_header += "    🔍 VISUAL EVIDENCE (Model Attention)\n"
    visual_header += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    visual_content = wrap_text(explanation.visual_evidence, max_width=50)
    
    # Column 2: Medical Context (parsed properly)
    medical_header = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    medical_header += "    📋 MEDICAL CONTEXT\n"
    medical_header += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # Parse medical context into cleaner format for display
    medical_raw = explanation.medical_context
    medical_clean = medical_raw.replace('【 CLINICAL KNOWLEDGE BASE 】', '📚 KNOWLEDGE BASE:')
    medical_clean = medical_clean.replace('【 RECENT RESEARCH (PubMed) 】', '📖 PUBMED RESEARCH:')
    medical_clean = medical_clean.replace('─' * 40, '')
    lines = [line.strip() for line in medical_clean.split('\n') if line.strip()]
    medical_content = '\n'.join(lines)
    medical_content = wrap_text(medical_content, max_width=75)
    
    # Column 3: References
    ref_header = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    ref_header += "    📚 REFERENCES (Latest First)\n"
    ref_header += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    ref_content = ""
    if references:
        for ref in references[:4]:
            ref_text = ref if len(ref) <= 45 else ref[:42] + "..."
            ref_content += f"  {ref_text}\n\n"
    else:
        ref_content = "  [1] Medical Knowledge Base\n"
    
    # Position text in three columns
    ax5.text(0.02, 0.95, visual_header + visual_content, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             fontfamily='monospace', linespacing=1.4,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=1.5, alpha=0.9))
    
    ax5.text(0.35, 0.95, medical_header + medical_content, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             fontfamily='monospace', linespacing=1.4,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0', edgecolor='#f57c00', linewidth=1.5, alpha=0.9))
    
    ax5.text(0.75, 0.95, ref_header + ref_content, transform=ax5.transAxes, fontsize=8,
             verticalalignment='top', horizontalalignment='left',
             fontfamily='monospace', linespacing=1.4,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f3e5f5', edgecolor='#7b1fa2', linewidth=1.5, alpha=0.9))
    
    ax5.set_title('AI-Generated Explanation', fontsize=14, fontweight='bold', pad=15)
    
    # Main title
    fig.suptitle(f'LungXAI: Explainable Lung Cancer Classification ({model_name})', fontsize=20, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.03)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Visualization saved to: {save_path}")
    
    return fig


def find_sample_image(config: Config, random_select: bool = True) -> str:
    """Find a sample image from the dataset."""
    all_images = []
    for class_name in config.class_names:
        class_dir = os.path.join(config.dataset_dir, class_name)
        if os.path.exists(class_dir):
            images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
            all_images.extend(images)
    
    if not all_images:
        raise FileNotFoundError("No sample images found in dataset")
    
    if random_select:
        return random.choice(all_images)
    return all_images[0]


def demo_single_model(model_name: str = "mobilenetv2", image_path: str = None):
    """
    Run demo with a single model.
    
    Args:
        model_name: Name of the model to use
        image_path: Optional path to image file
    """
    print("=" * 60)
    print("EXPLAINABLE LUNG CANCER CLASSIFICATION - VISUAL DEMO")
    print("=" * 60)
    
    # Get device
    device = get_device()
    
    # Initialize configuration
    config = Config()
    config.model_name = model_name
    
    # Check if model is trained
    if not config.is_model_trained(model_name):
        print(f"\n⚠ Model '{model_name}' not found in cache.")
        print(f"  Checkpoint path: {config.get_model_checkpoint_path(model_name, 'best')}")
        print(f"\n  To train this model, run:")
        print(f"    python train_all_models.py --models {model_name}")
        print(f"\n  Or train all models:")
        print(f"    python train_all_models.py")
        return
    
    # Initialize pipeline
    print(f"\nInitializing pipeline with {config.model_display_names.get(model_name, model_name)}...")
    pipeline = ExplainablePipeline(model_name=model_name, config=config, device=device)
    
    # Find sample image
    if image_path and os.path.exists(image_path):
        sample_image = image_path
    else:
        print("\nSelecting random test image...")
        sample_image = find_sample_image(config, random_select=True)
    
    print(f"Image: {sample_image}")
    
    # Run prediction
    print("\nRunning prediction...")
    
    # Run prediction without showing/saving yet
    result = pipeline.predict(
        sample_image,
        show_visualization=False,
        save_visualization=None
    )
    
    print(f"\n{'=' * 60}")
    print(f"PREDICTION: {result.predicted_class.replace('_', ' ').title()}")
    print(f"CONFIDENCE: {result.confidence*100:.1f}%")
    print("=" * 60)
    
    # Print all probabilities
    print("\nAll Class Probabilities:")
    for class_name, prob in sorted(result.all_probabilities.items(), key=lambda x: x[1], reverse=True):
        marker = "→" if class_name == result.predicted_class else " "
        print(f"  {marker} {class_name}: {prob*100:.2f}%")
    
    # Print explanation
    print("\n" + "-" * 60)
    print("EXPLANATION")
    print("-" * 60)
    result.print_explanation()
    
    # Create output folder
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predicted_class_clean = result.predicted_class.replace(' ', '_').lower()
    output_filename = f"{model_name}_{predicted_class_clean}_{timestamp}.png"
    save_path = os.path.join(output_dir, output_filename)
    
    # Generate comprehensive visualization (like demo.py)
    print("\nGenerating visualization...")
    model_display = config.model_display_names.get(model_name, model_name)
    fig = visualize_prediction(result, model_display, save_path=save_path)
    
    print(f"\n{'=' * 60}")
    print(f"OUTPUT SAVED TO:")
    print(f"{save_path}")
    print(f"{'=' * 60}")
    
    # Auto-open the saved image
    subprocess.Popen(['start', '', save_path], shell=True)
    print("\n✓ Opening output image...")
    
    # Close the matplotlib figure (don't show the small window)
    plt.close(fig)
    
    # Clear memory
    del pipeline
    clear_memory()
    
    print("\n✓ Demo completed!")


def demo_compare_models(image_path: str = None):
    """
    Run demo comparing all available models on the same image.
    """
    print("=" * 60)
    print("LUNG CANCER CLASSIFICATION - MODEL COMPARISON DEMO")
    print("=" * 60)
    
    # Get device
    device = get_device()
    
    config = Config()
    
    # Find sample image
    if image_path and os.path.exists(image_path):
        sample_image = image_path
    else:
        print("\nSelecting random test image...")
        sample_image = find_sample_image(config, random_select=True)
    
    print(f"Image: {sample_image}")
    
    results = {}
    available_models = []
    
    # Check which models are trained
    print("\nChecking available models...")
    for model_name in config.available_models:
        if config.is_model_trained(model_name):
            available_models.append(model_name)
            print(f"  ✓ {model_name} - trained")
        else:
            print(f"  ✗ {model_name} - not trained")
    
    if not available_models:
        print("\n⚠ No trained models found!")
        print("  Run: python train_all_models.py")
        return
    
    # Create output folder
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test each model and generate visualizations
    for model_name in available_models:
        model_display = config.model_display_names.get(model_name, model_name)
        print(f"\n--- Testing {model_display} ---")
        
        try:
            pipeline = ExplainablePipeline(model_name=model_name, config=config, device=device)
            result = pipeline.predict(sample_image, show_visualization=False)
            
            results[model_name] = {
                'result': result,
                'predicted_class': result.predicted_class,
                'confidence': result.confidence,
                'display_name': model_display
            }
            
            print(f"  Prediction: {result.predicted_class} ({result.confidence*100:.1f}%)")
            
            # Generate and save visualization for each model
            predicted_class_clean = result.predicted_class.replace(' ', '_').lower()
            output_filename = f"compare_{model_name}_{predicted_class_clean}_{timestamp}.png"
            save_path = os.path.join(output_dir, output_filename)
            
            fig = visualize_prediction(result, model_display, save_path=save_path)
            plt.close(fig)
            
            # Clear memory between models
            del pipeline
            clear_memory()
                
        except Exception as e:
            print(f"  Error: {e}")
            clear_memory()
    
    # Print comparison summary
    if results:
        print(f"\n{'=' * 60}")
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"\n{'Model':<35} {'Prediction':<25} {'Confidence':<12}")
        print("-" * 72)
        for model_name, data in results.items():
            marker = "★" if data['confidence'] == max(r['confidence'] for r in results.values()) else " "
            print(f"{marker} {data['display_name']:<34} {data['predicted_class']:<25} {data['confidence']*100:.1f}%")
        
        print(f"\n{'=' * 60}")
        print(f"OUTPUT SAVED TO: {output_dir}")
        print("=" * 60)
        
        # Auto-open the output folder
        subprocess.Popen(['explorer', output_dir], shell=True)
        print("\n✓ Opening output folder...")
        
        print("\n✓ Comparison completed!")


def list_available_models():
    """List all models and their training status."""
    config = Config()
    
    print("=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Display Name':<35} {'Status':<15}")
    print("-" * 75)
    
    for model_name in config.available_models:
        display_name = config.model_display_names.get(model_name, model_name)
        if config.is_model_trained(model_name):
            status = "✓ Trained"
        else:
            status = "✗ Not trained"
        print(f"{model_name:<25} {display_name:<35} {status:<15}")
    
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Model Demo for Lung Cancer Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo_multi_model.py                          # Default model (mobilenetv2)
    python demo_multi_model.py --model resnet50         # Use ResNet-50
    python demo_multi_model.py --model vit_b_16         # Use Vision Transformer
    python demo_multi_model.py --model swin_t           # Use Swin Transformer
    python demo_multi_model.py --compare                # Compare all models
    python demo_multi_model.py --list                   # List available models
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenetv2",
        choices=["resnet50", "mobilenetv2", "vit_b_16", "swin_t"],
        help="Model to use for demo"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all available models"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models and their training status"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file (optional)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    elif args.compare:
        demo_compare_models(args.image)
    else:
        demo_single_model(args.model, args.image)


if __name__ == "__main__":
    main()
