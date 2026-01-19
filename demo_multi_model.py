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
    python demo_multi_model.py                          # Demo with default (resnet50)
    python demo_multi_model.py --model mobilenetv2      # Use specific model
    python demo_multi_model.py --model vit_b_16         # Use Vision Transformer
    python demo_multi_model.py --compare                # Compare all models

All data is stored on D: drive to prevent C: drive issues.
"""

import os
import sys
import argparse
import gc
from pathlib import Path
from datetime import datetime

# Force cache to D: drive
os.environ['TORCH_HOME'] = r'D:\Major Project\.cache\torch'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.pipeline import ExplainablePipeline
from src.utils.config import Config


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_sample_image(config: Config) -> str:
    """Find a sample image from the dataset."""
    for class_name in config.class_names:
        class_dir = os.path.join(config.dataset_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                return os.path.join(class_dir, images[0])
    raise FileNotFoundError("No sample images found in dataset")


def demo_single_model(model_name: str = "resnet50", image_path: str = None):
    """
    Run demo with a single model.
    
    Args:
        model_name: Name of the model to use
        image_path: Optional path to image file
    """
    print("\n" + "=" * 70)
    print(f"LUNG CANCER CLASSIFICATION DEMO - {model_name.upper()}")
    print("=" * 70)
    
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
    print(f"\nInitializing pipeline with {model_name}...")
    pipeline = ExplainablePipeline(model_name=model_name, config=config)
    
    # Find sample image
    if image_path and os.path.exists(image_path):
        sample_image = image_path
    else:
        sample_image = find_sample_image(config)
    print(f"\nUsing sample image: {sample_image}")
    
    # Run prediction
    print("\nRunning prediction...")
    results_dir = config.get_model_results_path(model_name)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run prediction first without saving
    result = pipeline.predict(
        sample_image,
        show_visualization=False,
        save_visualization=None
    )
    
    # Create descriptive filename: model_class_timestamp.png
    predicted_class_clean = result.predicted_class.replace(' ', '_').lower()
    output_filename = f"{model_name}_{predicted_class_clean}_{timestamp}.png"
    output_path = os.path.join(results_dir, output_filename)
    
    # Save the visualization using the result's show_visualization method
    result.show_visualization(save_path=output_path)
    
    # Print results
    print("\n" + "-" * 50)
    print("PREDICTION RESULTS")
    print("-" * 50)
    print(f"Model: {config.model_display_names.get(model_name, model_name)}")
    print(f"Predicted Class: {result.predicted_class}")
    print(f"Confidence: {result.confidence*100:.2f}%")
    print("\nAll Probabilities:")
    for class_name, prob in sorted(result.all_probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {prob*100:.2f}%")
    
    print("\n" + "-" * 50)
    print("EXPLANATION")
    print("-" * 50)
    result.print_explanation()
    
    # Save results
    print(f"\n✓ Visualization saved to {output_path}")
    
    # Clear memory
    del pipeline
    clear_memory()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


def demo_compare_models(image_path: str = None):
    """
    Run demo comparing all available models on the same image.
    """
    print("\n" + "=" * 70)
    print("LUNG CANCER CLASSIFICATION - MODEL COMPARISON DEMO")
    print("=" * 70)
    
    config = Config()
    
    # Find sample image
    if image_path and os.path.exists(image_path):
        sample_image = image_path
    else:
        sample_image = find_sample_image(config)
    print(f"\nUsing sample image: {sample_image}")
    
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
    
    # Test each model
    for model_name in available_models:
        print(f"\n--- Testing {config.model_display_names.get(model_name, model_name)} ---")
        
        try:
            pipeline = ExplainablePipeline(model_name=model_name, config=config)
            result = pipeline.predict(sample_image, show_visualization=False)
            
            results[model_name] = {
                'predicted_class': result.predicted_class,
                'confidence': result.confidence,
                'display_name': config.model_display_names.get(model_name, model_name)
            }
            
            print(f"  Prediction: {result.predicted_class} ({result.confidence*100:.1f}%)")
            
            # Clear memory between models
            del pipeline
            clear_memory()
                
        except Exception as e:
            print(f"  Error: {e}")
            clear_memory()
    
    # Print comparison summary
    if results:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\n{'Model':<35} {'Prediction':<25} {'Confidence':<12}")
        print("-" * 72)
        for model_name, data in results.items():
            print(f"{data['display_name']:<35} {data['predicted_class']:<25} {data['confidence']*100:.1f}%")
        
        print("\n" + "=" * 70)
        print("COMPARISON COMPLETE")
        print("=" * 70)


def list_available_models():
    """List all models and their training status."""
    config = Config()
    
    print("\n" + "=" * 70)
    print("AVAILABLE MODELS")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Display Name':<35} {'Status':<15}")
    print("-" * 75)
    
    for model_name in config.available_models:
        display_name = config.model_display_names.get(model_name, model_name)
        if config.is_model_trained(model_name):
            status = "✓ Trained"
        else:
            status = "✗ Not trained"
        print(f"{model_name:<25} {display_name:<35} {status:<15}")
    
    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Model Demo for Lung Cancer Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo_multi_model.py                          # Default model (resnet50)
    python demo_multi_model.py --model mobilenetv2      # Use MobileNetV2
    python demo_multi_model.py --model vit_b_16         # Use Vision Transformer
    python demo_multi_model.py --model swin_t           # Use Swin Transformer
    python demo_multi_model.py --compare                # Compare all models
    python demo_multi_model.py --list                   # List available models
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
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
