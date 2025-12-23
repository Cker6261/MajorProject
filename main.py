# =============================================================================
# MAIN.PY - Entry Point for the Lung Cancer Classification System
# =============================================================================
"""
Explainable AI for Multi-Class Lung Cancer Classification
Using Deep Learning and RAG-Based Knowledge Retrieval

This is the main entry point for the project.
Run this file to execute the complete pipeline.

Usage:
    python main.py --mode train      # Train the model
    python main.py --mode evaluate   # Evaluate on test set
    python main.py --mode predict    # Run inference on a single image
    python main.py --mode demo       # Run the complete demo pipeline

Author: Major Project Team
Date: December 2024
"""

import argparse
import sys
import os

# Force PyTorch cache to D: drive to avoid C: drive space issues
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '.cache', 'torch')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import Config
from src.utils.helpers import set_seed, get_device


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Explainable AI for Lung Cancer Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --mode train --epochs 10
    python main.py --mode predict --image path/to/ct_scan.png
    python main.py --mode demo
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'predict', 'demo'],
        default='demo',
        help='Execution mode (default: demo)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to image for prediction mode'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    
    return parser.parse_args()


def run_training(config: Config, args, device):
    """Run the training pipeline."""
    from src.data.dataloader import create_dataloaders
    from src.models.model_factory import create_model, get_optimizer, get_loss_function, get_scheduler
    from src.training import train_model, evaluate_model, plot_training_history
    
    # Create data loaders
    print("\n📁 Loading dataset...")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        dataset_dir=config.dataset_dir,
        class_names=config.class_names,
        image_size=config.image_size,
        batch_size=args.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        num_workers=config.num_workers,
        random_seed=config.random_seed
    )
    
    # Create model
    print("\n🧠 Creating model...")
    model = create_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_base,
        device=device
    )
    
    # Setup training components
    criterion = get_loss_function()
    optimizer = get_optimizer(
        model,
        optimizer_name="adamw",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = get_scheduler(optimizer, scheduler_name="step", step_size=5)
    
    # Train the model
    print("\n🚀 Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        scheduler=scheduler,
        checkpoint_dir=config.checkpoint_dir,
        class_names=config.class_names
    )
    
    # Plot training curves
    plot_path = os.path.join(config.results_dir, "training_curves.png")
    plot_training_history(history, save_path=plot_path)
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=config.class_names
    )
    
    return results


def run_evaluation(config: Config, args, device):
    """Run evaluation on test set using a saved checkpoint."""
    from src.data.dataloader import create_dataloaders
    from src.models.model_factory import create_model, get_loss_function
    from src.training import evaluate_model
    from src.utils.helpers import load_checkpoint
    
    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first or provide a valid checkpoint path.")
        return
    
    # Create data loaders (only need test loader)
    print("\n📁 Loading dataset...")
    _, _, test_loader, dataset = create_dataloaders(
        dataset_dir=config.dataset_dir,
        class_names=config.class_names,
        image_size=config.image_size,
        batch_size=args.batch_size,
        num_workers=config.num_workers,
        random_seed=config.random_seed
    )
    
    # Create model and load checkpoint
    print("\n🧠 Loading model...")
    model = create_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=False,  # Will load weights from checkpoint
        device=device
    )
    load_checkpoint(checkpoint_path, model)
    
    # Evaluate
    criterion = get_loss_function()
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=config.class_names
    )
    
    return results


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize configuration
    config = Config()
    
    # Set random seed for reproducibility
    set_seed(config.random_seed)
    
    # Get device (GPU/CPU)
    device = get_device()
    
    print("\n" + "=" * 60)
    print("EXPLAINABLE AI FOR LUNG CANCER CLASSIFICATION")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print("=" * 60 + "\n")
    
    # Execute based on mode
    if args.mode == 'train':
        run_training(config, args, device)
        
    elif args.mode == 'evaluate':
        run_evaluation(config, args, device)
        
    elif args.mode == 'predict':
        run_prediction(config, args, device)
        
    elif args.mode == 'demo':
        run_demo(config, args, device)


def run_prediction(config: Config, args, device):
    """Run prediction on a single image."""
    from src.pipeline import ExplainablePipeline, create_demo_visualization
    
    if args.image is None:
        print("❌ Error: Please provide --image path for prediction mode")
        print("   Usage: python main.py --mode predict --image path/to/ct_scan.png")
        return
    
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found at {args.image}")
        return
    
    print(f"📷 Processing image: {args.image}")
    
    # Create pipeline
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    
    pipeline = ExplainablePipeline(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device
    )
    
    # Run prediction
    result = pipeline.predict(args.image)
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Prediction: {result.predicted_class.replace('_', ' ').title()}")
    print(f"Confidence: {result.confidence * 100:.2f}%")
    print("\nAll Probabilities:")
    for class_name, prob in sorted(result.all_probabilities.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {class_name:25s}: {prob*100:5.1f}% {bar}")
    print("=" * 60)
    
    # Show visualization
    save_path = os.path.join(config.results_dir, "prediction_result.png")
    fig = create_demo_visualization(result, save_path=save_path)
    
    # Print explanation
    result.print_explanation()
    
    import matplotlib.pyplot as plt
    plt.show()
    
    return result


def run_demo(config: Config, args, device):
    """Run demo pipeline."""
    from src.pipeline import ExplainablePipeline, create_demo_visualization
    
    print("🎯 DEMO MODE: Explainable Lung Cancer Classification")
    print("=" * 60)
    
    # Check for sample image
    sample_image = None
    
    # First try user-provided image
    if args.image:
        sample_image = args.image
    else:
        # Look for images in dataset
        from pathlib import Path
        for class_name in config.class_names:
            class_dir = Path(config.dataset_dir) / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.png")) + \
                        list(class_dir.glob("*.jpg")) + \
                        list(class_dir.glob("*.jpeg"))
                if images:
                    sample_image = str(images[0])
                    print(f"📂 Found sample image: {sample_image}")
                    break
    
    if sample_image is None:
        print("\n⚠️ No sample image found!")
        print("\nTo run the demo, either:")
        print("  1. Add images to the dataset/ folder")
        print("  2. Provide an image path: python main.py --mode demo --image path/to/ct.png")
        print("\n💡 Running in demonstration mode with synthetic data...\n")
        
        # Create a synthetic demo
        _run_synthetic_demo(config, device)
        return
    
    # Create pipeline
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    
    pipeline = ExplainablePipeline(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device
    )
    
    # Run prediction
    print(f"\n🔍 Analyzing: {sample_image}")
    result = pipeline.predict(sample_image)
    
    # Create comprehensive visualization
    save_path = os.path.join(config.results_dir, "demo_output.png")
    fig = create_demo_visualization(result, save_path=save_path)
    
    # Print explanation
    result.print_explanation()
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE")
    print("=" * 60)
    print(f"📊 Visualization saved to: {save_path}")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    plt.show()
    
    return result


def _run_synthetic_demo(config: Config, device):
    """Run a demo with synthetic data to show pipeline capabilities."""
    import numpy as np
    
    print("=" * 60)
    print("SYNTHETIC DEMO")
    print("=" * 60)
    print("\nThis demo shows the pipeline structure without real data.\n")
    
    # Show what the pipeline does
    print("📋 PIPELINE COMPONENTS:")
    print("-" * 60)
    print("1. IMAGE PREPROCESSING")
    print("   - Resize to 224x224")
    print("   - Normalize with ImageNet statistics")
    print()
    print("2. MODEL INFERENCE (ResNet-50)")
    print("   - Transfer learning from ImageNet")
    print("   - 4-class classification")
    print("   - Classes: adenocarcinoma, squamous_cell_carcinoma,")
    print("             large_cell_carcinoma, normal")
    print()
    print("3. GRAD-CAM VISUALIZATION")
    print("   - Target layer: layer4 (last conv block)")
    print("   - Generates attention heatmap")
    print("   - Shows where model is looking")
    print()
    print("4. RAG EXPLANATION GENERATION")
    print("   - Converts visual attention to text")
    print("   - Retrieves relevant medical knowledge")
    print("   - Generates evidence-based explanation")
    print()
    
    # Show example output
    print("-" * 60)
    print("📝 EXAMPLE OUTPUT:")
    print("-" * 60)
    print("""
============================================================
EXPLAINABLE AI ANALYSIS REPORT
============================================================

PREDICTION: Adenocarcinoma
CONFIDENCE: 87.3%

------------------------------------------------------------
VISUAL EVIDENCE (What the model focused on):
------------------------------------------------------------
Attention is primarily focused on the upper right region, 
with peripheral distribution pattern. Model shows high 
attention intensity, concentrated in specific areas.

------------------------------------------------------------
MEDICAL CONTEXT (Retrieved knowledge):
------------------------------------------------------------
Adenocarcinoma typically presents in the peripheral regions 
of the lung, often in the outer third of the lung parenchyma.
Ground-glass opacity (GGO) on CT imaging is frequently 
associated with adenocarcinoma.

------------------------------------------------------------
SOURCES:
------------------------------------------------------------
  [1] Travis WD et al., WHO Classification of Tumours, 2021
  [2] Hansell DM et al., Fleischner Society Glossary, 2008

============================================================
""")
    
    print("-" * 60)
    print("\n💡 To run with real data:")
    print("   1. Add CT scan images to dataset/ folder")
    print("   2. Train the model: python main.py --mode train")
    print("   3. Run demo: python main.py --mode demo")
    print()


if __name__ == "__main__":
    main()
