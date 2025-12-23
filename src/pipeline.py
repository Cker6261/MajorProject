# =============================================================================
# PIPELINE MODULE
# End-to-end inference pipeline with explainability
# =============================================================================
"""
Complete Inference Pipeline for Explainable Lung Cancer Classification.

This module brings together all components:
    1. Image preprocessing
    2. Model prediction
    3. Grad-CAM visualization
    4. RAG-based explanation generation

USAGE:
    from src.pipeline import ExplainablePipeline
    
    pipeline = ExplainablePipeline(checkpoint_path="checkpoints/best_model.pth")
    result = pipeline.predict("path/to/ct_scan.png")
    
    print(result.explanation)
    result.show_visualization()
"""

import os
from pathlib import Path
from typing import Optional, Union, Dict, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.models.model_factory import create_model
from src.data.transforms import get_inference_transforms, denormalize_image
from src.xai.gradcam import GradCAM
from src.xai.visualize import (
    create_heatmap_overlay, 
    visualize_gradcam, 
    tensor_to_numpy_image
)
from src.rag.explanation_generator import ExplanationGenerator, Explanation
from src.utils.config import Config
from src.utils.helpers import get_device


@dataclass
class PredictionResult:
    """
    Complete prediction result with all outputs.
    
    Attributes:
        image_path: Path to the input image
        predicted_class: Predicted cancer type
        confidence: Prediction confidence (0-1)
        all_probabilities: Probabilities for all classes
        heatmap: Grad-CAM heatmap
        overlay: Heatmap overlay on original image
        explanation: RAG-based explanation object
        original_image: Original image as numpy array
    """
    image_path: str
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    heatmap: np.ndarray
    overlay: np.ndarray
    explanation: Explanation
    original_image: np.ndarray
    
    def show_visualization(self, save_path: Optional[str] = None):
        """Display the Grad-CAM visualization."""
        fig = visualize_gradcam(
            image=self.original_image,
            heatmap=self.heatmap,
            predicted_class=self.predicted_class,
            confidence=self.confidence,
            save_path=save_path
        )
        plt.show()
        return fig
    
    def print_explanation(self):
        """Print the full explanation."""
        print(self.explanation.full_explanation)
    
    def get_short_summary(self) -> str:
        """Get a short summary of the prediction."""
        class_display = self.predicted_class.replace('_', ' ').title()
        return f"{class_display} ({self.confidence*100:.1f}%)"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for JSON serialization)."""
        return {
            'image_path': self.image_path,
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'all_probabilities': self.all_probabilities,
            'explanation': self.explanation.to_dict()
        }


class ExplainablePipeline:
    """
    End-to-end pipeline for explainable lung cancer classification.
    
    This class encapsulates the entire workflow:
        1. Load and preprocess image
        2. Run model inference
        3. Generate Grad-CAM heatmap
        4. Generate RAG-based explanation
        5. Create visualizations
    
    Example:
        >>> pipeline = ExplainablePipeline()
        >>> result = pipeline.predict("ct_scan.png")
        >>> result.show_visualization()
        >>> result.print_explanation()
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[Config] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the explainable pipeline.
        
        Args:
            checkpoint_path: Path to model checkpoint. If None, uses default.
            config: Configuration object. If None, uses default config.
            device: Device to run on. If None, auto-detects.
        """
        self.config = config or Config()
        self.device = device or get_device()
        
        # Initialize model
        self._init_model(checkpoint_path)
        
        # Initialize transforms
        self.transforms = get_inference_transforms(self.config.image_size)
        
        # Initialize Grad-CAM
        self._init_gradcam()
        
        # Initialize explanation generator
        self.explanation_generator = ExplanationGenerator()
        
        print("\n" + "=" * 60)
        print("EXPLAINABLE PIPELINE READY")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Device: {self.device}")
        print(f"Classes: {', '.join(self.config.class_names)}")
        print("=" * 60 + "\n")
    
    def _init_model(self, checkpoint_path: Optional[str]) -> None:
        """Initialize and load the model."""
        # Create model
        self.model = create_model(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes,
            pretrained=False,  # Will load weights from checkpoint
            device=self.device
        )
        
        # Load checkpoint if provided
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir, "best_model.pth"
            )
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"⚠ Warning: No checkpoint found at {checkpoint_path}")
            print("  Using model with random weights (for demo purposes)")
        
        self.model.eval()
    
    def _init_gradcam(self) -> None:
        """Initialize Grad-CAM with target layer."""
        target_layer = self.model.get_gradcam_target_layer()
        self.gradcam = GradCAM(self.model, target_layer)
        print("✓ Grad-CAM initialized")
    
    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        show_visualization: bool = False,
        save_visualization: Optional[str] = None
    ) -> PredictionResult:
        """
        Run complete prediction pipeline on an image.
        
        Args:
            image_input: Can be:
                - Path to image file (str or Path)
                - PIL Image
                - Numpy array [H, W, 3]
                - PyTorch tensor [C, H, W] or [B, C, H, W]
            show_visualization: Whether to display visualization
            save_visualization: Optional path to save visualization
        
        Returns:
            PredictionResult with all outputs
        """
        # Step 1: Load and preprocess image
        image_path, original_image, input_tensor = self._preprocess_image(image_input)
        
        # Step 2: Get prediction
        predicted_class, confidence, all_probs = self._get_prediction(input_tensor)
        
        # Step 3: Generate Grad-CAM heatmap
        heatmap = self.gradcam.generate(input_tensor, target_class=None)
        
        # Step 4: Create overlay
        overlay = create_heatmap_overlay(
            (original_image * 255).astype(np.uint8) if original_image.max() <= 1 else original_image,
            heatmap,
            alpha=0.4
        )
        
        # Step 5: Generate explanation
        explanation = self.explanation_generator.generate(
            heatmap=heatmap,
            predicted_class=predicted_class,
            confidence=confidence
        )
        
        # Create result object
        result = PredictionResult(
            image_path=str(image_path) if image_path else "unknown",
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probs,
            heatmap=heatmap,
            overlay=overlay,
            explanation=explanation,
            original_image=original_image
        )
        
        # Optionally show/save visualization
        if show_visualization or save_visualization:
            result.show_visualization(save_path=save_visualization)
        
        return result
    
    def _preprocess_image(
        self, 
        image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]
    ) -> tuple:
        """
        Preprocess image input to tensor.
        
        Returns:
            Tuple of (image_path, original_image_numpy, preprocessed_tensor)
        """
        image_path = None
        
        # Handle different input types
        if isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            image = Image.open(image_path).convert('RGB')
            original_image = np.array(image) / 255.0
            
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
            original_image = np.array(image) / 255.0
            
        elif isinstance(image_input, np.ndarray):
            if image_input.max() > 1:
                original_image = image_input / 255.0
            else:
                original_image = image_input
            image = Image.fromarray((original_image * 255).astype(np.uint8))
            
        elif isinstance(image_input, torch.Tensor):
            # Assume normalized tensor
            original_image = tensor_to_numpy_image(image_input)
            image = Image.fromarray((original_image * 255).astype(np.uint8))
            
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Apply transforms
        input_tensor = self.transforms(image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        return image_path, original_image, input_tensor
    
    def _get_prediction(self, input_tensor: torch.Tensor) -> tuple:
        """
        Get model prediction.
        
        Returns:
            Tuple of (predicted_class_name, confidence, all_probabilities_dict)
        """
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]
        
        # Get predicted class
        confidence, predicted_idx = probabilities.max(dim=0)
        predicted_class = self.config.class_names[predicted_idx.item()]
        
        # Get all probabilities
        all_probs = {
            self.config.class_names[i]: prob.item()
            for i, prob in enumerate(probabilities)
        }
        
        return predicted_class, confidence.item(), all_probs
    
    def predict_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[PredictionResult]:
        """
        Run prediction on multiple images.
        
        Args:
            image_paths: List of paths to images
            show_progress: Whether to show progress bar
        
        Returns:
            List of PredictionResult objects
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(image_paths, desc="Processing") if show_progress else image_paths
        
        for path in iterator:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        return results
    
    def demo(self, image_path: Optional[str] = None) -> Optional[PredictionResult]:
        """
        Run a demo prediction.
        
        If no image is provided, looks for a sample image in the dataset.
        
        Args:
            image_path: Optional path to demo image
        
        Returns:
            PredictionResult or None if no image found
        """
        # If no image provided, try to find one in the dataset
        if image_path is None:
            for class_name in self.config.class_names:
                class_dir = Path(self.config.dataset_dir) / class_name
                if class_dir.exists():
                    images = list(class_dir.glob("*.png")) + \
                            list(class_dir.glob("*.jpg")) + \
                            list(class_dir.glob("*.jpeg"))
                    if images:
                        image_path = str(images[0])
                        print(f"Using sample image: {image_path}")
                        break
        
        if image_path is None:
            print("❌ No image found for demo. Please provide an image path.")
            print("   Usage: pipeline.demo('path/to/ct_scan.png')")
            return None
        
        # Run prediction
        print("\n🔍 Running explainable prediction...\n")
        result = self.predict(image_path, show_visualization=True)
        
        # Print explanation
        result.print_explanation()
        
        return result


def run_single_prediction(
    image_path: str,
    checkpoint_path: Optional[str] = None,
    save_dir: Optional[str] = None
) -> PredictionResult:
    """
    Convenience function to run a single prediction.
    
    Args:
        image_path: Path to the CT scan image
        checkpoint_path: Optional path to model checkpoint
        save_dir: Optional directory to save visualization
    
    Returns:
        PredictionResult object
    """
    pipeline = ExplainablePipeline(checkpoint_path=checkpoint_path)
    
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = Path(image_path).stem
        save_path = os.path.join(save_dir, f"{filename}_explanation.png")
    
    result = pipeline.predict(
        image_path,
        show_visualization=True,
        save_visualization=save_path
    )
    
    result.print_explanation()
    
    return result


def create_demo_visualization(
    result: PredictionResult,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10)
) -> plt.Figure:
    """
    Create a comprehensive demo visualization with all outputs.
    
    Args:
        result: PredictionResult from pipeline
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(result.original_image)
    ax1.set_title('Original CT Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(result.heatmap, cmap='jet', vmin=0, vmax=1)
    ax2.set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(result.overlay)
    ax3.set_title('Overlay', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Probabilities bar chart
    ax4 = fig.add_subplot(gs[1, 0])
    classes = list(result.all_probabilities.keys())
    probs = list(result.all_probabilities.values())
    colors = ['green' if c == result.predicted_class else 'steelblue' for c in classes]
    
    # Format class names for display
    display_names = [c.replace('_', '\n').title() for c in classes]
    
    bars = ax4.bar(display_names, [p * 100 for p in probs], color=colors)
    ax4.set_ylabel('Probability (%)')
    ax4.set_title('Class Probabilities', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Explanation text
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')
    
    # Format explanation for display
    explanation_text = f"""
PREDICTION: {result.predicted_class.replace('_', ' ').title()} ({result.confidence*100:.1f}%)

VISUAL EVIDENCE:
{result.explanation.visual_evidence}

MEDICAL CONTEXT:
{result.explanation.medical_context[:300]}{'...' if len(result.explanation.medical_context) > 300 else ''}

SOURCES:
{chr(10).join(['• ' + s for s in result.explanation.sources[:2]])}
"""
    
    ax5.text(0.02, 0.98, explanation_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Main title
    fig.suptitle('Explainable AI for Lung Cancer Classification',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Demo visualization saved to {save_path}")
    
    return fig