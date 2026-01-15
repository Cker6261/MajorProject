# =============================================================================
# CONFIGURATION FILE
# Centralized configuration for the entire project
# =============================================================================
"""
Configuration Module for Lung Cancer Classification.

WHY CENTRALIZED CONFIG?
    - Single source of truth for all hyperparameters
    - Easy to modify settings without changing code
    - Makes experiments reproducible
    - Essential for academic documentation
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    """
    Centralized configuration class for the lung cancer classification project.
    
    Using dataclass for:
        - Clean, readable configuration
        - Type hints for documentation
        - Easy to modify and extend
    """
    
    # ==========================================================================
    # PATH CONFIGURATION
    # ==========================================================================
    # Base directory (modify this according to your setup)
    base_dir: str = r"d:\Major Project"
    
    # Dataset path (Kaggle CT Scan Images)
    dataset_dir: str = field(default="")
    
    # Output directories
    checkpoint_dir: str = field(default="")
    results_dir: str = field(default="")
    
    # ==========================================================================
    # DATASET CONFIGURATION
    # ==========================================================================
    # Class names (order matters for label encoding)
    # These must match the actual folder names in the dataset
    class_names: List[str] = field(default_factory=lambda: [
        "adenocarcinoma",
        "Benign cases",
        "large cell carcinoma", 
        "Normal cases",
        "squamous cell carcinoma"
    ])
    
    # Number of classes
    num_classes: int = 5
    
    # Image dimensions (224x224 is standard for pretrained models)
    image_size: Tuple[int, int] = (224, 224)
    
    # Data split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # ==========================================================================
    # TRAINING CONFIGURATION
    # ==========================================================================
    # Batch size (reduce if GPU memory is limited)
    batch_size: int = 32
    
    # Number of epochs for full training
    num_epochs: int = 50
    
    # Learning rate (AdamW default)
    learning_rate: float = 1e-4
    
    # Weight decay for regularization
    weight_decay: float = 1e-4
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Number of workers for data loading
    num_workers: int = 4
    
    # Early stopping patience
    early_stopping_patience: int = 10
    
    # Learning rate scheduler step size
    lr_scheduler_step: int = 10
    
    # Learning rate scheduler gamma
    lr_scheduler_gamma: float = 0.5
    
    # ==========================================================================
    # MODEL CONFIGURATION
    # ==========================================================================
    # Model architecture choice
    model_name: str = "resnet50"  # Options: "resnet50", "vit_b_16", "efficientnet_b0"
    
    # Use pretrained weights (ImageNet)
    pretrained: bool = True
    
    # Freeze base layers during initial training
    freeze_base: bool = False  # Set True if dataset is very small
    
    # Dropout rate for regularization
    dropout_rate: float = 0.5
    
    # ==========================================================================
    # GRAD-CAM CONFIGURATION
    # ==========================================================================
    # Target layer for Grad-CAM (for ResNet-50: "layer4")
    gradcam_target_layer: str = "layer4"
    
    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================
    def __post_init__(self):
        """Set derived paths after initialization."""
        if not self.dataset_dir:
            # Use the archive (1)/Lung Cancer Dataset folder
            self.dataset_dir = os.path.join(self.base_dir, "archive (1)", "Lung Cancer Dataset")
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        if not self.results_dir:
            self.results_dir = os.path.join(self.base_dir, "results")
        
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


# Create a default config instance
default_config = Config()
