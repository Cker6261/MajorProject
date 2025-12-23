# =============================================================================
# MODEL FACTORY
# Factory functions for creating model instances
# =============================================================================
"""
Model Factory for Lung Cancer Classification.

WHY A FACTORY PATTERN?
    1. Centralized Model Creation: Single place to instantiate models
    2. Easy Experimentation: Switch models by changing a string
    3. Configuration Management: Apply settings consistently
    4. Future Extensibility: Easy to add ViT or other architectures

Currently Supported:
    - resnet50: ResNet-50 (default, recommended)
    
Future Enhancements:
    - vit_b_16: Vision Transformer Base
    - efficientnet: EfficientNet variants
"""

import torch
import torch.nn as nn
from typing import Optional

from .classifier import LungCancerClassifier


def create_model(
    model_name: str = "resnet50",
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Factory function to create a classification model.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout probability
        freeze_backbone: Whether to freeze base layers
        device: Device to move model to
    
    Returns:
        Initialized model
    
    Raises:
        ValueError: If model_name is not supported
    
    Example:
        >>> model = create_model("resnet50", num_classes=4, pretrained=True)
        >>> model = model.to(device)
    """
    # Normalize model name
    model_name = model_name.lower().strip()
    
    print(f"\n{'=' * 50}")
    print(f"Creating Model: {model_name.upper()}")
    print(f"{'=' * 50}")
    
    if model_name == "resnet50":
        model = LungCancerClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone
        )
    
    # Future Enhancement: Add Vision Transformer
    # elif model_name == "vit_b_16":
    #     model = create_vit_classifier(num_classes, pretrained)
    
    else:
        supported = ["resnet50"]
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Choose from: {supported}"
        )
    
    # Print model summary
    model.print_model_summary()
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
        print(f"✓ Model moved to {device}")
    
    return model


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4
) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    
    WHY ADAMW?
        - Adam with decoupled weight decay
        - Better generalization than vanilla Adam
        - Recommended for fine-tuning pretrained models
    
    Args:
        model: The model to optimize
        optimizer_name: Name of optimizer ("adamw", "adam", "sgd")
        learning_rate: Learning rate
        weight_decay: L2 regularization strength
    
    Returns:
        Configured optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported")
    
    print(f"✓ Created {optimizer_name.upper()} optimizer (lr={learning_rate})")
    return optimizer


def get_loss_function() -> nn.Module:
    """
    Get the loss function for training.
    
    WHY CROSSENTROPYLOSS?
        - Standard loss for multi-class classification
        - Combines LogSoftmax and NLLLoss
        - Works directly with class indices (not one-hot)
    
    Returns:
        CrossEntropyLoss instance
    """
    criterion = nn.CrossEntropyLoss()
    print("✓ Using CrossEntropyLoss")
    return criterion


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "step",
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler.
    
    WHY USE A SCHEDULER?
        - Reduces learning rate during training
        - Helps model converge to better minima
        - Prevents oscillation in later epochs
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_name: Name of scheduler ("step", "cosine", "none")
        **kwargs: Additional arguments for the scheduler
    
    Returns:
        Configured scheduler or None
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "none":
        return None
    
    elif scheduler_name == "step":
        # Reduce LR by factor of 0.1 every 5 epochs
        step_size = kwargs.get('step_size', 5)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma
        )
        print(f"✓ Using StepLR scheduler (step={step_size}, gamma={gamma})")
    
    elif scheduler_name == "cosine":
        # Cosine annealing
        T_max = kwargs.get('T_max', 10)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=T_max
        )
        print(f"✓ Using CosineAnnealingLR scheduler (T_max={T_max})")
    
    else:
        raise ValueError(f"Scheduler '{scheduler_name}' not supported")
    
    return scheduler
