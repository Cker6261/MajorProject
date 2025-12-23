# =============================================================================
# DATA TRANSFORMS
# Image preprocessing and augmentation pipelines
# =============================================================================
"""
Image Transforms for Lung Cancer CT Classification.

WHY THESE SPECIFIC TRANSFORMS?

1. Resize to 224x224:
   - Standard input size for pretrained ImageNet models (ResNet, VGG, etc.)
   - Allows use of transfer learning with pretrained weights

2. Normalization (ImageNet mean/std):
   - Pretrained models expect inputs normalized with ImageNet statistics
   - Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
   
3. Data Augmentation (Training only):
   - RandomHorizontalFlip: Lung CT scans can be flipped horizontally
   - RandomRotation (±15°): Small rotations for robustness
   - ColorJitter: Handles slight variations in CT scanner settings
   
   WHY NOT vertical flip? 
   - Anatomically, lungs have top-bottom orientation that matters
   
   WHY NOT aggressive augmentation?
   - Medical images require careful augmentation to preserve diagnostic features
"""

from torchvision import transforms
from typing import Tuple


# ImageNet normalization values (required for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Get training transforms with data augmentation.
    
    Augmentation Strategy:
        - Horizontal flip: 50% probability
        - Rotation: ±15 degrees
        - Color jitter: Slight brightness/contrast variations
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Composed transforms for training
    """
    return transforms.Compose([
        # Resize to standard input size
        transforms.Resize(image_size),
        
        # Data Augmentation (only for training)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05
        ),
        
        # Convert to tensor (scales to [0, 1])
        transforms.ToTensor(),
        
        # Normalize with ImageNet statistics
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    WHY NO AUGMENTATION FOR VAL/TEST?
        - We want consistent, deterministic evaluation
        - Augmentation is only for increasing training data variety
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Composed transforms for validation/testing
    """
    return transforms.Compose([
        # Resize to standard input size
        transforms.Resize(image_size),
        
        # Convert to tensor (scales to [0, 1])
        transforms.ToTensor(),
        
        # Normalize with ImageNet statistics
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_inference_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Get transforms for single image inference.
    Same as validation transforms but explicitly named for clarity.
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Composed transforms for inference
    """
    return get_val_transforms(image_size)


def denormalize_image(tensor):
    """
    Reverse normalization for visualization.
    
    Useful for displaying images after they've been normalized.
    
    Args:
        tensor: Normalized image tensor [C, H, W]
    
    Returns:
        Denormalized tensor suitable for display
    """
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    # Denormalize: x = x * std + mean
    denorm = tensor.clone()
    denorm = denorm * std + mean
    
    # Clip to valid range [0, 1]
    denorm = torch.clamp(denorm, 0, 1)
    
    return denorm
