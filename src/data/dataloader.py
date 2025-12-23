# =============================================================================
# DATALOADER UTILITIES
# Creates train/val/test DataLoaders with proper splits
# =============================================================================
"""
DataLoader creation utilities for Lung Cancer Classification.

WHY THESE SPLIT RATIOS?
    - Train (70%): Sufficient data for model learning
    - Validation (15%): For hyperparameter tuning and early stopping
    - Test (15%): For final unbiased evaluation
    
    This is a standard split for academic projects with limited data.

WHY STRATIFIED SPLIT?
    - Ensures each split maintains the same class distribution
    - Critical for imbalanced medical datasets
    - Prevents bias in evaluation
"""

import os
from typing import Tuple, List, Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split

from .dataset import LungCancerDataset
from .transforms import get_train_transforms, get_val_transforms


def create_dataloaders(
    dataset_dir: str,
    class_names: Optional[List[str]] = None,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, LungCancerDataset]:
    """
    Create train, validation, and test DataLoaders.
    
    This function:
        1. Loads the full dataset
        2. Performs stratified train/val/test split
        3. Applies appropriate transforms (augmentation for train only)
        4. Returns DataLoaders ready for training
    
    Args:
        dataset_dir: Path to dataset directory
        class_names: List of class folder names
        image_size: Target image size (height, width)
        batch_size: Batch size for DataLoaders
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        num_workers: Number of workers for data loading
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, full_dataset)
    """
    # Validate split ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Load full dataset without transforms first (for splitting)
    full_dataset = LungCancerDataset(
        root_dir=dataset_dir,
        class_names=class_names,
        transform=None  # Will apply transforms later
    )
    
    # Print class distribution
    full_dataset.print_class_distribution()
    
    # Get indices and labels for stratified split
    indices = list(range(len(full_dataset)))
    labels = full_dataset.labels
    
    # First split: train vs (val + test)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices,
        labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=random_seed
    )
    
    # Second split: val vs test (from the remaining data)
    # Adjust ratio: val_ratio / (val_ratio + test_ratio)
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_ratio_adjusted,
        stratify=temp_labels,
        random_state=random_seed
    )
    
    print(f"✓ Dataset split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Create datasets with appropriate transforms
    train_dataset = TransformedSubset(
        full_dataset, 
        train_indices, 
        transform=get_train_transforms(image_size)
    )
    
    val_dataset = TransformedSubset(
        full_dataset, 
        val_indices, 
        transform=get_val_transforms(image_size)
    )
    
    test_dataset = TransformedSubset(
        full_dataset, 
        test_indices, 
        transform=get_val_transforms(image_size)
    )
    
    # Determine if CUDA is available for pin_memory
    import torch
    use_pin_memory = torch.cuda.is_available()
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=use_pin_memory  # Only if GPU available
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for test
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, test_loader, full_dataset


class TransformedSubset(Dataset):
    """
    A Dataset that wraps a subset of another dataset with transforms.
    
    WHY THIS CLASS?
        - Standard Subset doesn't properly support different transforms
        - We need different transforms for train (with augmentation) vs val/test
        - This wrapper applies the correct transform when fetching items
        
    Note: Inherits from Dataset (not Subset) to ensure __getitem__ is properly used.
    """
    
    def __init__(self, dataset, indices, transform=None):
        """
        Initialize TransformedSubset.
        
        Args:
            dataset: The full dataset
            indices: Indices of samples in this subset
            transform: Transforms to apply
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        """Return the number of samples in this subset."""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get item with transform applied."""
        # Get the actual index in the full dataset
        actual_idx = self.indices[idx]
        
        # Load raw image
        img_path = self.dataset.image_paths[actual_idx]
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        # Get label
        label = self.dataset.labels[actual_idx]
        
        # Apply transform
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


def visualize_sample_batch(dataloader: DataLoader, class_names: List[str], num_samples: int = 8):
    """
    Visualize a sample batch from the DataLoader.
    
    Useful for sanity checking that data loading is correct.
    
    Args:
        dataloader: DataLoader to sample from
        class_names: List of class names
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    from .transforms import denormalize_image
    
    # Get a batch
    images, labels = next(iter(dataloader))
    
    # Limit to num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        if idx >= len(axes):
            break
        
        # Denormalize for display
        img_display = denormalize_image(img)
        img_display = img_display.permute(1, 2, 0).numpy()
        
        axes[idx].imshow(img_display)
        axes[idx].set_title(class_names[label])
        axes[idx].axis('off')
    
    plt.suptitle('Sample Batch Visualization', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return fig
