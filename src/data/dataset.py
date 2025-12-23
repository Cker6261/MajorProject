# =============================================================================
# LUNG CANCER DATASET
# Custom PyTorch Dataset for CT scan images
# =============================================================================
"""
Custom PyTorch Dataset for Lung Cancer CT Image Classification.

WHY CUSTOM DATASET?
    - Full control over data loading and preprocessing
    - Can add custom logic for handling medical images
    - Easy to extend for additional metadata

Dataset Structure Expected:
    dataset/
    ├── adenocarcinoma/
    ├── large_cell_carcinoma/
    ├── normal/
    └── squamous_cell_carcinoma/
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image


class LungCancerDataset(Dataset):
    """
    PyTorch Dataset for Lung Cancer CT Scan Classification.
    
    This dataset:
        - Loads images from class-organized folders
        - Applies transforms for preprocessing
        - Returns (image, label) pairs for training
    
    Attributes:
        root_dir: Path to the dataset directory
        class_names: List of class folder names
        transform: Optional transforms to apply
        image_paths: List of all image file paths
        labels: List of corresponding labels
    """
    
    # Supported image extensions
    VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    def __init__(
        self,
        root_dir: str,
        class_names: Optional[List[str]] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the Lung Cancer Dataset.
        
        Args:
            root_dir: Path to dataset directory containing class folders
            class_names: List of class folder names (auto-detected if None)
            transform: Optional transforms to apply to images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Auto-detect class names if not provided
        if class_names is None:
            self.class_names = sorted([
                d.name for d in self.root_dir.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ])
        else:
            self.class_names = class_names
        
        # Create class to index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        # Collect all image paths and labels
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        
        self._load_dataset()
        
        print(f"✓ Loaded {len(self)} images from {len(self.class_names)} classes")
    
    def _load_dataset(self) -> None:
        """Scan directories and collect image paths with labels."""
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                print(f"⚠ Warning: Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Collect all valid images in this class folder
            for file_path in class_dir.iterdir():
                if file_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.image_paths.append(file_path)
                    self.labels.append(class_idx)
    
    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (image_tensor, label)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Get label
        label = self.labels[idx]
        
        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def get_image_path(self, idx: int) -> Path:
        """Get the file path of an image by index."""
        return self.image_paths[idx]
    
    def get_class_name(self, idx: int) -> str:
        """Get the class name for a given label index."""
        return self.idx_to_class[idx]
    
    def get_class_distribution(self) -> dict:
        """
        Get the distribution of samples across classes.
        
        Returns:
            Dictionary mapping class names to sample counts
        """
        from collections import Counter
        label_counts = Counter(self.labels)
        
        distribution = {
            self.idx_to_class[label]: count 
            for label, count in sorted(label_counts.items())
        }
        return distribution
    
    def print_class_distribution(self) -> None:
        """Print a formatted class distribution summary."""
        distribution = self.get_class_distribution()
        total = sum(distribution.values())
        
        print("\n" + "=" * 50)
        print("CLASS DISTRIBUTION")
        print("=" * 50)
        
        for class_name, count in distribution.items():
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 2)
            print(f"{class_name:25s} | {count:5d} ({percentage:5.1f}%) {bar}")
        
        print("-" * 50)
        print(f"{'TOTAL':25s} | {total:5d}")
        print("=" * 50 + "\n")
