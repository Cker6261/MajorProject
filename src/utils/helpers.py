# =============================================================================
# HELPER FUNCTIONS
# Common utilities used across the project
# =============================================================================
"""
Helper utilities for the lung cancer classification project.
"""

import os
import random
import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    WHY IS THIS IMPORTANT?
        - Ensures experiments are reproducible
        - Critical for academic work and thesis defense
        - Allows fair comparison between different runs
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Random seed set to {seed}")


def get_device() -> torch.device:
    """
    Get the available device (GPU if available, else CPU).
    
    Returns:
        torch.device: The device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU (GPU not available)")
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    filepath: str
) -> None:
    """
    Save model checkpoint for resuming training or inference.
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy
        filepath: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        model: The PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
    
    Returns:
        dict: The checkpoint dictionary
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded from epoch {checkpoint['epoch']}")
    return checkpoint
