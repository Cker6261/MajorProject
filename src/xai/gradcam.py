# =============================================================================
# GRAD-CAM IMPLEMENTATION
# Gradient-weighted Class Activation Mapping for model interpretability
# =============================================================================
"""
Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

WHAT IS GRAD-CAM?
    Grad-CAM uses the gradients flowing into the final convolutional layer
    to produce a coarse localization map highlighting important regions
    in the image for predicting the target class.

HOW IT WORKS:
    1. Forward pass: Get feature maps from target layer
    2. Backward pass: Get gradients of target class w.r.t. feature maps
    3. Weight calculation: Global average pool the gradients
    4. Weighted combination: Multiply weights with feature maps
    5. ReLU: Keep only positive influences
    6. Resize: Upsample to input image size

WHY GRAD-CAM FOR THIS PROJECT?
    1. No model modification required (works with any CNN)
    2. Class-discriminative (shows regions for specific class)
    3. Produces intuitive visual explanations
    4. Well-established in medical imaging literature

LIMITATIONS:
    1. Coarse resolution (limited by conv layer size)
    2. May miss fine-grained details
    3. Can highlight spurious correlations
    4. Doesn't explain "why" only "where"

Reference:
    Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from 
    Deep Networks via Gradient-based Localization." ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization.
    
    This class:
        - Registers hooks on the target layer to capture activations and gradients
        - Computes class-specific heatmaps
        - Works with any CNN architecture
    
    Example:
        >>> model = LungCancerClassifier()
        >>> gradcam = GradCAM(model, target_layer=model.get_gradcam_target_layer())
        >>> heatmap = gradcam.generate(input_image, target_class=0)
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The trained CNN model
            target_layer: The layer to compute Grad-CAM for
                         (typically the last convolutional layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        # Storage for activations and gradients
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        
        # Register hooks
        self._register_hooks()
        
        # Put model in eval mode
        self.model.eval()
    
    def _register_hooks(self) -> None:
        """
        Register forward and backward hooks on the target layer.
        
        Forward hook: Captures feature map activations
        Backward hook: Captures gradients flowing back
        """
        def forward_hook(module, input, output):
            """Store activations during forward pass."""
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """Store gradients during backward pass."""
            self.gradients = grad_output[0].detach()
        
        # Register the hooks
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self) -> None:
        """Remove registered hooks (call when done to free memory)."""
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input image.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Class index to generate heatmap for
                         If None, uses the predicted class
        
        Returns:
            Heatmap as numpy array [H, W] with values in [0, 1]
        """
        # Ensure input has batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Get device
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Enable gradients for input (needed for backward pass)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class (predicted if not specified)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Get the score for target class
        target_score = output[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward()
        
        # Get the stored activations and gradients
        activations = self.activations  # [1, C, H, W] for CNNs or [1, N, D] for ViT
        gradients = self.gradients      # [1, C, H, W] for CNNs or [1, N, D] for ViT
        
        # Handle Vision Transformer (ViT) outputs
        # ViT produces [batch, num_patches+1, embedding_dim] tensors
        # Standard Grad-CAM doesn't work well for transformers, so we use input gradient saliency
        if activations.dim() == 3:
            # For Vision Transformers, use input gradient saliency instead
            # This is more reliable than trying to adapt Grad-CAM for attention layers
            if input_tensor.grad is not None:
                # Use input gradients to create saliency map
                input_grad = input_tensor.grad.data
                saliency = input_grad.abs().mean(dim=1, keepdim=True)  # Average over channels
                
                # Normalize
                saliency = saliency - saliency.min()
                if saliency.max() > 0:
                    saliency = saliency / saliency.max()
                
                return saliency.squeeze().cpu().numpy()
            else:
                # Fallback: use activation norms per patch
                batch_size, num_tokens, embed_dim = activations.shape
                num_patches = num_tokens - 1
                patch_size = int(num_patches ** 0.5)
                
                if patch_size * patch_size == num_patches:
                    # Use activation magnitude as importance
                    activations_patch = activations[:, 1:, :]  # Exclude CLS token
                    importance = activations_patch.norm(dim=-1)  # L2 norm per patch
                    
                    cam = importance.reshape(batch_size, 1, patch_size, patch_size)
                    cam = cam - cam.min()
                    if cam.max() > 0:
                        cam = cam / cam.max()
                    
                    cam = F.interpolate(
                        cam,
                        size=(input_tensor.shape[2], input_tensor.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                    return cam.squeeze().cpu().numpy()
                else:
                    # Return uniform heatmap as last resort
                    return np.ones((input_tensor.shape[2], input_tensor.shape[3])) * 0.5
        
        # Compute weights: global average pooling of gradients
        # Shape: [1, C, 1, 1] -> [C]
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Compute weighted combination of activation maps
        # weighted_activations: [1, C, H, W] * [1, C, 1, 1] -> [1, C, H, W]
        # Sum over channels -> [1, 1, H, W]
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU (only keep positive influences)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input image size
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert to numpy
        heatmap = cam.squeeze().cpu().numpy()
        
        return heatmap
    
    def generate_with_confidence(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap along with prediction and confidence.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Class index to generate heatmap for
        
        Returns:
            Tuple of (heatmap, predicted_class, confidence)
        """
        # Ensure input has batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Get prediction first
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = probabilities.max(dim=1)
            predicted_class = predicted_class.item()
            confidence = confidence.item()
        
        # Generate heatmap
        if target_class is None:
            target_class = predicted_class
        
        heatmap = self.generate(input_tensor, target_class)
        
        return heatmap, predicted_class, confidence
    
    def __del__(self):
        """Cleanup hooks when object is deleted."""
        try:
            self.remove_hooks()
        except:
            pass


def get_gradcam_for_batch(
    model: nn.Module,
    target_layer: nn.Module,
    images: torch.Tensor,
    target_classes: Optional[List[int]] = None
) -> List[np.ndarray]:
    """
    Generate Grad-CAM heatmaps for a batch of images.
    
    Args:
        model: The trained model
        target_layer: Target layer for Grad-CAM
        images: Batch of images [B, C, H, W]
        target_classes: List of target classes (one per image)
    
    Returns:
        List of heatmaps
    """
    gradcam = GradCAM(model, target_layer)
    heatmaps = []
    
    for i in range(images.shape[0]):
        image = images[i:i+1]
        target = target_classes[i] if target_classes else None
        heatmap = gradcam.generate(image, target)
        heatmaps.append(heatmap)
    
    gradcam.remove_hooks()
    
    return heatmaps
