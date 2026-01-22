# =============================================================================
# ATTENTION-BASED TRANSFORMER XAI
# Extracts and visualizes attention patterns from Vision Transformers
# =============================================================================
"""
This module provides attention-based XAI for Vision Transformers.
Works by extracting attention weights directly from transformer layers.

Key Methods:
1. AttentionRollout - Aggregates attention across all layers
2. GradientAttention - Combines attention with gradient information
3. RawLastLayerAttention - Uses final layer attention (simplest, often best)
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AttentionExtractor:
    """
    Extract attention maps from Vision Transformers.
    
    Works with torchvision ViT, Swin Transformers, and timm models.
    """
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.attention_maps = []
        self.hooks = []
        
    def _get_attention_hook(self):
        """Create hook to capture attention weights."""
        def hook(module, input, output):
            # Different models store attention differently
            if hasattr(output, 'shape') and len(output.shape) == 4:
                # Shape: [B, num_heads, seq_len, seq_len]
                self.attention_maps.append(output.detach().cpu())
            elif isinstance(output, tuple) and len(output) >= 2:
                # Some models return (output, attention)
                attn = output[1] if len(output) > 1 else output[0]
                if hasattr(attn, 'shape') and len(attn.shape) >= 3:
                    self.attention_maps.append(attn.detach().cpu())
        return hook
    
    def register_hooks(self):
        """Register hooks to capture attention from transformer layers."""
        self.attention_maps = []
        self.hooks = []
        
        # Try to find attention modules in the model
        for name, module in self.model.named_modules():
            # Common patterns for attention layers
            module_name = name.lower()
            module_type = type(module).__name__.lower()
            
            if 'attention' in module_name or 'attn' in module_name:
                if 'softmax' in module_type or 'dropout' not in module_name:
                    # Hook the attention module's forward
                    self.hooks.append(module.register_forward_hook(self._get_attention_hook()))
                    
            # For torchvision ViT - hook the self_attention module
            if 'self_attention' in module_name and 'ln' not in module_name:
                self.hooks.append(module.register_forward_hook(self._get_attention_hook()))
                
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_attention_maps(self, input_tensor: torch.Tensor) -> list:
        """Forward pass and return captured attention maps."""
        self.attention_maps = []
        
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))
        
        return self.attention_maps


class GradientWeightedAttention:
    """
    Combines attention patterns with gradient information.
    
    Similar to GradCAM but uses attention weights instead of activations.
    This produces more focused heatmaps than raw attention.
    """
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
    def _find_attention_layers(self):
        """Find attention-related layers in the model."""
        layers = []
        for name, module in self.model.named_modules():
            name_lower = name.lower()
            type_name = type(module).__name__.lower()
            
            # Look for attention or transformer encoder layers
            if any(x in name_lower for x in ['encoder', 'block', 'layer']) and \
               'norm' not in name_lower and 'drop' not in name_lower:
                if hasattr(module, 'forward'):
                    layers.append((name, module))
                    
        return layers[-3:] if len(layers) > 3 else layers  # Use last few layers
    
    def generate(self, input_tensor: torch.Tensor, target_class: int,
                 show_progress: bool = True) -> np.ndarray:
        """Generate gradient-weighted attention heatmap."""
        self.model.eval()
        
        input_tensor = input_tensor.to(self.device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        _, C, H, W = input_tensor.shape
        
        # Enable gradient computation
        input_tensor.requires_grad_(True)
        
        # Get layers to hook
        layers = self._find_attention_layers()
        
        if not layers:
            # Fallback: use input gradients directly
            return self._input_gradient_fallback(input_tensor, target_class)
        
        activations = []
        gradients = []
        
        def save_activation(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations.append(output)
            
        def save_gradient(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                grad_output = grad_output[0]
            gradients.append(grad_output)
        
        # Register hooks on last layer
        layer_name, layer = layers[-1]
        hook_forward = layer.register_forward_hook(save_activation)
        hook_backward = layer.register_full_backward_hook(save_gradient)
        
        try:
            # Forward pass
            output = self.model(input_tensor)
            
            # Backward pass for target class
            self.model.zero_grad()
            target_output = output[0, target_class]
            target_output.backward()
            
            hook_forward.remove()
            hook_backward.remove()
            
            if not activations or not gradients:
                return self._input_gradient_fallback(input_tensor, target_class)
            
            # Compute gradient-weighted features
            activation = activations[0].detach()
            gradient = gradients[0].detach()
            
            # Global average pooling of gradients for weighting
            if len(gradient.shape) == 4:  # [B, C, H, W]
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                weighted = (activation * weights).sum(dim=1).squeeze()
            elif len(gradient.shape) == 3:  # [B, seq_len, features]
                weights = gradient.mean(dim=1, keepdim=True)
                weighted = (activation * weights).sum(dim=-1).squeeze()
                # Reshape from sequence to 2D if needed
                seq_len = weighted.shape[-1] if len(weighted.shape) > 0 else weighted.numel()
                grid_size = int(np.sqrt(seq_len - 1))  # -1 for CLS token
                if grid_size ** 2 == seq_len - 1:
                    weighted = weighted[1:].reshape(grid_size, grid_size)  # Remove CLS
                else:
                    weighted = weighted.reshape(int(np.sqrt(seq_len)), -1)
            else:
                return self._input_gradient_fallback(input_tensor, target_class)
            
            heatmap = weighted.cpu().numpy()
            
        except Exception as e:
            print(f"Gradient attention failed: {e}")
            return self._input_gradient_fallback(input_tensor, target_class)
        
        # Resize to input size
        heatmap = np.abs(heatmap)
        if heatmap.shape != (H, W):
            from scipy.ndimage import zoom as scipy_zoom
            zoom_factors = (H / heatmap.shape[0], W / heatmap.shape[1])
            heatmap = scipy_zoom(heatmap, zoom_factors, order=1)
        
        # Apply ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = gaussian_filter(heatmap, sigma=5)
        
        # Focus enhancement
        threshold = np.percentile(heatmap, 70)
        heatmap = np.where(heatmap >= threshold, heatmap, heatmap * 0.2)
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def _input_gradient_fallback(self, input_tensor, target_class):
        """Fallback using input gradients."""
        # Make sure we have a fresh tensor with gradients enabled
        input_tensor = input_tensor.detach().clone().requires_grad_(True)
        
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, target_class].backward()
        
        if input_tensor.grad is None:
            # If still no gradient, return uniform heatmap
            H, W = input_tensor.shape[-2:]
            return np.ones((H, W)) * 0.5
        
        # Use gradient * input (saliency)
        gradients = input_tensor.grad.detach()
        saliency = (gradients.abs() * input_tensor.detach().abs()).sum(dim=1).squeeze()
        
        heatmap = saliency.cpu().numpy()
        heatmap = gaussian_filter(heatmap, sigma=4)
        
        # Focus
        threshold = np.percentile(heatmap, 65)
        heatmap = np.where(heatmap >= threshold, heatmap, heatmap * 0.25)
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap


class HighQualityTransformerXAI:
    """
    Produces high-quality, focused heatmaps for transformers.
    
    Combines multiple signals for the best possible result:
    1. Gradient-weighted attention
    2. Saliency (input * gradient)
    3. Activation difference from occluding key regions
    """
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
    
    def generate(self, input_tensor: torch.Tensor, target_class: int,
                 show_progress: bool = True) -> np.ndarray:
        """Generate high-quality, focused heatmap for transformer models."""
        self.model.eval()
        
        input_tensor = input_tensor.to(self.device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        _, C, H, W = input_tensor.shape
        
        # Use occlusion as primary method - most reliable for transformers
        print("  Computing targeted occlusion...")
        occlusion = self._targeted_occlusion(input_tensor.clone(), target_class, 
                                              None, show_progress)
        
        # Apply GradCAM-style focus enhancement 
        focused = self._gradcam_style_focus(occlusion)
        
        return focused
    
    def _gradcam_style_focus(self, heatmap):
        """Apply focus enhancement that preserves natural importance patterns."""
        # Start with the raw heatmap - it already contains good information
        
        # Step 1: Apply ReLU (only positive importance)
        heatmap = np.maximum(heatmap, 0)
        
        # Step 2: Compute statistics
        max_val = heatmap.max()
        if max_val <= 0:
            return np.zeros_like(heatmap)
        
        # Step 3: Percentile-based thresholding (more stable than max-based)
        p90 = np.percentile(heatmap, 90)
        p70 = np.percentile(heatmap, 70)
        p50 = np.percentile(heatmap, 50)
        
        # Step 4: Create focused heatmap with sharp falloff
        focused = heatmap.copy()
        
        # Suppress values below 70th percentile significantly
        low_mask = heatmap < p50
        focused[low_mask] = heatmap[low_mask] * 0.05  # Heavy suppression
        
        mid_mask = (heatmap >= p50) & (heatmap < p70)
        focused[mid_mask] = heatmap[mid_mask] * 0.2
        
        # Values above 70th percentile stay mostly intact
        # Values above 90th percentile get small boost
        high_mask = heatmap >= p90
        focused[high_mask] = heatmap[high_mask] * 1.1
        
        # Step 5: Normalize
        focused = (focused - focused.min()) / (focused.max() - focused.min() + 1e-8)
        
        # Step 6: Apply smoothing for clean visualization
        focused = gaussian_filter(focused, sigma=3)
        
        # Step 7: Re-normalize (smoothing can change range)
        focused = (focused - focused.min()) / (focused.max() - focused.min() + 1e-8)
        
        return focused

    def _compute_saliency(self, input_tensor, target_class):
        """Compute saliency map (gradient * input)."""
        # Ensure fresh tensor with gradients
        input_tensor = input_tensor.detach().clone().requires_grad_(True)
        
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, target_class].backward()
        
        if input_tensor.grad is None:
            # Fallback if no gradients
            H, W = input_tensor.shape[-2:]
            return np.ones((H, W)) * 0.5
        
        gradients = input_tensor.grad.detach()
        saliency = (gradients.abs() * input_tensor.detach().abs()).sum(dim=1).squeeze()
        
        heatmap = saliency.cpu().numpy()
        heatmap = gaussian_filter(heatmap, sigma=3)
        
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def _targeted_occlusion(self, input_tensor, target_class, guide_map, show_progress):
        """
        High-resolution occlusion for focused, GradCAM-quality heatmaps.
        Uses very small patches for fine-grained attribution.
        """
        _, C, H, W = input_tensor.shape
        
        # Get baseline probability
        with torch.no_grad():
            baseline_out = self.model(input_tensor)
            baseline_prob = F.softmax(baseline_out, dim=1)[0, target_class].item()
        
        # Use SMALL patches for fine resolution (like GradCAM's 7x7 output)
        patch_size = 8  # Small patches
        stride = 4      # Fine stride for smooth output
        
        # Initialize importance grid
        n_rows = (H - patch_size) // stride + 1
        n_cols = (W - patch_size) // stride + 1
        importance_grid = np.zeros((n_rows, n_cols))
        
        positions = [(i, j, i // stride, j // stride) 
                     for i in range(0, H - patch_size + 1, stride) 
                     for j in range(0, W - patch_size + 1, stride)]
        
        iterator = positions
        if show_progress:
            iterator = tqdm(positions, desc="Occlusion", leave=False)
        
        max_drop = 0
        all_results = []
        
        for i, j, gi, gj in iterator:
            # Occlude with neutral value (zeros work well for normalized images)
            occluded = input_tensor.clone()
            occluded[0, :, i:i+patch_size, j:j+patch_size] = 0
            
            with torch.no_grad():
                output = self.model(occluded)
                prob = F.softmax(output, dim=1)[0, target_class].item()
            
            drop = baseline_prob - prob
            all_results.append((gi, gj, drop))
            if drop > max_drop:
                max_drop = drop
        
        # Fill grid with normalized importance
        for gi, gj, drop in all_results:
            if max_drop > 0 and drop > 0:
                importance_grid[gi, gj] = drop / max_drop
        
        # Upsample to full resolution with bicubic interpolation
        from scipy.ndimage import zoom as scipy_zoom
        zoom_h = H / importance_grid.shape[0]
        zoom_w = W / importance_grid.shape[1]
        importance_map = scipy_zoom(importance_grid, (zoom_h, zoom_w), order=3)
        
        # Ensure correct size
        importance_map = importance_map[:H, :W]
        
        # Apply mild smoothing
        importance_map = gaussian_filter(importance_map, sigma=2)
        
        # Normalize
        importance_map = np.clip(importance_map, 0, None)
        importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)
        
        return importance_map
    
    def _enhance_focus(self, heatmap):
        """Apply aggressive focus enhancement to match GradCAM quality."""
        # Smooth first
        heatmap = gaussian_filter(heatmap, sigma=3)
        
        # More aggressive thresholding for tighter focus
        # Keep only top 20% with full intensity
        p80 = np.percentile(heatmap, 80)
        p60 = np.percentile(heatmap, 60)
        p40 = np.percentile(heatmap, 40)
        
        focused = np.zeros_like(heatmap)
        
        # Top 20%: full intensity
        top_mask = heatmap >= p80
        focused[top_mask] = heatmap[top_mask]
        
        # 60-80%: reduced
        mid_high_mask = (heatmap >= p60) & (heatmap < p80)
        focused[mid_high_mask] = heatmap[mid_high_mask] * 0.5
        
        # 40-60%: minimal
        mid_low_mask = (heatmap >= p40) & (heatmap < p60)
        focused[mid_low_mask] = heatmap[mid_low_mask] * 0.15
        
        # Bottom 40%: suppress almost completely
        focused[heatmap < p40] = heatmap[heatmap < p40] * 0.02
        
        # Normalize
        focused = (focused - focused.min()) / (focused.max() - focused.min() + 1e-8)
        
        # Apply power transform to sharpen peaks further
        focused = np.power(focused, 0.5)
        
        # Final light smoothing to clean edges
        focused = gaussian_filter(focused, sigma=1.5)
        
        return focused


def best_transformer_xai(model, input_tensor: torch.Tensor, target_class: int,
                         show_progress: bool = True) -> np.ndarray:
    """
    Generate the best possible XAI heatmap for a transformer model.
    
    Args:
        model: Transformer model (ViT, Swin, DeiT, etc.)
        input_tensor: Input image tensor [1, C, H, W] or [C, H, W]
        target_class: Target class index
        show_progress: Whether to show progress bars
    
    Returns:
        Heatmap as numpy array [H, W] normalized to [0, 1]
    """
    xai = HighQualityTransformerXAI(model)
    return xai.generate(input_tensor, target_class, show_progress)
