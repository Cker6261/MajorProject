# =============================================================================
# LOCALIZED XAI FOR TRANSFORMERS
# Methods that produce focused, region-specific heatmaps
# =============================================================================
"""
Better XAI Methods for Vision Transformers

The gradient-based methods produce diffuse heatmaps because transformers 
have global attention. These methods produce more LOCALIZED explanations:

1. Occlusion Sensitivity - Masks regions and measures prediction drop
2. RISE - Randomized mask-based attribution  
3. ScoreCAM-like approach - Uses feature importance without gradients

These are slower but produce much better localized heatmaps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm
import warnings


class OcclusionSensitivity:
    """
    Occlusion Sensitivity Analysis.
    
    Slides a gray/black patch across the image and measures how much
    the prediction drops when each region is occluded.
    
    PROS:
        - Model agnostic (works with ANY model)
        - Produces focused, localized heatmaps
        - Very interpretable (shows "if I hide this, prediction drops")
    
    CONS:
        - Slower than gradient methods
        - Patch size affects resolution
    
    This is often the BEST method for medical imaging!
    """
    
    def __init__(
        self,
        model: nn.Module,
        patch_size: int = 16,
        stride: int = 8,
        occlusion_value: float = 0.0
    ):
        """
        Args:
            model: The model to explain
            patch_size: Size of occlusion patch
            stride: Step size for sliding window
            occlusion_value: Value to fill occluded region (0=black, 0.5=gray)
        """
        self.model = model
        self.model.eval()
        self.patch_size = patch_size
        self.stride = stride
        self.occlusion_value = occlusion_value
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate occlusion sensitivity map.
        
        Returns heatmap where HIGH values = important regions
        (regions that cause large prediction drops when occluded)
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        _, C, H, W = input_tensor.shape
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(input_tensor)
            baseline_probs = F.softmax(baseline_output, dim=1)
            
            if target_class is None:
                target_class = baseline_probs.argmax(dim=1).item()
            
            baseline_score = baseline_probs[0, target_class].item()
        
        # Create sensitivity map
        sensitivity_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        
        # Calculate number of positions
        n_rows = (H - self.patch_size) // self.stride + 1
        n_cols = (W - self.patch_size) // self.stride + 1
        total = n_rows * n_cols
        
        positions = []
        for i in range(0, H - self.patch_size + 1, self.stride):
            for j in range(0, W - self.patch_size + 1, self.stride):
                positions.append((i, j))
        
        # Process in batches for efficiency
        batch_size = 32
        iterator = range(0, len(positions), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Occlusion", total=len(positions)//batch_size + 1)
        
        for batch_start in iterator:
            batch_positions = positions[batch_start:batch_start + batch_size]
            batch_inputs = []
            
            for (i, j) in batch_positions:
                occluded = input_tensor.clone()
                occluded[:, :, i:i+self.patch_size, j:j+self.patch_size] = self.occlusion_value
                batch_inputs.append(occluded)
            
            batch_tensor = torch.cat(batch_inputs, dim=0)
            
            with torch.no_grad():
                batch_outputs = self.model(batch_tensor)
                batch_probs = F.softmax(batch_outputs, dim=1)
                batch_scores = batch_probs[:, target_class].cpu().numpy()
            
            for idx, (i, j) in enumerate(batch_positions):
                # Importance = how much prediction drops when occluded
                importance = baseline_score - batch_scores[idx]
                importance = max(0, importance)  # Only positive (drops)
                
                sensitivity_map[i:i+self.patch_size, j:j+self.patch_size] += importance
                count_map[i:i+self.patch_size, j:j+self.patch_size] += 1
        
        # Average overlapping regions
        count_map = np.maximum(count_map, 1)
        sensitivity_map = sensitivity_map / count_map
        
        # Apply Gaussian smoothing for cleaner output
        sensitivity_map = self._smooth(sensitivity_map)
        
        # Normalize
        if sensitivity_map.max() > 0:
            sensitivity_map = sensitivity_map / sensitivity_map.max()
        
        return sensitivity_map
    
    def _smooth(self, heatmap: np.ndarray, sigma: float = None) -> np.ndarray:
        """Apply Gaussian smoothing."""
        from scipy.ndimage import gaussian_filter
        if sigma is None:
            sigma = self.patch_size / 4.0  # Adaptive sigma based on patch size
        return gaussian_filter(heatmap, sigma=sigma)


class RISE:
    """
    RISE: Randomized Input Sampling for Explanation.
    
    Uses random masks to probe which regions are important.
    Faster than occlusion with similar quality.
    
    From: "RISE: Randomized Input Sampling for Explanation of Black-box Models"
    (Petsiuk et al., 2018)
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_masks: int = 2000,
        mask_prob: float = 0.5,
        cell_size: int = 8
    ):
        """
        Args:
            model: Model to explain
            n_masks: Number of random masks to use
            mask_prob: Probability of each cell being visible
            cell_size: Size of mask cells (smaller = finer resolution)
        """
        self.model = model
        self.model.eval()
        self.n_masks = n_masks
        self.mask_prob = mask_prob
        self.cell_size = cell_size
    
    def _generate_masks(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate random masks."""
        # Small mask grid
        h = H // self.cell_size
        w = W // self.cell_size
        
        # Generate small random masks
        small_masks = (torch.rand(self.n_masks, 1, h, w, device=device) < self.mask_prob).float()
        
        # Upscale to full size with bilinear interpolation (smooth edges)
        masks = F.interpolate(small_masks, size=(H, W), mode='bilinear', align_corners=False)
        
        return masks
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate RISE attribution map."""
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        _, C, H, W = input_tensor.shape
        
        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Generate masks
        masks = self._generate_masks(H, W, device)
        
        # Accumulate weighted masks
        attribution = torch.zeros(H, W, device=device)
        
        batch_size = 64
        iterator = range(0, self.n_masks, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="RISE", total=self.n_masks//batch_size + 1)
        
        for i in iterator:
            batch_masks = masks[i:i+batch_size]  # [B, 1, H, W]
            
            # Apply masks
            masked_inputs = input_tensor * batch_masks  # Broadcasting
            
            with torch.no_grad():
                outputs = self.model(masked_inputs)
                probs = F.softmax(outputs, dim=1)
                scores = probs[:, target_class]  # [B]
            
            # Weight masks by scores
            weighted_masks = batch_masks.squeeze(1) * scores.view(-1, 1, 1)
            attribution += weighted_masks.sum(dim=0)
        
        # Normalize by mask expectations
        attribution = attribution / (self.n_masks * self.mask_prob)
        
        # Convert to numpy
        attribution = attribution.cpu().numpy()
        
        # Apply Gaussian smoothing for cleaner output
        from scipy.ndimage import gaussian_filter
        attribution = gaussian_filter(attribution, sigma=self.cell_size / 2.0)
        
        # Normalize to [0, 1]
        attribution = attribution - attribution.min()
        if attribution.max() > 0:
            attribution = attribution / attribution.max()
        
        return attribution


class ScoreCAMForTransformer:
    """
    Score-CAM adapted for Transformers.
    
    Instead of using gradients, uses activation maps directly
    and weights them by their contribution to the class score.
    
    This avoids the gradient dispersion problem in transformers.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: Vision Transformer model
            target_layer: Layer to extract activations from (auto-detected if None)
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._find_target_layer()
        self.activations = None
        self.hook = None
        
        if self.target_layer is not None:
            self._register_hook()
    
    def _find_target_layer(self) -> Optional[nn.Module]:
        """Find suitable target layer."""
        model = self.model
        if hasattr(model, 'backbone'):
            model = model.backbone
        
        # For ViT
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            return list(model.encoder.layers)[-1]
        
        # For Swin
        if hasattr(model, 'features'):
            return model.features[-1]
        
        return None
    
    def _register_hook(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        self.hook = self.target_layer.register_forward_hook(hook_fn)
    
    def remove_hook(self):
        if self.hook:
            self.hook.remove()
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Score-CAM attribution."""
        if self.target_layer is None:
            # Fallback to occlusion
            occ = OcclusionSensitivity(self.model, patch_size=16, stride=8)
            return occ.generate(input_tensor, target_class, show_progress=False)
        
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        _, C, H, W = input_tensor.shape
        
        # Forward pass to get activations
        with torch.no_grad():
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
        
        if self.activations is None:
            # Fallback
            occ = OcclusionSensitivity(self.model, patch_size=16, stride=8)
            return occ.generate(input_tensor, target_class, show_progress=False)
        
        activations = self.activations  # [1, N, D] or [1, C, H', W']
        
        # Handle transformer activations (sequence format)
        if activations.dim() == 3:
            # [1, num_tokens, embed_dim]
            num_tokens = activations.shape[1]
            
            # Remove CLS token if present
            if num_tokens == 197:  # 14*14 + 1
                activations = activations[:, 1:, :]  # [1, 196, D]
            
            num_patches = activations.shape[1]
            patch_size = int(np.sqrt(num_patches))
            
            if patch_size * patch_size != num_patches:
                # Non-square, use fallback
                occ = OcclusionSensitivity(self.model, patch_size=16, stride=8)
                return occ.generate(input_tensor, target_class, show_progress=False)
            
            # Reshape to spatial
            D = activations.shape[2]
            activations = activations.reshape(1, patch_size, patch_size, D)
            activations = activations.permute(0, 3, 1, 2)  # [1, D, H', W']
        
        # Now activations is [1, C, H', W']
        n_channels = activations.shape[1]
        h, w = activations.shape[2], activations.shape[3]
        
        # Normalize each channel to [0, 1]
        act_min = activations.view(1, n_channels, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        act_max = activations.view(1, n_channels, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        act_range = act_max - act_min
        act_range[act_range == 0] = 1
        norm_acts = (activations - act_min) / act_range
        
        # Upsample to input size
        upsampled = F.interpolate(norm_acts, size=(H, W), mode='bilinear', align_corners=False)
        
        # Get scores for each masked input
        scores = []
        batch_size = 32
        
        for i in range(0, n_channels, batch_size):
            batch_masks = upsampled[0, i:i+batch_size]  # [B, H, W]
            
            # Apply masks to input
            masked = input_tensor * batch_masks.unsqueeze(1)  # [B, C, H, W]
            
            with torch.no_grad():
                outputs = self.model(masked)
                probs = F.softmax(outputs, dim=1)
                batch_scores = probs[:, target_class]
            
            scores.append(batch_scores)
        
        scores = torch.cat(scores)  # [n_channels]
        
        # Weight activation maps by scores
        scores = F.relu(scores)  # Only positive contributions
        weights = scores.view(1, -1, 1, 1)
        
        # Weighted sum
        cam = (weights * upsampled).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.squeeze().cpu().numpy()
    
    def __del__(self):
        self.remove_hook()


class LocalizedXAI:
    """
    Unified interface for localized XAI methods.
    
    Recommended: 'occlusion' for best quality, 'rise' for speed
    """
    
    METHODS = {
        'occlusion': OcclusionSensitivity,
        'rise': RISE,
        'scorecam': ScoreCAMForTransformer,
    }
    
    def __init__(self, model: nn.Module, method: str = 'occlusion', **kwargs):
        """
        Args:
            model: Model to explain
            method: 'occlusion', 'rise', or 'scorecam'
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}")
        
        self.method_name = method
        self.xai = self.METHODS[method](model, **kwargs)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate attribution map."""
        if hasattr(self.xai, 'generate'):
            if 'show_progress' in self.xai.generate.__code__.co_varnames:
                return self.xai.generate(input_tensor, target_class, show_progress)
            return self.xai.generate(input_tensor, target_class)
        raise NotImplementedError


# =============================================================================
# VISUALIZATION
# =============================================================================

def enhance_heatmap(heatmap: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Enhance heatmap to focus on high-attention regions.
    
    Args:
        heatmap: Input heatmap [H, W] in [0, 1]
        threshold: Values below this are suppressed
    
    Returns:
        Enhanced heatmap
    """
    # Apply threshold
    enhanced = np.where(heatmap > threshold, heatmap, 0)
    
    # Re-normalize
    if enhanced.max() > 0:
        enhanced = enhanced / enhanced.max()
    
    return enhanced


def apply_gaussian_blur(heatmap: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian blur to smooth heatmap."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(heatmap, sigma=sigma)


# =============================================================================
# TEST
# =============================================================================

def test_localized_xai():
    """Test the localized XAI methods."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    from torchvision import transforms
    from PIL import Image
    
    print("=" * 60)
    print("Testing Localized XAI Methods")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load test image
    dataset_path = Path("archive (1)/Lung Cancer Dataset/adenocarcinoma")
    images = list(dataset_path.glob("*.png"))
    
    if not images:
        print("No test images found!")
        return
    
    image_path = images[0]
    print(f"Using: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    original = np.array(image.resize((224, 224))) / 255.0
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    # Load ViT model
    from src.models.model_factory import ViTClassifier
    
    model = ViTClassifier(num_classes=5, pretrained=True)
    model.to(device)
    model.eval()
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor.to(device))
        pred = output.argmax().item()
        conf = F.softmax(output, dim=1).max().item()
    
    class_names = ['Adenocarcinoma', 'Benign', 'Large Cell', 'Normal', 'Squamous']
    print(f"Prediction: {class_names[pred]} ({conf*100:.1f}%)")
    
    # Test methods
    methods = {}
    
    print("\n[1] Occlusion (patch=16, stride=8)...")
    occ = OcclusionSensitivity(model, patch_size=16, stride=8)
    methods['Occlusion'] = occ.generate(input_tensor.clone(), pred)
    
    print("\n[2] RISE (1000 masks)...")
    rise = RISE(model, n_masks=1000, cell_size=8)
    methods['RISE'] = rise.generate(input_tensor.clone(), pred)
    
    print("\n[3] Occlusion (fine: patch=8, stride=4)...")
    occ_fine = OcclusionSensitivity(model, patch_size=8, stride=4)
    methods['Occlusion\n(fine)'] = occ_fine.generate(input_tensor.clone(), pred)
    
    # Visualize
    import cv2
    
    fig, axes = plt.subplots(2, len(methods) + 1, figsize=(14, 7))
    
    # Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original', fontsize=11)
    axes[0, 0].axis('off')
    axes[1, 0].imshow(original)
    axes[1, 0].set_title(f'{class_names[pred]}\n({conf*100:.1f}%)', fontsize=10)
    axes[1, 0].axis('off')
    
    for i, (name, heatmap) in enumerate(methods.items(), 1):
        # Heatmap
        axes[0, i].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        axes[0, i].set_title(name, fontsize=11)
        axes[0, i].axis('off')
        
        # Overlay
        heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        img_uint8 = (original * 255).astype(np.uint8)
        overlay = cv2.addWeighted(img_uint8, 0.5, heatmap_colored, 0.5, 0)
        
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'{name}\nOverlay', fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle('Localized XAI Methods for Vision Transformer\n(Region-focused explanations)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path("results/xai_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "localized_transformer_xai.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: results/xai_comparison/localized_transformer_xai.png")
    
    plt.show()


if __name__ == "__main__":
    test_localized_xai()
