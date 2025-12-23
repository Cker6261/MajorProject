# =============================================================================
# LUNG CANCER CLASSIFIER
# ResNet-50 based model for multi-class lung cancer classification
# =============================================================================
"""
Main Classification Model for Lung Cancer CT Images.

WHY RESNET-50?
    1. Proven Performance: State-of-the-art results on ImageNet
    2. Residual Connections: Solves vanishing gradient problem in deep networks
    3. Transfer Learning: Pretrained weights capture useful visual features
    4. Grad-CAM Compatible: Works well with our XAI requirements
    5. Well-Documented: Easy to explain in viva

WHY TRANSFER LEARNING FOR MEDICAL IMAGING?
    1. Limited Data: Medical datasets are typically small
    2. Feature Reuse: Low-level features (edges, textures) transfer well
    3. Faster Convergence: Pretrained weights provide good initialization
    4. Better Generalization: Reduces overfitting on small datasets

Reference:
    He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class LungCancerClassifier(nn.Module):
    """
    ResNet-50 based classifier for lung cancer CT images.
    
    Architecture:
        - Base: ResNet-50 pretrained on ImageNet
        - Modified: Final fully connected layer for 4-class classification
        - Optional: Dropout for regularization
    
    Attributes:
        num_classes: Number of output classes (default: 4)
        backbone: The ResNet-50 feature extractor
        fc: The final classification layer
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Initialize the Lung Cancer Classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout probability before final layer
            freeze_backbone: Whether to freeze base layers (for fine-tuning)
        """
        super(LungCancerClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained ResNet-50
        # Using weights parameter (new PyTorch API)
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
            print("✓ Loaded ResNet-50 with ImageNet pretrained weights")
        else:
            self.backbone = models.resnet50(weights=None)
            print("✓ Loaded ResNet-50 without pretrained weights")
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features  # 2048 for ResNet-50
        
        # Replace the final fully connected layer
        # Original: fc(2048 → 1000) for ImageNet
        # Modified: fc(2048 → num_classes) for our task
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        # Optionally freeze backbone layers
        if freeze_backbone:
            self._freeze_backbone()
            print("✓ Backbone layers frozen (only training classifier head)")
    
    def _freeze_backbone(self) -> None:
        """
        Freeze all layers except the final classification layer.
        
        WHY FREEZE?
            - Useful when dataset is very small
            - Prevents overfitting by keeping pretrained features
            - Faster training (fewer gradients to compute)
        """
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the final fc layer
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all layers for full fine-tuning.
        
        Call this after initial training to fine-tune the entire network.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ All layers unfrozen for fine-tuning")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 224, 224]
        
        Returns:
            Output logits of shape [batch_size, num_classes]
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classification layer.
        
        Useful for:
            - Feature visualization
            - Transfer to other tasks
            - Similarity comparisons
        
        Args:
            x: Input tensor of shape [batch_size, 3, 224, 224]
        
        Returns:
            Feature tensor of shape [batch_size, 2048]
        """
        # Forward through all layers except fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def get_gradcam_target_layer(self):
        """
        Get the target layer for Grad-CAM visualization.
        
        For ResNet-50, we use layer4 (the last convolutional block).
        This gives us the most semantically meaningful activations.
        
        Returns:
            The target layer for Grad-CAM
        """
        return self.backbone.layer4
    
    def count_parameters(self) -> dict:
        """
        Count trainable and total parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
    
    def print_model_summary(self) -> None:
        """Print a summary of the model architecture."""
        params = self.count_parameters()
        
        print("\n" + "=" * 50)
        print("MODEL SUMMARY: LungCancerClassifier")
        print("=" * 50)
        print(f"Base Architecture: ResNet-50")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Total Parameters: {params['total']:,}")
        print(f"Trainable Parameters: {params['trainable']:,}")
        print(f"Frozen Parameters: {params['frozen']:,}")
        print("=" * 50 + "\n")
