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
    4. Future Extensibility: Easy to add new architectures

Supported Models:
    - resnet50: ResNet-50 (default, recommended)
    - mobilenetv2: MobileNetV2 (lightweight, fast)
    - vit_b_16: Vision Transformer Base-16 (attention-based)
    - swin_t: Swin Transformer Tiny (hierarchical attention)
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple

from .classifier import LungCancerClassifier


# =============================================================================
# MODEL CLASSES FOR DIFFERENT ARCHITECTURES
# =============================================================================

class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2-based classifier for lung cancer CT images.
    
    WHY MOBILENETV2?
        - Lightweight and fast (good for deployment)
        - Inverted residuals with linear bottlenecks
        - Good accuracy with fewer parameters
        - Suitable for mobile/edge devices
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        super(MobileNetV2Classifier, self).__init__()
        self.num_classes = num_classes
        self.model_name = "mobilenetv2"
        
        # Load pretrained MobileNetV2
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
            self.backbone = models.mobilenet_v2(weights=weights)
            print("✓ Loaded MobileNetV2 with ImageNet pretrained weights")
        else:
            self.backbone = models.mobilenet_v2(weights=None)
            print("✓ Loaded MobileNetV2 without pretrained weights")
        
        # Get the number of features from the last layer
        num_features = self.backbone.classifier[1].in_features  # 1280
        
        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        # Store features layer for Grad-CAM
        self.features = self.backbone.features
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        print("✓ Backbone layers frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_gradcam_target_layer(self):
        """Get the target layer for Grad-CAM visualization."""
        # For MobileNetV2, use the last feature layer
        return self.backbone.features[-1]
    
    def print_model_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*50}")
        print(f"Model: MobileNetV2")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Output Classes: {self.num_classes}")
        print(f"{'='*50}\n")


class ViTClassifier(nn.Module):
    """
    Vision Transformer (ViT) classifier for lung cancer CT images.
    
    WHY VISION TRANSFORMER?
        - Attention-based architecture (no convolutions)
        - Captures global relationships in images
        - State-of-the-art performance on many benchmarks
        - Novel approach worth exploring in research
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        super(ViTClassifier, self).__init__()
        self.num_classes = num_classes
        self.model_name = "vit_b_16"
        
        # Load pretrained ViT-B/16
        if pretrained:
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            self.backbone = models.vit_b_16(weights=weights)
            print("✓ Loaded ViT-B/16 with ImageNet pretrained weights")
        else:
            self.backbone = models.vit_b_16(weights=None)
            print("✓ Loaded ViT-B/16 without pretrained weights")
        
        # Get the number of features from the head
        num_features = self.backbone.heads.head.in_features  # 768
        
        # Replace the classification head
        self.backbone.heads = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        # Store encoder for Grad-CAM
        self.encoder = self.backbone.encoder
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.backbone.encoder.parameters():
            param.requires_grad = False
        for param in self.backbone.heads.parameters():
            param.requires_grad = True
        print("✓ Backbone layers frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_gradcam_target_layer(self):
        """Get the target layer for Grad-CAM visualization."""
        # For ViT, use the last encoder layer
        return self.backbone.encoder.layers[-1]
    
    def print_model_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*50}")
        print(f"Model: Vision Transformer (ViT-B/16)")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Output Classes: {self.num_classes}")
        print(f"{'='*50}\n")


class SwinTransformerClassifier(nn.Module):
    """
    Swin Transformer classifier for lung cancer CT images.
    
    WHY SWIN TRANSFORMER?
        - Hierarchical feature maps (like CNNs)
        - Shifted window attention (efficient)
        - State-of-the-art on many vision tasks
        - Good balance between accuracy and efficiency
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        super(SwinTransformerClassifier, self).__init__()
        self.num_classes = num_classes
        self.model_name = "swin_t"
        
        # Load pretrained Swin-T
        if pretrained:
            weights = models.Swin_T_Weights.IMAGENET1K_V1
            self.backbone = models.swin_t(weights=weights)
            print("✓ Loaded Swin-T with ImageNet pretrained weights")
        else:
            self.backbone = models.swin_t(weights=None)
            print("✓ Loaded Swin-T without pretrained weights")
        
        # Get the number of features from the head
        num_features = self.backbone.head.in_features  # 768
        
        # Replace the classification head
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        # Store features for Grad-CAM
        self.features = self.backbone.features
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.head.parameters():
            param.requires_grad = True
        print("✓ Backbone layers frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_gradcam_target_layer(self):
        """Get the target layer for Grad-CAM visualization."""
        # For Swin Transformer, use the last feature layer
        return self.backbone.features[-1]
    
    def print_model_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*50}")
        print(f"Model: Swin Transformer (Tiny)")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Output Classes: {self.num_classes}")
        print(f"{'='*50}\n")


class DeiTClassifier(nn.Module):
    """
    DeiT (Data-efficient Image Transformer) classifier for lung cancer CT images.
    
    WHY DeiT?
        - Data-efficient training of Vision Transformers
        - Uses knowledge distillation from CNNs
        - Better performance with limited data than vanilla ViT
        - Designed for scenarios with smaller datasets
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        super(DeiTClassifier, self).__init__()
        self.num_classes = num_classes
        self.model_name = "deit_small"
        
        try:
            import timm
        except ImportError:
            raise ImportError("timm library required for DeiT. Install with: pip install timm")
        
        # Load DeiT Small
        if pretrained:
            self.backbone = timm.create_model('deit_small_patch16_224', pretrained=True)
            print("✓ Loaded DeiT-Small with ImageNet pretrained weights")
        else:
            self.backbone = timm.create_model('deit_small_patch16_224', pretrained=False)
            print("✓ Loaded DeiT-Small without pretrained weights")
        
        # Get the number of features from the head
        num_features = self.backbone.head.in_features  # 384
        
        # Replace the classification head
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        print("✓ Backbone layers frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_gradcam_target_layer(self):
        """Get the target layer for Grad-CAM visualization."""
        return self.backbone.blocks[-1]
    
    def print_model_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*50}")
        print(f"Model: DeiT-Small (Data-efficient Image Transformer)")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Output Classes: {self.num_classes}")
        print(f"{'='*50}\n")


class MobileViTClassifier(nn.Module):
    """
    MobileViT classifier for lung cancer CT images.
    
    WHY MobileViT?
        - Combines CNN efficiency with Transformer power
        - Lightweight architecture for mobile deployment
        - Local + global feature learning
        - Good balance between accuracy and efficiency
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        super(MobileViTClassifier, self).__init__()
        self.num_classes = num_classes
        self.model_name = "mobilevit_s"
        
        try:
            import timm
        except ImportError:
            raise ImportError("timm library required for MobileViT. Install with: pip install timm")
        
        # Load MobileViT Small
        if pretrained:
            self.backbone = timm.create_model('mobilevit_s', pretrained=True)
            print("✓ Loaded MobileViT-S with ImageNet pretrained weights")
        else:
            self.backbone = timm.create_model('mobilevit_s', pretrained=False)
            print("✓ Loaded MobileViT-S without pretrained weights")
        
        # Get the number of features from the head
        num_features = self.backbone.head.fc.in_features  # 640
        
        # Replace the classification head
        self.backbone.head.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        print("✓ Backbone layers frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_gradcam_target_layer(self):
        """Get the target layer for Grad-CAM visualization."""
        return self.backbone.stages[-1]
    
    def print_model_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*50}")
        print(f"Model: MobileViT-S (Mobile Vision Transformer)")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Output Classes: {self.num_classes}")
        print(f"{'='*50}\n")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_model(
    model_name: str = "resnet50",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Factory function to create a classification model.
    
    Args:
        model_name: Name of the model architecture
            - "resnet50": ResNet-50 (default)
            - "mobilenetv2": MobileNetV2
            - "vit_b_16": Vision Transformer Base-16
            - "swin_t": Swin Transformer Tiny
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
        >>> model = create_model("resnet50", num_classes=5, pretrained=True)
        >>> model = model.to(device)
    """
    # Normalize model name
    model_name = model_name.lower().strip()
    
    print(f"\n{'=' * 60}")
    print(f"Creating Model: {model_name.upper()}")
    print(f"{'=' * 60}")
    
    # Create the appropriate model
    if model_name == "resnet50":
        model = LungCancerClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone
        )
    
    elif model_name == "mobilenetv2":
        model = MobileNetV2Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone
        )
    
    elif model_name == "vit_b_16":
        model = ViTClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone
        )
    
    elif model_name == "swin_t":
        model = SwinTransformerClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone
        )
    
    elif model_name == "deit_small":
        model = DeiTClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone
        )
    
    elif model_name == "mobilevit_s":
        model = MobileViTClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone
        )
    
    else:
        supported = ["resnet50", "mobilenetv2", "vit_b_16", "swin_t", "deit_small", "mobilevit_s"]
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


def get_model_info(model_name: str) -> dict:
    """
    Get information about a model without creating it.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary with model information
    """
    info = {
        "resnet50": {
            "name": "ResNet-50",
            "params": "~25.6M",
            "gradcam_layer": "layer4",
            "description": "Deep residual network with skip connections"
        },
        "mobilenetv2": {
            "name": "MobileNetV2",
            "params": "~3.5M",
            "gradcam_layer": "features",
            "description": "Lightweight network with inverted residuals"
        },
        "vit_b_16": {
            "name": "Vision Transformer (ViT-B/16)",
            "params": "~86M",
            "gradcam_layer": "encoder",
            "description": "Attention-based transformer architecture"
        },
        "swin_t": {
            "name": "Swin Transformer (Tiny)",
            "params": "~28M",
            "gradcam_layer": "features",
            "description": "Hierarchical transformer with shifted windows"
        },
        "deit_small": {
            "name": "DeiT-Small",
            "params": "~22M",
            "gradcam_layer": "blocks",
            "description": "Data-efficient Image Transformer with distillation"
        },
        "mobilevit_s": {
            "name": "MobileViT-S",
            "params": "~5.6M",
            "gradcam_layer": "stages",
            "description": "Mobile-friendly Vision Transformer hybrid"
        }
    }
    return info.get(model_name.lower(), {"name": "Unknown", "params": "N/A"})


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
