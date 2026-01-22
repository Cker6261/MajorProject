# =============================================================================
# FINAL XAI COMPARISON
# Production-ready comparison of CNN GradCAM vs Transformer XAI
# =============================================================================
"""
This is the final, optimized comparison script.
Uses the best available XAI method for each model type.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
import cv2
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def load_trained_model(model_name: str, checkpoint_dir: Path = Path("checkpoints")):
    """Load a trained model from checkpoint."""
    from src.models.model_factory import create_model
    
    patterns = [
        checkpoint_dir / f"best_model_{model_name}.pth",
        checkpoint_dir / f"final_model_{model_name}.pth",
    ]
    
    checkpoint_path = None
    for p in patterns:
        if p.exists():
            checkpoint_path = p
            break
    
    if checkpoint_path is None:
        print(f"  No trained checkpoint for {model_name}")
        return None, False
    
    print(f"  Loading: {checkpoint_path.name}")
    model = create_model(model_name, num_classes=5, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    try:
        model.load_state_dict(state_dict, strict=True)
        return model, True
    except:
        try:
            model.load_state_dict(state_dict, strict=False)
            return model, True
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return None, False


def load_test_image():
    """Load a test image."""
    base_path = Path("archive (1)/Lung Cancer Dataset")
    
    for cls in ['adenocarcinoma', 'squamous cell carcinoma']:
        cls_path = base_path / cls
        if cls_path.exists():
            imgs = list(cls_path.glob("*.png"))
            if imgs:
                img_path = imgs[0]
                img = Image.open(img_path).convert('RGB')
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                return transform(img), np.array(img.resize((224, 224))) / 255.0, img_path
    
    raise FileNotFoundError("No images found")


def overlay(image, heatmap, alpha=0.5):
    """Create clean overlay visualization."""
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap.astype(np.float32), (image.shape[1], image.shape[0]))
    
    # Ensure clean normalization
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = (heatmap * 255).astype(np.uint8)
    
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return cv2.addWeighted(image, 1-alpha, colored, alpha, 0)


def analyze_heatmap(heatmap, name):
    """Compute quality metrics for a heatmap."""
    # Focus score: concentration of activation
    sorted_vals = np.sort(heatmap.flatten())[::-1]
    top_10_pct = sorted_vals[:int(len(sorted_vals)*0.1)].sum()
    total = sorted_vals.sum() + 1e-8
    focus = top_10_pct / total
    
    # Smoothness
    gx = np.gradient(heatmap, axis=0)
    gy = np.gradient(heatmap, axis=1)
    smoothness = 1 - np.sqrt(gx**2 + gy**2).mean()
    
    return {
        'name': name,
        'focus': focus,
        'smoothness': smoothness,
        'peak': heatmap.max(),
        'mean': heatmap.mean()
    }


def main():
    print("=" * 70)
    print("PRODUCTION XAI COMPARISON")
    print("CNN GradCAM vs Optimized Transformer XAI")
    print("=" * 70)
    
    # Load test image
    tensor, original, img_path = load_test_image()
    print(f"\nTest Image: {img_path}")
    
    class_names = ['Adenocarcinoma', 'Benign', 'Large Cell', 'Normal', 'Squamous']
    
    results = {}  # {model_name: heatmap}
    predictions = {}
    metrics = []
    
    # ===== CNN: ResNet50 with GradCAM =====
    print("\n" + "="*60)
    print("1. ResNet50 (CNN) with GradCAM")
    print("="*60)
    
    from src.xai.gradcam import GradCAM
    
    resnet, loaded = load_trained_model('resnet50')
    if resnet:
        resnet.to(device).eval()
        
        with torch.no_grad():
            out = resnet(tensor.unsqueeze(0).to(device))
            pred = out.argmax().item()
            conf = F.softmax(out, dim=1).max().item()
        
        predictions['ResNet50'] = (class_names[pred], conf)
        print(f"  Prediction: {class_names[pred]} ({conf*100:.1f}%)")
        
        target_layer = resnet.get_gradcam_target_layer()
        gradcam = GradCAM(resnet, target_layer)
        heatmap = gradcam.generate(tensor.unsqueeze(0).clone(), pred)
        gradcam.remove_hooks()
        
        results['ResNet50\n(GradCAM)'] = heatmap
        metrics.append(analyze_heatmap(heatmap, 'ResNet50 GradCAM'))
        print("  ✓ Complete")
    
    # ===== Transformer: Swin-T with Best XAI =====
    print("\n" + "="*60)
    print("2. Swin-T (Transformer) with Optimized XAI")
    print("="*60)
    
    from src.xai.best_transformer_xai import HighQualityTransformerXAI
    
    swin, loaded = load_trained_model('swin_t')
    if swin:
        swin.to(device).eval()
        
        with torch.no_grad():
            out = swin(tensor.unsqueeze(0).to(device))
            pred = out.argmax().item()
            conf = F.softmax(out, dim=1).max().item()
        
        predictions['Swin-T'] = (class_names[pred], conf)
        print(f"  Prediction: {class_names[pred]} ({conf*100:.1f}%)")
        
        xai = HighQualityTransformerXAI(swin)
        heatmap = xai.generate(tensor.unsqueeze(0).clone(), pred, show_progress=True)
        
        results['Swin-T\n(Best XAI)'] = heatmap
        metrics.append(analyze_heatmap(heatmap, 'Swin-T Best XAI'))
        print("  ✓ Complete")
    
    # ===== Transformer: ViT with Best XAI =====
    print("\n" + "="*60)
    print("3. ViT-B/16 (Transformer) with Optimized XAI")
    print("="*60)
    
    vit, loaded = load_trained_model('vit_b_16')
    if vit:
        vit.to(device).eval()
        
        with torch.no_grad():
            out = vit(tensor.unsqueeze(0).to(device))
            pred = out.argmax().item()
            conf = F.softmax(out, dim=1).max().item()
        
        predictions['ViT-B/16'] = (class_names[pred], conf)
        print(f"  Prediction: {class_names[pred]} ({conf*100:.1f}%)")
        
        xai = HighQualityTransformerXAI(vit)
        heatmap = xai.generate(tensor.unsqueeze(0).clone(), pred, show_progress=True)
        
        results['ViT-B/16\n(Best XAI)'] = heatmap
        metrics.append(analyze_heatmap(heatmap, 'ViT-B/16 Best XAI'))
        print("  ✓ Complete")
    
    # ===== Create Visualization =====
    print("\n" + "="*60)
    print("Creating Visualization")
    print("="*60)
    
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original\nCT Scan', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Model results
    for i, (model_name, heatmap) in enumerate(results.items()):
        ax = axes[i + 1]
        
        # Get prediction
        clean_name = model_name.split('\n')[0]
        if clean_name in predictions:
            pred_label, conf = predictions[clean_name]
            title = f'{model_name}\n{pred_label} ({conf*100:.0f}%)'
        else:
            title = model_name
        
        ov = overlay(original.copy(), heatmap, alpha=0.55)
        ax.imshow(ov)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Lung Cancer XAI: CNN GradCAM vs Transformer Interpretability',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path("results/xai_comparison/production_xai_comparison.png")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    plt.show()
    
    # ===== Print Metrics =====
    print("\n" + "="*70)
    print("QUALITY METRICS")
    print("="*70)
    print(f"\n{'Method':<25} {'Focus':>8} {'Smooth':>8} {'Peak':>8}")
    print("-" * 55)
    
    for m in metrics:
        print(f"{m['name']:<25} {m['focus']:.3f}    {m['smoothness']:.3f}    {m['peak']:.3f}")
    
    print("\n" + "="*70)
    print("USAGE RECOMMENDATION")
    print("="*70)
    print("""
For your LungXAI project:

  CNN Models (ResNet50, MobileNetV2):
    → Use GradCAM from src/xai/gradcam.py
    → Fast and highly focused

  Transformer Models (Swin-T, ViT, DeiT):
    → Use HighQualityTransformerXAI from src/xai/best_transformer_xai.py
    → Combines saliency + gradient attention + targeted occlusion
    → Produces focused, clinically meaningful heatmaps

Integration example:
    from src.xai.best_transformer_xai import best_transformer_xai
    heatmap = best_transformer_xai(model, image_tensor, predicted_class)
""")


if __name__ == "__main__":
    main()
