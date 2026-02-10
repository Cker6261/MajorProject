# Baseline vs Fine-tuned Model Comparison

## CNN Model Analysis for Lung Cancer Classification

This document compares baseline (trained from scratch) vs fine-tuned (ImageNet pretrained) performance for our CNN models.

---

## Summary

| Model | Baseline Accuracy | Fine-tuned Accuracy | Improvement | Parameters |
|-------|-------------------|---------------------|-------------|------------|
| **MobileNetV2 (Primary)** | 89.61% | **97.40%** | +7.79% | 2.2M |
| ResNet-50 | 78.79% | 96.97% | +18.18% | 23.5M |
| VGG-16 | 71.43% | - | - | 138M |
| DenseNet-121 | 84.42% | - | - | 7.0M |
| EfficientNet-B0 | 72.29% | - | - | 5.3M |

---

## 1. Baseline Results (Training from Scratch)

Models trained from random initialization without pretrained weights.

### Training Configuration
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 30
- Pretrained: **False**

### Results

| Model | Test Accuracy | Precision | Recall | F1-Score |
|-------|---------------|-----------|--------|----------|
| **MobileNetV2** | **89.61%** | 0.899 | 0.896 | 0.894 |
| DenseNet-121 | 84.42% | 0.857 | 0.844 | 0.826 |
| ResNet-50 | 78.79% | 0.794 | 0.788 | 0.790 |
| EfficientNet-B0 | 72.29% | 0.736 | 0.723 | 0.726 |
| VGG-16 | 71.43% | 0.698 | 0.714 | 0.694 |

---

## 2. Fine-tuned Results (Transfer Learning)

Models initialized with ImageNet weights and fine-tuned.

### Training Configuration
- Optimizer: AdamW
- Learning Rate: 1e-4
- Weight Decay: 0.01
- Batch Size: 32
- Epochs: 50 (early stopping, patience=10)
- Pretrained: **True (ImageNet)**

### Results

| Model | Test Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|---------------|-----------|--------|----------|---------|
| **MobileNetV2** | **97.40%** | **0.975** | **0.974** | **0.974** | 0.999 |
| ResNet-50 | 96.97% | 0.970 | 0.970 | 0.970 | 0.999 |

---

## 3. Transfer Learning Impact

| Model | Baseline | Fine-tuned | Improvement |
|-------|----------|------------|-------------|
| **MobileNetV2** | 89.61% | **97.40%** | **+7.79%** |
| ResNet-50 | 78.79% | 96.97% | +18.18% |

### Key Insights
- MobileNetV2 has highest baseline (89.61%) - efficient architecture learns well from scratch
- DenseNet-121 baseline (84.42%) - dense connections provide good feature reuse
- ResNet-50 benefits most from transfer learning (+18.18%)
- VGG-16 and EfficientNet-B0 struggle without pretrained weights (71-72%)
- **MobileNetV2 achieves best final accuracy** (97.40%)

---

## 4. Model Selection

### Primary Model: MobileNetV2
- Best accuracy: **97.40%** (fine-tuned)
- Best baseline: **89.61%** (from scratch)
- Smallest size: **2.2M parameters**
- Best GradCAM: Focus score 0.51-0.58
- Ideal for deployment

### CNN Baselines (for comparison)
| Model | Baseline Acc | Parameters | Use Case |
|-------|--------------|------------|----------|
| DenseNet-121 | 84.42% | 7.0M | Good feature reuse |
| ResNet-50 | 78.79% | 23.5M | Deep residual learning |
| EfficientNet-B0 | 72.29% | 5.3M | Compound scaling |
| VGG-16 | 71.43% | 138M | Classic deep CNN |

---

*LungXAI Project - February 2026*
