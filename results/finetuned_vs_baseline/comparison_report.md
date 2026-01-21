# Fine-tuned vs Baseline Model Comparison Report

**Generated:** 2026-01-20 22:11:13

## Overview

This report compares models trained with two approaches:
- **Fine-tuned (Transfer Learning):** Models initialized with ImageNet pretrained weights
- **Baseline (From Scratch):** Models trained from random initialization

---

## Summary Table

| Model | Fine-tuned Acc | Baseline Acc | Improvement |
|-------|----------------|--------------|-------------|
| ResNet-50 | 96.97% | 82.25% | +14.72% |
| MobileNetV2 | 97.84% | 83.98% | +13.85% |
| ViT-B/16 | 91.34% | 64.07% | +27.27% |
| Swin-T | 96.54% | 58.44% | +38.10% |

---

## Detailed Results

### ResNet-50

**Fine-tuned (Transfer Learning):**
- Test Accuracy: 96.97%
- Test Loss: 0.0849
- Precision: 96.99%
- Recall: 96.97%
- F1 Score: 96.95%

**Baseline (From Scratch):**
- Test Accuracy: 82.25%
- Test Loss: 0.4726
- Precision: 83.37%
- Recall: 82.25%
- F1 Score: 80.96%

**Improvement (Fine-tuned over Baseline):**
- Accuracy Improvement: +14.72%
- Relative Improvement: 17.9%

### MobileNetV2

**Fine-tuned (Transfer Learning):**
- Test Accuracy: 97.84%
- Test Loss: 0.1137
- Precision: 97.87%
- Recall: 97.84%
- F1 Score: 97.83%

**Baseline (From Scratch):**
- Test Accuracy: 83.98%
- Test Loss: 0.4096
- Precision: 86.61%
- Recall: 83.98%
- F1 Score: 84.40%

**Improvement (Fine-tuned over Baseline):**
- Accuracy Improvement: +13.85%
- Relative Improvement: 16.5%

### ViT-B/16

**Fine-tuned (Transfer Learning):**
- Test Accuracy: 91.34%
- Test Loss: 0.2779
- Precision: 91.63%
- Recall: 91.34%
- F1 Score: 91.32%

**Baseline (From Scratch):**
- Test Accuracy: 64.07%
- Test Loss: 0.8276
- Precision: 48.89%
- Recall: 64.07%
- F1 Score: 54.83%

**Improvement (Fine-tuned over Baseline):**
- Accuracy Improvement: +27.27%
- Relative Improvement: 42.6%

### Swin-T

**Fine-tuned (Transfer Learning):**
- Test Accuracy: 96.54%
- Test Loss: 0.1382
- Precision: 96.55%
- Recall: 96.54%
- F1 Score: 96.53%

**Baseline (From Scratch):**
- Test Accuracy: 58.44%
- Test Loss: 1.0070
- Precision: 38.84%
- Recall: 58.44%
- F1 Score: 46.35%

**Improvement (Fine-tuned over Baseline):**
- Accuracy Improvement: +38.10%
- Relative Improvement: 65.2%

---

## Baseline-Only Models

These models were only trained from scratch (no fine-tuned versions):

| Model | Baseline Acc | F1 Score |
|-------|--------------|----------|
| DeiT-Small | 63.64% | 54.34% |
| MobileViT-S | 67.97% | 63.00% |

### DeiT-Small (Baseline Only)

*Note: Baseline only - no fine-tuned version available*

- Test Accuracy: 63.64%
- Test Loss: 0.8266
- Precision: 48.46%
- Recall: 63.64%
- F1 Score: 54.34%

### MobileViT-S (Baseline Only)

*Note: Baseline only - no fine-tuned version available*

- Test Accuracy: 67.97%
- Test Loss: 0.6742
- Precision: 61.70%
- Recall: 67.97%
- F1 Score: 63.00%

---

## Conclusion

Transfer learning with pretrained ImageNet weights provides significant benefits for medical image classification:
- Faster convergence
- Higher accuracy
- Better generalization

This demonstrates the effectiveness of using pretrained models for lung cancer CT image classification.