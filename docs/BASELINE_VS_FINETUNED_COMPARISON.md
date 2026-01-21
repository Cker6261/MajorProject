# Model Comparison Study: Fine-tuned vs Baseline (From Scratch)

## Lung Cancer CT Image Classification Project

**Date:** January 20, 2026  
**Project:** LungXAI - Explainable AI for Lung Cancer Classification

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [What We Did](#what-we-did)
3. [Models Compared](#models-compared)
4. [Training Configuration](#training-configuration)
5. [Results Summary](#results-summary)
6. [Detailed Analysis](#detailed-analysis)
7. [Key Findings](#key-findings)
8. [Files Created](#files-created)
9. [How to Reproduce](#how-to-reproduce)

---

## Overview

This document explains the comparison study between **fine-tuned models** (using ImageNet pretrained weights) and **baseline models** (trained from scratch with random initialization).

### Why This Comparison Matters

In deep learning, especially for medical imaging:
- **Transfer Learning** uses knowledge from large datasets (like ImageNet) to help with smaller, specialized datasets
- **Training from Scratch** starts with random weights and learns everything from the target dataset only

This comparison demonstrates the **value of transfer learning** for medical image classification tasks with limited data.

---

## What We Did

### Step 1: Trained Fine-tuned Models (Already Existing)

The project already had models trained with **pretrained ImageNet weights**:
- Models were initialized with weights learned from 1.2 million ImageNet images
- These weights were then **fine-tuned** on our lung cancer CT dataset
- This is called "transfer learning"

### Step 2: Trained Baseline Models (New)

We created new models trained **from scratch**:
- Models were initialized with **random weights**
- No pretrained knowledge was used
- Models had to learn everything from our 1,535 lung cancer CT images only

### Step 3: Compared Both Approaches

We evaluated all models on the same test set and compared:
- Test Accuracy
- Precision, Recall, F1 Score
- Per-class performance

---

## Models Compared

### Models with Both Versions (Fine-tuned + Baseline)

| Model | Architecture Type | Parameters | Description |
|-------|------------------|------------|-------------|
| **ResNet-50** | CNN | ~25.6M | Deep residual network with skip connections |
| **MobileNetV2** | CNN | ~2.2M | Lightweight network with inverted residuals |
| **ViT-B/16** | Transformer | ~86M | Vision Transformer with 16x16 patches |
| **Swin-T** | Transformer | ~28M | Hierarchical transformer with shifted windows |

### Baseline-Only Models (Additional Comparison)

| Model | Architecture Type | Parameters | Description |
|-------|------------------|------------|-------------|
| **DeiT-Small** | Transformer | ~22M | Data-efficient Image Transformer |
| **MobileViT-S** | Hybrid (CNN+Transformer) | ~5M | Mobile Vision Transformer |

---

## Training Configuration

### Dataset Information

```
Dataset: Lung Cancer CT Scan Images
Total Images: 1,535
Classes: 5
  - adenocarcinoma (337 images, 22.0%)
  - Benign cases (120 images, 7.8%)
  - large cell carcinoma (187 images, 12.2%)
  - Normal cases (631 images, 41.1%)
  - squamous cell carcinoma (260 images, 16.9%)

Data Split:
  - Training: 70% (1,074 images)
  - Validation: 15% (230 images)
  - Test: 15% (231 images)
```

### Fine-tuned Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Pretrained | **True** | Uses ImageNet weights |
| Epochs | 50 | Maximum training epochs |
| Learning Rate | 1e-4 (0.0001) | AdamW optimizer |
| Weight Decay | 1e-4 | L2 regularization |
| Batch Size | 32 | Images per batch |
| Optimizer | AdamW | Adam with weight decay |
| Scheduler | StepLR | Learning rate decay |
| Dropout | 0.5 | Regularization |
| Early Stopping | Yes | Patience: 10 epochs |

### Baseline Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Pretrained | **False** | Random initialization |
| Epochs | 30 | Training epochs |
| Learning Rate | 1e-3 (0.001) | Simple Adam optimizer |
| Weight Decay | 0 | No weight decay |
| Batch Size | 32 | Images per batch |
| Optimizer | Adam | Simple Adam (no AdamW) |
| Scheduler | None | No learning rate scheduling |
| Dropout | 0.5 | Regularization |
| Early Stopping | No | Train for fixed epochs |

### Key Differences

| Aspect | Fine-tuned | Baseline |
|--------|-----------|----------|
| **Starting Point** | ImageNet pretrained weights | Random weights |
| **Prior Knowledge** | 1.2M ImageNet images | None |
| **Learning Rate** | Lower (1e-4) | Higher (1e-3) |
| **Optimizer** | AdamW (with weight decay) | Adam (simple) |
| **Scheduler** | StepLR decay | None |
| **Weight Decay** | 1e-4 | 0 |

---

## Results Summary

### Main Comparison Table

| Model | Fine-tuned Accuracy | Baseline Accuracy | **Improvement** |
|-------|--------------------:|------------------:|----------------:|
| **MobileNetV2** | 97.84% | 83.98% | **+13.85%** |
| **ResNet-50** | 96.97% | 82.25% | **+14.72%** |
| **Swin-T** | 96.54% | 58.44% | **+38.10%** |
| **ViT-B/16** | 91.34% | 64.07% | **+27.27%** |

### Baseline-Only Models

| Model | Baseline Accuracy | F1 Score |
|-------|------------------:|---------:|
| **MobileViT-S** | 67.97% | 63.00% |
| **DeiT-Small** | 63.64% | 54.34% |

### Visual Improvement Chart

```
                        Accuracy Improvement (Fine-tuned over Baseline)
Model          |  Improvement
---------------|----------------------------------------
MobileNetV2    | ████████████████ +13.85%
ResNet-50      | █████████████████ +14.72%
ViT-B/16       | ██████████████████████████████ +27.27%
Swin-T         | ██████████████████████████████████████ +38.10%
```

---

## Detailed Analysis

### CNN Models (ResNet-50, MobileNetV2)

**Fine-tuned Performance:**
- Both achieved **>96% accuracy**
- Excellent performance across all classes
- Low test loss (<0.15)

**Baseline Performance:**
- Both achieved **~82-84% accuracy**
- Reasonable performance even without pretraining
- CNNs can learn basic features from scratch with enough data

**Analysis:**
- CNNs are relatively robust to training from scratch
- The convolutional architecture naturally captures local patterns
- Still, transfer learning provides ~14% improvement

### Transformer Models (ViT, Swin-T, DeiT)

**Fine-tuned Performance:**
- Swin-T: 96.54% (excellent)
- ViT-B/16: 91.34% (very good)

**Baseline Performance:**
- Swin-T: 58.44% (poor - only slightly better than random)
- ViT-B/16: 64.07% (poor)
- DeiT-Small: 63.64% (poor)

**Analysis:**
- **Transformers struggle massively without pretrained weights**
- They need massive amounts of data to learn from scratch
- With only 1,074 training images, transformers cannot learn meaningful patterns
- Transfer learning is **essential** for transformers on small datasets

### Hybrid Model (MobileViT-S)

**Baseline Performance:**
- 67.97% accuracy (better than pure transformers)
- The CNN component helps with limited data

**Analysis:**
- Hybrid architectures (CNN + Transformer) perform better from scratch
- CNN layers provide inductive biases that help with small datasets
- Still far behind fine-tuned CNN performance

---

## Key Findings

### 1. Transfer Learning is Crucial
- Average improvement: **+23.5% accuracy**
- All models benefit significantly from pretrained weights

### 2. Transformers Need Pretrained Weights
- Without pretraining: ~58-64% accuracy
- With pretraining: ~91-97% accuracy
- **Improvement of 27-38%** for transformer models

### 3. CNNs are More Robust
- Even without pretraining, CNNs achieve ~82-84%
- Their architecture naturally suits image tasks
- Pretrained weights still help (+14% improvement)

### 4. Model Size vs Data
- ViT-B/16 has 86M parameters but only 1,074 training images
- This is a **severe mismatch** without pretraining
- Pretrained weights bridge this gap

### 5. Best Overall Model
- **MobileNetV2 (Fine-tuned)**: 97.84% accuracy
- Smallest model (2.2M params) with best accuracy
- Shows that efficient architectures + transfer learning = excellent results

---

## Files Created

### Training Scripts

| File | Description |
|------|-------------|
| `train_baseline_models.py` | Main script to train all baseline models |
| `train_new_baselines.py` | Script to train DeiT and MobileViT baselines |
| `compare_finetuned_vs_baseline.py` | Comparison and evaluation script |

### Model Checkpoints

**Location:** `checkpoints/baseline/`

| File | Model |
|------|-------|
| `best_model_resnet50_baseline.pth` | ResNet-50 (best validation) |
| `best_model_mobilenetv2_baseline.pth` | MobileNetV2 (best validation) |
| `best_model_vit_b_16_baseline.pth` | ViT-B/16 (best validation) |
| `best_model_swin_t_baseline.pth` | Swin-T (best validation) |
| `best_model_deit_small_baseline.pth` | DeiT-Small (best validation) |
| `best_model_mobilevit_s_baseline.pth` | MobileViT-S (best validation) |
| `final_model_*_baseline.pth` | Final epoch checkpoints |

### Results Files

**Location:** `results/baseline/`

| File | Description |
|------|-------------|
| `all_baseline_results.json` | All baseline training results |
| `*_baseline_results.json` | Individual model results |

**Location:** `results/finetuned_vs_baseline/`

| File | Description |
|------|-------------|
| `comparison_results.json` | Complete comparison data |
| `comparison_report.md` | Markdown comparison report |

---

## How to Reproduce

### Prerequisites

```bash
# Ensure you're in the project directory
cd "D:\Major Project"

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

### Step 1: Train Baseline Models

```bash
# Train all baseline models (ResNet50, MobileNetV2, ViT, Swin, DeiT, MobileViT)
python train_baseline_models.py

# Or to force retrain even if cached:
python train_baseline_models.py --force
```

### Step 2: Run Comparison

```bash
# Compare fine-tuned vs baseline models
python compare_finetuned_vs_baseline.py
```

### Step 3: View Results

Results are saved in:
- `results/finetuned_vs_baseline/comparison_report.md`
- `results/finetuned_vs_baseline/comparison_results.json`

---

## Technical Notes

### Why Baseline Models Use Different Hyperparameters

1. **Higher Learning Rate (1e-3 vs 1e-4)**
   - Random weights need larger updates to find good solutions
   - Pretrained weights are already near good solutions, need fine adjustments

2. **No Weight Decay**
   - Simpler optimization for baseline comparison
   - Shows raw model capability without regularization tricks

3. **No Learning Rate Scheduler**
   - Keeps training simple and comparable
   - Shows what models can achieve with minimal tuning

4. **Fewer Epochs (30 vs 50)**
   - Most baseline models plateau or overfit quickly
   - Avoids wasting compute on stuck training

### GPU Memory Considerations

- Models are trained one at a time to avoid memory issues
- GPU memory is cleared between models
- 6GB VRAM is sufficient for all models at batch size 32

---

## Conclusion

This comparison study clearly demonstrates that:

1. **Transfer learning is essential** for medical imaging with limited data
2. **Transformers particularly benefit** from pretrained weights (~30% improvement)
3. **CNNs are more forgiving** but still benefit from pretraining (~14% improvement)
4. **MobileNetV2 with fine-tuning** achieves the best accuracy (97.84%) despite being the smallest model

For lung cancer CT classification with ~1,500 images, **always use pretrained models** - the accuracy gain is substantial and consistent across all architectures.

---

*This documentation was generated as part of the LungXAI project for lung cancer classification using explainable AI.*
