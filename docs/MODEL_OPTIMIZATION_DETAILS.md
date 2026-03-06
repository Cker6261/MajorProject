# Model Optimization Details

## LungXAI - Lung Cancer Classification Project

This document provides a comprehensive overview of all optimization techniques and hyperparameter configurations used across the fine-tuned models in this project.

---

## Table of Contents

1. [Overview](#overview)
2. [Bayesian Optimization (Swin-T Only)](#bayesian-optimization-swin-t-only)
3. [Standard Fine-Tuned Models Configuration](#standard-fine-tuned-models-configuration)
4. [Baseline Models Configuration](#baseline-models-configuration)
5. [Common Optimization Techniques](#common-optimization-techniques)
6. [Model Performance Summary](#model-performance-summary)
7. [Recommendations](#recommendations)

---

## Overview

### Models Trained in This Project

| Category | Models |
|----------|--------|
| **CNN Models** | ResNet-50, MobileNetV2, VGG-16, DenseNet-121, EfficientNet-B0 |
| **Transformer Models** | ViT-B/16, Swin-T, DeiT-Small, MobileViT-S |

### Bayesian Optimization Summary

| Model | Bayesian Optimization | Optimization Method |
|-------|:---------------------:|---------------------|
| **Swin-T (Transformer)** | ✅ Yes | Optuna TPE (30 trials) |
| MobileNetV2 | ❌ No | Manual hyperparameters |
| ResNet-50 | ❌ No | Manual hyperparameters |
| ViT-B/16 | ❌ No | Manual hyperparameters |
| DeiT-Small | ❌ No | Manual hyperparameters |
| MobileViT-S | ❌ No | Manual hyperparameters |
| DenseNet-121 | ❌ No | Manual hyperparameters |
| EfficientNet-B0 | ❌ No | Manual hyperparameters |
| VGG-16 | ❌ No | Manual hyperparameters |

---

## Bayesian Optimization (Swin-T Only)

### Overview

Only the **Swin Transformer (Tiny)** model underwent Bayesian hyperparameter optimization using the Optuna framework with Tree-structured Parzen Estimator (TPE) sampling.

**Training Script**: `archive_transformer_files/train_swin_bayesian.py`

### Optimization Configuration

| Parameter | Value |
|-----------|-------|
| **Method** | Tree-structured Parzen Estimator (TPE) |
| **Framework** | Optuna |
| **Number of Trials** | 30 |
| **Epochs per Trial** | 15 |
| **Final Training Epochs** | 50 |
| **Pruning Strategy** | MedianPruner (n_startup_trials=5, n_warmup_steps=3) |
| **Optimization Time** | ~328 minutes |
| **Final Training Time** | ~53 minutes |
| **Total Time** | ~381 minutes (~6.4 hours) |

### Hyperparameter Search Space

| Hyperparameter | Search Range | Scale | Description |
|----------------|--------------|-------|-------------|
| Learning Rate | 1e-6 to 1e-3 | Log | Base learning rate for AdamW |
| Weight Decay | 1e-6 to 1e-2 | Log | L2 regularization strength |
| Dropout Rate | 0.1 to 0.7 | Linear | Classifier dropout probability |
| Beta1 (Adam) | 0.85 to 0.99 | Linear | First moment decay rate |
| Beta2 (Adam) | 0.9 to 0.9999 | Linear | Second moment decay rate |
| LR Scheduler Gamma | 0.1 to 0.9 | Linear | LR decay factor |
| LR Scheduler Step | 3 to 10 | Integer | Epochs between LR decay |
| Warmup Epochs | 0 to 5 | Integer | Learning rate warmup period |
| Label Smoothing | 0.0 to 0.2 | Linear | Soft label regularization |

### Optimal Hyperparameters Found

| Hyperparameter | Bayesian Optimized | Default Value | Change |
|----------------|-------------------|---------------|--------|
| **Learning Rate** | 0.000155 | 0.0001 | +55% |
| **Weight Decay** | 0.000023 | 0.0001 | -77% |
| **Dropout Rate** | 0.333 | 0.5 | -33% |
| **Beta1 (Adam)** | 0.879 | 0.9 | -2.3% |
| **Beta2 (Adam)** | 0.9997 | 0.999 | +0.07% |
| **LR Scheduler Gamma** | 0.70 | 0.5 | +40% |
| **LR Scheduler Step** | 8 | 10 | -20% |
| **Warmup Epochs** | 0 | N/A | - |
| **Label Smoothing** | 0.076 | 0.0 | New |

### Key Insights from Bayesian Optimization

1. **Higher Learning Rate**: A 55% higher learning rate (0.000155 vs 0.0001) proved beneficial
2. **Lower Regularization**: Lower dropout (0.333) and weight decay (0.000023) with label smoothing (0.076) as compensation
3. **Gradual LR Decay**: Higher gamma (0.70) provides more gradual learning rate decay
4. **More Frequent Decay**: Step size of 8 (vs 10) provides more frequent LR adjustments

### Model Checkpoints

| File | Description |
|------|-------------|
| `checkpoints/bayesian_swin_t/best_model_swin_t_bayesian.pth` | Best validation checkpoint |
| `checkpoints/bayesian_swin_t/final_model_swin_t_bayesian.pth` | Final epoch checkpoint |
| `results/bayesian_swin_t/bayesian_optimization_results.json` | All trial results |
| `results/bayesian_swin_t/training_history_bayesian.json` | Full training history |

---

## Standard Fine-Tuned Models Configuration

All models except the Bayesian-optimized Swin-T use the following standard configuration.

### Training Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| **Optimizer** | AdamW | `src/models/model_factory.py` |
| **Learning Rate** | 1e-4 (0.0001) | `src/utils/config.py` |
| **Weight Decay** | 1e-4 (0.0001) | `src/utils/config.py` |
| **Epochs** | 50 | `src/utils/config.py` |
| **Batch Size (CNNs)** | 32 | `train_all_models.py` |
| **Batch Size (ViT)** | 8 | `train_all_models.py` |
| **Batch Size (Swin-T)** | 16 | `train_all_models.py` |
| **Dropout Rate** | 0.5 | `src/utils/config.py` |
| **Early Stopping Patience** | 10 epochs | `src/utils/config.py` |
| **Pretrained Weights** | ImageNet | All models |

### Learning Rate Scheduler

| Parameter | Value |
|-----------|-------|
| **Scheduler Type** | StepLR |
| **Step Size** | 10 epochs |
| **Gamma (Decay Factor)** | 0.5 |

### Loss Function

| Parameter | Value |
|-----------|-------|
| **Loss Function** | CrossEntropyLoss |
| **Class Weighting** | Used for handling imbalance |

### Data Configuration

| Parameter | Value |
|-----------|-------|
| **Image Size** | 224 × 224 |
| **Train Split** | 70% |
| **Validation Split** | 15% |
| **Test Split** | 15% |
| **Random Seed** | 42 |

---

## Baseline Models Configuration

Baseline models are trained **from scratch** (without pretrained weights) for comparison purposes.

### Training Configuration

| Parameter | Value | Difference from Fine-Tuned |
|-----------|-------|---------------------------|
| **Pretrained Weights** | ❌ None (Random Init) | No transfer learning |
| **Optimizer** | Adam (simple) | Not AdamW |
| **Learning Rate** | 0.001 | 10× higher |
| **Weight Decay** | 0 | No L2 regularization |
| **Epochs** | 30 | 20 fewer epochs |
| **LR Scheduler** | ❌ None | No LR decay |
| **Early Stopping** | ❌ No | Trains all epochs |

### Baseline Models Trained

| Model | Checkpoint File |
|-------|-----------------|
| ResNet-50 | `checkpoints/baseline/best_model_resnet50_baseline.pth` |
| MobileNetV2 | `checkpoints/baseline/best_model_mobilenetv2_baseline.pth` |
| ViT-B/16 | `checkpoints/baseline/best_model_vit_b_16_baseline.pth` |
| Swin-T | `checkpoints/baseline/best_model_swin_t_baseline.pth` |
| DeiT-Small | `checkpoints/baseline/best_model_deit_small_baseline.pth` |
| MobileViT-S | `checkpoints/baseline/best_model_mobilevit_s_baseline.pth` |
| DenseNet-121 | `checkpoints/baseline/best_model_densenet121_baseline.pth` |
| EfficientNet-B0 | `checkpoints/baseline/best_model_efficientnet_b0_baseline.pth` |
| VGG-16 | `checkpoints/baseline/best_model_vgg16_baseline.pth` |

---

## Common Optimization Techniques

### Applied to All Models

| Technique | Description | Purpose |
|-----------|-------------|---------|
| **Transfer Learning** | ImageNet pretrained weights (fine-tuned only) | Leverage learned features |
| **Data Augmentation** | Random rotation (±15°), horizontal flip, scaling (0.9-1.1) | Prevent overfitting |
| **Stratified Sampling** | Maintain class distribution across splits | Handle class imbalance |
| **Dropout** | 0.5 probability in classifier head | Regularization |

### Applied to Fine-Tuned Models Only

| Technique | Description | Purpose |
|-----------|-------------|---------|
| **Early Stopping** | Patience of 10 epochs | Prevent overfitting |
| **StepLR Scheduler** | Reduce LR by 0.5× every 10 epochs | Improve convergence |
| **AdamW Optimizer** | Adam with decoupled weight decay | Better generalization |
| **Weight Decay** | 1e-4 L2 regularization | Regularization |

### Applied to Bayesian Swin-T Only

| Technique | Description | Purpose |
|-----------|-------------|---------|
| **Label Smoothing** | 0.076 smoothing factor | Soft regularization |
| **Learning Rate Warmup** | Gradual LR increase (0 epochs optimal) | Stable training start |
| **Adaptive Hyperparameters** | Optuna TPE optimization | Find optimal configuration |
| **Trial Pruning** | MedianPruner | Efficient search |

---

## Model Performance Summary

### Fine-Tuned Models (with ImageNet Pretrained Weights)

| Model | Test Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|---------------|-----------|--------|----------|------------|
| **Swin-T (Bayesian)** | 97.84% | 97.86% | 97.84% | 97.83% | ~28M |
| **MobileNetV2** | 97.40% | 97.5% | 97.4% | 97.4% | ~2.2M |
| **ResNet-50** | 96.97% | 97.0% | 97.0% | 97.0% | ~23.5M |
| Swin-T (Standard) | 97.84% | 97.86% | 97.84% | 97.84% | ~28M |

### Baseline Models (without Pretrained Weights)

| Model | Test Accuracy | Precision | Recall | F1-Score |
|-------|---------------|-----------|--------|----------|
| **MobileNetV2** | 89.61% | 89.9% | 89.6% | 89.4% |
| **DenseNet-121** | 84.42% | 85.7% | 84.4% | 82.6% |
| **ResNet-50** | 78.79% | 79.4% | 78.8% | 79.0% |
| **EfficientNet-B0** | 72.29% | 73.6% | 72.3% | 72.6% |
| **VGG-16** | 71.43% | 69.8% | 71.4% | 69.4% |

### Transfer Learning Impact

| Model | Baseline | Fine-Tuned | Improvement |
|-------|----------|------------|-------------|
| **ResNet-50** | 78.79% | 96.97% | **+18.18%** |
| **MobileNetV2** | 89.61% | 97.40% | **+7.79%** |

---

## Recommendations

### For Future Model Improvements

1. **Extend Bayesian Optimization**: Apply Optuna-based hyperparameter tuning to other high-performing models (MobileNetV2, ResNet-50)

2. **Grid Search Alternative**: For models where Bayesian optimization is too time-consuming, consider a targeted grid search over:
   - Learning rate: [1e-5, 5e-5, 1e-4, 5e-4]
   - Weight decay: [1e-5, 1e-4, 1e-3]
   - Dropout: [0.3, 0.4, 0.5]

3. **Advanced Techniques to Consider**:
   - Mixup/CutMix augmentation
   - Cosine annealing with warm restarts
   - Stochastic Weight Averaging (SWA)
   - Knowledge distillation from best model

4. **Ensemble Methods**: Combine predictions from top-3 models (Swin-T, MobileNetV2, ResNet-50) for potentially higher accuracy

---

## References

### Key Files

| File | Purpose |
|------|---------|
| `src/utils/config.py` | Centralized configuration |
| `src/models/model_factory.py` | Model creation and optimizer setup |
| `train_all_models.py` | Multi-model training script |
| `train_baseline_models.py` | Baseline training script |
| `archive_transformer_files/train_swin_bayesian.py` | Bayesian optimization script |

---

*Document Generated: March 2026*  
*LungXAI - Lung Cancer Classification with Explainable AI*
