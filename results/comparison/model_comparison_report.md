# Lung Cancer Classification - Model Comparison Report

**Generated:** 2026-01-18 20:32:33

## Summary

| Model | Test Accuracy | Precision | Recall | F1 Score | Training Time |
|-------|--------------|-----------|--------|----------|---------------|
| ResNet-50 | 96.97% | 96.99% | 96.97% | 96.95% | ~7 min |
| MobileNetV2 | 97.40% | 97.50% | 97.40% | 97.40% | ~17 min |
| Vision Transformer (ViT-B/16) | 93.51% | 93.74% | 93.51% | 93.48% | ~80 min |
| Swin Transformer (Tiny) | 97.84% | 97.86% | 97.84% | 97.84% | ~28 min |

## Best Model: **Swin Transformer (Tiny)** (97.84% accuracy)

## Model Details

### ResNet-50

- **Parameters:** ~25.6M
- **Description:** Deep residual network with skip connections
- **Test Accuracy:** 96.97%
- **Test Loss:** 0.0849
- **Precision:** 96.99%
- **Recall:** 96.97%
- **F1 Score:** 96.95%
- **Training Time:** ~7 minutes

### MobileNetV2

- **Parameters:** ~3.5M
- **Description:** Lightweight network with inverted residuals
- **Test Accuracy:** 97.40%
- **Test Loss:** 0.1202
- **Precision:** 97.50%
- **Recall:** 97.40%
- **F1 Score:** 97.40%
- **Training Time:** ~17 minutes

### Vision Transformer (ViT-B/16)

- **Parameters:** ~86M
- **Description:** Attention-based transformer architecture
- **Test Accuracy:** 93.51%
- **Test Loss:** 0.2466
- **Precision:** 93.74%
- **Recall:** 93.51%
- **F1 Score:** 93.48%
- **Training Time:** ~80 minutes

### Swin Transformer (Tiny)

- **Parameters:** ~28M
- **Description:** Hierarchical transformer with shifted windows
- **Test Accuracy:** 97.84%
- **Test Loss:** 0.1469
- **Precision:** 97.86%
- **Recall:** 97.84%
- **F1 Score:** 97.84%
- **Training Time:** ~28 minutes

## Recommendations

Based on the comparison results:

1. **Best Overall Performance:** Swin Transformer (Tiny) with 97.84% accuracy
2. **Best Accuracy-Efficiency Ratio:** MobileNetV2 (97.40% with only 3.5M parameters)
3. **For Deployment:** MobileNetV2 for resource-constrained/edge environments
4. **For Explainability:** ResNet-50 works best with Grad-CAM visualizations
5. **For Research:** Vision Transformer provides novel attention-based approach

## Key Insights

- **Swin Transformer** achieved the highest accuracy due to its hierarchical attention mechanism
- **MobileNetV2** is highly efficient with minimal accuracy trade-off (only 0.44% less than Swin-T)
- **ResNet-50** provides consistent, reliable results with excellent interpretability
- **ViT-B/16** requires more training data and compute but captures global features well
