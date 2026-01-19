# Notebooks Directory

This directory contains Jupyter notebooks for the LungXAI project.

## Available Notebooks

| Notebook | Description |
|----------|-------------|
| `05_full_pipeline.ipynb` | Data exploration and basic pipeline testing |
| `LungXAI_Complete_Pipeline.ipynb` | Complete end-to-end demo with all models |

## Model Performance Summary

After training all models on the Lung Cancer CT Scan Dataset:

| Model | Test Accuracy | Parameters | Best For |
|-------|---------------|------------|----------|
| **Swin-T** | **97.84%** | ~28M | Best overall accuracy |
| MobileNetV2 | 97.40% | ~3.5M | Edge deployment |
| ResNet-50 | 96.97% | ~25.6M | Explainability (Grad-CAM) |
| ViT-B/16 | 93.51% | ~86M | Research/global features |

## Key Features Demonstrated

- Multi-class lung cancer classification (5 classes)
- Grad-CAM visual explanations
- RAG-based textual explanations with PubMed integration
- Model comparison and selection
