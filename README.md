# Explainable AI for Multi-Class Lung Cancer Classification

## Using Deep Learning and RAG-Based Knowledge Retrieval

---

## 🎯 Project Overview

This project implements an **explainable AI system** for classifying lung CT images into five categories:
- **Adenocarcinoma** - Most common type of lung cancer
- **Squamous Cell Carcinoma** - Often found in central lung areas
- **Large Cell Carcinoma** - Fast-growing cancer type
- **Benign Cases** - Non-cancerous tissue
- **Normal Cases** - Healthy lung tissue

### What Makes This Project Unique?

Traditional medical AI systems provide predictions but lack interpretability. This project bridges that gap by:

1. **Multi-Model Classification**: Comparing 4 different deep learning architectures:
   - **ResNet-50**: Classic residual network (96.97% accuracy)
   - **MobileNetV2**: Lightweight model for deployment (97.40% accuracy)
   - **Vision Transformer (ViT)**: Attention-based architecture (93.51% accuracy)
   - **Swin Transformer**: Hierarchical transformer with shifted windows (97.84% accuracy - **Best!**)

2. **Visual Explanation**: Generating Grad-CAM heatmaps to show WHERE the model is looking
3. **Textual Explanation**: Using RAG to explain WHY those regions are significant
4. **Model Comparison**: Built-in tools to compare all models and select the best one

### Key Features

- **Caching Support**: Models are cached after training - no retraining needed!
- **Multi-Model Training**: Train all models with a single command
- **Automatic Comparison**: Generate comparison charts and reports
- **Memory Efficient**: Sequential training to prevent GPU memory issues
- **D: Drive Storage**: All data stored on D: drive to prevent C: drive issues

---

## 📁 Project Structure

```
Major Project/
│
├── main.py                    # Main entry point
├── train_all_models.py        # Train all models with caching
├── compare_models.py          # Compare all trained models
├── demo_multi_model.py        # Demo with model selection
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── src/                       # Source code
│   ├── __init__.py
│   │
│   ├── data/                  # Data handling
│   │   ├── __init__.py
│   │   ├── dataset.py         # Custom PyTorch Dataset
│   │   ├── transforms.py      # Image augmentation
│   │   └── dataloader.py      # DataLoader utilities
│   │
│   ├── models/                # Neural network models
│   │   ├── __init__.py
│   │   ├── classifier.py      # ResNet-50 classifier
│   │   └── model_factory.py   # Factory for all models
│   │
│   ├── xai/                   # Explainable AI
│   │   ├── __init__.py
│   │   ├── gradcam.py         # Grad-CAM implementation
│   │   └── visualize.py       # Visualization utilities
│   │
│   ├── rag/                   # RAG Pipeline
│   │   ├── __init__.py
│   │   ├── knowledge_base.py        # Medical knowledge store
│   │   ├── pubmed_retriever.py      # PubMed API integration
│   │   ├── xai_to_text.py           # XAI → Text conversion
│   │   └── explanation_generator.py # Full explanation generation
│   │
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── config.py          # Centralized configuration
│       ├── helpers.py         # Helper functions
│       └── metrics.py         # Evaluation metrics
│
├── checkpoints/               # Saved model checkpoints
│   ├── best_model_resnet50.pth
│   ├── best_model_mobilenetv2.pth
│   ├── best_model_vit_b_16.pth
│   └── best_model_swin_t.pth
│
├── results/                   # Output results
│   ├── comparison/            # Model comparison charts
│   ├── resnet50/              # ResNet-50 specific results
│   ├── mobilenetv2/           # MobileNetV2 specific results
│   ├── vit_b_16/              # ViT specific results
│   └── swin_t/                # Swin Transformer specific results
│
└── archive (1)/               # Dataset
    └── Lung Cancer Dataset/
        ├── adenocarcinoma/
        ├── Benign cases/
        ├── large cell carcinoma/
        ├── Normal cases/
        └── squamous cell carcinoma/
```

### Why This Structure?

| Directory | Purpose | Academic Justification |
|-----------|---------|----------------------|
| `src/data/` | Data loading & preprocessing | Separates data concerns from model logic |
| `src/models/` | Neural network architectures | Allows easy model comparison |
| `src/xai/` | Explainability methods | Isolates XAI implementation for clarity |
| `src/rag/` | Knowledge retrieval | Novel contribution - bridges XAI to explanations |
| `src/utils/` | Common utilities | Reduces code duplication |
| `notebooks/` | Experiments | Interactive development and visualization |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: CT SCAN IMAGE                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING (224x224, Normalize)                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              DEEP LEARNING MODEL (Pretrained, Fine-tuned)            │
│                                                                       │
│    ┌──────────────┐                          ┌──────────────┐        │
│    │   Features   │ ──────────────────────── │  Prediction  │        │
│    │   (layer4)   │                          │   (4 class)  │        │
│    └──────────────┘                          └──────────────┘        │
│           │                                         │                 │
└───────────│─────────────────────────────────────────│─────────────────┘
            │                                         │
            ▼                                         ▼
┌──────────────────────┐                    ┌──────────────────────┐
│      GRAD-CAM        │                    │  CLASS PREDICTION    │
│    (Visual XAI)      │                    │  + Confidence Score  │
└──────────────────────┘                    └──────────────────────┘
            │                                         │
            ▼                                         │
┌──────────────────────┐                              │
│  XAI → TEXT BRIDGE   │                              │
│  "peripheral opacity"│                              │
└──────────────────────┘                              │
            │                                         │
            ▼                                         │
┌──────────────────────┐                              │
│  KNOWLEDGE RETRIEVAL │                              │
│   (Medical Facts)    │                              │
└──────────────────────┘                              │
            │                                         │
            └────────────────────┬────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FINAL OUTPUT                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ Prediction  │  │  Grad-CAM   │  │      RAG Explanation        │  │
│  │Adenocarcinoma│  │  Heatmap    │  │ "Ground-glass opacity..."  │  │
│  │   (92%)     │  │             │  │                             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Major Project"

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset

**Source**: Kaggle - CT Scan Images of Lung Cancer Patients

**Structure**: Place the dataset in the `dataset/` folder with the following structure:
```
dataset/
├── adenocarcinoma/
├── squamous_cell_carcinoma/
├── large_cell_carcinoma/
└── normal/
```

**Download**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)

---

## 🎮 Usage

### Training All Models

```bash
# Train all models (ResNet-50, MobileNetV2, ViT, Swin Transformer)
# Uses caching - already trained models are skipped automatically
python train_all_models.py

# Force retrain all models
python train_all_models.py --force-retrain

# Train specific models only
python train_all_models.py --models resnet50 mobilenetv2
```

### Model Comparison

```bash
# Compare all trained models
python compare_models.py

# This generates:
# - results/comparison/model_comparison_charts.png
# - results/comparison/model_comparison_radar.png
# - results/comparison/confusion_matrices_comparison.png
# - results/comparison/model_comparison_report.md
```

### Demo with Model Selection

```bash
# Demo with default model (ResNet-50)
python demo_multi_model.py

# Demo with specific model
python demo_multi_model.py --model mobilenetv2
python demo_multi_model.py --model vit_b_16
python demo_multi_model.py --model swin_t

# Compare all models on same image
python demo_multi_model.py --compare

# List available models and training status
python demo_multi_model.py --list
```

### Visual Demo

```bash
# Run visual demo with Grad-CAM visualization
python demo.py
python demo.py path/to/your/image.png
```

### Legacy Commands

```bash
# Train single model (legacy)
python main.py --mode train --epochs 10

# Evaluate on test set
python main.py --mode evaluate --checkpoint checkpoints/best_model_resnet50.pth

# Predict single image
python main.py --mode predict --image path/to/ct_scan.png
```

---

## 📊 Model Comparison

| Model | Parameters | Test Acc | Description | Best For |
|-------|-----------|----------|-------------|----------|
| ResNet-50 | ~25.6M | 96.97% | Deep residual network with skip connections | Default choice, excellent Grad-CAM visualizations |
| MobileNetV2 | ~3.5M | 97.40% | Lightweight network with inverted residuals | Deployment, edge devices, mobile apps |
| ViT-B/16 | ~86M | 93.51% | Attention-based transformer architecture | Research, capturing global image features |
| **Swin-T** | ~28M | **97.84%** | Hierarchical transformer with shifted windows | **Best accuracy**, production deployment |

---

## 📈 Results

*Results after training all models on the Lung Cancer CT Scan Dataset*

| Model | Test Accuracy | Precision | Recall | F1 Score | Training Time |
|-------|---------------|-----------|--------|----------|---------------|
| ResNet-50 | 96.97% | 96.99% | 96.97% | 96.95% | ~7 min |
| **MobileNetV2** | **97.40%** | **97.50%** | **97.40%** | **97.40%** | ~17 min |
| ViT-B/16 | 93.51% | 93.74% | 93.51% | 93.48% | ~80 min |
| Swin-T | 97.84% | 97.86% | 97.84% | 97.84% | ~28 min |

### 🏆 Best Model: **Swin Transformer (Tiny)** with 97.84% accuracy

**Key Findings:**
- **Swin-T** achieved the highest test accuracy (97.84%) with excellent precision and recall
- **MobileNetV2** offers the best accuracy-to-efficiency ratio with only 3.5M parameters
- **ResNet-50** provides reliable performance with excellent Grad-CAM visualizations
- **ViT-B/16** requires more data/training time but captures global features well

---

## ⚠️ Limitations

1. **Dataset Size**: Limited medical imaging data may affect generalization
2. **Grad-CAM**: Shows correlation, not causation; may highlight spurious features
3. **RAG Simplicity**: Keyword matching is basic; doesn't capture semantic meaning
4. **Clinical Validation**: Not validated by medical professionals

---

## 🔮 Future Enhancements

1. **Semantic RAG**: Upgrade to sentence transformers for better retrieval
2. **Multiple XAI Methods**: Add LIME, SHAP for comparison
3. **Larger Dataset**: Include more diverse CT scan sources
4. **Clinical Validation**: Partner with radiologists for validation
5. **Web Interface**: Build a user-friendly web application

---

## 📚 References

1. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017.
2. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR 2016.
3. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.

---

## 👥 Authors

Major Project Team - Final Year B.Tech

---

## 📄 License

This project is for academic purposes only. 
