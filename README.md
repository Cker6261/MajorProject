# Explainable AI for Multi-Class Lung Cancer Classification

## Using Deep Learning and RAG-Based Knowledge Retrieval

---

## 🎯 Project Overview

This project implements an **explainable AI system** for classifying lung CT images into four categories:
- **Adenocarcinoma** - Most common type of lung cancer
- **Squamous Cell Carcinoma** - Often found in central lung areas
- **Large Cell Carcinoma** - Fast-growing cancer type
- **Normal/Benign** - Healthy lung tissue

### What Makes This Project Unique?

Traditional medical AI systems provide predictions but lack interpretability. This project bridges that gap by:

1. **Classification**: Using deep learning (ResNet-50) to classify CT images
2. **Visual Explanation**: Generating Grad-CAM heatmaps to show WHERE the model is looking
3. **Textual Explanation**: Using RAG to explain WHY those regions are significant

---

## 📁 Project Structure

```
Major Project/
│
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── src/                   # Source code
│   ├── __init__.py
│   │
│   ├── data/              # Data handling
│   │   ├── __init__.py
│   │   ├── dataset.py     # Custom PyTorch Dataset
│   │   ├── transforms.py  # Image augmentation
│   │   └── dataloader.py  # DataLoader utilities
│   │
│   ├── models/            # Neural network models
│   │   ├── __init__.py
│   │   ├── classifier.py  # Main classification model
│   │   └── model_factory.py
│   │
│   ├── xai/               # Explainable AI
│   │   ├── __init__.py
│   │   ├── gradcam.py     # Grad-CAM implementation
│   │   └── visualize.py   # Visualization utilities
│   │
│   ├── rag/               # RAG Pipeline
│   │   ├── __init__.py
│   │   ├── knowledge_base.py      # Medical knowledge store
│   │   ├── xai_to_text.py         # XAI → Text conversion
│   │   └── explanation_generator.py
│   │
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── config.py      # Centralized configuration
│       ├── helpers.py     # Helper functions
│       └── metrics.py     # Evaluation metrics
│
├── notebooks/             # Jupyter notebooks for experiments
│
├── dataset/               # Dataset directory (not in repo)
│   ├── adenocarcinoma/
│   ├── squamous_cell_carcinoma/
│   ├── large_cell_carcinoma/
│   └── normal/
│
├── checkpoints/           # Saved model checkpoints
│
└── results/               # Output results and visualizations
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
│                    RESNET-50 (Pretrained, Fine-tuned)                │
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

```bash
# Train the model
python main.py --mode train --epochs 10

# Evaluate on test set
python main.py --mode evaluate --checkpoint checkpoints/best_model.pth

# Predict single image
python main.py --mode predict --image path/to/ct_scan.png

# Run demo pipeline
python main.py --mode demo
```

---

## 📈 Results

*To be updated after training*

| Metric | Value |
|--------|-------|
| Accuracy | - |
| Precision | - |
| Recall | - |
| F1 Score | - |

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
