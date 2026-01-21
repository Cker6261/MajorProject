# 🫁 LungXAI - Explainable AI for Lung Cancer Classification

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Bridging the gap between AI predictions and clinical understanding through visual and textual explanations**

An advanced deep learning system that classifies lung CT scan images into cancer types while providing explainable AI insights through visual heatmaps and medical knowledge retrieval.

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

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/LungXAI.git
cd LungXAI

# Create and activate virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (see Dataset section below)
# Then train all models
python train_all_models.py

# Run demo with all models
python demo_multi_model.py --compare
```

## 📊 Key Results

| Model | Test Accuracy | Parameters | Best For |
|-------|---------------|------------|----------|
| **Swin-T** 🏆 | **97.84%** | ~28M | **Best overall performance** |
| MobileNetV2 ⚡ | 97.40% | ~3.5M | Edge deployment |
| ResNet-50 🔍 | 96.97% | ~25.6M | Excellent explainability |
| ViT-B/16 🧠 | 93.51% | ~86M | Research applications |

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

## � Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- At least 8GB RAM
- 10GB free disk space

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/LungXAI.git
cd LungXAI
```

### Step 2: Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows CMD
.venv\Scripts\activate.bat
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📊 Dataset

**Source**: [CT Scan Images of Lung Cancer Patients](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) (Kaggle)

### Download Instructions
1. Visit the Kaggle dataset link above
2. Download and extract the dataset
3. Place in the project directory as:
```
LungXAI/
└── archive (1)/
    └── Lung Cancer Dataset/
        ├── adenocarcinoma/
        ├── Benign cases/
        ├── large cell carcinoma/
        ├── Normal cases/
        └── squamous cell carcinoma/
```

### Dataset Statistics
| Class | Images | Description |
|-------|--------|-------------|
| Adenocarcinoma | ~150 | Most common lung cancer type |
| Squamous Cell | ~150 | Central lung areas |
| Large Cell | ~150 | Fast-growing type |
| Benign | ~150 | Non-cancerous tissue |
| Normal | ~150 | Healthy lung tissue |

---

## 🎮 Usage

### 🚀 Training All Models (Recommended)

```bash
# Train all models with caching (skip already trained models)
python train_all_models.py

# Force retrain all models
python train_all_models.py --force-retrain

# Train specific models only
python train_all_models.py --models resnet50 mobilenetv2
```

### 🔍 Demo & Inference

```bash
# Demo with default model (ResNet-50)
python demo_multi_model.py

# Demo with specific model
python demo_multi_model.py --model swin_t        # Best accuracy
python demo_multi_model.py --model mobilenetv2  # Fastest
python demo_multi_model.py --model vit_b_16      # Research

# Compare all models on same image
python demo_multi_model.py --compare

# List available models and training status
python demo_multi_model.py --list

# Visual demo with custom image
python demo.py path/to/your/ct_scan.png
```

### 📊 Model Comparison & Evaluation

```bash
# Generate comprehensive model comparison
python compare_models.py

# Evaluate specific model
python evaluate_model.py --model swin_t

# Test RAG explanation system
python test_rag_pipeline.py
```

### 🔧 Individual Operations

```bash
# Train single model (legacy method)
python main.py --mode train --epochs 30

# Evaluate on test set
python main.py --mode evaluate --checkpoint checkpoints/best_model_resnet50.pth

# Predict single image
python main.py --mode predict --image path/to/ct_scan.png

# Run interactive demo
python main.py --mode demo
```

---

## 📊 Model Performance

*Results after training on the Lung Cancer CT Scan Dataset (5 classes)*

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Parameters |
|-------|----------|-----------|--------|----------|---------------|------------|
| **Swin-T** 🥇 | **97.84%** | **97.86%** | **97.84%** | **97.84%** | ~28 min | 28M |
| MobileNetV2 🥈 | **97.40%** | **97.50%** | **97.40%** | **97.40%** | ~17 min | 3.5M |
| ResNet-50 🥉 | 96.97% | 96.99% | 96.97% | 96.95% | ~7 min | 25.6M |
| ViT-B/16 | 93.51% | 93.74% | 93.51% | 93.48% | ~80 min | 86M |

### 🎯 Model Selection Guide

- **🏆 Best Overall**: **Swin Transformer (Tiny)** - Highest accuracy with reasonable training time
- **⚡ Deployment**: **MobileNetV2** - Excellent accuracy-to-efficiency ratio
- **🔍 Explainability**: **ResNet-50** - Superior Grad-CAM visualizations
- **🧠 Research**: **ViT-B/16** - Cutting-edge transformer architecture

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Start for Contributors
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 🐛 Known Issues & Limitations

1. **Dataset Size**: Limited medical imaging data may affect generalization
2. **Grad-CAM**: Shows correlation, not causation; may highlight spurious features
3. **RAG Simplicity**: Current implementation uses keyword matching; semantic search would be better
4. **Clinical Validation**: Not validated by medical professionals - **NOT FOR CLINICAL USE**
5. **GPU Memory**: Large models (ViT) require significant GPU memory

---

## 🔮 Roadmap & Future Enhancements

### Planned Features
- [ ] **Semantic RAG**: Upgrade to sentence transformers for better knowledge retrieval
- [ ] **Multiple XAI Methods**: Add LIME, SHAP for comprehensive explanations
- [ ] **Web Interface**: User-friendly web application for easier access
- [ ] **Larger Dataset**: Integration with additional medical image datasets
- [ ] **Clinical Validation**: Collaboration with medical professionals

### Technical Improvements
- [ ] **Model Ensemble**: Combine predictions from multiple models
- [ ] **Real-time Inference**: Optimize for faster prediction times
- [ ] **Cloud Deployment**: Docker containerization and cloud deployment guides
- [ ] **Mobile App**: Mobile application for edge deployment

---

## ⚠️ Important Disclaimers

> **⚠️ NOT FOR CLINICAL USE**: This project is for research and educational purposes only. It has not been validated by medical professionals and should not be used for actual medical diagnosis or treatment decisions.

> **📚 Academic Use**: This project is developed for academic research and learning purposes. Always consult qualified medical professionals for health-related decisions.

---

## 📚 Documentation

- [**Complete User Guide**](docs/PROJECT_REVIEW_GUIDE.md) - Comprehensive project documentation
- [**Command Reference**](COMMANDS.md) - All available commands and usage examples
- [**Research Paper**](docs/LungXAI_Research_Paper.md) - Academic paper with technical details
- [**Pipeline Architecture**](docs/PSEUDOCODE.md) - Technical implementation details
- [**Model Comparison**](docs/BASELINE_VS_FINETUNED_COMPARISON.md) - Detailed model analysis

---

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@misc{lungxai2024,
  title={Explainable AI for Multi-Class Lung Cancer Classification Using Deep Learning and RAG-Based Knowledge Retrieval},
  author={Major Project Team},
  year={2024},
  howpublished={\url{https://github.com/yourusername/LungXAI}}
}
```

---

## 📚 References

1. **Selvaraju, R. R., et al.** (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV 2017*.
2. **He, K., et al.** (2016). "Deep Residual Learning for Image Recognition." *CVPR 2016*.
3. **Lewis, P., et al.** (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.
4. **Liu, Z., et al.** (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *ICCV 2021*.
5. **Dosovitskiy, A., et al.** (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/LungXAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/LungXAI/discussions)
- **Email**: your.email@university.edu

---

## 👥 Authors & Contributors

**Major Project Team** - *Final Year B.Tech Computer Science*
- Lead Developer: [Your Name]
- Contributors: [Team Member 1], [Team Member 2]

See the full list of [contributors](https://github.com/yourusername/LungXAI/contributors) who participated in this project.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Academic Use Encouraged**: This project is developed for educational and research purposes. We encourage its use in academic settings with proper attribution.

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/LungXAI&type=Date)](https://star-history.com/#yourusername/LungXAI&Date)

---

## 🙏 Acknowledgments

- **Kaggle Community** for providing the lung cancer CT scan dataset
- **PyTorch Team** for the excellent deep learning framework
- **Hugging Face** for transformer models and tools
- **Scientific Community** for open-source medical AI research
- **University Faculty** for guidance and support

---

<div align="center">

**Made with ❤️ for the advancement of medical AI research**

[⭐ Star this repo](https://github.com/yourusername/LungXAI/stargazers) | [🐛 Report Bug](https://github.com/yourusername/LungXAI/issues) | [✨ Request Feature](https://github.com/yourusername/LungXAI/issues)

</div> 
